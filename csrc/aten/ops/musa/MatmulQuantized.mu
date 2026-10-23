#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/quantized/mudnn/Qlinear.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include "musa.h"
#include "musa_runtime.h"
#include "musa_runtime_api.h"

namespace at::native {
template <typename T>
__global__ void W8A8MatmulKernel_Float(
    const uint8_t* A,
    const int8_t* B,
    T* C,
    int M,
    int N,
    int K,
    float A_scale,
    int A_zeropoint,
    float* B_scale,
    int32_t* B_zeropoint,
    float C_scale,
    int C_zeropoint,
    const float* bias_data,
    bool use_relu) {
  const int start_col = threadIdx.x + blockIdx.x * blockDim.x; // col c
  const int start_row = threadIdx.y + blockIdx.y * blockDim.y; // row c
  const int stride_x = blockDim.x * gridDim.x;
  const int stride_y = blockDim.y * gridDim.y;
  for (int row = start_row; row < M; row += stride_y) {
    for (int col = start_col; col < N; col += stride_x) {
      int32_t acc = 0;
      for (int i = 0; i < K; i++) {
        int32_t a_val = (int32_t)A[row * K + i] - A_zeropoint;
        int32_t b_val = (int32_t)B[i * N + col] - B_zeropoint[col];
        acc += a_val * b_val;
      }
      float C_float = (float)acc * A_scale * B_scale[col];
      if (bias_data) {
        C_float = C_float + bias_data[col];
      }
      if (use_relu) {
        C_float = fmaxf(0.0f, C_float);
      }
      C[row * N + col] = static_cast<T>(C_float);
    }
  }
}

template <typename T>
__global__ void W8A8MatmulKernel_UINT(
    const uint8_t* A,
    const int8_t* B,
    T* C,
    int M,
    int N,
    int K,
    float A_scale,
    int A_zeropoint,
    float* B_scale,
    int32_t* B_zeropoint,
    float C_scale,
    int C_zeropoint,
    float* bias_data,
    bool use_relu) {
  const int start_col = threadIdx.x + blockIdx.x * blockDim.x; // col c
  const int start_row = threadIdx.y + blockIdx.y * blockDim.y; // row c
  const int stride_x = blockDim.x * gridDim.x;
  const int stride_y = blockDim.y * gridDim.y;
  for (int row = start_row; row < M; row += stride_y) {
    for (int col = start_col; col < N; col += stride_x) {
      int32_t acc = 0;
      for (int i = 0; i < K; i++) {
        int32_t a_val = (int32_t)A[row * K + i] - A_zeropoint;
        int32_t b_val = (int32_t)B[i * N + col] - B_zeropoint[col];
        acc += a_val * b_val;
      }
      float C_float = (float)acc * A_scale * B_scale[col];
      if (bias_data) {
        C_float = C_float + bias_data[col];
      }
      if (use_relu) {
        C_float = fmaxf(0.0, C_float);
      }
      float quantized =
          roundf(C_float / C_scale) + static_cast<float>(C_zeropoint);
      float min_bound = 0.0f;
      float max_bound = 255.0f;
      float clamped = fmaxf(min_bound, fminf(quantized, max_bound));
      C[row * N + col] = static_cast<T>(clamped);
    }
  }
}

template <typename T>
__global__ void W8A8MatmulKernel_INT(
    const uint8_t* A,
    const int8_t* B,
    T* C,
    int M,
    int N,
    int K,
    float A_scale,
    int A_zeropoint,
    float* B_scale,
    int32_t* B_zeropoint,
    float C_scale,
    int C_zeropoint,
    float* bias_data,
    bool use_relu) {
  const int start_col = threadIdx.x + blockIdx.x * blockDim.x; // col c
  const int start_row = threadIdx.y + blockIdx.y * blockDim.y; // row c
  const int stride_x = blockDim.x * gridDim.x;
  const int stride_y = blockDim.y * gridDim.y;
  for (int row = start_row; row < M; row += stride_y) {
    for (int col = start_col; col < N; col += stride_x) {
      int32_t acc = 0;
      for (int i = 0; i < K; i++) {
        int32_t a_val = (int32_t)A[row * K + i] - A_zeropoint;
        int32_t b_val = (int32_t)B[i * N + col] - B_zeropoint[col];
        acc += a_val * b_val;
      }
      float C_float = (float)acc * A_scale * B_scale[col];
      if (bias_data) {
        C_float = C_float + bias_data[col];
      }
      if (use_relu) {
        C_float = fmaxf(0.0, C_float);
      }
      float quantized =
          roundf(C_float / C_scale) + static_cast<float>(C_zeropoint);
      float min_bound = -128.0f;
      float max_bound = 127.0f;
      float clamped = fmaxf(min_bound, fminf(quantized, max_bound));
      C[row * N + col] = static_cast<T>(clamped);
    }
  }
}

#define GEN_MATMUL_FUNC(_TYPE)                                     \
  [](int M,                                                        \
     int N,                                                        \
     int K,                                                        \
     at::Tensor mat1,                                              \
     double input_scale,                                           \
     int64_t input_zero_point,                                     \
     at::Tensor mat2,                                              \
     at::Tensor weight_scales,                                     \
     at::Tensor weight_zero_points,                                \
     at::Tensor bias,                                              \
     at::Tensor& result,                                           \
     double output_scale,                                          \
     int64_t output_zero_point,                                    \
     bool use_relu) {                                              \
    auto stream = c10::musa::getCurrentMUSAStream();               \
    dim3 block(32, 32);                                            \
    int elements = M * N;                                          \
    int grid_x = std::min<int>((N + block.x - 1) / block.x, 4096); \
    int grid_y = std::min<int>((M + block.y - 1) / block.y, 4096); \
    dim3 grid(grid_x, grid_y);                                     \
    float* bias_ptr = nullptr;                                     \
    if (bias.defined() && bias.numel() > 0) {                      \
      bias_ptr = bias.data_ptr<float>();                           \
    }                                                              \
    if constexpr (std::is_same_v<_TYPE, int8_t>) {                 \
      W8A8MatmulKernel_INT<_TYPE><<<grid, block, 0, stream>>>(     \
          static_cast<const uint8_t*>(mat1.data_ptr()),            \
          static_cast<const int8_t*>(mat2.data_ptr()),             \
          static_cast<_TYPE*>(result.data_ptr()),                  \
          M,                                                       \
          N,                                                       \
          K,                                                       \
          (float)input_scale,                                      \
          (int)input_zero_point,                                   \
          static_cast<float*>(weight_scales.data_ptr()),           \
          static_cast<int32_t*>(weight_zero_points.data_ptr()),    \
          (float)output_scale,                                     \
          (int)output_zero_point,                                  \
          bias_ptr,                                                \
          use_relu);                                               \
    } else if constexpr (std::is_same_v<_TYPE, uint8_t>) {         \
      W8A8MatmulKernel_UINT<_TYPE><<<grid, block, 0, stream>>>(    \
          static_cast<const uint8_t*>(mat1.data_ptr()),            \
          static_cast<const int8_t*>(mat2.data_ptr()),             \
          static_cast<_TYPE*>(result.data_ptr()),                  \
          M,                                                       \
          N,                                                       \
          K,                                                       \
          (float)input_scale,                                      \
          (int)input_zero_point,                                   \
          static_cast<float*>(weight_scales.data_ptr()),           \
          static_cast<int32_t*>(weight_zero_points.data_ptr()),    \
          (float)output_scale,                                     \
          (int)output_zero_point,                                  \
          bias_ptr,                                                \
          use_relu);                                               \
    } else {                                                       \
      W8A8MatmulKernel_Float<_TYPE><<<grid, block, 0, stream>>>(   \
          static_cast<const uint8_t*>(mat1.data_ptr()),            \
          static_cast<const int8_t*>(mat2.data_ptr()),             \
          static_cast<_TYPE*>(result.data_ptr()),                  \
          M,                                                       \
          N,                                                       \
          K,                                                       \
          (float)input_scale,                                      \
          (int)input_zero_point,                                   \
          static_cast<float*>(weight_scales.data_ptr()),           \
          static_cast<int32_t*>(weight_zero_points.data_ptr()),    \
          (float)output_scale,                                     \
          (int)output_zero_point,                                  \
          bias_ptr,                                                \
          use_relu);                                               \
    }                                                              \
  }

// type to CTYPE
#define REGISTER_KERNEL(_TYPE, _CTYPE) \
  matmul_kernels[(int)_TYPE] = GEN_MATMUL_FUNC(_CTYPE);

struct MatmulKernelTable {
  using KernelFunc = std::function<void(
      int,
      int,
      int,
      at::Tensor,
      double,
      int64_t,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor&,
      double,
      int64_t,
      bool use_relu)>;

#define REGISTER_KERNEL_DTYPE                       \
  REGISTER_KERNEL(at::ScalarType::Half, float16_t); \
  REGISTER_KERNEL(at::ScalarType::Float, float);    \
  REGISTER_KERNEL(at::ScalarType::Char, int8_t);    \
  REGISTER_KERNEL(at::ScalarType::Byte, uint8_t);

  MatmulKernelTable() {
    REGISTER_KERNEL_DTYPE;
  }

#undef REGISTER_KERNEL_DTYPE

  void launch(
      int M,
      int N,
      int K,
      at::Tensor mat1, // act
      double input_scale,
      int64_t input_zero_point,
      at::Tensor mat2, // weight
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor bias,
      at::Tensor& result, // output
      double output_scale,
      int64_t output_zero_point,
      c10::ScalarType output_dtype,
      const std::string_view& unary_post_op) {
    auto& func = matmul_kernels[(int)output_dtype];
    if (unary_post_op == "relu") {
      func(
          M,
          N,
          K,
          mat1,
          input_scale,
          input_zero_point,
          mat2,
          weight_scales,
          weight_zero_points,
          bias,
          result,
          output_scale,
          output_zero_point,
          true);
    } else {
      func(
          M,
          N,
          K,
          mat1,
          input_scale,
          input_zero_point,
          mat2,
          weight_scales,
          weight_zero_points,
          bias,
          result,
          output_scale,
          output_zero_point,
          false);
    }
  }

  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  KernelFunc matmul_kernels[nr_dtype];
};

void matmul_kernel(
    int M,
    int N,
    int K,
    at::Tensor mat1, // act
    double input_scale,
    int64_t input_zero_point,
    at::Tensor mat2, // weight
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    at::Tensor bias,
    at::Tensor& result, // output
    double output_scale,
    int64_t output_zero_point,
    c10::ScalarType output_dtype,
    const std::string_view& unary_post_op) {
  MatmulKernelTable kernels;
  kernels.launch(
      M,
      N,
      K,
      mat1,
      input_scale,
      input_zero_point,
      mat2,
      weight_scales,
      weight_zero_points,
      bias,
      result,
      output_scale,
      output_zero_point,
      output_dtype,
      unary_post_op);
}

REGISTER_MUSA_DISPATCH(matmul_stub, &matmul_kernel);
} // namespace at::native
