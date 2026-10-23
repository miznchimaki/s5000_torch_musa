#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include "torch_musa/csrc/aten/musa/Exceptions.h"

namespace at {
namespace musa {
namespace detail {

__device__ __constant__ float cublas_one_device;
__device__ __constant__ float cublas_zero_device;

float *get_cublas_device_one() {
  static float *ptr = nullptr;
  static auto init_flag = [&]() {
    const float one = 1.f;
    AT_MUSA_CHECK(musaMemcpyToSymbol(cublas_one_device, &one, sizeof(float)));
    AT_MUSA_CHECK(musaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_one_device));
    return true;
  }();

  return ptr;
}

float *get_cublas_device_zero() {
  static float *ptr = nullptr;
  static auto init_flag = [&]() {
    const float zero = 0.f;
    AT_MUSA_CHECK(musaMemcpyToSymbol(cublas_zero_device, &zero, sizeof(float)));
    AT_MUSA_CHECK(musaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_zero_device));
    return true;
  }();

  return ptr;
}

float *get_user_alpha_ptr() {
  static float *alpha_ptr;

  static bool init_flag [[maybe_unused]] = []() {
    AT_MUSA_CHECK(musaMalloc(&alpha_ptr, sizeof(float)));
    return true;
  }();

  return alpha_ptr;
}

} // namespace detail
} // namespace musa
} // namespace at
