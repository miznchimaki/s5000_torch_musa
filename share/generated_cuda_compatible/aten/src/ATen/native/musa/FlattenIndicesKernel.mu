#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/musa/KernelUtils.muh>
#include <ATen/musa/detail/OffsetCalculator.muh>
#include <ATen/AccumulateType.h>

namespace at::native {

namespace {

template <typename func_t>
struct CUDAKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};

Tensor flatten_indices_cuda_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<CUDAKernelLauncher>(indices, size);
}

}

REGISTER_MUSA_DISPATCH(flatten_indices_stub, &flatten_indices_cuda_kernel)

} // namespace at::native
