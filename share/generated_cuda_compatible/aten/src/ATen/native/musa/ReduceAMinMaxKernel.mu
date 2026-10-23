#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/musa/ReduceOps.h>
#include <ATen/musa/NumericLimits.muh>
#include <ATen/native/musa/Reduce.muh>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/musa/NumericLimits.muh>

namespace at::native {

template <typename scalar_t>
void _min_max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      thrust::pair<scalar_t, scalar_t>(
          at::numeric_limits<scalar_t>::upper_bound(),
          at::numeric_limits<scalar_t>::lower_bound()));
}

void aminmax_allreduce_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "aminmax_all_cuda", [&] {
        _min_max_values_kernel_cuda_impl<scalar_t>(iter);
      });
}

void aminmax_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "aminmax_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MinMaxOps<scalar_t, scalar_t, int32_t>{},
            thrust::pair<scalar_t, scalar_t>(
                at::numeric_limits<scalar_t>::upper_bound(),
                at::numeric_limits<scalar_t>::lower_bound()));
      });
}

} // namespace at::native
