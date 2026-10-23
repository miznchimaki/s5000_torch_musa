#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/musa/MUSAMathCompat.h>
#include <ATen/musa/ApplyGridUtils.muh>
#include <ATen/musa/detail/OffsetCalculator.muh>
#include <ATen/native/musa/Loops.muh>

namespace at::native {
namespace {

void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardshrink_cuda",
      [&]() {
        auto lambd = value.to<scalar_t>();
        gpu_kernel(iter, [lambd] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return (a >= -lambd && a <= lambd) ? scalar_t(0) : a;
        });
      });
}
} // namespace

REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel)

} // namespace at::native
