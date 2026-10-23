#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/musa/MUSAApplyUtils.muh>
#include <ATen/AccumulateType.h>
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/DistributionTemplates.h>

#include <murand.h>
#include <murand_kernel.h>
#include <murand_philox4x32_10.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/TensorIterator.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace at::native {

void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen_) {
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen_, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::bernoulli_kernel(iter, p, generator);
}

REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel)
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel)

} // namespace at::native
