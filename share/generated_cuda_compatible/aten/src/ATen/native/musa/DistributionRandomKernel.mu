#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/DistributionTemplates.h>

namespace at::native {

void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<MUSAGeneratorImpl>(gen_, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::random_from_to_kernel(iter, range, base, gen);
}

void random_full_64_bits_range_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<MUSAGeneratorImpl>(gen_, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::random_full_64_bits_range_kernel(iter, gen);
}

void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<MUSAGeneratorImpl>(gen_, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::random_kernel(iter, gen);
}

REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel)
REGISTER_DISPATCH(random_stub, &random_kernel)
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel)

} // namespace at::native
