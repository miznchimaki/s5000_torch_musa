#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/DistributionTemplates.h>

namespace at::native {

void exponential_kernel(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::exponential_kernel(iter, lambda, generator);
}

REGISTER_DISPATCH(exponential_stub, &exponential_kernel)

} // namespace at::native
