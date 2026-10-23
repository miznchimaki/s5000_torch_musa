#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/DistributionTemplates.h>

namespace at::native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen, musa::detail::getDefaultMUSAGenerator());
  templates::musa::uniform_kernel(iter, from, to, generator);
}

REGISTER_DISPATCH(uniform_stub, &uniform_kernel)

} // namespace at::native
