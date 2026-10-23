#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/musa/DistributionTemplates.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::normal_kernel(self, mean, std, generator);
}

REGISTER_DISPATCH(normal_stub, &normal_kernel)

} // namespace at::native
