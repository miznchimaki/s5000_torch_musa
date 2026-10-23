#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/DistributionTemplates.h>

namespace at::native {

void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::cauchy_kernel(iter, median, sigma, generator);
}

REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel)

} // namespace at::native
