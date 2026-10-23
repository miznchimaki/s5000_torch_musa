#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/DistributionTemplates.h>

namespace at::native {

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<MUSAGeneratorImpl>(gen, musa::detail::getDefaultMUSAGenerator());
  at::native::templates::musa::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel)

} // namespace at::native
