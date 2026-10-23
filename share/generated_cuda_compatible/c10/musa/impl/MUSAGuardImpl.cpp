#include "torch_musa/csrc/core/GuardImpl.h"

namespace c10::musa::impl {

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl)

} // namespace c10::musa::impl
