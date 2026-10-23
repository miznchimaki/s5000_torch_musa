#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once
#include <ATen/detail/MUSAHooksInterface.h>
namespace at::musa {
// Forward-declares at::musa::NVRTC
struct NVRTC;

namespace detail {
extern NVRTC lazyNVRTC;
} // namespace detail

}  // namespace at::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
