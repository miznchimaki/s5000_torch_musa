#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include "torch_musa/csrc/core/CachingHostAllocator.h"

namespace at::musa {

inline TORCH_CUDA_CPP_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kCUDA);
}
} // namespace at::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
