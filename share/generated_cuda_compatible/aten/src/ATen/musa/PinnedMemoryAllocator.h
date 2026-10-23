#pragma once

#include "torch_musa/csrc/core/CachingHostAllocator.h"

namespace at::musa {

inline TORCH_CUDA_CPP_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kCUDA);
}
} // namespace at::musa
