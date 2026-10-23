#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <cstddef>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace at::musa {

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  typedef char value_type;

  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(c10::musa::MUSACachingAllocator::raw_alloc(size));
  }

  void deallocate(char* p, size_t size) {
    c10::musa::MUSACachingAllocator::raw_delete(p);
  }
};

} // namespace at::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
