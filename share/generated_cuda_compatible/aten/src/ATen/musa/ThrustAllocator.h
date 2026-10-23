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
