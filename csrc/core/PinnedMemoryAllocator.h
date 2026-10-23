#ifndef TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_

#include "torch_musa/csrc/core/CachingHostAllocator.h"

namespace at::musa {

inline at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kMUSA);
}

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_
