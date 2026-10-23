#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include "torch_musa/csrc/aten/musa/Exceptions.h"

#include <musa.h>
#include <musa_runtime.h>

namespace at::musa {

inline Device getDeviceFromPtr(void* ptr) {
  musaPointerAttributes attr{};

  AT_MUSA_CHECK(musaPointerGetAttributes(&attr, ptr));

#if !defined(USE_ROCM)
  TORCH_CHECK(attr.type != musaMemoryTypeUnregistered,
    "The specified pointer resides on host memory and is not registered with any CUDA device.");
#endif

  return {c10::DeviceType::PrivateUse1, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
