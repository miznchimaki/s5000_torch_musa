#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <torch/csrc/inductor/aoti_runtime/utils_musa.h>
#include <torch_musa/csrc/inductor/aoti_torch/generated/c_shim_musa.h>

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
