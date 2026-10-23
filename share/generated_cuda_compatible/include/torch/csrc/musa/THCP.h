#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <torch/csrc/THP.h>
#include <torch/csrc/musa/Event.h>
#include <torch/csrc/musa/Module.h>
#include <torch/csrc/musa/Stream.h>
#include <torch/csrc/musa/utils.h>
#include <torch/csrc/python_headers.h>

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
