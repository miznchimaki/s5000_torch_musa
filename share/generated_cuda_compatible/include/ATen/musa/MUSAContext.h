#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include "torch_musa/csrc/aten/musa/MUSAContextLight.h"

// Preserved for BC, as many files depend on these includes
#include <ATen/Context.h>
#include "torch_musa/csrc/core/MUSAStream.h"
#include <c10/util/Logging.h>
#include "torch_musa/csrc/aten/musa/Exceptions.h"

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
