#pragma once

#include "torch_musa/csrc/aten/musa/MUSAContextLight.h"

// Preserved for BC, as many files depend on these includes
#include <ATen/Context.h>
#include "torch_musa/csrc/core/MUSAStream.h"
#include <c10/util/Logging.h>
#include "torch_musa/csrc/aten/musa/Exceptions.h"
