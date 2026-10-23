#pragma once

#include <c10/musa/MUSAMacros.h>
#include <bitset>
#include <cstddef>

namespace torch {

using device_set = std::bitset<C10_COMPILE_TIME_MAX_GPUS>;

} // namespace torch
