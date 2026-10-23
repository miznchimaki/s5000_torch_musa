#pragma once
#include <ATen/detail/MUSAHooksInterface.h>
namespace at::musa {
// Forward-declares at::musa::NVRTC
struct NVRTC;

namespace detail {
extern NVRTC lazyNVRTC;
} // namespace detail

}  // namespace at::musa
