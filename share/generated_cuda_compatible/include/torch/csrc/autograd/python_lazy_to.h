#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>

#include <optional>

namespace torch::autograd {

TORCH_PYTHON_API void lazy_replace_device(
    const at::Tensor& self,
    c10::Device device);

TORCH_PYTHON_API bool can_lazy_replace_device(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<c10::Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format);

} // namespace torch::autograd
#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
