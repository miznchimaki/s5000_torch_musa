#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include "torch_musa/csrc/core/MUSAStream.h"
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class TORCH_CUDA_CPP_API AOTIModelContainerRunnerMusa
    : public AOTIModelContainerRunner {
 public:
  // @param device_str: musa device string, e.g. "musa", "musa:0"
  AOTIModelContainerRunnerMusa(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "musa",
      const std::string& cubin_dir = "",
      const bool run_single_threaded = false);

  ~AOTIModelContainerRunnerMusa() override;

  std::vector<at::Tensor> run_impl(
      std::vector<AtenTensorHandle>& input_handles,
      void* stream_handle) override;

  std::vector<at::Tensor> run_with_musa_stream(
      const std::vector<at::Tensor>& inputs,
      const at::musa::MUSAStream& musa_stream);
};

} // namespace torch::inductor
#endif

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
