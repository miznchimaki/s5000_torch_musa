#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>

#include <musa.h>
#include <musa_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch::jit::fuser::musa {

// query codegen output arch and target
TORCH_CUDA_CU_API void codegenOutputQuery(
    const musaDeviceProp* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass);

// A class holding metadata for an actual CUDA function.
// Note: CUDA functions are per device.
struct TORCH_CUDA_CU_API FusedKernelCUDA
    : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCUDA(
      at::DeviceIndex device,
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  ~FusedKernelCUDA() override;

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override;

  at::Backend backend() const override {
    return at::Backend::CUDA;
  }

 private:
  static constexpr auto kBlockSize = 128;

  // Note: per device to store device properties and compute launch heuristics
  //  Acquiring these values at launch time would be too slow
  at::DeviceIndex device_;
  int maxBlocks_{};
  musaDeviceProp* prop_{};
  std::vector<char> ptx_;
  MUmodule module_{};
  MUfunction function_{};
};

} // namespace torch::jit::fuser::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
