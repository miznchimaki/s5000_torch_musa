#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#include <ATen/native/transformers/musa/mem_eff_attention/kernel_forward.h>
using namespace PyTorchMemEffAttention;
__global__ void __launch_bounds__(
    AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(typename AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 800
#if __MUSA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 64, 64, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x128_rf_sm80(typename AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 800
#if __MUSA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 64, 128, 128, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_bf16_aligned_64x128_rf_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 800
#if __MUSA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<mutlass::bfloat16_t, mutlass::arch::Sm80, true, 32, 128, 65536, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
