#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#include <ATen/native/transformers/musa/mem_eff_attention/kernel_backward.h>
using namespace PyTorchMemEffAttention;
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70(typename AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32, true>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 700
#if __MUSA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80(typename AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32, true>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 800
#if __MUSA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm50, mutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm50, mutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm50(typename AttentionBackwardKernel<mutlass::arch::Sm50, mutlass::half_t, true, false, false, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 500
#if __MUSA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm50, mutlass::half_t, true, false, false, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k32_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm70(typename AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 700
#if __MUSA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm70, mutlass::half_t, true, false, false, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k32_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm75, mutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm75, mutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm75(typename AttentionBackwardKernel<mutlass::arch::Sm75, mutlass::half_t, true, false, false, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 750
#if __MUSA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm75, mutlass::half_t, true, false, false, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k32_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm80(typename AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 800
#if __MUSA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm80, mutlass::half_t, true, false, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k32_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
