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
    AttentionBackwardKernel<mutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k32_sm50(typename AttentionBackwardKernel<mutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 500
#if __MUSA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f32_notaligned_64x64_k32_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k32_sm70(typename AttentionBackwardKernel<mutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 700
#if __MUSA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f32_notaligned_64x64_k32_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<mutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<mutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k32_sm75(typename AttentionBackwardKernel<mutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::Params p) {
#ifdef __MUSA_ARCH__
#if __MUSA_ARCH__ >= 750
#if __MUSA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<mutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f32_notaligned_64x64_k32_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__MUSA_ARCH__ + 0) / 10);
#endif
}
