// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include <ATen/musa/tunable/StreamTimer.h>
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at::musa::tunable {

StreamTimer::StreamTimer() {
  AT_MUSA_CHECK(musaEventCreate(&start_));
  AT_MUSA_CHECK(musaEventCreate(&end_));
}

StreamTimer::~StreamTimer() = default;

void StreamTimer::Start() {
  AT_MUSA_CHECK(musaEventSynchronize(start_));
  AT_MUSA_CHECK(musaEventRecord(start_, at::musa::getCurrentMUSAStream()));
}

void StreamTimer::End() {
  AT_MUSA_CHECK(musaEventRecord(end_, at::musa::getCurrentMUSAStream()));
  AT_MUSA_CHECK(musaEventSynchronize(end_));
}

float StreamTimer::Duration() {
  auto time = std::numeric_limits<float>::quiet_NaN();
  // time is in ms with a resolution of 1 us
  AT_MUSA_CHECK(musaEventElapsedTime(&time, start_, end_));
  return time;
}

StreamTimerNoSync::StreamTimerNoSync() {
  AT_MUSA_CHECK(musaEventCreate(&start_));
  AT_MUSA_CHECK(musaEventCreate(&end_));
}

StreamTimerNoSync::~StreamTimerNoSync() = default;

void StreamTimerNoSync::Start() {
  AT_MUSA_CHECK(musaEventRecord(start_, at::musa::getCurrentMUSAStream()));
}

void StreamTimerNoSync::End() {
  AT_MUSA_CHECK(musaEventRecord(end_, at::musa::getCurrentMUSAStream()));
}

float StreamTimerNoSync::Duration() {
  auto time = std::numeric_limits<float>::quiet_NaN();
  AT_MUSA_CHECK(musaEventSynchronize(end_));
  // time is in ms with a resolution of 1 us
  AT_MUSA_CHECK(musaEventElapsedTime(&time, start_, end_));
  return time;
}

} // namespace at::musa::tunable
