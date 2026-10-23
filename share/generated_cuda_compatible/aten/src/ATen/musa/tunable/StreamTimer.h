// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <musa_runtime.h>

#include <ATen/musa/tunable/Tunable.h>

namespace at::musa::tunable {

class StreamTimer : public ITimer {
  public:
    StreamTimer();
    ~StreamTimer() override;

    void Start() override;

    void End() override;

    float Duration() override;

  private:
    musaEvent_t start_{};
    musaEvent_t end_{};
};

class StreamTimerNoSync : public ITimer {
  public:
    StreamTimerNoSync();
    ~StreamTimerNoSync() override;

    void Start() override;

    void End() override;

    float Duration() override;

  private:
    musaEvent_t start_{};
    musaEvent_t end_{};
};

} // namespace at::musa::tunable
