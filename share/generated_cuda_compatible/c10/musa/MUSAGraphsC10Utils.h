#pragma once

#include "torch_musa/csrc/core/MUSAStream.h"
#include <iostream>
#include <utility>

// CUDA Graphs utils used by c10 and aten.
// aten/musa/MUSAGraphsUtils.cuh adds utils used by aten only.

namespace c10::musa {

// RAII guard for "musaStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct C10_MUSA_API CUDAStreamCaptureModeGuard {
  CUDAStreamCaptureModeGuard(musaStreamCaptureMode desired)
      : strictness_(desired) {
    C10_MUSA_CHECK(musaThreadExchangeStreamCaptureMode(&strictness_));
  }
  CUDAStreamCaptureModeGuard(const CUDAStreamCaptureModeGuard&) = delete;
  CUDAStreamCaptureModeGuard(CUDAStreamCaptureModeGuard&&) = delete;
  CUDAStreamCaptureModeGuard& operator=(const CUDAStreamCaptureModeGuard&) =
      delete;
  CUDAStreamCaptureModeGuard& operator=(CUDAStreamCaptureModeGuard&&) = delete;
  ~CUDAStreamCaptureModeGuard() {
    TORCH_MUSA_WARN(musaThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  musaStreamCaptureMode strictness_;
};

// Protects against enum musaStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(
    int(musaStreamCaptureStatus::musaStreamCaptureStatusNone) == 0,
    "unexpected int(musaStreamCaptureStatusNone) value");
static_assert(
    int(musaStreamCaptureStatus::musaStreamCaptureStatusActive) == 1,
    "unexpected int(musaStreamCaptureStatusActive) value");
static_assert(
    int(musaStreamCaptureStatus::musaStreamCaptureStatusInvalidated) == 2,
    "unexpected int(musaStreamCaptureStatusInvalidated) value");

enum class CaptureStatus : int {
  None = int(musaStreamCaptureStatus::musaStreamCaptureStatusNone),
  Active = int(musaStreamCaptureStatus::musaStreamCaptureStatusActive),
  Invalidated = int(musaStreamCaptureStatus::musaStreamCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "musaStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "musaStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "musaStreamCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown CUDA graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a CUDA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  musaStreamCaptureStatus is_capturing{musaStreamCaptureStatusNone};
  C10_MUSA_CHECK(
      musaStreamIsCapturing(c10::musa::getCurrentMUSAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

} // namespace c10::musa
