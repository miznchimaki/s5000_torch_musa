#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/PyInterpreter.h>
#include <musa_runtime_api.h>
#include <cstdint>
#include <optional>

namespace c10::musa::impl {

struct CUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::PrivateUse1;

  CUDAGuardImpl() = default;
  explicit CUDAGuardImpl(DeviceType t) {
    TORCH_CHECK(
        t == DeviceType::PrivateUse1,
        "CUDAGuardImpl initialized with non-CUDA DeviceType: ",
        t);
  }
  DeviceType type() const override {
    return DeviceType::PrivateUse1;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Expected a CUDA device, but got ", d);
    auto old_device_index = c10::musa::ExchangeDevice(d.index());
    return Device(DeviceType::PrivateUse1, old_device_index);
  }
  Device getDevice() const override {
    DeviceIndex device = 0;
    C10_MUSA_CHECK(c10::musa::GetDevice(&device));
    return Device(DeviceType::PrivateUse1, device);
  }
  std::optional<Device> uncheckedGetDevice() const noexcept {
    DeviceIndex device{-1};
    const auto err = C10_MUSA_ERROR_HANDLED(c10::musa::GetDevice(&device));
    TORCH_MUSA_WARN(err);
    if (err != musaSuccess) {
      return std::nullopt;
    }
    return Device(DeviceType::PrivateUse1, device);
  }
  void setDevice(Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Expected a CUDA device, but got ", d);
    C10_MUSA_CHECK(c10::musa::SetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    TORCH_MUSA_WARN(c10::musa::MaybeSetDevice(d.index()));
  }
  Stream getStream(Device d) const override {
    return getCurrentMUSAStream(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultMUSAStream(d.index());
  }
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const override {
    MUSAStream cs(s);
    auto old_stream = getCurrentMUSAStream(s.device().index());
    setCurrentMUSAStream(cs);
    return old_stream.unwrap();
  }
  void* getStreamNativeHandle(const Stream s) const override {
    MUSAStream stream{s};
    return reinterpret_cast<void*>(stream.stream());
  }
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event-related functions
  void createEvent(musaEvent_t* cuda_event, const EventFlag flag) const {
    // Maps PyTorch's Event::Flag to CUDA flag
    auto cuda_flag = musaEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        cuda_flag = musaEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        cuda_flag = musaEventDefault;
        break;
      default:
        TORCH_CHECK(false, "CUDA event received unknown flag");
    }

    C10_MUSA_CHECK(musaEventCreateWithFlags(cuda_event, cuda_flag));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
  }

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto cuda_event = static_cast<musaEvent_t>(event);
    DeviceIndex orig_device{-1};
    TORCH_MUSA_WARN(c10::musa::GetDevice(&orig_device));
    TORCH_MUSA_WARN(c10::musa::SetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
    TORCH_MUSA_WARN(musaEventDestroy(cuda_event));
    TORCH_MUSA_WARN(c10::musa::SetDevice(orig_device));
  }

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    musaEvent_t cuda_event = static_cast<musaEvent_t>(*event);
    MUSAStream cuda_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!cuda_event)
      createEvent(&cuda_event, flag);
    C10_MUSA_CHECK(musaEventRecord(cuda_event, cuda_stream));
    // Makes the void* point to the (possibly just allocated) CUDA event
    *event = cuda_event;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }

    // Resets device
    setDevice(orig_device);
  }

  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;
    musaEvent_t cuda_event = static_cast<musaEvent_t>(event);
    MUSAStream cuda_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_MUSA_CHECK(musaStreamWaitEvent(
        cuda_stream,
        cuda_event,
        /*flags (must be zero)=*/0));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    musaEvent_t cuda_event = static_cast<musaEvent_t>(event);
    // Note: musaEventQuery can be safely called from any device
    const musaError_t err = C10_MUSA_ERROR_HANDLED(musaEventQuery(cuda_event));
    if (err != musaErrorNotReady) {
      C10_MUSA_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)musaGetLastError();
    }
    return (err == musaSuccess);
  }

  // Stream-related functions
  bool queryStream(const Stream& stream) const override {
    MUSAStream cuda_stream{stream};
    return cuda_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    MUSAStream cuda_stream{stream};
    cuda_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    musaEvent_t cuda_event = static_cast<musaEvent_t>(event);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_synchronization(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
    // Note: musaEventSynchronize can be safely called from any device
    C10_MUSA_CHECK(musaEventSynchronize(cuda_event));
  }

  // Note: synchronizeDevice can be safely called from any device
  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    DeviceIndex orig_device{-1};
    C10_MUSA_CHECK(c10::musa::GetDevice(&orig_device));
    C10_MUSA_CHECK(c10::musa::SetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_device_synchronization(c10::kCUDA);
    }
    C10_MUSA_CHECK(musaDeviceSynchronize());
    C10_MUSA_CHECK(c10::musa::SetDevice(orig_device));
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    MUSAStream cuda_stream{stream};
    MUSACachingAllocator::recordStream(data_ptr, cuda_stream);
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    // Even though musaEventElapsedTime can be safely called from any device, if
    // the current device is not initialized, it will create a new musa context,
    // which will consume a lot of memory.
    DeviceIndex orig_device{-1};
    C10_MUSA_CHECK(c10::musa::GetDevice(&orig_device));
    C10_MUSA_CHECK(c10::musa::SetDevice(device_index));
    musaEvent_t cuda_event1 = static_cast<musaEvent_t>(event1);
    musaEvent_t cuda_event2 = static_cast<musaEvent_t>(event2);
    float time_ms = 0;
    // raise musaErrorNotReady if either event is recorded but not yet completed
    C10_MUSA_CHECK(musaEventElapsedTime(&time_ms, cuda_event1, cuda_event2));
    C10_MUSA_CHECK(c10::musa::SetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
};

} // namespace c10::musa::impl
