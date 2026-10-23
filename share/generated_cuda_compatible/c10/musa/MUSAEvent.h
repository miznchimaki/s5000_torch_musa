#pragma once

#include <c10/core/alignment.h>
#include <c10/core/impl/GPUTrace.h>
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

/*
 * `cudaEventExternal` is a torch-specific flag that is used to
 * indicate that the CUDAEvent will be used only for synchronization
 * with work outside of the musa graph, rather than creation of
 * cross-stream dependencies within a musa graph. Resources:
 * https://docs.nvidia.com/musa/archive/12.9.0/musa-c-programming-guide/index.html#cross-stream-dependencies-and-events
 * https://docs.nvidia.com/musa/archive/12.9.0/musa-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3457b81d1d32c6a00f6132fbc2693d47
 * https://docs.nvidia.com/musa/archive/12.9.0/musa-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0c23426b7252eaa9cef695859991304e
 */
#define cudaEventExternal 0x08

namespace c10::musa {

/*
 * CUDAEvents are movable not copyable wrappers around CUDA's events.
 *
 * CUDAEvents are constructed lazily when first recorded unless it is
 * reconstructed from a musaIpcEventHandle_t. The event has a device, and this
 * device is acquired from the first recording stream. However, if reconstructed
 * from a handle, the device should be explicitly specified; or if ipc_handle()
 * is called before the event is ever recorded, it will use the current device.
 * Later streams that record the event must match this device.
 */
struct CUDAEvent {
  // Constructors
  // Default value for `flags` is specified below - it's musaEventDisableTiming
  CUDAEvent() noexcept = default;
  CUDAEvent(unsigned int flags) noexcept : flags_{flags} {}

  CUDAEvent(DeviceIndex device_index, const musaIpcEventHandle_t* handle)
      : device_index_(device_index) {
    MUSAGuard guard(device_index_);

    C10_MUSA_CHECK(musaIpcOpenEventHandle(&event_, *handle));
    is_created_ = true;
  }

  // Note: event destruction done on creating device to avoid creating a
  // CUDA context on other devices.
  ~CUDAEvent() {
    if (is_created_) {
      MUSAGuard guard(device_index_);
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_deletion(
            c10::kCUDA, reinterpret_cast<uintptr_t>(event_));
      }
      TORCH_MUSA_WARN(musaEventDestroy(event_));
    }
  }

  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  CUDAEvent(CUDAEvent&& other) noexcept {
    moveHelper(std::move(other));
  }
  CUDAEvent& operator=(CUDAEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator musaEvent_t() const {
    return event();
  }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAEvent& left, const CUDAEvent& right) {
    return left.event_ < right.event_;
  }

  std::optional<c10::Device> device() const {
    if (is_created_) {
      return c10::Device(c10::kCUDA, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const {
    return is_created_;
  }
  DeviceIndex device_index() const {
    return device_index_;
  }
  musaEvent_t event() const {
    return event_;
  }

  // Note: musaEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }

    musaError_t err = musaEventQuery(event_);
    if (err == musaSuccess) {
      return true;
    } else if (err != musaErrorNotReady) {
      C10_MUSA_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)musaGetLastError();
    }

    return false;
  }

  void record() {
    record(getCurrentMUSAStream());
  }

  void recordOnce(const MUSAStream& stream) {
    if (!was_recorded_)
      record(stream);
  }

  // Note: musaEventRecord must be called on the same device as the event.
  void record(const MUSAStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");
    MUSAGuard guard(device_index_);

#ifndef USE_ROCM
    // it is an error to use musaEventRecordExternal when not doing stream
    // capture
    unsigned int flags = (c10::musa::currentStreamCaptureStatusMayInitCtx() !=
                              c10::musa::CaptureStatus::None &&
                          external_)
        ? musaEventRecordExternal
        : musaEventRecordDefault;
    C10_MUSA_CHECK(musaEventRecordWithFlags(event_, stream, flags));
#else
    C10_MUSA_CHECK(musaEventRecord(event_, stream));
#endif
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(event_),
          reinterpret_cast<uintptr_t>(stream.stream()));
    }
    was_recorded_ = true;
  }

  // Note: musaStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GPU resources associated with it.
  void block(const MUSAStream& stream) {
    if (is_created_) {
      MUSAGuard guard(stream.device_index());
#ifndef USE_ROCM
      // it is an error to use musaEventWaitExternal when not doing stream
      // capture
      unsigned int flags = (c10::musa::currentStreamCaptureStatusMayInitCtx() !=
                                c10::musa::CaptureStatus::None &&
                            external_)
          ? musaEventWaitExternal
          : musaEventWaitDefault;
      C10_MUSA_CHECK(musaStreamWaitEvent(stream, event_, flags));
#else
      C10_MUSA_CHECK(musaStreamWaitEvent(stream, event_));
#endif
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(
            c10::kCUDA,
            reinterpret_cast<uintptr_t>(event_),
            reinterpret_cast<uintptr_t>(stream.stream()));
      }
    }
  }

  // Note: musaEventElapsedTime can be safely called from any device
  float elapsed_time(const CUDAEvent& other) const {
    TORCH_CHECK_VALUE(
        !(flags_ & musaEventDisableTiming) &&
            !(other.flags_ & musaEventDisableTiming),
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK_VALUE(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");

    float time_ms = 0;
    // We do not strictly have to set the device index to the same as our event,
    // but if we don't and the current device is not initialized, it will
    // create a new musa context, which will consume a lot of memory.
    MUSAGuard guard(device_index_);
    // raise musaErrorNotReady if either event is recorded but not yet completed
    C10_MUSA_CHECK(musaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  // Note: musaEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_synchronization(
            c10::kCUDA, reinterpret_cast<uintptr_t>(event_));
      }
      C10_MUSA_CHECK(musaEventSynchronize(event_));
    }
  }

  // Note: musaIpcGetEventHandle must be called on the same device as the event
  void ipc_handle(musaIpcEventHandle_t* handle) {
    if (!is_created_) {
      // this CUDAEvent object was initially constructed from flags but event_
      // is not created yet.
      createEvent(getCurrentMUSAStream().device_index());
    }
    MUSAGuard guard(device_index_);
    C10_MUSA_CHECK(musaIpcGetEventHandle(handle, event_));
  }

  void create(DeviceIndex device_index) {
    if (!is_created_) {
      createEvent(device_index);
    }
  }

 private:
  unsigned int flags_ = musaEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  bool external_ = false;
  DeviceIndex device_index_ = -1;
  musaEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    external_ = (flags_ & cudaEventExternal) != 0;
#ifdef USE_ROCM
    TORCH_CHECK(!external_, "External events are disallowed in rocm");
#endif
    flags_ &= ~cudaEventExternal;
    device_index_ = device_index;
    MUSAGuard guard(device_index_);
    C10_MUSA_CHECK(musaEventCreateWithFlags(&event_, flags_));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          c10::kCUDA, reinterpret_cast<uintptr_t>(event_));
    }
    is_created_ = true;
  }

  void moveHelper(CUDAEvent&& other) {
    // Transfer ownership of all state from other to this
    flags_ = other.flags_;
    is_created_ = other.is_created_;
    was_recorded_ = other.was_recorded_;
    external_ = other.external_;
    device_index_ = other.device_index_;
    event_ = other.event_;

    // Reset other to a valid empty state to prevent double-free
    // The moved-from object must not attempt to destroy the event
    other.is_created_ = false;
    other.event_ = musaEvent_t{};
  }
};

// CUDAEventPool - A thread-safe pool of CUDA events to avoid the overhead of
// repeatedly calling musaEventCreate(). Concurrent musaEventCreate() calls
// can incur significant cost on some device/driver combinations.
//
// This pool maintains per-device lists of pre-created CUDA events.
// Borrowed events are returned to the pool via a custom unique_ptr deleter.

class CUDAEventPool {
 public:
  using Event = std::unique_ptr<
      c10::musa::CUDAEvent,
      std::function<void(c10::musa::CUDAEvent*)>>;

  CUDAEventPool(size_t init_num_events = 0)
      : pools_(c10::musa::device_count()) {
    if (init_num_events > 0) {
      reserve_events_on_pools(init_num_events);
    }
  }

  // Acquire an event associated with a given device. If device is invalid, fall
  // back to a regular CUDAEvent and no pooling.
  Event get(const DeviceIndex device) {
    if (device < 0 || device >= (DeviceIndex)pools_.size()) {
      auto deleter = [](CUDAEvent* event) { delete event; };
      return Event(std::make_unique<CUDAEvent>().release(), deleter);
    }

    auto& pool = pools_[device];

    // Create a destructor that returns the event to the appropriate device pool
    auto destructor = [&pool](CUDAEvent* event) noexcept {
      if (event != nullptr) {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        pool.event_pool_.emplace_back(event);
      }
    };

    {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto event = std::move(pool.event_pool_.back());
        pool.event_pool_.pop_back();
        return Event(event.release(), destructor);
      }
    }

    // Pool is empty then create a new Event
    return Event(std::make_unique<CUDAEvent>().release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  // Pre-initialize each device pool with N events. This prevents
  // musaEventCreate() from invoking during steady-state execution.
  void reserve_events_on_pools(size_t num_events) {
    for (const auto device : c10::irange(pools_.size())) {
      std::vector<Event> temp_events;
      temp_events.reserve(num_events);
      pools_[device].event_pool_.reserve(num_events);
      for ([[maybe_unused]] const auto _ : c10::irange(num_events)) {
        auto event = get(device);
        event->create(device);
        temp_events.emplace_back(std::move(event));
      }
      // Events will be returned to pool when temp_events is destroyed.
    }
  }

  struct alignas(c10::hardware_destructive_interference_size) PerDevicePool {
    alignas(c10::hardware_destructive_interference_size) std::mutex mutex_;
    std::vector<std::unique_ptr<CUDAEvent>> event_pool_;
  };

  std::vector<PerDevicePool> pools_;
};

} // namespace c10::musa
