#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#include <ATen/musa/MUSAEvent.h>
#include <c10/core/Device.h>
#include "torch_musa/csrc/core/MUSAStream.h"
#include <torch/custom_class.h>

namespace torch::jit {

class CUDAEvent;
// This class is a wrapper around c10::musa::MUSAStream.
// It is needed because TorchBind does not support all of the argument types
// for c10::musa::MUSAStream. For more details, please refer to
// c10/musa/MUSAStream.h.
class MUSAStream final : public CustomClassHolder {
 public:
  MUSAStream(
      std::optional<c10::Device> device = std::nullopt,
      int64_t priority = 0) {
    c10::DeviceIndex device_index =
        device.has_value() ? device->index() : c10::musa::current_device();
    stream_ = std::make_unique<c10::musa::MUSAStream>(
        c10::musa::getStreamFromPool(static_cast<int>(priority), device_index));
  }

  MUSAStream(c10::musa::MUSAStream s) {
    stream_ = std::make_unique<c10::musa::MUSAStream>(s);
  }

  bool query() {
    return stream_->query();
  }

  c10::intrusive_ptr<CUDAEvent> recordEvent(
      c10::intrusive_ptr<CUDAEvent> event);

  void synchronize() {
    stream_->synchronize();
  }

  void waitEvent(const c10::intrusive_ptr<CUDAEvent>& event);

  void waitStream(const c10::intrusive_ptr<MUSAStream>& stream);

  /// Get the CUDA device index that this stream is associated with.
  int64_t device_index() const {
    return stream_->device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a CUDA device.
  c10::Device device() const {
    return stream_->device();
  }

  /// Return the stream ID corresponding to this particular stream.
  int64_t id() const {
    return stream_->id();
  }

 private:
  std::unique_ptr<c10::musa::MUSAStream> stream_;
  friend class CUDAEvent;
};

// This class is a wrapper around at::musa::MUSAStream.
// It is needed because TorchBind does not support all of the argument types
// for at::musa::CUDAEvent. For more details, please refer to
// aten/src/ATen/musa/MUSAEvent.h.
class CUDAEvent final : public CustomClassHolder {
 public:
  CUDAEvent(
      bool enable_timing = false,
      bool blocking = false,
      bool interprocess = false) {
    int flags = musaEventDisableTiming;
    if (enable_timing) {
      flags = musaEventDefault;
    }
    if (blocking) {
      flags |= musaEventBlockingSync;
    }
    if (interprocess) {
      TORCH_CHECK(!enable_timing);
      flags |= musaEventInterprocess;
    }

    event_ = std::make_unique<at::musa::CUDAEvent>(flags);
  }

  double elapsedTime(const c10::intrusive_ptr<CUDAEvent>& end) {
    return event_->elapsed_time(*end->event_);
  }

  std::string ipcHandle() {
    musaIpcEventHandle_t handle{};
    event_->ipc_handle(&handle);
    std::string str_handle((const char*)&handle, sizeof(handle));
    return str_handle;
  }

  bool query() {
    return event_->query();
  }

  void record(const c10::intrusive_ptr<MUSAStream>& stream);

  void synchronize() {
    event_->synchronize();
  }
  void wait(const c10::intrusive_ptr<MUSAStream>& stream);

 private:
  void recordInternal(MUSAStream* stream);
  std::unique_ptr<at::musa::CUDAEvent> event_;

  friend class MUSAStream;
};

inline c10::intrusive_ptr<CUDAEvent> MUSAStream::recordEvent(
    c10::intrusive_ptr<CUDAEvent> event) {
  if (!event) {
    event = c10::make_intrusive<CUDAEvent>();
  }

  event->recordInternal(this);
  return event;
}

inline void MUSAStream::waitEvent(const c10::intrusive_ptr<CUDAEvent>& event) {
  event->event_->block(*stream_);
}

inline void MUSAStream::waitStream(
    const c10::intrusive_ptr<MUSAStream>& stream) {
  auto ev = c10::make_intrusive<CUDAEvent>();
  stream->recordEvent(ev);
  waitEvent(ev);
}

inline void CUDAEvent::record(const c10::intrusive_ptr<MUSAStream>& stream) {
  event_->record(*stream->stream_);
}

inline void CUDAEvent::recordInternal(MUSAStream* stream) {
  event_->record(*stream->stream_);
}

inline void CUDAEvent::wait(const c10::intrusive_ptr<MUSAStream>& stream) {
  event_->block(*stream->stream_);
}

TORCH_LIBRARY(musa, m) {
  auto stream_class = m.class_<torch::jit::MUSAStream>("Stream").def(
      torch::init<std::optional<c10::Device>, int64_t>(),
      "",
      {torch::arg("device") = std::nullopt, torch::arg("priority") = 0});
  auto event_class = m.class_<torch::jit::CUDAEvent>("Event").def(
      torch::init<bool, bool, bool>(),
      "",
      {torch::arg("enable_timing") = false,
       torch::arg("blocking") = false,
       torch::arg("interprocess") = false});

  stream_class.def("query", &MUSAStream::query)
      .def("record_event", &MUSAStream::recordEvent)
      .def("synchronize", &MUSAStream::synchronize)
      .def("wait_event", &MUSAStream::waitEvent)
      .def("wait_stream", &MUSAStream::waitStream)
      .def("device_index", &MUSAStream::device_index)
      .def_property("device", &MUSAStream::device)
      .def("id", &MUSAStream::id);

  event_class.def("elapsed_time", &CUDAEvent::elapsedTime)
      .def("query", &CUDAEvent::query)
      .def("record", &CUDAEvent::record)
      .def("synchronize", &CUDAEvent::synchronize)
      .def("wait", &CUDAEvent::wait);
}

} // namespace torch::jit

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
