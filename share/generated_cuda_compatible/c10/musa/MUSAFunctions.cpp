#include "torch_musa/csrc/core/MUSAFunctions.h"
#include <c10/macros/Macros.h>
#include <c10/util/WaitCounter.h>

#include <limits>

namespace c10::musa {

namespace {
// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  C10_MUSA_IGNORE_ERROR(musaDriverGetVersion(&driver_version));
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = C10_MUSA_ERROR_HANDLED(c10::musa::GetDeviceCount(&count));
  if (err == musaSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much CUDA
  // code in this regime.)
  [[maybe_unused]] musaError_t last_err = musaGetLastError();
  switch (err) {
    case musaErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case musaErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No CUDA driver means no devices
          count = 0;
          break;
        }
        TORCH_CHECK(
            false,
            "Found no NVIDIA driver on your system. Please check that you "
            "have an NVIDIA GPU and installed a driver from "
            "http://www.nvidia.com/Download/index.aspx");
      } else {
        TORCH_CHECK(
            false,
            "The NVIDIA driver on your system is too old (found version ",
            version,
            "). Please update your GPU driver by downloading and installing "
            "a new version from the URL: "
            "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
            "https://pytorch.org to install a PyTorch version that has been "
            "compiled with your version of the CUDA driver.");
      }
    }
    case musaErrorInitializationError:
      TORCH_CHECK(
          false,
          "CUDA driver initialization failed, you might not "
          "have a CUDA gpu.");
    case musaErrorUnknown:
      TORCH_CHECK(
          false,
          "CUDA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable CUDA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
#if C10_ASAN_ENABLED
    case musaErrorMemoryAllocation:
      // In ASAN mode, we know that a musaErrorMemoryAllocation error will
      // pop up if compiled with NVCC (clang-musa is fine)
      TORCH_CHECK(
          false,
          "Got 'out of memory' error while trying to initialize CUDA. "
          "CUDA with nvcc does not work well with ASAN and it's probably "
          "the reason. We will simply shut down CUDA support. If you "
          "would like to use GPUs, turn off ASAN.");
      break;
#endif // C10_ASAN_ENABLED
#if defined(_WIN32) && MUSA_COMP_VERSION >= 13000
    // Workaround for CUDA-13.0 error handling on Windows, see
    // https://github.com/pytorch/pytorch/issues/162333#issuecomment-3267929585
    case musaErrorNotSupported:
      if (!fail_if_no_driver) {
        TORCH_WARN(
            "musaGetDeviceCount() returned musaErrorNotSupported, "
            "likely using older driver or on CPU machine");
        count = 0;
        break;
      }
#endif
    default:
      TORCH_CHECK(
          false,
          "Unexpected error from musaGetDeviceCount(). Did you run "
          "some musa functions before calling NumCudaDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          musaGetErrorString(err));
  }
  return count;
}
} // namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many CUDA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("CUDA initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No CUDA GPUs are available");
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<DeviceIndex>::max(),
      "Too many CUDA devices, DeviceIndex overflowed");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_MUSA_CHECK(c10::musa::GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device, const bool force) {
  C10_MUSA_CHECK(c10::musa::SetDevice(device, force));
}

void device_synchronize() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kCUDA);
  }
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.cuda_device_synchronize);
  C10_MUSA_CHECK(musaDeviceSynchronize());
}

// this function has to be called from callers performing musa synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing CUDA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing CUDA operation");
  }
}

std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(at::musa::device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return std::nullopt;
}

namespace _internal {
static bool dummyHasPrimaryContext([[maybe_unused]] DeviceIndex device_index) {
  TORCH_CHECK(false, "Should never been called");
}
static bool (*hasPrimaryContext)(DeviceIndex) = dummyHasPrimaryContext;

// Private api to be called from CUDAHooks.cpp
// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_MUSA_API void setHasPrimaryContext(bool (*func)(DeviceIndex)) {
  hasPrimaryContext = func ? func : dummyHasPrimaryContext;
}
} // namespace _internal

bool hasPrimaryContext(DeviceIndex device_index) {
  return _internal::hasPrimaryContext(device_index);
}

// Wrappers for raw CUDA device management functions
musaError_t GetDeviceCount(int* dev_count) {
  return musaGetDeviceCount(dev_count);
}

// This is a codepath for CUDA 12 that comes with a critical change in behavior
// of `musaSetDevice`. Unlike to previous CUDA versions that allocate context
// lazily CUDA 12.x eagerly allocates primary context the moment `musaSetDevice`
// is called. This can lead to dramatic consequences and pollute the device
// memory in distributed runs. To avoid unnecessary context creation a new
// function called `MaybeSetDevice` was introduced. This function is to be
// called in device guard destructor and at the exit of torch.musa.device
// context manager. The behavior of `MaybeSetDevice` is quite simple, it calls
// to `musaSetDevice` if context already exist or if context was not allocated
// on targeted device it simply saves the device index. This way we can keep
// PyTorch backward compatible for applications like this:
//
// ```
// import torch
// x = torch.empty(1, device=“musa:1”) # no CUDA context on musa:0 after this
// call y = torch.empty(1, device=“musa”) # CUDA context is created on musa:0
// ```
#if MUSA_COMP_VERSION >= 12000
thread_local static DeviceIndex targetDeviceIndex = -1;

musaError_t GetDevice(DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return musaSuccess;
  }
  int tmp_device = -1;
  auto err = musaGetDevice(&tmp_device);
  if (err == musaSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "musaGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

musaError_t SetDevice(DeviceIndex device, const bool force) {
  TORCH_CHECK(
      device >= 0, "device id must be non-negative!", static_cast<int>(device));
  targetDeviceIndex = -1;
  if (force) {
    return musaSetDevice(device);
  }
  int cur_device = -1;
  C10_MUSA_CHECK(musaGetDevice(&cur_device));
  if (device == cur_device) {
    return musaSuccess;
  }
  return musaSetDevice(device);
}

musaError_t MaybeSetDevice(DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return c10::musa::SetDevice(device);
  }
  targetDeviceIndex = device;
  return musaSuccess;
}

// This function always initializes the CUDA context
// on to_device
DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_MUSA_CHECK(musaGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_MUSA_CHECK(musaSetDevice(to_device));
  return cur_device;
}

// This function does not initialize the CUDA context
// on to_device if it does not already exist
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  int tmp_cur_device = -1;
  C10_MUSA_CHECK(musaGetDevice(&tmp_cur_device));
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 &&
          tmp_cur_device <= std::numeric_limits<DeviceIndex>::max(),
      "musaGetDevice returns invalid device ",
      tmp_cur_device);
  auto cur_device = static_cast<DeviceIndex>(tmp_cur_device);
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  if (hasPrimaryContext(to_device)) {
    C10_MUSA_CHECK(musaSetDevice(to_device));
  } else {
    targetDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    C10_MUSA_CHECK(c10::musa::SetDevice(targetDeviceIndex));
  }
}
#else
musaError_t GetDevice(DeviceIndex* device) {
  int tmp_device = -1;
  auto err = musaGetDevice(&tmp_device);
  if (err == musaSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "musaGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

musaError_t SetDevice(DeviceIndex device, const bool force) {
  TORCH_CHECK(
      device >= 0, "device id must be non-negative!", static_cast<int>(device));
  if (force) {
    return musaSetDevice(device);
  }
  int cur_device = -1;
  C10_MUSA_CHECK(musaGetDevice(&cur_device));
  if (device == cur_device) {
    return musaSuccess;
  }
  return musaSetDevice(device);
}

musaError_t MaybeSetDevice(DeviceIndex device) {
  return c10::musa::SetDevice(device);
}

DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  DeviceIndex cur_device = -1;
  C10_MUSA_CHECK(c10::musa::GetDevice(&cur_device));
  if (to_device == cur_device) {
    return cur_device;
  }
  C10_MUSA_CHECK(musaSetDevice(to_device));
  return cur_device;
}

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  return c10::musa::ExchangeDevice(to_device);
}

void SetTargetDevice() {
  // no-op on CUDA version < 12.x
}
#endif

} // namespace c10::musa
