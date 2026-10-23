#include "torch_musa/csrc/core/MUSAException.h"

#include "torch_musa/csrc/core/MUSADeviceAssertionHost.h"
#include <c10/util/Exception.h>

#include <string>

namespace c10::musa {

void c10_musa_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions) {
  const auto cuda_error = static_cast<musaError_t>(err);
  const auto cuda_kernel_failure = include_device_assertions
      ? c10::musa::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  if (C10_LIKELY(cuda_error == musaSuccess && !cuda_kernel_failure)) {
    return;
  }

  [[maybe_unused]] auto error_unused = musaGetLastError();

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("CUDA error: ");
  const char* error_string = musaGetErrorString(cuda_error);
  check_message.append(error_string);
  check_message.append(c10::musa::get_cuda_error_help(cuda_error));
  check_message.append(c10::musa::get_cuda_check_suffix());
  check_message.append("\n");
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif
  throw c10::AcceleratorError(
      {function_name, filename, line_number}, err, check_message);
}

} // namespace c10::musa
