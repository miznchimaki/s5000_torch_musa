#include "torch_musa/csrc/aten/musa/MUSAContextLight.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/utils/cpp_stacktraces.h>
#include <cstring>

namespace {
// Helper to call the appropriate check implementation for CUDA vs ROCm.
// This is done in a separate function to avoid preprocessor directives inside
// macro (AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE) arguments, which
// is undefined behavior and fails on MSVC.
inline void call_c10_accelerator_check_implementation(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions) {
  c10::musa::c10_musa_check_implementation(
      err, filename, function_name, line_number, include_device_assertions);
}
} // namespace

AOTITorchError torch_get_current_musa_blas_handle(void** ret_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(mublasHandle_t*)(ret_handle) = at::musa::getCurrentMUSABlasHandle();
  });
}

AOTITorchError torch_set_current_musa_stream(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::musa::setCurrentMUSAStream(at::musa::getStreamFromExternal(
        static_cast<musaStream_t>(stream), device_index));
  });
}

AOTITorchError torch_get_musa_stream_from_pool(
    const bool isHighPriority,
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(musaStream_t*)(ret_stream) =
        at::musa::getStreamFromPool(isHighPriority, device_index);
  });
}

AOTITorchError torch_musa_stream_synchronize(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::musa::getStreamFromExternal(
        static_cast<musaStream_t>(stream), device_index)
        .synchronize();
  });
}

AOTITorchError torch_c10_musa_check_msg(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions,
    char** error_msg) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *error_msg = nullptr;

    try {
      call_c10_accelerator_check_implementation(
          err, filename, function_name, line_number, include_device_assertions);
    } catch (const c10::AcceleratorError& e) {
      // Match the behavior of Python exception translation:
      // use what() if C++ stacktraces are enabled, otherwise
      // what_without_backtrace()
      const char* what_str = torch::get_cpp_stacktraces_enabled()
          ? e.what()
          : e.what_without_backtrace();
      size_t msg_len = std::strlen(what_str);
      *error_msg = new char[msg_len + 1];
      std::memcpy(*error_msg, what_str, msg_len + 1);
    }
  });
}

void torch_c10_musa_free_error_msg(char* error_msg) {
  delete[] error_msg;
}
