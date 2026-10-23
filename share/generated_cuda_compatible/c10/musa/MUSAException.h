#pragma once

#include "torch_musa/csrc/core/MUSADeviceAssertionHost.h"
#include <c10/musa/MUSAMacros.h>
#include "torch_musa/csrc/core/MUSAMiscFunctions.h"
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <musa.h>

// Note [CHECK macro]
// ~~~~~~~~~~~~~~~~~~
// This is a macro so that AT_ERROR can get accurate __LINE__
// and __FILE__ information.  We could split this into a short
// macro and a function implementation if we pass along __LINE__
// and __FILE__, but no one has found this worth doing.

// Used to denote errors from CUDA framework.
// This needs to be declared here instead util/Exception.h for proper conversion
// during hipify.
namespace c10 {
class C10_MUSA_API CUDAError : public c10::Error {
  using Error::Error;
};
} // namespace c10

#define C10_MUSA_CHECK(EXPR)                                        \
  do {                                                              \
    const musaError_t __err = EXPR;                                 \
    c10::musa::c10_musa_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        true);                                                      \
  } while (0)
// backwards compat due to hipify v2 changes, for extension projects
#define C10_HIP_CHECK C10_CUDA_CHECK

#define TORCH_MUSA_WARN(EXPR)                              \
  do {                                                         \
    const musaError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != musaSuccess)) {                  \
      [[maybe_unused]] auto error_unused = musaGetLastError(); \
      TORCH_WARN("CUDA warning: ", musaGetErrorString(__err)); \
    }                                                          \
  } while (0)

// Indicates that a CUDA error is handled in a non-standard way
#define C10_MUSA_ERROR_HANDLED(EXPR) EXPR

// Intentionally ignore a CUDA error
#define C10_MUSA_IGNORE_ERROR(EXPR)                                   \
  do {                                                                \
    const musaError_t __err = EXPR;                                   \
    if (C10_UNLIKELY(__err != musaSuccess)) {                         \
      [[maybe_unused]] musaError_t error_unused = musaGetLastError(); \
    }                                                                 \
  } while (0)

// Clear the last CUDA error
#define C10_CUDA_CLEAR_ERROR()                                      \
  do {                                                              \
    [[maybe_unused]] musaError_t error_unused = musaGetLastError(); \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_MUSA_KERNEL_LAUNCH_CHECK() C10_MUSA_CHECK(musaGetLastError())

/// Launches a CUDA kernel appending to it all the information need to handle
/// device-side assertion failures. Checks that the launch was successful.
#define TORCH_DSA_KERNEL_LAUNCH(                                      \
    kernel, blocks, threads, shared_mem, stream, ...)                 \
  do {                                                                \
    auto& launch_registry =                                           \
        c10::musa::CUDAKernelLaunchRegistry::get_singleton_ref();     \
    kernel<<<blocks, threads, shared_mem, stream>>>(                  \
        __VA_ARGS__,                                                  \
        launch_registry.get_uvm_assertions_ptr_for_current_device(),  \
        launch_registry.insert(                                       \
            __FILE__, __FUNCTION__, __LINE__, #kernel, stream.id())); \
    C10_MUSA_KERNEL_LAUNCH_CHECK();                                   \
  } while (0)

namespace c10::musa {

/// In the event of a CUDA failure, formats a nice error message about that
/// failure and also checks for device-side assertion failures
C10_MUSA_API void c10_musa_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions);

} // namespace c10::musa
