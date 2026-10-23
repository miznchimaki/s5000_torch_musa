#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once
#include <musa.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#include <c10/util/Exception.h>

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    MUresult __err = EXPR;                                                 \
    if (__err != MUSA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      MUresult get_error_str_err [[maybe_unused]] =                        \
          c10::musa::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != MUSA_SUCCESS) {                             \
        TORCH_CHECK(false, "CUDA driver error: unknown error");            \
      } else {                                                             \
        TORCH_CHECK(false, "CUDA driver error: ", err_str);                \
      }                                                                    \
    }                                                                      \
  } while (0)

#define C10_CUDA_DRIVER_CHECK_GOTO(EXPR, NEXT)                             \
  do {                                                                     \
    MUresult __err = EXPR;                                                 \
    if (__err != MUSA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      MUresult get_error_str_err [[maybe_unused]] =                        \
          c10::musa::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != MUSA_SUCCESS) {                             \
        TORCH_WARN("CUDA driver error: unknown error");                    \
      } else {                                                             \
        TORCH_WARN("CUDA driver error: ", err_str);                        \
      }                                                                    \
      goto NEXT;                                                           \
    }                                                                      \
  } while (0)

// The integer in the second column specifies the requested CUDA Driver API
// version. The dynamic loader will accept a driver with a newer version, but it
// ensures that the requested symbol exists in *at least* the specified version
// or earlier.

// Keep these requested versions as low as possible to maximize compatibility
// across different driver versions.

// Why do we pin to an older version instead of using the latest?
// If a user installs a newer driver, blindly resolving the symbol may bind to a
// newer version of the function with different behavior, potentially breaking
// PyTorch.

#define C10_LIBCUDA_DRIVER_API_REQUIRED(_)         \
  _(muDeviceGet, 12000)                            \
  _(muDeviceGetAttribute, 12000)                   \
  _(muMemAddressReserve, 12000)                    \
  _(muMemRelease, 12000)                           \
  _(muMemMap, 12000)                               \
  _(muMemAddressFree, 12000)                       \
  _(muMemSetAccess, 12000)                         \
  _(muMemUnmap, 12000)                             \
  _(muMemCreate, 12000)                            \
  _(muMemGetAllocationGranularity, 12000)          \
  _(muMemExportToShareableHandle, 12000)           \
  _(muMemImportFromShareableHandle, 12000)         \
  _(muMemRetainAllocationHandle, 12000)            \
  _(muMemGetAllocationPropertiesFromHandle, 12000) \
  _(muMemsetD32Async, 12000)                       \
  _(muStreamWriteValue32, 12000)                   \
  _(muGetErrorString, 12000)

#if defined(MUSA_COMP_VERSION) && (MUSA_COMP_VERSION >= 12080)
#define C10_LIBCUDA_DRIVER_API_OPTIONAL(_) \
  _(cuCtxFromGreenCtx, 12080)              \
  _(muCtxGetCurrent, 12080)                \
  _(muCtxPopCurrent, 12080)                \
  _(muCtxPushCurrent, 12080)               \
  _(muCtxSetCurrent, 12080)                \
  _(cuGreenCtxCreate, 12080)               \
  _(cuGreenCtxDestroy, 12080)              \
  _(cuGreenCtxStreamCreate, 12080)         \
  _(cuDevSmResourceSplitByCount, 12080)    \
  _(cuDeviceGetDevResource, 12080)         \
  _(cuDevResourceGenerateDesc, 12080)      \
  _(cuMulticastAddDevice, 12030)           \
  _(cuMulticastBindMem, 12030)             \
  _(cuMulticastCreate, 12030)              \
  _(cuMulticastUnbind, 12030)
#elif defined(MUSA_COMP_VERSION) && (MUSA_COMP_VERSION >= 12030)
#define C10_LIBCUDA_DRIVER_API_OPTIONAL(_) \
  _(cuMulticastAddDevice, 12030)           \
  _(cuMulticastBindMem, 12030)             \
  _(cuMulticastCreate, 12030)              \
  _(cuMulticastUnbind, 12030)
#else
#define C10_LIBCUDA_DRIVER_API_OPTIONAL(_)
#endif

#define C10_NVML_DRIVER_API(_)            \
  _(nvmlInit_v2)                          \
  _(nvmlDeviceGetHandleByPciBusId_v2)     \
  _(nvmlDeviceGetNvLinkRemoteDeviceType)  \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2)  \
  _(nvmlDeviceGetComputeRunningProcesses) \
  _(nvmlSystemGetCudaDriverVersion_v2)

#if defined(MUSA_COMP_VERSION) && (MUSA_COMP_VERSION >= 12040)
#define C10_NVML_DRIVER_API_OPTIONAL(_) _(nvmlDeviceGetGpuFabricInfoV)
#else
#define C10_NVML_DRIVER_API_OPTIONAL(_)
#endif

namespace c10::musa {

struct DriverAPI {
#define CREATE_MEMBER_VERSIONED(name, version) decltype(&name) name##_;
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API_REQUIRED(CREATE_MEMBER_VERSIONED)
  C10_LIBCUDA_DRIVER_API_OPTIONAL(CREATE_MEMBER_VERSIONED)
  C10_NVML_DRIVER_API(CREATE_MEMBER)
  C10_NVML_DRIVER_API_OPTIONAL(CREATE_MEMBER)
#undef CREATE_MEMBER_VERSIONED
#undef CREATE_MEMBER

  static DriverAPI* get();
  static void* get_nvml_handle();
};

} // namespace c10::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
