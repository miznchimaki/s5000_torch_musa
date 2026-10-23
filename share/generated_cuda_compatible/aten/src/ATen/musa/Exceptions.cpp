//NS: MUSACachingAllocator must be included before to get MUSART_VERSION definedi

#include "torch_musa/csrc/aten/musa/Exceptions.h"

namespace at::musa {
namespace blas {

C10_EXPORT const char* _cublasGetErrorEnum(mublasStatus_t error) {
  if (error == MUBLAS_STATUS_SUCCESS) {
    return "MUBLAS_STATUS_SUCCESS";
  }
  if (error == MUBLAS_STATUS_NOT_INITIALIZED) {
    return "MUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == MUBLAS_STATUS_ALLOC_FAILED) {
    return "MUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == MUBLAS_STATUS_INVALID_VALUE) {
    return "MUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == MUBLAS_STATUS_ARCH_MISMATCH) {
    return "MUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == MUBLAS_STATUS_MAPPING_ERROR) {
    return "MUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == MUBLAS_STATUS_EXECUTION_FAILED) {
    return "MUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == MUBLAS_STATUS_INTERNAL_ERROR) {
    return "MUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == MUBLAS_STATUS_NOT_SUPPORTED) {
    return "MUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef MUBLAS_STATUS_LICENSE_ERROR
  if (error == MUBLAS_STATUS_LICENSE_ERROR) {
    return "MUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

} // namespace blas

namespace solver {
#if !defined(USE_ROCM)

C10_EXPORT const char* cusolverGetErrorMessage(musolverStatus_t status) {
  switch (status) {
    case MUSOLVER_STATUS_SUCCESS:                     return "MUSOLVER_STATUS_SUCCESS";
    case MUSOLVER_STATUS_NOT_INITIALIZED:             return "MUSOLVER_STATUS_NOT_INITIALIZED";
    case MUSOLVER_STATUS_ALLOC_FAILED:                return "MUSOLVER_STATUS_ALLOC_FAILED";
    case MUSOLVER_STATUS_INVALID_VALUE:               return "MUSOLVER_STATUS_INVALID_VALUE";
    case MUSOLVER_STATUS_ARCH_MISMATCH:               return "MUSOLVER_STATUS_ARCH_MISMATCH";
    case MUSOLVER_STATUS_EXECUTION_FAILED:            return "MUSOLVER_STATUS_EXECUTION_FAILED";
    case MUSOLVER_STATUS_INTERNAL_ERROR:              return "MUSOLVER_STATUS_INTERNAL_ERROR";
    case MUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "MUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:                                          return "Unknown cusolver error number";
  }
}

#else

C10_EXPORT const char* hipsolverGetErrorMessage(hipsolverStatus_t status) {
  switch (status) {
    case HIPSOLVER_STATUS_SUCCESS:                    return "HIPSOLVER_STATUS_SUCCESS";
    case HIPSOLVER_STATUS_NOT_INITIALIZED:            return "HIPSOLVER_STATUS_NOT_INITIALIZED";
    case HIPSOLVER_STATUS_ALLOC_FAILED:               return "HIPSOLVER_STATUS_ALLOC_FAILED";
    case HIPSOLVER_STATUS_INVALID_VALUE:              return "HIPSOLVER_STATUS_INVALID_VALUE";
    case HIPSOLVER_STATUS_MAPPING_ERROR:              return "HIPSOLVER_STATUS_MAPPING_ERROR";
    case HIPSOLVER_STATUS_EXECUTION_FAILED:           return "HIPSOLVER_STATUS_EXECUTION_FAILED";
    case HIPSOLVER_STATUS_INTERNAL_ERROR:             return "HIPSOLVER_STATUS_INTERNAL_ERROR";
    case HIPSOLVER_STATUS_NOT_SUPPORTED:              return "HIPSOLVER_STATUS_NOT_SUPPORTED";
    case HIPSOLVER_STATUS_ARCH_MISMATCH:              return "HIPSOLVER_STATUS_ARCH_MISMATCH";
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:          return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
    case HIPSOLVER_STATUS_INVALID_ENUM:               return "HIPSOLVER_STATUS_INVALID_ENUM";
    case HIPSOLVER_STATUS_UNKNOWN:                    return "HIPSOLVER_STATUS_UNKNOWN";
    case HIPSOLVER_STATUS_ZERO_PIVOT:                 return "HIPSOLVER_STATUS_ZERO_PIVOT";
    default:                                          return "Unknown hipsolver error number";
  }
}

#endif
} // namespace solver

#if defined(USE_CUDSS)
namespace cudss {

C10_EXPORT const char* cudssGetErrorMessage(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS:                     return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:             return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:                return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:               return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:               return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:            return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:              return "CUDSS_STATUS_INTERNAL_ERROR";
    default:                                       return "Unknown cudss error number";
  }
}

} // namespace cudss
#endif

} // namespace at::musa
