#pragma once

#include <ATen/Config.h>

#include <string>
#include <stdexcept>
#include <sstream>
#include <mufft.h>
#include <mufftXt.h>

namespace at { namespace native {

// This means that max dim is 3 + 2 = 5 with batch dimension and possible
// complex dimension
constexpr int max_rank = 3;

static inline std::string _cudaGetErrorEnum(mufftResult error)
{
  switch (error)
  {
    case MUFFT_SUCCESS:
      return "MUFFT_SUCCESS";
    case MUFFT_INVALID_PLAN:
      return "MUFFT_INVALID_PLAN";
    case MUFFT_ALLOC_FAILED:
      return "MUFFT_ALLOC_FAILED";
    case MUFFT_INVALID_TYPE:
      return "MUFFT_INVALID_TYPE";
    case MUFFT_INVALID_VALUE:
      return "MUFFT_INVALID_VALUE";
    case MUFFT_INTERNAL_ERROR:
      return "MUFFT_INTERNAL_ERROR";
    case MUFFT_EXEC_FAILED:
      return "MUFFT_EXEC_FAILED";
    case MUFFT_SETUP_FAILED:
      return "MUFFT_SETUP_FAILED";
    case MUFFT_INVALID_SIZE:
      return "MUFFT_INVALID_SIZE";
    case MUFFT_UNALIGNED_DATA:
      return "MUFFT_UNALIGNED_DATA";
    case MUFFT_INVALID_DEVICE:
      return "MUFFT_INVALID_DEVICE";
    case MUFFT_NO_WORKSPACE:
      return "MUFFT_NO_WORKSPACE";
    case MUFFT_NOT_IMPLEMENTED:
      return "MUFFT_NOT_IMPLEMENTED";
#if MUSA_COMP_VERSION <= 12090
    case MUFFT_INCOMPLETE_PARAMETER_LIST:
      return "MUFFT_INCOMPLETE_PARAMETER_LIST";
    case MUFFT_PARSE_ERROR:
      return "MUFFT_PARSE_ERROR";
#endif
#if !defined(USE_ROCM) && MUSA_COMP_VERSION <= 12090
    case MUFFT_LICENSE_ERROR:
      return "MUFFT_LICENSE_ERROR";
#endif
    case MUFFT_NOT_SUPPORTED:
      return "MUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(mufftResult error)
{
  if (error != MUFFT_SUCCESS) {
    std::ostringstream ss;
    ss << "cuFFT error: " << _cudaGetErrorEnum(error);
    TORCH_CHECK(false, ss.str());
  }
}

}} // at::native
