#pragma once
// this file is to avoid circular dependency between CUDAFunctions.h and
// CUDAExceptions.h

#include <c10/musa/MUSAMacros.h>
#include <musa_runtime.h>

#include <mutex>
#include <string>

namespace c10::musa {
C10_MUSA_API std::string get_cuda_error_help(musaError_t /*error*/) noexcept;
C10_MUSA_API const char* get_cuda_check_suffix() noexcept;
C10_MUSA_API std::mutex* getFreeMutex();
} // namespace c10::musa
