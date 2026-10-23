#pragma once
#include <c10/macros/Export.h>
#include <cstdint>

namespace at::musa {

// enqueues a kernel that spins for the specified number of cycles
TORCH_CUDA_CU_API void sleep(int64_t cycles);

// flushes instruction cache for ROCm; no-op for CUDA
TORCH_CUDA_CU_API void flush_icache();

}  // namespace at::musa
