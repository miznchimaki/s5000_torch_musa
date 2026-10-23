// Just a little test file to make sure that the CUDA library works

#include "torch_musa/csrc/core/MUSAException.h"
#include <c10/musa/impl/CUDATest.h>

#include <musa_runtime.h>

namespace c10::musa::impl {

static bool has_cuda_gpu() {
  int count = 0;
  C10_MUSA_IGNORE_ERROR(musaGetDeviceCount(&count));

  return count != 0;
}

int c10_cuda_test() {
  int r = 0;
  if (has_cuda_gpu()) {
    C10_MUSA_CHECK(musaGetDevice(&r));
  }
  return r;
}

} // namespace c10::musa::impl
