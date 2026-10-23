#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#if defined(__MUSACC__)
#include <musa.h>
#include <musa_fp16.h>
#include <c10/musa/MUSAMathCompat.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <c10/hip/HIPMathCompat.h>
#endif

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

void copysign_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return c10::musa::compat::copysign(a, b);
    });
  });
}

REGISTER_DISPATCH(copysign_stub, &copysign_kernel_cuda)

} // namespace at::native
