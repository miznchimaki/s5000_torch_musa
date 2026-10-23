#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/musa/JitLoops.muh>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/musa/Math.muh>
#include <ATen/native/musa/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/musa/MUSAMathCompat.h>
#include <c10/util/complex.h>

namespace at::native {
        namespace {
            constexpr char modified_bessel_k1_name[] = "modified_bessel_k1_forward";

            void modified_bessel_k1_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k1_cuda", [&]() {
                    jitted_gpu_kernel<modified_bessel_k1_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_k1_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k1_cuda", [&]() {
                    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
                        return modified_bessel_k1_forward(a);
                    });
                });
#endif // AT_USE_JITERATOR()
            }
        }

        REGISTER_DISPATCH(special_modified_bessel_k1_stub, &modified_bessel_k1_kernel_cuda)
} // namespace at::native
