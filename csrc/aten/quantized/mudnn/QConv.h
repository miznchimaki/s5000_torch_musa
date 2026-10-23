#ifndef Q_CONV
#define Q_CONV

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {
using conv_kernel_fn = void (*)(
    at::Tensor,
    double,
    int64_t,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor&,
    IntArrayRef,
    IntArrayRef,
    IntArrayRef,
    int64_t,
    double,
    int64_t,
    c10::ScalarType,
    const std::string_view&);
DECLARE_DISPATCH(conv_kernel_fn, conv_stub);

using conv_binary_kernel_fn = void (*)(
    at::Tensor,
    double,
    int64_t,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor&,
    IntArrayRef,
    IntArrayRef,
    IntArrayRef,
    int64_t,
    double,
    int64_t,
    c10::ScalarType,
    const std::string_view&,
    double,
    int64_t,
    double,
    const std::string_view&);
DECLARE_DISPATCH(conv_binary_kernel_fn, conv_binary_stub);
} // namespace native
} // namespace at
#endif