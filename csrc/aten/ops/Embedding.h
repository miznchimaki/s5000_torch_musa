#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using embedding_dense_backward_fn =
    Tensor (*)(const Tensor&, const Tensor&, int64_t, int64_t, bool);

DECLARE_DISPATCH(embedding_dense_backward_fn, embedding_dense_backward_stub);

} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_
