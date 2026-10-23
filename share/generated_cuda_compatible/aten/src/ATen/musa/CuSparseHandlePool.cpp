#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <ATen/musa/detail/DeviceThreadHandles.h>

namespace at::musa {
namespace {

void createCusparseHandle(musparseHandle_t *handle) {
  TORCH_MUSASPARSE_CHECK(musparseCreate(handle));
}

void destroyCusparseHandle(musparseHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the musa context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_MUDNN_DESTROY_HANDLE
#else
    musparseDestroy(handle);
#endif
}

using CuSparsePoolType = DeviceThreadHandlePool<musparseHandle_t, createCusparseHandle, destroyCusparseHandle>;

} // namespace

musparseHandle_t getCurrentCUDASparseHandle() {
  c10::DeviceIndex device = 0;
  AT_MUSA_CHECK(c10::musa::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSparsePoolType>();
  thread_local std::unique_ptr<CuSparsePoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  TORCH_MUSASPARSE_CHECK(musparseSetStream(handle, c10::musa::getCurrentMUSAStream()));
  return handle;
}

} // namespace at::musa
