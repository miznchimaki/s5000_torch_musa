#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <ATen/musa/detail/DeviceThreadHandles.h>

namespace at::musa {
namespace {

void createCusolverDnHandle(musolverDnHandle_t *handle) {
  TORCH_MUSOLVER_CHECK(musolverDnCreate(handle));
}

void destroyCusolverDnHandle(musolverDnHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the musa context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_MUDNN_DESTROY_HANDLE
  (void)handle; // Suppress unused variable warning
#else
    musolverDnDestroy(handle);
#endif
}

using CuSolverDnPoolType = DeviceThreadHandlePool<musolverDnHandle_t, createCusolverDnHandle, destroyCusolverDnHandle>;

} // namespace

musolverDnHandle_t getCurrentCUDASolverDnHandle() {
  c10::DeviceIndex device = 0;
  AT_MUSA_CHECK(c10::musa::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSolverDnPoolType>();
  thread_local std::unique_ptr<CuSolverDnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::musa::getCurrentMUSAStream();
  TORCH_MUSOLVER_CHECK(musolverDnSetStream(handle, stream));
  return handle;
}

} // namespace at::musa
