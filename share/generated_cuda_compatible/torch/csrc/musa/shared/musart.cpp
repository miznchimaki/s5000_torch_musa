#include <musa.h>
#include <musa_runtime.h>
#include <torch/csrc/utils/pybind.h>
#if !defined(USE_ROCM)
#include <musa_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace torch::musa::shared {

#ifdef USE_ROCM
namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace
#endif

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

  py::enum_<musaError_t>(
      cudart,
      "musa"
      "Error")
      .value("success", musaSuccess);

  cudart.def(
      "musa"
      "GetErrorString",
      musaGetErrorString);
  cudart.def(
      "musa"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      musaProfilerStart
#endif
  );
  cudart.def(
      "musa"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      musaProfilerStop
#endif
  );
  cudart.def(
      "musa"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> musaError_t {
        py::gil_scoped_release no_gil;
        return C10_MUSA_ERROR_HANDLED(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            musaHostRegister((void*)ptr, size, flags));
      });
  cudart.def(
      "musa"
      "HostUnregister",
      [](uintptr_t ptr) -> musaError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_MUSA_ERROR_HANDLED(musaHostUnregister((void*)ptr));
      });
  cudart.def(
      "musa"
      "StreamCreate",
      [](uintptr_t ptr) -> musaError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_MUSA_ERROR_HANDLED(musaStreamCreate((musaStream_t*)ptr));
      });
  cudart.def(
      "musa"
      "StreamDestroy",
      [](uintptr_t ptr) -> musaError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_MUSA_ERROR_HANDLED(musaStreamDestroy((musaStream_t)ptr));
      });
  cudart.def(
      "musa"
      "MemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        c10::musa::MUSAGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        py::gil_scoped_release no_gil;
        C10_MUSA_CHECK(musaMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::musa::shared
