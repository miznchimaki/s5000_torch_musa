#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/musa/MUSAGraph.h>
#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/musa/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::musa::graph_pool_handle);

  shared_ptr_class_<::at::musa::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::at::musa::CUDAGraph& self,
             std::optional<c10::musa::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            musaStreamCaptureMode capture_mode{};
            c10::musa::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::musa::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = musaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = musaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = musaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::musa::CUDAGraph::capture_end))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(&at::musa::CUDAGraph::instantiate))
      .def(
          "register_generator_state",
          [](::at::musa::CUDAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::musa::CUDAGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::musa::CUDAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::musa::CUDAGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::musa::CUDAGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::musa::CUDAGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::musa::CUDAGraph::debug_dump),
          py::arg("debug_path"))
      .def(
          "raw_cuda_graph",
          [](::at::musa::CUDAGraph& self) {
            musaGraph_t graph = self.raw_cuda_graph();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of musaGraph_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "raw_cuda_graph_exec",
          [](::at::musa::CUDAGraph& self) {
            musaGraphExec_t graph_exec = self.raw_cuda_graph_exec();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of musaGraphExec_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph_exec);
          },
          py::call_guard<py::gil_scoped_release>());
}
