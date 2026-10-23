#include <ATen/musa/CUDAGreenContext.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

// Cargo culted partially from csrc/musa/Stream.cpp

void THCPGreenContext_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<at::musa::GreenContext>(m, "_CUDAGreenContext")
      .def_static("create", &::at::musa::GreenContext::create)
      .def("set_context", &::at::musa::GreenContext::setContext)
      .def("pop_context", &::at::musa::GreenContext::popContext)
      .def("Stream", [](at::musa::GreenContext& self) {
        auto s = self.Stream();
        musaStream_t raw = self.Stream();
        auto ptr_val = reinterpret_cast<uintptr_t>(raw);

        py::object torch_cuda = py::module::import("torch.musa");
        py::object ExternalStream = torch_cuda.attr("ExternalStream");

        return ExternalStream(ptr_val, py::int_(s.device_index()));
      });
}
