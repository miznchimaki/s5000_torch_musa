#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/musa/MemPool.h>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::at::musa::MemPool>(torch_C_m, "_MemPool")
      .def(py::init(
          [](std::shared_ptr<c10::musa::MUSACachingAllocator::CUDAAllocator>
                 allocator,
             bool is_user_created,
             bool use_on_oom,
             bool no_split) {
            torch::utils::device_lazy_init(at::kCUDA);
            return std::make_shared<::at::musa::MemPool>(
                std::move(allocator), is_user_created, use_on_oom, no_split);
          }))
      .def_property_readonly("id", &::at::musa::MemPool::id)
      .def("use_count", &::at::musa::MemPool::use_count);
}
