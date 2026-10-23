#pragma once

#include <torch/csrc/utils/pythoncapi_compat.h>
namespace torch::musa::python {

void initCommMethods(PyObject* module);

} // namespace torch::musa::python
