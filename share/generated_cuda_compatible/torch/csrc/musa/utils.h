#pragma once

#include "torch_musa/csrc/core/MUSAStream.h"
#include <torch/csrc/utils/python_numbers.h>

#include <vector>

std::vector<std::optional<at::musa::MUSAStream>>
THPUtils_PySequence_to_CUDAStreamList(PyObject* obj);
