#pragma once

#include <ATen/core/TensorBase.h>

namespace at::musa::detail {

float *get_cublas_device_one();
float *get_cublas_device_zero();
float *get_user_alpha_ptr();

} // namespace at::musa::detail
