#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <c10/core/ScalarType.h>

#include <musa.h>
#include <library_types.h>

namespace at::musa {

template <typename scalar_t>
musaDataType getCudaDataType() {
  static_assert(false && sizeof(scalar_t), "Cannot convert type to musaDataType.");
  return {};
}

template<> inline musaDataType getCudaDataType<at::Half>() {
  return MUSA_R_16F;
}
template<> inline musaDataType getCudaDataType<float>() {
  return MUSA_R_32F;
}
template<> inline musaDataType getCudaDataType<double>() {
  return MUSA_R_64F;
}
template<> inline musaDataType getCudaDataType<c10::complex<c10::Half>>() {
  return MUSA_C_16F;
}
template<> inline musaDataType getCudaDataType<c10::complex<float>>() {
  return MUSA_C_32F;
}
template<> inline musaDataType getCudaDataType<c10::complex<double>>() {
  return MUSA_C_64F;
}

template<> inline musaDataType getCudaDataType<uint8_t>() {
  return MUSA_R_8U;
}
template<> inline musaDataType getCudaDataType<int8_t>() {
  return MUSA_R_8I;
}
template<> inline musaDataType getCudaDataType<int>() {
  return MUSA_R_32I;
}

template<> inline musaDataType getCudaDataType<int16_t>() {
  return MUSA_R_16I;
}
template<> inline musaDataType getCudaDataType<int64_t>() {
  return MUSA_R_64I;
}
template<> inline musaDataType getCudaDataType<at::BFloat16>() {
  return MUSA_R_16BF;
}

inline musaDataType ScalarTypeToCudaDataType(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return MUSA_R_8U;
    case c10::ScalarType::Char:
      return MUSA_R_8I;
    case c10::ScalarType::Int:
      return MUSA_R_32I;
    case c10::ScalarType::Half:
      return MUSA_R_16F;
    case c10::ScalarType::Float:
      return MUSA_R_32F;
    case c10::ScalarType::Double:
      return MUSA_R_64F;
    case c10::ScalarType::ComplexHalf:
      return MUSA_C_16F;
    case c10::ScalarType::ComplexFloat:
      return MUSA_C_32F;
    case c10::ScalarType::ComplexDouble:
      return MUSA_C_64F;
    case c10::ScalarType::Short:
      return MUSA_R_16I;
    case c10::ScalarType::Long:
      return MUSA_R_64I;
    case c10::ScalarType::BFloat16:
      return MUSA_R_16BF;
#if !defined(USE_ROCM) || ROCM_VERSION >= 60300
    case c10::ScalarType::Float8_e4m3fn:
      return MUSA_R_8F_E4M3;
    case c10::ScalarType::Float8_e5m2:
      return MUSA_R_8F_E5M2;
#endif
#if defined(USE_ROCM)
    case c10::ScalarType::Float8_e4m3fnuz:
      return HIP_R_8F_E4M3_FNUZ;
    case c10::ScalarType::Float8_e5m2fnuz:
      return HIP_R_8F_E5M2_FNUZ;
#endif
#if 0 || (defined(USE_ROCM) && ROCM_VERSION >= 70000)
    case c10::ScalarType::Float4_e2m1fn_x2:
      return CUDA_R_4F_E2M1;
#endif
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to musaDataType.")
  }
}

} // namespace at::musa

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
