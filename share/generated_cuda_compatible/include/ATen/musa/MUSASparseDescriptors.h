#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <ATen/Tensor.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <ATen/musa/MUSASparse.h>

#include <c10/core/ScalarType.h>

#if defined(USE_ROCM)
#include <type_traits>
#endif

namespace at::musa::sparse {

template <typename T, musparseStatus_t (*destructor)(T*)>
struct CuSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_MUSASPARSE_CHECK(destructor(x));
    }
  }
};

template <typename T, musparseStatus_t (*destructor)(T*)>
class CuSparseDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuSparseDescriptorDeleter<T, destructor>> descriptor_;
};

template <typename T, musparseStatus_t (*destructor)(const T*)>
struct ConstCuSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_MUSASPARSE_CHECK(destructor(x));
    }
  }
};

template <typename T, musparseStatus_t (*destructor)(const T*)>
class ConstCuSparseDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, ConstCuSparseDescriptorDeleter<T, destructor>> descriptor_;
};

#if defined(USE_ROCM)
using musparseMatDescr = std::remove_pointer_t<hipsparseMatDescr_t>;
using musparseDnMatDescr = std::remove_pointer_t<hipsparseDnMatDescr_t>;
using musparseDnVecDescr = std::remove_pointer_t<hipsparseDnVecDescr_t>;
using musparseSpMatDescr = std::remove_pointer_t<hipsparseSpMatDescr_t>;
using musparseSpMatDescr = std::remove_pointer_t<hipsparseSpMatDescr_t>;
using musparseSpGEMMDescr = std::remove_pointer_t<hipsparseSpGEMMDescr_t>;
using musparseSpSVDescr = std::remove_pointer_t<hipsparseSpSVDescr_t>;
using musparseSpSMDescr = std::remove_pointer_t<hipsparseSpSMDescr_t>;
using bsrsv2Info = std::remove_pointer_t<bsrsv2Info_t>;
using bsrsm2Info = std::remove_pointer_t<bsrsm2Info_t>;
#endif

// NOTE: This is only needed for CUDA 11 and earlier, since CUDA 12 introduced
// API for const descriptors
musparseStatus_t destroyConstDnMat(const musparseDnMatDescr* dnMatDescr);

class TORCH_CUDA_CPP_API CuSparseMatDescriptor
    : public CuSparseDescriptor<musparseMatDescr, &musparseDestroyMatDescr> {
 public:
  CuSparseMatDescriptor() {
    musparseMatDescr_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseCreateMatDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }

  CuSparseMatDescriptor(bool upper, bool unit) {
    musparseFillMode_t fill_mode =
        upper ? MUSPARSE_FILL_MODE_UPPER : MUSPARSE_FILL_MODE_LOWER;
    musparseDiagType_t diag_type =
        unit ? MUSPARSE_DIAG_TYPE_UNIT : MUSPARSE_DIAG_TYPE_NON_UNIT;
    musparseMatDescr_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseCreateMatDescr(&raw_descriptor));
    TORCH_MUSASPARSE_CHECK(musparseSetMatFillMode(raw_descriptor, fill_mode));
    TORCH_MUSASPARSE_CHECK(musparseSetMatDiagType(raw_descriptor, diag_type));
    descriptor_.reset(raw_descriptor);
  }
};


class TORCH_CUDA_CPP_API CuSparseBsrsv2Info
    : public CuSparseDescriptor<bsrsv2Info, &musparseDestroyBsrsv2Info> {
 public:
  CuSparseBsrsv2Info() {
    bsrsv2Info_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseCreateBsrsv2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

class TORCH_CUDA_CPP_API CuSparseBsrsm2Info
    : public CuSparseDescriptor<bsrsm2Info, &musparseDestroyBsrsm2Info> {
 public:
  CuSparseBsrsm2Info() {
    bsrsm2Info_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseCreateBsrsm2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};


musparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type);

  class TORCH_CUDA_CPP_API CuSparseDnMatDescriptor
      : public ConstCuSparseDescriptor<
            musparseDnMatDescr,
            &musparseDestroyDnMat> {
   public:
    explicit CuSparseDnMatDescriptor(
        const Tensor& input,
        int64_t batch_offset = -1);
  };

  class TORCH_CUDA_CPP_API CuSparseConstDnMatDescriptor
      : public ConstCuSparseDescriptor<
            const musparseDnMatDescr,
            &destroyConstDnMat> {
   public:
    explicit CuSparseConstDnMatDescriptor(
        const Tensor& input,
        int64_t batch_offset = -1);
  musparseDnMatDescr* unsafe_mutable_descriptor() const {
    return const_cast<musparseDnMatDescr*>(descriptor());
  }
  musparseDnMatDescr* unsafe_mutable_descriptor() {
    return const_cast<musparseDnMatDescr*>(descriptor());
  }
  };

  class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
      : public ConstCuSparseDescriptor<
            musparseDnVecDescr,
            &musparseDestroyDnVec> {
   public:
    explicit CuSparseDnVecDescriptor(const Tensor& input);
  };

  class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
      : public ConstCuSparseDescriptor<
            musparseSpMatDescr,
            &musparseDestroySpMat> {};

class TORCH_CUDA_CPP_API CuSparseSpMatCsrDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatCsrDescriptor(const Tensor& input, int64_t batch_offset = -1);

  std::tuple<int64_t, int64_t, int64_t> get_size() {
    int64_t rows = 0, cols = 0, nnz = 0;
    TORCH_MUSASPARSE_CHECK(musparseSpMatGetSize(
        this->descriptor(),
        &rows,
        &cols,
        &nnz));
    return std::make_tuple(rows, cols, nnz);
  }

  void set_tensor(const Tensor& input) {
    auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    TORCH_MUSASPARSE_CHECK(musparseCsrSetPointers(
        this->descriptor(),
        crow_indices.data_ptr(),
        col_indices.data_ptr(),
        values.data_ptr()));
  }

#if AT_USE_CUSPARSE_GENERIC_SPSV()
  void set_mat_fill_mode(bool upper) {
    musparseFillMode_t fill_mode =
        upper ? MUSPARSE_FILL_MODE_UPPER : MUSPARSE_FILL_MODE_LOWER;
    TORCH_MUSASPARSE_CHECK(musparseSpMatSetAttribute(
        this->descriptor(),
        MUSPARSE_SPMAT_FILL_MODE,
        &fill_mode,
        sizeof(fill_mode)));
  }

  void set_mat_diag_type(bool unit) {
    musparseDiagType_t diag_type =
        unit ? MUSPARSE_DIAG_TYPE_UNIT : MUSPARSE_DIAG_TYPE_NON_UNIT;
    TORCH_MUSASPARSE_CHECK(musparseSpMatSetAttribute(
        this->descriptor(),
        MUSPARSE_SPMAT_DIAG_TYPE,
        &diag_type,
        sizeof(diag_type)));
  }
#endif
};

#if AT_USE_CUSPARSE_GENERIC_SPSV()
class TORCH_CUDA_CPP_API CuSparseSpSVDescriptor
    : public CuSparseDescriptor<musparseSpSVDescr, &musparseSpSV_destroyDescr> {
 public:
  CuSparseSpSVDescriptor() {
    musparseSpSVDescr_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseSpSV_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

#if AT_USE_CUSPARSE_GENERIC_SPSM()
class TORCH_CUDA_CPP_API CuSparseSpSMDescriptor
    : public CuSparseDescriptor<musparseSpSMDescr, &musparseSpSM_destroyDescr> {
 public:
  CuSparseSpSMDescriptor() {
    musparseSpSMDescr_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseSpSM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

class TORCH_CUDA_CPP_API CuSparseSpGEMMDescriptor
    : public CuSparseDescriptor<musparseSpGEMMDescr, &musparseSpGEMM_destroyDescr> {
 public:
  CuSparseSpGEMMDescriptor() {
    musparseSpGEMMDescr_t raw_descriptor = nullptr;
    TORCH_MUSASPARSE_CHECK(musparseSpGEMM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

} // namespace at::musa::sparse

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
