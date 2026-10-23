#include <ATen/musa/MUSADataType.h>
#include <ATen/musa/MUSASparseDescriptors.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/musa/MiscUtils.h>

namespace at::musa::sparse {

musparseStatus_t destroyConstDnMat(const musparseDnMatDescr* dnMatDescr) {
  // NOLINTNEXTLINE(*const-cast)
  return musparseDestroyDnMat(const_cast<musparseDnMatDescr*>(dnMatDescr));
}

namespace {

// If a specific GPU model does not provide native support for a given data
// type, cuSparse routines return MUSPARSE_STATUS_ARCH_MISMATCH error
void check_supported_cuda_type(musaDataType cuda_type) {
  if (cuda_type == MUSA_R_16F) {
    musaDeviceProp* prop = at::musa::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major >= 3 && ((10 * prop->major + prop->minor) >= 31),
        "Sparse operations with CUDA tensors of Float16 type are not supported on GPUs with compute capability < 3.1 (current: ",
        prop->major,
        ".",
        prop->minor,
        ")");
  }
#if !defined(USE_ROCM)
  if (cuda_type == MUSA_R_16BF) {
    musaDeviceProp* prop = at::musa::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major >= 3,
        "Sparse operations with CUDA tensors of BFloat16 type are not supported on GPUs with compute capability < 3.1 (current: ",
        prop->major,
        ".",
        prop->minor,
        ")");
  }
#endif
}

} // anonymous namespace

musparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type) {
  if (scalar_type == c10::ScalarType::Int) {
    return MUSPARSE_INDEX_32I;
  } else if (scalar_type == c10::ScalarType::Long) {
    return MUSPARSE_INDEX_64I;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Cannot convert type ", scalar_type, " to cusparseIndexType.");
  }
}

musparseDnMatDescr_t createRawDnMatDescriptor(const Tensor& input, int64_t batch_offset, bool is_const=false) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == kStrided);
  IntArrayRef input_strides = input.strides();
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim >= 2);
  auto rows = input_sizes[ndim - 2];
  auto cols = input_sizes[ndim - 1];

  bool is_column_major =
      at::native::is_blas_compatible_column_major_order(input);
  bool is_row_major = at::native::is_blas_compatible_row_major_order(input);
  TORCH_INTERNAL_ASSERT(
      is_column_major || is_row_major,
      "Expected either row or column major contiguous input.");

  auto leading_dimension =
      is_row_major ? input_strides[ndim - 2] : input_strides[ndim - 1];

  auto order = is_row_major ? MUSPARSE_ORDER_ROW : MUSPARSE_ORDER_COL;

  auto batch_stride = ndim > 2 && batch_offset >= 0 ? input_strides[ndim - 3] : 0;
  // NOLINTNEXTLINE(*const-cast)
  void* data_ptr = is_const ? const_cast<void*>(input.const_data_ptr()) : input.data_ptr();
  void* values_ptr = static_cast<char*>(data_ptr) +
      batch_offset * batch_stride * input.itemsize();

  musaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
  check_supported_cuda_type(value_type);

  // NOTE: Ideally, in the const case, we would use cusparseConstDnMatDescr_t
  // and cusparseCreateConstDnMat, but those were introduced in CUDA 12, and we
  // still need to support CUDA 11
  musparseDnMatDescr_t raw_descriptor = nullptr;
  TORCH_MUSASPARSE_CHECK(musparseCreateDnMat(
      &raw_descriptor,
      rows,
      cols,
      leading_dimension,
      values_ptr,
      value_type,
      order));

  if (ndim >= 3 && batch_offset == -1) {
    int batch_count =
        at::native::cuda_int_cast(at::native::batchCount(input), "batch_count");
    TORCH_MUSASPARSE_CHECK(musparseDnMatSetStridedBatch(
        raw_descriptor, batch_count, input_strides[ndim - 3]));
  }
  return raw_descriptor;
}

CuSparseDnMatDescriptor::CuSparseDnMatDescriptor(const Tensor& input, int64_t batch_offset) {
  descriptor_.reset(createRawDnMatDescriptor(input, batch_offset));
}

CuSparseConstDnMatDescriptor::CuSparseConstDnMatDescriptor(const Tensor& input, int64_t batch_offset) {
  descriptor_.reset(createRawDnMatDescriptor(input, batch_offset, /*is_const*/true));
}

CuSparseDnVecDescriptor::CuSparseDnVecDescriptor(const Tensor& input) {
  // cuSPARSE doesn't support batched vectors
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      input.dim() == 1 || (input.dim() == 2 && input.size(-1) == 1));

  // cuSPARSE doesn't support non-contiguous vectors
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_contiguous());

  musaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
  check_supported_cuda_type(value_type);

  musparseDnVecDescr_t raw_descriptor = nullptr;
  TORCH_MUSASPARSE_CHECK(musparseCreateDnVec(
      &raw_descriptor, input.numel(), input.data_ptr(), value_type));
  descriptor_.reset(raw_descriptor);
}

CuSparseSpMatCsrDescriptor::CuSparseSpMatCsrDescriptor(const Tensor& input, int64_t batch_offset) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_sparse_csr());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);

  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  auto rows = input_sizes[ndim - 2];
  auto cols = input_sizes[ndim - 1];

  auto crow_indices = input.crow_indices();
  auto col_indices = input.col_indices();
  auto values = input.values();
  auto nnz = values.size(-1);
  c10::MaybeOwned<Tensor> values_ = values.expect_contiguous();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());

  musparseIndexType_t index_type =
      getCuSparseIndexType(crow_indices.scalar_type());
  musaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
  check_supported_cuda_type(value_type);

  auto crow_indices_batch_stride = crow_indices.dim() >= 2 && batch_offset >= 0
      ? crow_indices.stride(-2)
      : 0;
  auto col_indices_batch_stride =
      col_indices.dim() >= 2 && batch_offset >= 0 ? col_indices.stride(-2) : 0;
  auto values_batch_stride =
      values.dim() >= 2 && batch_offset >= 0 ? values_->stride(-2) : 0;

  musparseSpMatDescr_t raw_descriptor = nullptr;
  TORCH_MUSASPARSE_CHECK(musparseCreateCsr(
      &raw_descriptor, // output descriptor
      rows,
      cols,
      nnz,
      // row offsets of the sparse matrix, size = rows + 1
      static_cast<char*>(crow_indices.data_ptr()) +
          batch_offset * crow_indices_batch_stride * crow_indices.itemsize(),
      // column indices of the sparse matrix, size = nnz
      static_cast<char*>(col_indices.data_ptr()) +
          batch_offset * col_indices_batch_stride * col_indices.itemsize(),
      // values of the sparse matrix, size = nnz
      static_cast<char*>(values_->data_ptr()) +
          batch_offset * values_batch_stride * values.itemsize(),
      index_type, // data type of row offsets index
      index_type, // data type of col indices
      MUSPARSE_INDEX_BASE_ZERO, // base index of row offset and col index
      value_type // data type of values
      ));

  if (ndim == 3 && batch_offset == -1) {
    int batch_count =
        at::native::cuda_int_cast(at::native::batchCount(input), "batch_count");
    if (crow_indices.dim() >= 2 || values.dim() >= 2 ||
        col_indices.dim() >= 2) {
      // cuSPARSE ignores the strides and uses only the first batch
      TORCH_INTERNAL_ASSERT(
          false,
          "Support for batched CSR indices and values is not implemented.");
      TORCH_MUSASPARSE_CHECK(musparseCsrSetStridedBatch(
          raw_descriptor,
          batch_count,
          crow_indices.stride(-2),
          values_->stride(-2)));
    } else {
      // cuSPARSE allows broadcasting of indices and values across batches for
      // batched matmul
      TORCH_MUSASPARSE_CHECK(
          musparseCsrSetStridedBatch(raw_descriptor, batch_count, 0, 0));
    }
  }

  descriptor_.reset(raw_descriptor);
}

} // namespace at::musa::sparse
