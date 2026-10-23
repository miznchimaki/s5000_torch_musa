/*
  Provides the implementations of cuSPARSE function templates.
*/

#include <ATen/musa/MUSASparseBlas.h>

namespace at::musa::sparse {

template <>
void csrgeam2_bufferSizeExt<float>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseScsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2_bufferSizeExt<double>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2_bufferSizeExt<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      reinterpret_cast<const muComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const muComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const muComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const muComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<const muComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2_bufferSizeExt<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const muDoubleComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const muDoubleComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const muDoubleComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<const muDoubleComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2<float>(CUSPARSE_CSRGEAM2_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseScsrgeam2(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void csrgeam2<double>(CUSPARSE_CSRGEAM2_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDcsrgeam2(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void csrgeam2<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCcsrgeam2(
      handle,
      m,
      n,
      reinterpret_cast<const muComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const muComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const muComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const muComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<muComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void csrgeam2<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZcsrgeam2(
      handle,
      m,
      n,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const muDoubleComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const muDoubleComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const muDoubleComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<muDoubleComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void bsrmm<float>(CUSPARSE_BSRMM_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      B,
      ldb,
      beta,
      C,
      ldc));
}

template <>
void bsrmm<double>(CUSPARSE_BSRMM_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      B,
      ldb,
      beta,
      C,
      ldc));
}

template <>
void bsrmm<c10::complex<float>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      reinterpret_cast<const muComplex*>(alpha),
      descrA,
      reinterpret_cast<const muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const muComplex*>(B),
      ldb,
      reinterpret_cast<const muComplex*>(beta),
      reinterpret_cast<muComplex*>(C),
      ldc));
}

template <>
void bsrmm<c10::complex<double>>(
    CUSPARSE_BSRMM_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const muDoubleComplex*>(B),
      ldb,
      reinterpret_cast<const muDoubleComplex*>(beta),
      reinterpret_cast<muDoubleComplex*>(C),
      ldc));
}

template <>
void bsrmv<float>(CUSPARSE_BSRMV_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      x,
      beta,
      y));
}

template <>
void bsrmv<double>(CUSPARSE_BSRMV_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      x,
      beta,
      y));
}

template <>
void bsrmv<c10::complex<float>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      reinterpret_cast<const muComplex*>(alpha),
      descrA,
      reinterpret_cast<const muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const muComplex*>(x),
      reinterpret_cast<const muComplex*>(beta),
      reinterpret_cast<muComplex*>(y)));
}

template <>
void bsrmv<c10::complex<double>>(
    CUSPARSE_BSRMV_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const muDoubleComplex*>(x),
      reinterpret_cast<const muDoubleComplex*>(beta),
      reinterpret_cast<muDoubleComplex*>(y)));
}

template <>
void bsrsv2_bufferSize<float>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<double>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_analysis<float>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_analysis<double>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<const muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<const muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<float>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      x,
      y,
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<double>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      x,
      y,
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<c10::complex<float>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      reinterpret_cast<const muComplex*>(alpha),
      descrA,
      reinterpret_cast<const muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const muComplex*>(x),
      reinterpret_cast<muComplex*>(y),
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<c10::complex<double>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const muDoubleComplex*>(x),
      reinterpret_cast<muDoubleComplex*>(y),
      policy,
      pBuffer));
}

template <>
void bsrsm2_bufferSize<float>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_bufferSize<double>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_analysis<float>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_analysis<double>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<const muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<const muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<float>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(float)) {
  TORCH_MUSASPARSE_CHECK(musparseSbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      B,
      ldb,
      X,
      ldx,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<double>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(double)) {
  TORCH_MUSASPARSE_CHECK(musparseDbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      B,
      ldb,
      X,
      ldx,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<c10::complex<float>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSASPARSE_CHECK(musparseCbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      reinterpret_cast<const muComplex*>(alpha),
      descrA,
      reinterpret_cast<const muComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const muComplex*>(B),
      ldb,
      reinterpret_cast<muComplex*>(X),
      ldx,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<c10::complex<double>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSASPARSE_CHECK(musparseZbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const muDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const muDoubleComplex*>(B),
      ldb,
      reinterpret_cast<muDoubleComplex*>(X),
      ldx,
      policy,
      pBuffer));
}

} // namespace at::musa::sparse
