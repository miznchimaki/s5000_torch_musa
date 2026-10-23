#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/musa/linalg/CUDASolver.h>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include <c10/macros/Export.h>


namespace at::musa::solver {

template <>
void getrf<double>(
    musolverDnHandle_t handle, int m, int n, double* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(
      musolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);
  TORCH_MUSOLVER_CHECK(musolverDnDgetrf(
      handle, m, n, dA, ldda, static_cast<double*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<float>(
    musolverDnHandle_t handle, int m, int n, float* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(
      musolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);
  TORCH_MUSOLVER_CHECK(musolverDnSgetrf(
      handle, m, n, dA, ldda, static_cast<float*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<c10::complex<double>>(
    musolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<double>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnZgetrf_bufferSize(
      handle, m, n, reinterpret_cast<muDoubleComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muDoubleComplex) * lwork);
  TORCH_MUSOLVER_CHECK(musolverDnZgetrf(
      handle,
      m,
      n,
      reinterpret_cast<muDoubleComplex*>(dA),
      ldda,
      static_cast<muDoubleComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrf<c10::complex<float>>(
    musolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<float>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnCgetrf_bufferSize(
      handle, m, n, reinterpret_cast<muComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muComplex) * lwork);
  TORCH_MUSOLVER_CHECK(musolverDnCgetrf(
      handle,
      m,
      n,
      reinterpret_cast<muComplex*>(dA),
      ldda,
      static_cast<muComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrs<double>(
    musolverDnHandle_t handle, int n, int nrhs, double* dA, int lda, int* ipiv, double* ret, int ldb, int* info, mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(musolverDnDgetrs(
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<float>(
    musolverDnHandle_t handle, int n, int nrhs, float* dA, int lda, int* ipiv, float* ret, int ldb, int* info, mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(musolverDnSgetrs(
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<c10::complex<double>>(
    musolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<double>* dA,
    int lda,
    int* ipiv,
    c10::complex<double>* ret,
    int ldb,
    int* info,
    mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(musolverDnZgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<muDoubleComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<muDoubleComplex*>(ret),
      ldb,
      info));
}

template <>
void getrs<c10::complex<float>>(
    musolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<float>* dA,
    int lda,
    int* ipiv,
    c10::complex<float>* ret,
    int ldb,
    int* info,
    mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(musolverDnCgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<muComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<muComplex*>(ret),
      ldb,
      info));
}

template <>
void sytrf_bufferSize<double>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDsytrf_bufferSize(handle, n, A, lda, lwork));
}

template <>
void sytrf_bufferSize<float>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSsytrf_bufferSize(handle, n, A, lda, lwork));
}

template <>
void sytrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZsytrf_bufferSize(
      handle, n, reinterpret_cast<muDoubleComplex*>(A), lda, lwork));
}

template <>
void sytrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCsytrf_bufferSize(
      handle, n, reinterpret_cast<muComplex*>(A), lda, lwork));
}

template <>
void sytrf<double>(CUDASOLVER_SYTRF_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
}

template <>
void sytrf<float>(CUDASOLVER_SYTRF_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
}

template <>
void sytrf<c10::complex<double>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZsytrf(
      handle,
      uplo,
      n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      ipiv,
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      devInfo));
}

template <>
void sytrf<c10::complex<float>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCsytrf(
      handle,
      uplo,
      n,
      reinterpret_cast<muComplex*>(A),
      lda,
      ipiv,
      reinterpret_cast<muComplex*>(work),
      lwork,
      devInfo));
}

template<>
void gesvd_buffersize<float>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_MUSOLVER_CHECK(musolverDnSgesvd_bufferSize(handle, m, n, lwork));
}

template<>
void gesvd_buffersize<double>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_MUSOLVER_CHECK(musolverDnDgesvd_bufferSize(handle, m, n, lwork));
}

template<>
void gesvd_buffersize<c10::complex<float>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_MUSOLVER_CHECK(musolverDnCgesvd_bufferSize(handle, m, n, lwork));
}

template<>
void gesvd_buffersize<c10::complex<double>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_MUSOLVER_CHECK(musolverDnZgesvd_bufferSize(handle, m, n, lwork));
}


template<>
void gesvd<float>(CUDASOLVER_GESVD_ARGTYPES(float, float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSgesvd(
      handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info));
}

template<>
void gesvd<double>(CUDASOLVER_GESVD_ARGTYPES(double, double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDgesvd(
      handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info));
}


template<>
void gesvd<c10::complex<float>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<float>, float)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgesvd(
      handle, jobu, jobvt, m, n,
      reinterpret_cast<muComplex*>(A),
      lda, S,
      reinterpret_cast<muComplex*>(U),
      ldu,
      reinterpret_cast<muComplex*>(VT),
      ldvt,
      reinterpret_cast<muComplex*>(work),
      lwork, rwork, info
  ));
}

template<>
void gesvd<c10::complex<double>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<double>, double)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgesvd(
      handle, jobu, jobvt, m, n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda, S,
      reinterpret_cast<muDoubleComplex*>(U),
      ldu,
      reinterpret_cast<muDoubleComplex*>(VT),
      ldvt,
      reinterpret_cast<muDoubleComplex*>(work),
      lwork, rwork, info
  ));
}


template<>
void gesvdj_buffersize<float>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, float *A, int lda, float *S,
    float *U, int ldu, float *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params));
}

template<>
void gesvdj_buffersize<double>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, double *A, int lda, double *S,
    double *U, int ldu, double *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params));
}

template<>
void gesvdj_buffersize<c10::complex<float>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, c10::complex<float> *A, int lda, float *S,
    c10::complex<float> *U, int ldu, c10::complex<float> *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n,
    reinterpret_cast<muComplex*>(A),
    lda,
    S,
    reinterpret_cast<muComplex*>(U),
    ldu,
    reinterpret_cast<muComplex*>(V),
    ldv, lwork, params));
}

template<>
void gesvdj_buffersize<c10::complex<double>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, c10::complex<double> *A, int lda, double *S,
    c10::complex<double> *U, int ldu, c10::complex<double> *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda,
    S,
    reinterpret_cast<muDoubleComplex*>(U),
    ldu,
    reinterpret_cast<muDoubleComplex*>(V),
    ldv, lwork, params));
}


template<>
void gesvdj<float>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, float* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params));
}

template<>
void gesvdj<double>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, double* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params));
}

template<>
void gesvdj<c10::complex<float>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, c10::complex<float>* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnCgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<muComplex*>(A),
    lda, S,
    reinterpret_cast<muComplex*>(U),
    ldu,
    reinterpret_cast<muComplex*>(V),
    ldv,
    reinterpret_cast<muComplex*>(work),
    lwork, info, params));
}

template<>
void gesvdj<c10::complex<double>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int econ, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, c10::complex<double>* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_MUSOLVER_CHECK(musolverDnZgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<muDoubleComplex*>(U),
    ldu,
    reinterpret_cast<muDoubleComplex*>(V),
    ldv,
    reinterpret_cast<muDoubleComplex*>(work),
    lwork, info, params));
}


template<>
void gesvdjBatched<float>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);

  TORCH_MUSOLVER_CHECK(musolverDnSgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<float*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<double>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);

  TORCH_MUSOLVER_CHECK(musolverDnDgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<double*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<c10::complex<float>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnCgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<muComplex*>(A),
    lda, S,
    reinterpret_cast<muComplex*>(U),
    ldu,
    reinterpret_cast<muComplex*>(V),
    ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muComplex)*lwork);

  TORCH_MUSOLVER_CHECK(musolverDnCgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<muComplex*>(A),
    lda, S,
    reinterpret_cast<muComplex*>(U),
    ldu,
    reinterpret_cast<muComplex*>(V),
    ldv,
    static_cast<muComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<c10::complex<double>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnZgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<muDoubleComplex*>(U),
    ldu,
    reinterpret_cast<muDoubleComplex*>(V),
    ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muDoubleComplex)*lwork);

  TORCH_MUSOLVER_CHECK(musolverDnZgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<muDoubleComplex*>(U),
    ldu,
    reinterpret_cast<muDoubleComplex*>(V),
    ldv,
    static_cast<muDoubleComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}


// ROCM does not implement gesdva yet
#if 0
template<>
void gesvdaStridedBatched_buffersize<float>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, float *A, int lda, long long int strideA,
    float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnSgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, lwork, batchSize
  ));
}

template<>
void gesvdaStridedBatched_buffersize<double>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, double *A, int lda, long long int strideA,
    double *S, long long int strideS, double *U, int ldu, long long int strideU, double *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnDgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, lwork, batchSize
  ));
}

template<>
void gesvdaStridedBatched_buffersize<c10::complex<float>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, c10::complex<float> *A, int lda, long long int strideA,
    float *S, long long int strideS, c10::complex<float> *U, int ldu, long long int strideU,
    c10::complex<float> *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnCgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n,
    reinterpret_cast<muComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<muComplex*>(U),
    ldu, strideU,
    reinterpret_cast<muComplex*>(V),
    ldv, strideV, lwork, batchSize
  ));
}

template<>
void gesvdaStridedBatched_buffersize<c10::complex<double>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, c10::complex<double> *A, int lda, long long int strideA,
    double *S, long long int strideS, c10::complex<double> *U, int ldu, long long int strideU,
    c10::complex<double> *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnZgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<muDoubleComplex*>(U),
    ldu, strideU,
    reinterpret_cast<muDoubleComplex*>(V),
    ldv, strideV, lwork, batchSize
  ));
}


template<>
void gesvdaStridedBatched<float>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, float *A, int lda, long long int strideA,
    float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV,
    float *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnSgesvdaStridedBatched(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize
  ));
}

template<>
void gesvdaStridedBatched<double>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, double *A, int lda, long long int strideA,
    double *S, long long int strideS, double *U, int ldu, long long int strideU, double *V, int ldv, long long int strideV,
    double *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnDgesvdaStridedBatched(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize
  ));
}

template<>
void gesvdaStridedBatched<c10::complex<float>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, c10::complex<float> *A, int lda, long long int strideA,
    float *S, long long int strideS, c10::complex<float> *U, int ldu, long long int strideU,
    c10::complex<float> *V, int ldv, long long int strideV,
    c10::complex<float> *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnCgesvdaStridedBatched(
    handle, jobz, rank, m, n,
    reinterpret_cast<muComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<muComplex*>(U),
    ldu, strideU,
    reinterpret_cast<muComplex*>(V),
    ldv, strideV,
    reinterpret_cast<muComplex*>(work),
    lwork, info, h_R_nrmF, batchSize
  ));
}

template<>
void gesvdaStridedBatched<c10::complex<double>>(
    musolverDnHandle_t handle, musolverEigMode_t jobz, int rank, int m, int n, c10::complex<double> *A, int lda, long long int strideA,
    double *S, long long int strideS, c10::complex<double> *U, int ldu, long long int strideU,
    c10::complex<double> *V, int ldv, long long int strideV,
    c10::complex<double> *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnZgesvdaStridedBatched(
    handle, jobz, rank, m, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<muDoubleComplex*>(U),
    ldu, strideU,
    reinterpret_cast<muDoubleComplex*>(V),
    ldv, strideV,
    reinterpret_cast<muDoubleComplex*>(work),
    lwork, info, h_R_nrmF, batchSize
  ));
}
#endif


template<>
void potrf<float>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* info
) {
  TORCH_MUSOLVER_CHECK(musolverDnSpotrf(
    handle, uplo, n, A, lda, work, lwork, info));
}

template<>
void potrf<double>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* info
) {
  TORCH_MUSOLVER_CHECK(musolverDnDpotrf(
    handle, uplo, n, A, lda, work, lwork, info));
}

template<>
void potrf<c10::complex<float>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, c10::complex<float>* A, int lda, c10::complex<float>* work, int lwork, int* info
) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrf(
    handle,
    uplo,
    n,
    reinterpret_cast<muComplex*>(A),
    lda,
    reinterpret_cast<muComplex*>(work),
    lwork,
    info));
}

template<>
void potrf<c10::complex<double>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, c10::complex<double>* A, int lda, c10::complex<double>* work, int lwork, int* info
) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrf(
    handle,
    uplo,
    n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda,
    reinterpret_cast<muDoubleComplex*>(work),
    lwork,
    info));
}


template<>
void potrf_buffersize<float>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, float* A, int lda, int* lwork
) {
  TORCH_MUSOLVER_CHECK(musolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template<>
void potrf_buffersize<double>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, double* A, int lda, int* lwork
) {
  TORCH_MUSOLVER_CHECK(musolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template<>
void potrf_buffersize<c10::complex<float>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, c10::complex<float>* A, int lda, int* lwork
) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrf_bufferSize(
    handle, uplo, n,
    reinterpret_cast<muComplex*>(A),
    lda, lwork));
}

template<>
void potrf_buffersize<c10::complex<double>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, c10::complex<double>* A, int lda, int* lwork
) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrf_bufferSize(
    handle, uplo, n,
    reinterpret_cast<muDoubleComplex*>(A),
    lda, lwork));
}


template<>
void potrfBatched<float>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, float** A, int lda, int* info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
}

template<>
void potrfBatched<double>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, double** A, int lda, int* info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
}

template<>
void potrfBatched<c10::complex<float>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, c10::complex<float>** A, int lda, int* info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrfBatched(
    handle, uplo, n,
    reinterpret_cast<muComplex**>(A),
    lda, info, batchSize));
}

template<>
void potrfBatched<c10::complex<double>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, c10::complex<double>** A, int lda, int* info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrfBatched(
    handle, uplo, n,
    reinterpret_cast<muDoubleComplex**>(A),
    lda, info, batchSize));
}

template <>
void geqrf_bufferSize<float>(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork));
}

template <>
void geqrf_bufferSize<double>(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<muComplex*>(A), lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<muDoubleComplex*>(A), lda, lwork));
}

template <>
void geqrf<float>(CUDASOLVER_GEQRF_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<double>(CUDASOLVER_GEQRF_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<c10::complex<float>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(tau),
      reinterpret_cast<muComplex*>(work),
      lwork,
      devInfo));
}

template <>
void geqrf<c10::complex<double>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(tau),
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      devInfo));
}

template<>
void potrs<float>(
    musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo
) {
  TORCH_MUSOLVER_CHECK(musolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template<>
void potrs<double>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo
) {
  TORCH_MUSOLVER_CHECK(musolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template<>
void potrs<c10::complex<float>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, const c10::complex<float> *A, int lda, c10::complex<float> *B, int ldb, int *devInfo
) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrs(
    handle, uplo, n, nrhs,
    reinterpret_cast<const muComplex*>(A),
    lda,
    reinterpret_cast<muComplex*>(B),
    ldb, devInfo));
}

template<>
void potrs<c10::complex<double>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, const c10::complex<double> *A, int lda, c10::complex<double> *B, int ldb, int *devInfo
) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrs(
    handle, uplo, n, nrhs,
    reinterpret_cast<const muDoubleComplex*>(A),
    lda,
    reinterpret_cast<muDoubleComplex*>(B),
    ldb, devInfo));
}

template<>
void potrsBatched<float>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, float *Aarray[], int lda, float *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnSpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template<>
void potrsBatched<double>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, double *Aarray[], int lda, double *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnDpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template<>
void potrsBatched<c10::complex<float>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, c10::complex<float> *Aarray[], int lda, c10::complex<float> *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrsBatched(
    handle, uplo, n, nrhs,
    reinterpret_cast<muComplex**>(Aarray),
    lda,
    reinterpret_cast<muComplex**>(Barray),
    ldb, info, batchSize));
}

template<>
void potrsBatched<c10::complex<double>>(
  musolverDnHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, c10::complex<double> *Aarray[], int lda, c10::complex<double> *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrsBatched(
    handle, uplo, n, nrhs,
    reinterpret_cast<muDoubleComplex**>(Aarray),
    lda,
    reinterpret_cast<muDoubleComplex**>(Barray),
    ldb, info, batchSize));
}


template <>
void orgqr_buffersize<float>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    const float* A, int lda,
    const float* tau, int* lwork) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

template <>
void orgqr_buffersize<double>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    const double* A, int lda,
    const double* tau, int* lwork) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

template <>
void orgqr_buffersize<c10::complex<float>>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    const c10::complex<float>* A, int lda,
    const c10::complex<float>* tau, int* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnMungqr_bufferSize(
      handle,
      m, n, k,
      reinterpret_cast<const muComplex*>(A), lda,
      reinterpret_cast<const muComplex*>(tau), lwork));
}

template <>
void orgqr_buffersize<c10::complex<double>>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    const c10::complex<double>* A, int lda,
    const c10::complex<double>* tau, int* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnZungqr_bufferSize(
      handle,
      m, n, k,
      reinterpret_cast<const muDoubleComplex*>(A), lda,
      reinterpret_cast<const muDoubleComplex*>(tau), lwork));
}

template <>
void orgqr<float>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    float* A, int lda,
    const float* tau,
    float* work, int lwork,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template <>
void orgqr<double>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    double* A, int lda,
    const double* tau,
    double* work, int lwork,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template <>
void orgqr<c10::complex<float>>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    c10::complex<float>* A, int lda,
    const c10::complex<float>* tau,
    c10::complex<float>* work, int lwork,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(musolverDnMungqr(
      handle,
      m, n, k,
      reinterpret_cast<muComplex*>(A), lda,
      reinterpret_cast<const muComplex*>(tau),
      reinterpret_cast<muComplex*>(work), lwork,
      devInfo));
}

template <>
void orgqr<c10::complex<double>>(
    musolverDnHandle_t handle,
    int m, int n, int k,
    c10::complex<double>* A, int lda,
    const c10::complex<double>* tau,
    c10::complex<double>* work, int lwork,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(musolverDnZungqr(
      handle,
      m, n, k,
      reinterpret_cast<muDoubleComplex*>(A), lda,
      reinterpret_cast<const muDoubleComplex*>(tau),
      reinterpret_cast<muDoubleComplex*>(work), lwork,
      devInfo));
}

template <>
void ormqr_bufferSize<float>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_bufferSize<double>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_bufferSize<c10::complex<float>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnMunmqr_bufferSize(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const muComplex*>(A), lda,
      reinterpret_cast<const muComplex*>(tau),
      reinterpret_cast<const muComplex*>(C), ldc,
      lwork));
}

template <>
void ormqr_bufferSize<c10::complex<double>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZunmqr_bufferSize(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const muDoubleComplex*>(A), lda,
      reinterpret_cast<const muDoubleComplex*>(tau),
      reinterpret_cast<const muDoubleComplex*>(C), ldc,
      lwork));
}

template <>
void ormqr<float>(CUDASOLVER_ORMQR_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
}

template <>
void ormqr<double>(CUDASOLVER_ORMQR_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
}

template <>
void ormqr<c10::complex<float>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnMunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const muComplex*>(A), lda,
      reinterpret_cast<const muComplex*>(tau),
      reinterpret_cast<muComplex*>(C), ldc,
      reinterpret_cast<muComplex*>(work), lwork,
      devInfo));
}

template <>
void ormqr<c10::complex<double>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const muDoubleComplex*>(A), lda,
      reinterpret_cast<const muDoubleComplex*>(tau),
      reinterpret_cast<muDoubleComplex*>(C), ldc,
      reinterpret_cast<muDoubleComplex*>(work), lwork,
      devInfo));
}

#ifdef USE_CUSOLVER_64_BIT

template<> musaDataType get_cusolver_datatype<float>() { return MUSA_R_32F; }
template<> musaDataType get_cusolver_datatype<double>() { return MUSA_R_64F; }
template<> musaDataType get_cusolver_datatype<c10::complex<float>>() { return MUSA_C_32F; }
template<> musaDataType get_cusolver_datatype<c10::complex<double>>() { return MUSA_C_64F; }

void xpotrf_buffersize(
    musolverDnHandle_t handle, musolverDnParams_t params, mublasFillMode_t uplo, int64_t n, musaDataType dataTypeA, const void *A,
    int64_t lda, musaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(musolverDnXpotrf_bufferSize(
    handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost
  ));
}

void xpotrf(
    musolverDnHandle_t handle, musolverDnParams_t params, mublasFillMode_t uplo, int64_t n, musaDataType dataTypeA, void *A,
    int64_t lda, musaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost,
    int *info) {
  TORCH_MUSOLVER_CHECK(musolverDnXpotrf(
    handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info
  ));
}
#endif // USE_CUSOLVER_64_BIT

template <>
void syevd_bufferSize<float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
}

template <>
void syevd_bufferSize<double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
}

template <>
void syevd_bufferSize<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnCheevd_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const muComplex*>(A),
      lda,
      W,
      lwork));
}

template <>
void syevd_bufferSize<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnZheevd_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const muDoubleComplex*>(A),
      lda,
      W,
      lwork));
}

template <>
void syevd<float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
}

template <>
void syevd<double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
}

template <>
void syevd<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  TORCH_MUSOLVER_CHECK(musolverDnCheevd(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<muComplex*>(A),
      lda,
      W,
      reinterpret_cast<muComplex*>(work),
      lwork,
      info));
}

template <>
void syevd<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  TORCH_MUSOLVER_CHECK(musolverDnZheevd(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      info));
}

template <>
void syevj_bufferSize<float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params));
}

template <>
void syevj_bufferSize<double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params));
}

template <>
void syevj_bufferSize<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnCheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const muComplex*>(A),
      lda,
      W,
      lwork,
      params));
}

template <>
void syevj_bufferSize<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnZheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const muDoubleComplex*>(A),
      lda,
      W,
      lwork,
      params));
}

template <>
void syevj<float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnSsyevj(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params));
}

template <>
void syevj<double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnDsyevj(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params));
}

template <>
void syevj<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnCheevj(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<muComplex*>(A),
      lda,
      W,
      reinterpret_cast<muComplex*>(work),
      lwork,
      info,
      params));
}

template <>
void syevj<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_MUSOLVER_CHECK(musolverDnZheevj(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      info,
      params));
}

template <>
void syevjBatched_bufferSize<float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnSsyevjBatched_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
}

template <>
void syevjBatched_bufferSize<double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnDsyevjBatched_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
}

template <>
void syevjBatched_bufferSize<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnCheevjBatched_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const muComplex*>(A),
      lda,
      W,
      lwork,
      params,
      batchsize));
}

template <>
void syevjBatched_bufferSize<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnZheevjBatched_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const muDoubleComplex*>(A),
      lda,
      W,
      lwork,
      params,
      batchsize));
}

template <>
void syevjBatched<float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnSsyevjBatched(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
}

template <>
void syevjBatched<double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnDsyevjBatched(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
}

template <>
void syevjBatched<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnCheevjBatched(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<muComplex*>(A),
      lda,
      W,
      reinterpret_cast<muComplex*>(work),
      lwork,
      info,
      params,
      batchsize));
}

template <>
void syevjBatched<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_MUSOLVER_CHECK(musolverDnZheevjBatched(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      info,
      params,
      batchsize));
}

#ifdef USE_CUSOLVER_64_BIT

void xpotrs(
    musolverDnHandle_t handle, musolverDnParams_t params, mublasFillMode_t uplo, int64_t n, int64_t nrhs, musaDataType dataTypeA, const void *A,
    int64_t lda, musaDataType dataTypeB, void *B, int64_t ldb, int *info) {
  TORCH_MUSOLVER_CHECK(musolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info));
}

template <>
void xgeqrf_bufferSize<float>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      MUSA_R_32F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<const void*>(tau),
      MUSA_R_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf_bufferSize<double>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      MUSA_R_64F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<const void*>(tau),
      MUSA_R_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      MUSA_C_32F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_C_32F,
      reinterpret_cast<const void*>(tau),
      MUSA_C_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      MUSA_C_64F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_C_64F,
      reinterpret_cast<const void*>(tau),
      MUSA_C_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf<float>(CUDASOLVER_XGEQRF_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      MUSA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<void*>(tau),
      MUSA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeqrf<double>(CUDASOLVER_XGEQRF_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      MUSA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<void*>(tau),
      MUSA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeqrf<c10::complex<float>>(CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      MUSA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_C_32F,
      reinterpret_cast<void*>(tau),
      MUSA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeqrf<c10::complex<double>>(CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      MUSA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_C_64F,
      reinterpret_cast<void*>(tau),
      MUSA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd_bufferSize<float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const float* A,
    int64_t lda,
    const float* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_R_32F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<const void*>(W),
      MUSA_R_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd_bufferSize<double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const double* A,
    int64_t lda,
    const double* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_R_64F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<const void*>(W),
      MUSA_R_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd_bufferSize<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const c10::complex<float>* A,
    int64_t lda,
    const float* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_C_32F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<const void*>(W),
      MUSA_C_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd_bufferSize<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const c10::complex<double>* A,
    int64_t lda,
    const double* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_C_64F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<const void*>(W),
      MUSA_C_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd<float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    float* A,
    int64_t lda,
    float* W,
    float* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    float* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<void*>(W),
      MUSA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd<double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    double* A,
    int64_t lda,
    double* W,
    double* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    double* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<void*>(W),
      MUSA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    c10::complex<float>* A,
    int64_t lda,
    float* W,
    c10::complex<float>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<float>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<void*>(W),
      MUSA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    c10::complex<double>* A,
    int64_t lda,
    double* W,
    c10::complex<double>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<double>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_MUSOLVER_CHECK(musolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      MUSA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<void*>(W),
      MUSA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

// cuSOLVER Xgeev bindings (requires cuSOLVER >= 11.7.2, i.e. CUDA 12.8+)
#if defined(MUSOLVER_VERSION) && (MUSOLVER_VERSION >= 11702)

template <>
void xgeev_bufferSize<float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    const float* A,
    int64_t lda,
    const float* W,
    const float* VL,
    int64_t ldvl,
    const float* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      MUSA_R_32F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<const void*>(W),
      MUSA_R_32F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      MUSA_R_32F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      MUSA_R_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeev_bufferSize<double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    const double* A,
    int64_t lda,
    const double* W,
    const double* VL,
    int64_t ldvl,
    const double* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      MUSA_R_64F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<const void*>(W),
      MUSA_R_64F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      MUSA_R_64F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      MUSA_R_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}


template <>
void xgeev_bufferSize<c10::complex<float>>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    const c10::complex<float>* A,
    int64_t lda,
    const c10::complex<float>* W,
    const c10::complex<float>* VL,
    int64_t ldvl,
    const c10::complex<float>* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      MUSA_C_32F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_C_32F,
      reinterpret_cast<const void*>(W),
      MUSA_C_32F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      MUSA_C_32F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      MUSA_C_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeev_bufferSize<c10::complex<double>>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    const c10::complex<double>* A,
    int64_t lda,
    const c10::complex<double>* W,
    const c10::complex<double>* VL,
    int64_t ldvl,
    const c10::complex<double>* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_MUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      MUSA_C_64F,
      reinterpret_cast<const void*>(A),
      lda,
      MUSA_C_64F,
      reinterpret_cast<const void*>(W),
      MUSA_C_64F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      MUSA_C_64F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      MUSA_C_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeev<float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    float* A,
    int64_t lda,
    float* W,
    float* VL,
    int64_t ldvl,
    float* VR,
    int64_t ldvr,
    float* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    float* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_MUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      MUSA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_32F,
      reinterpret_cast<void*>(W),
      MUSA_R_32F,
      reinterpret_cast<void*>(VL),
      ldvl,
      MUSA_R_32F,
      reinterpret_cast<void*>(VR),
      ldvr,
      MUSA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}




template <>
void xgeev<double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    double* A,
    int64_t lda,
    double* W,
    double* VL,
    int64_t ldvl,
    double* VR,
    int64_t ldvr,
    double* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    double* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_MUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      MUSA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_R_64F,
      reinterpret_cast<void*>(W),
      MUSA_R_64F,
      reinterpret_cast<void*>(VL),
      ldvl,
      MUSA_R_64F,
      reinterpret_cast<void*>(VR),
      ldvr,
      MUSA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));

}

template <>
void xgeev<c10::complex<float>>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    c10::complex<float>* A,
    int64_t lda,
    c10::complex<float>* W,
    c10::complex<float>* VL,
    int64_t ldvl,
    c10::complex<float>* VR,
    int64_t ldvr,
    c10::complex<float>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<float>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_MUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      MUSA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_C_32F,
      reinterpret_cast<void*>(W),
      MUSA_C_32F,
      reinterpret_cast<void*>(VL),
      ldvl,
      MUSA_C_32F,
      reinterpret_cast<void*>(VR),
      ldvr,
      MUSA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeev<c10::complex<double>>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobvl,
    musolverEigMode_t jobvr,
    int64_t n,
    c10::complex<double>* A,
    int64_t lda,
    c10::complex<double>* W,
    c10::complex<double>* VL,
    int64_t ldvl,
    c10::complex<double>* VR,
    int64_t ldvr,
    c10::complex<double>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<double>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_MUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      MUSA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      MUSA_C_64F,
      reinterpret_cast<void*>(W),
      MUSA_C_64F,
      reinterpret_cast<void*>(VL),
      ldvl,
      MUSA_C_64F,
      reinterpret_cast<void*>(VR),
      ldvr,
      MUSA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}


#endif // defined(MUSOLVER_VERSION) && (MUSOLVER_VERSION >= 11702)


#endif // USE_CUSOLVER_64_BIT

#ifdef USE_CUSOLVER_64_BIT_XSYEV_BATCHED

template <>
void xsyevBatched_bufferSize<float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t  jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const float *A,
    int64_t lda,
    const float *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_R_32F,
       reinterpret_cast<const void*>(A),
       lda,
       MUSA_R_32F,
       reinterpret_cast<const void*>(W),
       MUSA_R_32F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched_bufferSize<double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t  jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const double *A,
    int64_t lda,
    const double *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_R_64F,
       reinterpret_cast<const void*>(A),
       lda,
       MUSA_R_64F,
       reinterpret_cast<const void*>(W),
       MUSA_R_64F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched_bufferSize<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t  jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const c10::complex<float> *A,
    int64_t lda,
    const float *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_C_32F,
       reinterpret_cast<const void*>(A),
       lda,
       MUSA_R_32F,
       reinterpret_cast<const void*>(W),
       MUSA_C_32F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched_bufferSize<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t  jobz,
    mublasFillMode_t uplo,
    int64_t n,
    const c10::complex<double> *A,
    int64_t lda,
    const double *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_C_64F,
       reinterpret_cast<const void*>(A),
       lda,
       MUSA_R_64F,
       reinterpret_cast<const void*>(W),
       MUSA_C_64F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched<float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    float *A,
    int64_t lda,
    float *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_R_32F,
       reinterpret_cast<void*>(A),
       lda,
       MUSA_R_32F,
       reinterpret_cast<void*>(W),
       MUSA_R_32F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

template <>
void xsyevBatched<double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    double *A,
    int64_t lda,
    double *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_R_64F,
       reinterpret_cast<void*>(A),
       lda,
       MUSA_R_64F,
       reinterpret_cast<void*>(W),
       MUSA_R_64F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

template <>
void xsyevBatched<c10::complex<float>, float>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    c10::complex<float> *A,
    int64_t lda,
    float *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_C_32F,
       reinterpret_cast<void*>(A),
       lda,
       MUSA_R_32F,
       reinterpret_cast<void*>(W),
       MUSA_C_32F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

template <>
void xsyevBatched<c10::complex<double>, double>(
    musolverDnHandle_t handle,
    musolverDnParams_t params,
    musolverEigMode_t jobz,
    mublasFillMode_t uplo,
    int64_t n,
    c10::complex<double> *A,
    int64_t lda,
    double *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_MUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       MUSA_C_64F,
       reinterpret_cast<void*>(A),
       lda,
       MUSA_R_64F,
       reinterpret_cast<void*>(W),
       MUSA_C_64F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

#endif // USE_CUSOLVER_64_BIT_XSYEV_BATCHED

} // namespace at::musa::solver
