#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

namespace at {
struct MUSAGeneratorImpl;
struct TensorIteratorBase;
class TensorBase;

namespace native {

void launch_poisson_cuda_kernel(
    const TensorBase &ret, const TensorBase &lambda, MUSAGeneratorImpl *gen);

void launch_gamma_kernel(
    const TensorBase &ret, const TensorBase &alpha, MUSAGeneratorImpl *gen);

void launch_binomial_cuda_kernel(
    TensorIteratorBase &iter, MUSAGeneratorImpl *gen);

void launch_dirichlet_kernel(TensorIteratorBase &iter);

void launch_standard_gamma_grad_kernel(TensorIteratorBase &iter);

void launch_dirichlet_grad_kernel(TensorIteratorBase &iter);

}}  // namespace at::native

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
