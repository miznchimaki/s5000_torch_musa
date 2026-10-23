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
