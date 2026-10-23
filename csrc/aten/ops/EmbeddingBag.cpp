#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_sparse_backward_native.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/empty.h>
#endif

#include "torch_musa/csrc/aten/ops/musa/EmbeddingBag.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

std::tuple<Tensor, Tensor, Tensor, Tensor> EmbeddingBag(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  UNUSED(scale_grad_by_freq);
  UNUSED(sparse);

  TORCH_CHECK(
      indices.dim() == 1,
      "input has to be a 1D Tensor, but got Tensor of dimension ",
      indices.dim());
  TORCH_CHECK(
      offsets.dim() == 1,
      "offsets has to be a 1D Tensor, but got Tensor of dimension ",
      offsets.dim());
  TORCH_CHECK(
      weight.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight.dim());

  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  const auto weight_arg = TensorArg(weight, "weight", 1);
  const auto indices_arg = TensorArg(indices, "indices", 2);
  const auto offsets_arg = TensorArg(offsets, "offsets", 3);
  checkScalarTypes("EmbeddingBag", indices_arg, {kLong, kInt});
  checkScalarTypes("EmbeddingBag", offsets_arg, {kLong, kInt});
  checkAllSameGPU("EmbeddingBag", {weight_arg, indices_arg, offsets_arg});

  const c10::musa::MUSAGuard device_guard(weight.device());

  int64_t numBags = offsets.size(0);
  if (include_last_offset) {
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  int64_t featureSize = weight.size(1);
  auto output = at::empty({numBags, featureSize}, weight.options());

  const auto promoteIdxType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  const auto IdxOpt = indices.options().dtype(promoteIdxType);
  auto offset2bag = at::empty({indices.size(0)}, IdxOpt);
  auto bag_size = at::empty({numBags}, IdxOpt);

  Tensor max_indices;
  if (mode == native::EmbeddingBagMode::MAX) {
    max_indices = at::empty({numBags, featureSize}, IdxOpt);
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::empty({0}, IdxOpt);
  }

  EmbeddingBagRun(
      output,
      offset2bag,
      bag_size,
      max_indices,
      weight,
      indices,
      offsets,
      per_sample_weights,
      mode,
      padding_idx);

  return std::make_tuple(
      std::move(output),
      std::move(offset2bag),
      std::move(bag_size),
      std::move(max_indices));
}

static Tensor apply_bag_size_backward(
    const int64_t mode,
    Tensor& output,
    const Tensor& offset2bag,
    const Tensor& bag_size) {
  if (mode == native::EmbeddingBagMode::MEAN) {
    auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
    output *= inv_bag_size_;
  }
  return output;
}

Tensor _embedding_bag_sparse_backward_symint(
    const Tensor& grad_,
    const Tensor& indices,
    [[maybe_unused]] const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    SymInt num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const std::optional<Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor index_grad = grad_.index_select(0, offset2bag);

  index_grad = apply_bag_size_backward(mode, index_grad, offset2bag, bag_size_);

  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == native::EmbeddingBagMode::SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }
  return native::embedding_backward_symint(
      index_grad,
      indices,
      std::move(num_weights),
      padding_idx,
      scale_grad_by_freq,
      true);
}

} // namespace at::musa
