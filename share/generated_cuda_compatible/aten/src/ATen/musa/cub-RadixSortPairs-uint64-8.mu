#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#include <ATen/musa/cub-RadixSortPairs.muh>

namespace at::musa::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(uint64_t, 8)

} // namespace at::musa::cub::detail
