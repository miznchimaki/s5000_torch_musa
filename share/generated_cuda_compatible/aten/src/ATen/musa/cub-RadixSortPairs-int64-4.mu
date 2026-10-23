#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#include <ATen/musa/cub-RadixSortPairs.muh>

namespace at::musa::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(int64_t, 4)

} // namespace at::musa::cub::detail
