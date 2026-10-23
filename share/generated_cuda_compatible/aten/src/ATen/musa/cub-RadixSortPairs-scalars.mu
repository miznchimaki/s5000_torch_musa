#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#include <ATen/musa/cub-RadixSortPairs.muh>

namespace at::musa::cub::detail {

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, AT_INSTANTIATE_SORT_PAIRS_8)

} // namespace at::musa::cub::detail
