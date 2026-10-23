#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/musa/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <ATen/musa/detail/TensorInfo.muh>
#include <ATen/native/musa/SortingCommon.muh>
#include <ATen/native/musa/SortingRadixSelect.muh>

#include "torch_musa/csrc/core/MUSAStream.h"

#include <cassert>
#include <cstdlib>

namespace at::native {

namespace {

// Finds the rank k element, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherKthValue(
    musa::detail::TensorInfo<const scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    musa::detail::TensorInfo<scalar_t, index_t> kthValue,
    musa::detail::TensorInfo<int64_t, index_t> indices) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of index_t
  __shared__ int smem[C10_WARP_SIZE]; // one per each warp, up to warp limit

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  index_t sliceStartIndex =
      musa::detail::IndexToOffset<const scalar_t, index_t, Dim>::get(slice, input);
  index_t kthValueSliceStartIndex =
      musa::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
  index_t indicesSliceStartIndex =
      musa::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

  const scalar_t* inputSliceStart = &input.data[sliceStartIndex];
  scalar_t* kthValueSliceStart = &kthValue.data[kthValueSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  scalar_t kValue = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t>(
      inputSliceStart,
      k,
      false,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &kValue);

  // Find the index of the k-th highest element
  __shared__ int32_t minIndexFound;

  if (threadIdx.x == 0) {
      minIndexFound = static_cast<int32_t>(inputSliceSize);
  }
  __SYNCTHREADS;

  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
      // Early exit based on best-so-far
      if (i >= minIndexFound) {
          break;
      }

      scalar_t v = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
      bool isKValue =
          ((v == kValue) || (at::_isnan(v) && at::_isnan(kValue)));

      if (isKValue) {
          atomicMin(&minIndexFound, static_cast<int32_t>(i));
          break;
      }
  }

  __SYNCTHREADS;

  if (threadIdx.x == 0) {
      indicesSliceStart[0] = static_cast<index_t>(minIndexFound);
      kthValueSliceStart[0] = kValue;
  }
}

// CUDA kernel to find the median, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherMedian(
    musa::detail::TensorInfo<scalar_t, index_t> values,
    musa::detail::TensorInfo<int64_t, index_t> indices,
    musa::detail::TensorInfo<const scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    bool ignore_nan) {
  // Shared memory for the subroutine RadixSelect. Note that RadixSelect converts the
  // floating point type to int with the same relative ordering.
  __shared__ int smem[C10_WARP_SIZE]; // one per each warp, up to warp limit

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Finds the start offset for our slice
  index_t valuesSliceStartIndex =
      musa::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, values);
  index_t indicesSliceStartIndex =
      musa::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);
  index_t inputSliceStartIndex =
      musa::detail::IndexToOffset<const scalar_t, index_t, Dim>::get(slice, input);

  scalar_t* valuesSliceStart = &values.data[valuesSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];
  const scalar_t* inputSliceStart = &input.data[inputSliceStartIndex];

  index_t nan_count = 0;
  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    nan_count += at::_isnan(val) ? 1 : 0;
  }

  // Counts number of nan values
  // This code performs a parallel sum reduction (not the most efficient code)
  __shared__ int64_t num_nan;
  if (threadIdx.x == 0) {
    num_nan = 0;
  }
  __SYNCTHREADS;
  if (nan_count > 0) {
    gpuAtomicAddNoReturn(&num_nan, nan_count);
  }
  __SYNCTHREADS;

  // For torch.median, if we found nan set k to last index so the computed value
  // is nan, otherwise set k to the middle element of the non-nan values
  index_t k = (!ignore_nan && num_nan > 0) ? inputSliceSize - 1
                                           : (inputSliceSize - num_nan - 1) / 2;

  // Find the median
  scalar_t median = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t>(
      inputSliceStart,
      k + 1,
      false,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &median);

  valuesSliceStart[0] = median;

  // Find the index of the median value in the slice
  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    if (val == median || (at::_isnan(val) && at::_isnan(median))) {
      indicesSliceStart[0] = i;
      break;
    }
  }
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      musa::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      musa::detail::TensorInfo<int64_t, index_t> indices_info,
      [[maybe_unused]] int collapse_indices_dim,
      musa::detail::TensorInfo<const scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      TORCH_CHECK(false, "slices are too many");
    }

    dim3 block(std::min(
        round_up(slice_size, (int64_t)at::musa::warp_size()), (int64_t)1024));
    auto stream = at::musa::getCurrentMUSAStream();
    gatherKthValue<scalar_t, index_t, all_dims><<<grid, block, 0, stream>>>(
        self_info,
        slice_size,
        k,
        num_slices,
        /* The actual dimension that the k-selection is running in */
        /* may have changed from collapseDims() */
        self_info.strides[collapse_self_dim],
        values_info,
        indices_info);
    C10_MUSA_KERNEL_LAUNCH_CHECK();
  }
};

struct MedianLauncher {
  bool ignore_nan;

  MedianLauncher(bool ignore_nan) : ignore_nan(ignore_nan) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      musa::detail::TensorInfo<scalar_t, index_t> values_info,
      [[maybe_unused]] int collapse_values_dim,
      musa::detail::TensorInfo<int64_t, index_t> indices_info,
      [[maybe_unused]] int collapse_indices_dim,
      musa::detail::TensorInfo<const scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      TORCH_CHECK(false, "slices are too many");
    }

    dim3 block(std::min(
        round_up(slice_size, (int64_t)at::musa::warp_size()), (int64_t)1024));
    auto stream = at::musa::getCurrentMUSAStream();
    gatherMedian<scalar_t, index_t, all_dims><<<grid, block, 0, stream>>>(
        values_info,
        indices_info,
        self_info,
        slice_size,
        num_slices,
        self_info.strides[collapse_self_dim],
        ignore_nan);
    C10_MUSA_KERNEL_LAUNCH_CHECK();
  }
};

}  // namespace (anonymous)

void launch_kthvalue_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t dim, int64_t k) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "kthvalue_cuda", [&] {
    AT_DISPATCH_INDEX_TYPES(
        musa::detail::canUse32BitIndexMath(self) &&
        musa::detail::canUse32BitIndexMath(values) &&
        musa::detail::canUse32BitIndexMath(indices) ? ScalarType::Int : ScalarType::Long,
        "kth_value_launcher", [&] {
          run_launcher<scalar_t, index_t>(
              values, indices, self, dim, KthValueLauncher(k));
    });
  });
}

void launch_median_kernel(
    const TensorBase &vals, const TensorBase &inds,
    const TensorBase &self, int64_t dim, bool ignore_nan) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "median_out_impl", [&] {
        if (musa::detail::canUse32BitIndexMath(vals) &&
            musa::detail::canUse32BitIndexMath(inds) &&
            musa::detail::canUse32BitIndexMath(self)) {
          run_launcher<scalar_t, uint32_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        } else {
          run_launcher<scalar_t, uint64_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        }
      });
}

} // namespace at::native
