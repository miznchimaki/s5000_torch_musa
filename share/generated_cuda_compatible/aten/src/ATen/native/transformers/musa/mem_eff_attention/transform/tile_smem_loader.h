/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/aligned_buffer.h>
#include <mutlass/array.h>
#include <mutlass/layout/matrix.h>
#include <mutlass/layout/pitch_linear.h>
#include <mutlass/numeric_types.h>
#include <mutlass/transform/pitch_linear_thread_map.h>
#include <mutlass/transform/threadblock/predicated_tile_iterator.h>
#include <mutlass/transform/threadblock/regular_tile_iterator.h>

template <
    typename scalar_t, // scalar type
    typename ThreadblockTileShape, // size of tile to load
    int Threads, // number of participating threads
    int ElementsPerAccess> // thread access width in elements
class TileSmemLoader {
 public:
  using SmemTile =
      mutlass::AlignedBuffer<scalar_t, ThreadblockTileShape::kCount>;

  using ThreadMap = mutlass::transform::PitchLinearStripminedThreadMap<
      mutlass::layout::PitchLinearShape<
          ThreadblockTileShape::kColumn, // contiguous
          ThreadblockTileShape::kRow>, // strided
      Threads, // Threads
      ElementsPerAccess>; // ElementsPerAccess

  using GmemTileIterator =
      mutlass::transform::threadblock::PredicatedTileIterator<
          ThreadblockTileShape, // Shape
          scalar_t, // Element
          mutlass::layout::RowMajor, // Layout
          0, // AdvanceRank
          ThreadMap>; // ThreadMap

  using SmemTileIterator = mutlass::transform::threadblock::RegularTileIterator<
      ThreadblockTileShape, // Shape
      scalar_t, // Element
      mutlass::layout::RowMajor, // Layout
      0, // AdvanceRank
      ThreadMap>; // ThreadMap

  using Fragment = typename GmemTileIterator::Fragment;

  /// load a tile from global memory into shared memory
  MUTLASS_DEVICE
  static void load(
      GmemTileIterator tile_load_iter,
      SmemTileIterator tile_store_iter) {
    Fragment tb_frag;
    tb_frag.clear();
    tile_load_iter.load(tb_frag);
    tile_store_iter.store(tb_frag);

    __SYNCTHREADS;
  }
};
