// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTER_2D_NEON_H
#define KLEIDICV_FILTER_2D_NEON_H

#include "filter_2d_window_loader_3x3.h"
#include "filter_2d_window_loader_5x5.h"
#include "filter_2d_window_loader_7x7.h"
#include "kleidicv/neon.h"
#include "process_filter_2d.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename InnerFilterType, size_t KSize, typename WindowLoaderType>
class Filter2d {
 public:
  using SourceType = typename InnerFilterType::SourceType;
  using DestinationType = typename InnerFilterType::DestinationType;
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using DestinationVecTraits = typename neon::VecTraits<DestinationType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using DestinationVectorType = typename DestinationVecTraits::VectorType;
  using BorderType = FixedBorderType;
  static constexpr size_t kMargin = KSize / 2;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo<SourceType, KSize>;
  using BorderOffsets = typename BorderInfoType::Offsets;

  explicit Filter2d(InnerFilterType filter) : filter_{filter} {}

  void process_pixels_without_horizontal_borders(
      size_t width, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) {
      SourceVectorType src[KSize][KSize];
      DestinationVectorType dst_vec;

      auto KernelWindow = [&](size_t row, size_t col) -> SourceVectorType& {
        return src[row][col];
      };

      auto load_array_element = [](const SourceType& x) { return vld1q(&x); };
      WindowLoaderType::load_window(KernelWindow, load_array_element, src_rows,
                                    window_row_offsets, window_col_offsets,
                                    index);

      filter_.vector_path(KernelWindow, dst_vec);
      vst1q(&dst_rows[index], dst_vec);
    });

    loop.tail([&](size_t index) {
      process_one_element_with_horizontal_borders(
          src_rows, dst_rows, window_row_offsets, window_col_offsets, index);
    });
  }

  void process_pixels_of_dual_rows_without_horizontal_borders(
      size_t width, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets_0,
      BorderOffsets window_row_offsets_1,
      BorderOffsets window_col_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) {
      SourceVectorType src[KSize + 1][KSize];
      DestinationVectorType dst_vec_0;
      DestinationVectorType dst_vec_1;
      auto KernelWindow = [&](size_t row, size_t col) -> SourceVectorType& {
        return src[row][col];
      };

      auto load_array_element = [](const SourceType& x) { return vld1q(&x); };
      WindowLoaderType::load_window_to_handle_dual_rows(
          KernelWindow, load_array_element, src_rows, window_row_offsets_0,
          window_row_offsets_1, window_col_offsets, index);

      filter_.vector_path_for_dual_row_handling(KernelWindow, dst_vec_0,
                                                dst_vec_1);
      vst1q(&dst_rows.at(0, 0)[index], dst_vec_0);
      vst1q(&dst_rows.at(1, 0)[index], dst_vec_1);
    });

    loop.tail([&](size_t index) {
      process_two_element_vertically_with_or_without_horizontal_borders(
          src_rows, dst_rows, window_row_offsets_0, window_row_offsets_1,
          window_col_offsets, index);
    });
  }

  void process_one_pixel_with_horizontal_borders(
      Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets) const KLEIDICV_STREAMING {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      disable_loop_vectorization();
      process_one_element_with_horizontal_borders(
          src_rows, dst_rows, window_row_offsets, window_col_offsets, index);
    }
  }

  // Processes two vertically adjacent pixels in a single column
  void process_two_pixels_with_horizontal_borders(
      Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets window_row_offsets_0, BorderOffsets window_row_offsets_1,
      BorderOffsets window_col_offsets) const {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      disable_loop_vectorization();
      process_two_element_vertically_with_or_without_horizontal_borders(
          src_rows, dst_rows, window_row_offsets_0, window_row_offsets_1,
          window_col_offsets, index);
    }
  }

 private:
  void process_one_element_with_horizontal_borders(
      Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets window_row_offsets, BorderOffsets window_col_offsets,
      size_t index) const KLEIDICV_STREAMING {
    SourceType src[KSize][KSize];

    auto KernelWindow =
        [&](size_t row, size_t col)
            KLEIDICV_STREAMING -> SourceType& { return src[row][col]; };

    auto load_array_element = [&](const SourceType& x)
                                  KLEIDICV_STREAMING { return x; };

    WindowLoaderType::load_window(KernelWindow, load_array_element, src_rows,
                                  window_row_offsets, window_col_offsets,
                                  index);

    filter_.scalar_path(KernelWindow, dst_rows[index]);
  }

  void process_two_element_vertically_with_or_without_horizontal_borders(
      Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets window_row_offsets_0, BorderOffsets window_row_offsets_1,
      BorderOffsets window_col_offsets, size_t index) const {
    SourceType src[KSize + 1][KSize];
    auto KernelWindow = [&](size_t row, size_t col) -> SourceType& {
      return src[row][col];
    };
    auto load_array_element = [](const SourceType& x) { return x; };
    WindowLoaderType::load_window_to_handle_dual_rows(
        KernelWindow, load_array_element, src_rows, window_row_offsets_0,
        window_row_offsets_1, window_col_offsets, index);

    filter_.scalar_path_for_dual_row_handling(
        KernelWindow, dst_rows.at(0, 0)[index], dst_rows.at(1, 0)[index]);
  }

  InnerFilterType filter_;
};

template <typename InnerFilterType>
using Filter2D3x3 =
    Filter2d<InnerFilterType, 3UL,
             Filter2dWindowLoader3x3<typename InnerFilterType::SourceType>>;

template <typename InnerFilterType>
using Filter2D5x5 =
    Filter2d<InnerFilterType, 5UL,
             Filter2dWindowLoader5x5<typename InnerFilterType::SourceType>>;

template <typename InnerFilterType>
using Filter2D7x7 =
    Filter2d<InnerFilterType, 7UL,
             Filter2dWindowLoader7x7<typename InnerFilterType::SourceType>>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_FILTER_2D_NEON_H
