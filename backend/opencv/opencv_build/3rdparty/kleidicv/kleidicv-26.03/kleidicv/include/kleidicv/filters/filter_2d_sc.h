// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTER_2D_SC_H
#define KLEIDICV_FILTER_2D_SC_H

#include "filter_2d_window_loader_3x3.h"
#include "filter_2d_window_loader_5x5.h"
#include "filter_2d_window_loader_7x7.h"
#include "kleidicv/sve2.h"
#include "process_filter_2d.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename SourceType, typename DestinationType,
          typename WindowLoaderType>
class Filter2D3x3VectorOperations {
 public:
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo<SourceType, 3>;
  using BorderOffsets = typename BorderInfoType::Offsets;

  template <typename InnerFilterType, typename SourceVectorType,
            typename DestinationVectorType>
  static void process_one_element_with_vector_operation(
      svbool_t pg, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets, size_t index,
      const InnerFilterType& filter_) KLEIDICV_STREAMING {
    SourceVectorType src_0_0, src_0_1, src_0_2, src_1_0, src_1_1, src_1_2,
        src_2_0, src_2_1, src_2_2;
    DestinationVectorType dst_vec;
    ScalableVectorArray2D<SourceVectorType, 3, 3> KernelWindow = {{
        {std::ref(src_0_0), std::ref(src_0_1), std::ref(src_0_2)},
        {std::ref(src_1_0), std::ref(src_1_1), std::ref(src_1_2)},
        {std::ref(src_2_0), std::ref(src_2_1), std::ref(src_2_2)},
    }};

    auto load_array_element = [&](const SourceType& x)
                                  KLEIDICV_STREAMING { return svld1(pg, &x); };

    WindowLoaderType::load_window(KernelWindow, load_array_element, src_rows,
                                  window_row_offsets, window_col_offsets,
                                  index);
    filter_.vector_path(pg, KernelWindow, dst_vec);

    svst1(pg, &dst_rows[index], dst_vec);
  }

  template <typename InnerFilterType, typename SourceVectorType,
            typename DestinationVectorType>
  static void process_two_elements_with_vector_operation(
      svbool_t pg, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets_0,
      BorderOffsets window_row_offsets_1, BorderOffsets window_col_offsets,
      size_t index, const InnerFilterType& filter_) KLEIDICV_STREAMING {
    SourceVectorType src_0_0, src_0_1, src_0_2, src_1_0, src_1_1, src_1_2,
        src_2_0, src_2_1, src_2_2, src_3_0, src_3_1, src_3_2;
    DestinationVectorType dst_vec_0, dst_vec_1;
    ScalableVectorArray2D<SourceVectorType, 4, 3> KernelWindow = {{
        {std::ref(src_0_0), std::ref(src_0_1), std::ref(src_0_2)},
        {std::ref(src_1_0), std::ref(src_1_1), std::ref(src_1_2)},
        {std::ref(src_2_0), std::ref(src_2_1), std::ref(src_2_2)},
        {std::ref(src_3_0), std::ref(src_3_1), std::ref(src_3_2)},
    }};

    auto load_array_element = [&](const SourceType& x)
                                  KLEIDICV_STREAMING { return svld1(pg, &x); };

    WindowLoaderType::load_window_to_handle_dual_rows(
        KernelWindow, load_array_element, src_rows, window_row_offsets_0,
        window_row_offsets_1, window_col_offsets, index);

    filter_.vector_path_for_dual_row_handling(pg, KernelWindow, dst_vec_0,
                                              dst_vec_1);
    svst1(pg, &dst_rows.at(0, 0)[index], dst_vec_0);
    svst1(pg, &dst_rows.at(1, 0)[index], dst_vec_1);
  }
};

template <typename SourceType, typename DestinationType,
          typename WindowLoaderType>
class Filter2D5x5VectorOperations {
 public:
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo<SourceType, 5>;
  using BorderOffsets = typename BorderInfoType::Offsets;

  template <typename InnerFilterType, typename SourceVectorType,
            typename DestinationVectorType>
  static void process_one_element_with_vector_operation(
      svbool_t pg, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets, size_t index,
      const InnerFilterType& filter_) KLEIDICV_STREAMING {
    SourceVectorType src_0_0, src_0_1, src_0_2, src_0_3, src_0_4, src_1_0,
        src_1_1, src_1_2, src_1_3, src_1_4, src_2_0, src_2_1, src_2_2, src_2_3,
        src_2_4, src_3_0, src_3_1, src_3_2, src_3_3, src_3_4, src_4_0, src_4_1,
        src_4_2, src_4_3, src_4_4;
    DestinationVectorType output_vector;
    // Initialization
    ScalableVectorArray2D<SourceVectorType, 5, 5> KernelWindow = {{
        {std::ref(src_0_0), std::ref(src_0_1), std::ref(src_0_2),
         std::ref(src_0_3), std::ref(src_0_4)},
        {std::ref(src_1_0), std::ref(src_1_1), std::ref(src_1_2),
         std::ref(src_1_3), std::ref(src_1_4)},
        {std::ref(src_2_0), std::ref(src_2_1), std::ref(src_2_2),
         std::ref(src_2_3), std::ref(src_2_4)},
        {std::ref(src_3_0), std::ref(src_3_1), std::ref(src_3_2),
         std::ref(src_3_3), std::ref(src_3_4)},
        {std::ref(src_4_0), std::ref(src_4_1), std::ref(src_4_2),
         std::ref(src_4_3), std::ref(src_4_4)},
    }};

    auto load_array_element = [&](const SourceType& x)
                                  KLEIDICV_STREAMING { return svld1(pg, &x); };

    WindowLoaderType::load_window(KernelWindow, load_array_element, src_rows,
                                  window_row_offsets, window_col_offsets,
                                  index);
    filter_.vector_path(pg, KernelWindow, output_vector);
    svst1(pg, &dst_rows[index], output_vector);
  }
};

template <typename SourceType, typename DestinationType,
          typename WindowLoaderType>
class Filter2D7x7VectorOperations {
 public:
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo<SourceType, 7>;
  using BorderOffsets = typename BorderInfoType::Offsets;

  template <typename InnerFilterType, typename SourceVectorType,
            typename DestinationVectorType>
  static void process_one_element_with_vector_operation(
      svbool_t pg, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets, size_t index,
      const InnerFilterType& filter_) KLEIDICV_STREAMING {
    SourceVectorType src_0_0, src_0_1, src_0_2, src_0_3, src_0_4, src_0_5,
        src_0_6, src_1_0, src_1_1, src_1_2, src_1_3, src_1_4, src_1_5, src_1_6,
        src_2_0, src_2_1, src_2_2, src_2_3, src_2_4, src_2_5, src_2_6, src_3_0,
        src_3_1, src_3_2, src_3_3, src_3_4, src_3_5, src_3_6, src_4_0, src_4_1,
        src_4_2, src_4_3, src_4_4, src_4_5, src_4_6, src_5_0, src_5_1, src_5_2,
        src_5_3, src_5_4, src_5_5, src_5_6, src_6_0, src_6_1, src_6_2, src_6_3,
        src_6_4, src_6_5, src_6_6;
    DestinationVectorType output_vector;

    // Initialization
    ScalableVectorArray2D<SourceVectorType, 7, 7> KernelWindow = {{
        {std::ref(src_0_0), std::ref(src_0_1), std::ref(src_0_2),
         std::ref(src_0_3), std::ref(src_0_4), std::ref(src_0_5),
         std::ref(src_0_6)},
        {std::ref(src_1_0), std::ref(src_1_1), std::ref(src_1_2),
         std::ref(src_1_3), std::ref(src_1_4), std::ref(src_1_5),
         std::ref(src_1_6)},
        {std::ref(src_2_0), std::ref(src_2_1), std::ref(src_2_2),
         std::ref(src_2_3), std::ref(src_2_4), std::ref(src_2_5),
         std::ref(src_2_6)},
        {std::ref(src_3_0), std::ref(src_3_1), std::ref(src_3_2),
         std::ref(src_3_3), std::ref(src_3_4), std::ref(src_3_5),
         std::ref(src_3_6)},
        {std::ref(src_4_0), std::ref(src_4_1), std::ref(src_4_2),
         std::ref(src_4_3), std::ref(src_4_4), std::ref(src_4_5),
         std::ref(src_4_6)},
        {std::ref(src_5_0), std::ref(src_5_1), std::ref(src_5_2),
         std::ref(src_5_3), std::ref(src_5_4), std::ref(src_5_5),
         std::ref(src_5_6)},
        {std::ref(src_6_0), std::ref(src_6_1), std::ref(src_6_2),
         std::ref(src_6_3), std::ref(src_6_4), std::ref(src_6_5),
         std::ref(src_6_6)},
    }};

    auto load_array_element = [&](const SourceType& x)
                                  KLEIDICV_STREAMING { return svld1(pg, &x); };

    WindowLoaderType::load_window(KernelWindow, load_array_element, src_rows,
                                  window_row_offsets, window_col_offsets,
                                  index);
    filter_.vector_path(pg, KernelWindow, output_vector);
    svst1(pg, &dst_rows[index], output_vector);
  }
};

template <typename InnerFilterType, size_t KSize,
          typename VectorOperationProviderType>
class Filter2d {
 public:
  using SourceType = typename InnerFilterType::SourceType;
  using DestinationType = typename InnerFilterType::DestinationType;
  using SourceVecTraits =
      typename KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using DestinationVecTraits =
      typename KLEIDICV_TARGET_NAMESPACE::VecTraits<DestinationType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using DestinationVectorType = typename DestinationVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo<SourceType, KSize>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;
  static constexpr size_t kMargin = KSize / 2UL;
  explicit Filter2d(InnerFilterType filter) KLEIDICV_STREAMING
      : filter_{filter} {}

  void process_pixels_without_horizontal_borders(
      size_t width, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets) const KLEIDICV_STREAMING {
    LoopUnroll2 loop{width * src_rows.channels(), SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) KLEIDICV_STREAMING {
      svbool_t pg = SourceVecTraits::svptrue();
      VectorOperationProviderType::
          template process_one_element_with_vector_operation<
              InnerFilterType, SourceVectorType, DestinationVectorType>(
              pg, src_rows, dst_rows, window_row_offsets, window_col_offsets,
              index, filter_);
    });

    loop.remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
      svbool_t pg = SourceVecTraits::svwhilelt(index, length);
      VectorOperationProviderType::
          template process_one_element_with_vector_operation<
              InnerFilterType, SourceVectorType, DestinationVectorType>(
              pg, src_rows, dst_rows, window_row_offsets, window_col_offsets,
              index, filter_);
    });
  }

  void process_one_pixel_with_horizontal_borders(
      Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets window_row_offsets,
      BorderOffsets window_col_offsets) const KLEIDICV_STREAMING {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      VectorOperationProviderType::
          template process_one_element_with_vector_operation<
              InnerFilterType, SourceVectorType, DestinationVectorType>(
              SourceVecTraits::template svptrue_pat<SV_VL1>(), src_rows,
              dst_rows, window_row_offsets, window_col_offsets, index, filter_);
    }
  }

  void process_pixels_of_dual_rows_without_horizontal_borders(
      size_t width, Rows<const SourceType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets window_row_offsets_0,
      BorderOffsets window_row_offsets_1,
      BorderOffsets window_col_offsets) const KLEIDICV_STREAMING {
    LoopUnroll2 loop{width * src_rows.channels(), SourceVecTraits::num_lanes()};
    loop.unroll_once([&](size_t index) KLEIDICV_STREAMING {
      svbool_t pg = SourceVecTraits::svptrue();
      VectorOperationProviderType::
          template process_two_elements_with_vector_operation<
              InnerFilterType, SourceVectorType, DestinationVectorType>(
              pg, src_rows, dst_rows, window_row_offsets_0,
              window_row_offsets_1, window_col_offsets, index, filter_);
    });

    loop.remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
      svbool_t pg = SourceVecTraits::svwhilelt(index, length);
      VectorOperationProviderType::
          template process_two_elements_with_vector_operation<
              InnerFilterType, SourceVectorType, DestinationVectorType>(
              pg, src_rows, dst_rows, window_row_offsets_0,
              window_row_offsets_1, window_col_offsets, index, filter_);
    });
  }

  // Processes two vertically adjacent pixels in a single column
  void process_two_pixels_with_horizontal_borders(
      Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets window_row_offsets_0, BorderOffsets window_row_offsets_1,
      BorderOffsets window_col_offsets) const KLEIDICV_STREAMING {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      VectorOperationProviderType::
          template process_two_elements_with_vector_operation<
              InnerFilterType, SourceVectorType, DestinationVectorType>(
              SourceVecTraits::template svptrue_pat<SV_VL1>(), src_rows,
              dst_rows, window_row_offsets_0, window_row_offsets_1,
              window_col_offsets, index, filter_);
    }
  }

 private:
  InnerFilterType filter_;
};

// Shorthand for 3x3 2D filters driver type.
template <class InnerFilterType>
using Filter2D3x3 = Filter2d<
    InnerFilterType, 3UL,
    Filter2D3x3VectorOperations<
        typename InnerFilterType::SourceType,
        typename InnerFilterType::DestinationType,
        Filter2dWindowLoader3x3<typename InnerFilterType::SourceType>>>;

template <typename InnerFilterType>
using Filter2D5x5 = Filter2d<
    InnerFilterType, 5UL,
    Filter2D5x5VectorOperations<
        typename InnerFilterType::SourceType,
        typename InnerFilterType::DestinationType,
        Filter2dWindowLoader5x5<typename InnerFilterType::SourceType>>>;

template <typename InnerFilterType>
using Filter2D7x7 = Filter2d<
    InnerFilterType, 7UL,
    Filter2D7x7VectorOperations<
        typename InnerFilterType::SourceType,
        typename InnerFilterType::DestinationType,
        Filter2dWindowLoader7x7<typename InnerFilterType::SourceType>>>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_FILTER_2D_SC_H
