// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MEDIAN_BLUR_SC_H
#define KLEIDICV_MEDIAN_BLUR_SC_H

#include <algorithm>

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/filter_2d_sc.h"
#include "kleidicv/filters/median_blur.h"
#include "kleidicv/filters/process_filter_2d.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/workspace/border_3x3.h"
#include "kleidicv/workspace/border_5x5.h"
#include "kleidicv/workspace/border_7x7.h"
#include "median_blur_sorting_network_3x3.h"
#include "median_blur_sorting_network_5x5.h"
#include "median_blur_sorting_network_7x7.h"
namespace KLEIDICV_TARGET_NAMESPACE {

// Primary template for Median Blur filters.
template <typename ScalarType, size_t KernelSize>
class MedianBlurSortingNetwork;

template <typename ScalarType>
class VectorComparator {
 public:
  using SourceVectorType = typename VecTraits<ScalarType>::VectorType;

  static void compare_and_swap(SourceVectorType& left, SourceVectorType& right,
                               svbool_t& pg) KLEIDICV_STREAMING {
    SourceVectorType max_value = svmax_m(pg, left, right);
    SourceVectorType min_value = svmin_m(pg, left, right);
    left = min_value;
    right = max_value;
  }

  static void min(SourceVectorType& left, SourceVectorType& right,
                  svbool_t& pg) KLEIDICV_STREAMING {
    left = svmin_m(pg, left, right);
  }

  static void max(SourceVectorType& left, SourceVectorType& right,
                  svbool_t& pg) KLEIDICV_STREAMING {
    right = svmax_m(pg, left, right);
  }
  static SourceVectorType get_min(SourceVectorType& left,
                                  SourceVectorType& right,
                                  svbool_t& pg) KLEIDICV_STREAMING {
    return svmin_m(pg, left, right);
  }
  static SourceVectorType get_max(SourceVectorType& left,
                                  SourceVectorType& right,
                                  svbool_t& pg) KLEIDICV_STREAMING {
    return svmax_m(pg, left, right);
  }
};

// Template for Median Blur 3x3 filters.
template <typename ScalarType>
class MedianBlurSortingNetwork<ScalarType, 3> {
 public:
  using SourceType = ScalarType;
  using DestinationType = SourceType;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using DestinationVectorType = typename VecTraits<DestinationType>::VectorType;

  template <typename KernelWindowFunctor>
  void vector_path(svbool_t& pg, KernelWindowFunctor& KernelWindow,
                   DestinationVectorType& output_vec) const KLEIDICV_STREAMING {
    sorting_network3x3_single_row<VectorComparator<ScalarType>>(KernelWindow,
                                                                output_vec, pg);
  }

  template <typename KernelWindowFunctor>
  void vector_path_for_dual_row_handling(
      svbool_t& pg, KernelWindowFunctor& KernelWindow,
      DestinationVectorType& output_vec_0,
      DestinationVectorType& output_vec_1) const KLEIDICV_STREAMING {
    sorting_network3x3_dual_rows<VectorComparator<ScalarType>>(
        KernelWindow, output_vec_0, output_vec_1, pg);
  }
};  // end of class MedianBlurSortingNetwork<ScalarType, 3>

// Template for Median Blur 5x5 filters.
template <typename ScalarType>
class MedianBlurSortingNetwork<ScalarType, 5> {
 public:
  using SourceType = ScalarType;
  using DestinationType = SourceType;
  using SourceVecTraits =
      typename KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using DestinationVectorType = typename KLEIDICV_TARGET_NAMESPACE::VecTraits<
      DestinationType>::VectorType;
  template <typename KernelWindowFunctor>
  void vector_path(svbool_t& pg, KernelWindowFunctor& KernelWindow,
                   DestinationVectorType& output_vec) const KLEIDICV_STREAMING {
    sorting_network5x5<VectorComparator<ScalarType>>(KernelWindow, output_vec,
                                                     pg);
  }
};  // end of class MedianBlurSortingNetworkSortingNetwork<ScalarType, 5>

// Template for Median Blur 7x7 filters.
template <typename ScalarType>
class MedianBlurSortingNetwork<ScalarType, 7> {
 public:
  using SourceType = ScalarType;
  using DestinationType = SourceType;
  using SourceVecTraits =
      typename KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using DestinationVectorType = typename KLEIDICV_TARGET_NAMESPACE::VecTraits<
      DestinationType>::VectorType;

  template <typename KernelWindowFunctor>
  void vector_path(svbool_t& pg, KernelWindowFunctor& KernelWindow,
                   DestinationVectorType& output_vec) const KLEIDICV_STREAMING {
    sorting_network7x7<VectorComparator<ScalarType>>(KernelWindow, output_vec,
                                                     pg);
  }
};  // end of class MedianBlurSortingNetworkSortingNetwork<ScalarType, 7>

template <typename T>
kleidicv_error_t median_blur_sorting_network_stripe_sc(
    const T* src, size_t src_stride, T* dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, [[maybe_unused]] size_t kernel_height,
    FixedBorderType border_type) KLEIDICV_STREAMING {
  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  if (kernel_width == 3) {
    MedianBlurSortingNetwork<T, 3> median_filter;
    Filter2D3x3<MedianBlurSortingNetwork<T, 3>> filter{median_filter};
    process_filter2d_by_dual_rows(rect, y_begin, y_end, src_rows, dst_rows,
                                  border_type, filter);
    return KLEIDICV_OK;
  }

  if (kernel_width == 5) {
    MedianBlurSortingNetwork<T, 5> median_filter;
    Filter2D5x5<MedianBlurSortingNetwork<T, 5>> filter{median_filter};
    process_filter2d(rect, y_begin, y_end, src_rows, dst_rows, border_type,
                     filter);
    return KLEIDICV_OK;
  }

  MedianBlurSortingNetwork<T, 7> median_filter;
  Filter2D7x7<MedianBlurSortingNetwork<T, 7>> filter{median_filter};
  process_filter2d(rect, y_begin, y_end, src_rows, dst_rows, border_type,
                   filter);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MEDIAN_BLUR_SC_H
