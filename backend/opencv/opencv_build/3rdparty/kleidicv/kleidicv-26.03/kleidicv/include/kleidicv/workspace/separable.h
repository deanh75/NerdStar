// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_SEPARABLE_H
#define KLEIDICV_WORKSPACE_SEPARABLE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <variant>

#include "kleidicv/types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Workspace for separable fixed-size filters.
//
// Theory of operation
//
// Given an NxM input matrix and a separable filter AxB = V x H, this workspace
// first processes N rows vertically into a separate horizontal buffer. Right
// after the vertical operation, the horizontal operation is applied and the
// result is written to the destination.
//
// Limitations
//
//  1. In-place operations are not supported.
//  2. The input's width and height have to be at least `filter's width - 1` and
//  `filter's height - 1`, respectively.
//
// Example
//
//  N = 2, M = 3, A = B = 3, border type replicate and 'x' is multiplication.
//
//  Input:              Separated filters:
//    [ M00, M01, M02 ]     V = [ V0 ]  H = [H0, H1, H2 ]
//    [ M10, M11, M12 ]         [ V1 ]
//    [ M20, M21, M22 ]         [ V2 ]
//
//  Buffer contents in iteration 0 after applying the vertical operation
//  taking "replicate" border type into account:
//
//    [ B0, B1, B2 ] =
//      [ M{0, 0, 1}0 x V, M{0, 0, 1}1 x V, M{0, 0, 1}2 x V ]
//
//  The horizontal operation is then semantically performed on the following
//  input taking "replicate" border type into account:
//
//    [ B0, B0, B1, B2, B2 ]
//
//  The destination contents after the 0th iteration is then:
//
//    [ D00, D01, D02 ] =
//      [ B{0, 0, 1} x H, B{0, 1, 2} x H, B{1, 2, 2} x H]
//
// Handling of borders is calculated based on offsets rather than setting up
// suitably-sized buffers which could hold both borders and data.
class SeparableFilterWorkspace {
 public:
  // Workspace is only constructible with create().
  SeparableFilterWorkspace() = delete;

  static std::variant<SeparableFilterWorkspace, kleidicv_error_t> create(
      Rectangle rect, size_t channels,
      size_t intermediate_size) KLEIDICV_STREAMING {
    auto [allocation, buffer_rows_stride] =
        allocate(rect, channels, intermediate_size);

    if (!allocation) {
      return KLEIDICV_ERROR_ALLOCATION;
    }

    return SeparableFilterWorkspace{rect, channels, allocation,
                                    buffer_rows_stride};
  }

 protected:
  static std::pair<uint8_t *, size_t> allocate(Rectangle rect, size_t channels,
                                               size_t intermediate_size)
      KLEIDICV_STREAMING {
    size_t buffer_rows_number_of_elements = rect.width() * channels;
    // Adding more elements because of SVE, where interleaving stores are
    // governed by one predicate. For example, if a predicate requires 7 uint8_t
    // elements and an algorithm performs widening to 16 bits, the resulting
    // interleaving store will still be governed by the same predicate, thus
    // storing 8 elements. Choosing '3' to account for svst4().
    buffer_rows_number_of_elements += 3;

    size_t buffer_rows_stride =
        buffer_rows_number_of_elements * intermediate_size;

    uint8_t *allocation =
        reinterpret_cast<uint8_t *>(std::malloc(buffer_rows_stride));

    return {allocation, buffer_rows_stride};
  }

  SeparableFilterWorkspace(Rectangle rect, size_t channels, uint8_t *allocation,
                           size_t buffer_rows_stride) KLEIDICV_STREAMING
      : rect_{rect},
        channels_{channels},
        buffer_{allocation, &std::free},
        buffer_rows_stride_{buffer_rows_stride} {}

 public:
  // Processes rows vertically first along the full width
  template <typename FilterType>
  void process(size_t y_begin, size_t y_end,
               Rows<const typename FilterType::SourceType> src_rows,
               Rows<typename FilterType::DestinationType> dst_rows,
               typename FilterType::BorderType border_type,
               FilterType filter) KLEIDICV_STREAMING {
    // Border helper which calculates border offsets.
    typename FilterType::BorderInfoType vertical_border{rect_.height(),
                                                        border_type};
    typename FilterType::BorderInfoType horizontal_border{rect_.width(),
                                                          border_type};

    // Buffer rows which hold intermediate widened data.
    auto buffer_rows =
        Rows{reinterpret_cast<typename FilterType::BufferType *>(buffer_.get()),
             buffer_rows_stride_, channels_};

    // Vertical processing loop.
    for (size_t vertical_index = y_begin; vertical_index < y_end;
         ++vertical_index) {
      // Recalculate vertical border offsets.
      auto offsets = vertical_border.offsets_with_border(vertical_index);
      // Process in the vertical direction first.
      filter.process_vertical(rect_.width(), src_rows.at(vertical_index),
                              buffer_rows, offsets);
      // Process in the horizontal direction last.
      process_horizontal(rect_.width(), buffer_rows,
                         dst_rows.at(vertical_index), filter,
                         horizontal_border);
    }
  }

  // Processes rows vertically first along the full width
  template <typename FilterType>
  void process_arbitrary(size_t kernel_size, size_t y_begin, size_t y_end,
                         Rows<const typename FilterType::SourceType> src_rows,
                         Rows<typename FilterType::DestinationType> dst_rows,
                         typename FilterType::BorderType /* border_type */,
                         FilterType filter) KLEIDICV_STREAMING {
    // Buffer rows which hold intermediate widened data.
    auto buffer_rows =
        Rows{reinterpret_cast<typename FilterType::BufferType *>(buffer_.get()),
             buffer_rows_stride_, channels_};
    size_t margin = kernel_size / 2;

    // Process top rows, affected by border
    for (size_t row_index = y_begin; row_index < std::max(y_begin, margin);
         ++row_index) {
      filter.process_arbitrary_border_vertical(rect_.width(), src_rows,
                                               row_index, buffer_rows);
      filter.process_arbitrary_horizontal(rect_.width(), kernel_size,
                                          buffer_rows, dst_rows.at(row_index));
    }

    // Process middle rows that are not affected by any borders
    for (size_t row_index = std::max(y_begin, margin);
         row_index < std::min(y_end, rect_.height() - margin); ++row_index) {
      filter.process_arbitrary_vertical(rect_.width(), src_rows.at(row_index),
                                        buffer_rows);
      filter.process_arbitrary_horizontal(rect_.width(), kernel_size,
                                          buffer_rows, dst_rows.at(row_index));
    }

    // Process bottom rows, affected by border
    for (size_t row_index = std::min(y_end, rect_.height() - margin);
         row_index < y_end; ++row_index) {
      filter.process_arbitrary_border_vertical(rect_.width(), src_rows,
                                               row_index, buffer_rows);
      filter.process_arbitrary_horizontal(rect_.width(), kernel_size,
                                          buffer_rows, dst_rows.at(row_index));
    }
  }

 private:
  template <typename FilterType>
  void process_horizontal(size_t width,
                          Rows<typename FilterType::BufferType> buffer_rows,
                          Rows<typename FilterType::DestinationType> dst_rows,
                          FilterType filter,
                          typename FilterType::BorderInfoType horizontal_border)
      KLEIDICV_STREAMING {
    // Margin associated with the filter.
    constexpr size_t margin = filter.margin;

    // Process data affected by left border.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t horizontal_index = 0; horizontal_index < margin;
         ++horizontal_index) {
      auto offsets =
          horizontal_border.offsets_with_left_border(horizontal_index);
      filter.process_horizontal_borders(buffer_rows.at(0, horizontal_index),
                                        dst_rows.at(0, horizontal_index),
                                        offsets);
    }

    // Process data which is not affected by any borders in bulk.
    {
      size_t width_without_borders = width - (2 * margin);
      auto offsets = horizontal_border.offsets_without_border();
      filter.process_horizontal(width_without_borders,
                                buffer_rows.at(0, margin),
                                dst_rows.at(0, margin), offsets);
    }

    // Process data affected by right border.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t horizontal_index = 0; horizontal_index < margin;
         ++horizontal_index) {
      size_t index = width - margin + horizontal_index;
      auto offsets = horizontal_border.offsets_with_right_border(index);
      filter.process_horizontal_borders(buffer_rows.at(0, index),
                                        dst_rows.at(0, index), offsets);
    }
  }

 protected:
  Rectangle rect_;
  size_t channels_;
  std::unique_ptr<uint8_t, decltype(&std::free)> buffer_;
  size_t buffer_rows_stride_;
};  // end of class SeparableFilterWorkspace

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_SEPARABLE_H
