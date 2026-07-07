// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/transform/rotate.h"
#include "rotate_transpose_common_neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
static kleidicv_error_t rotate_cw(Rectangle rect,
                                  Rows<const ScalarType> src_rows,
                                  Rows<ScalarType> dst_rows) {
  // 90-degree clockwise rotate mapping:
  auto vector_dst_row_from_src_col = [](size_t hindex, size_t) {
    return hindex;
  };
  auto vector_dst_col_from_src_row = [height = rect.height()](size_t vindex,
                                                              size_t lanes) {
    return height - vindex - lanes;
  };
  auto scalar_dst_row_from_src_col = [](size_t hindex, size_t) {
    return hindex;
  };
  auto scalar_dst_col_from_src_row = [height = rect.height()](size_t vindex,
                                                              size_t lanes) {
    return height - vindex - lanes;
  };
  auto vertical_dst_col_base = []([[maybe_unused]] size_t vindex) {
    return size_t{0};
  };
  auto row_index = [](size_t index, size_t) { return index; };
  auto scalar_store = [](Rows<ScalarType> &rows, size_t col, size_t row,
                         size_t height, const ScalarType value, size_t) {
    // dst[j][src_height - i - 1] = src[i][j]
    rows.at(col)[height - row - 1] = value;
  };
  return rotate_transpose_rect<true>(
      rect, src_rows, dst_rows, vector_dst_row_from_src_col,
      vector_dst_col_from_src_row, scalar_dst_row_from_src_col,
      scalar_dst_col_from_src_row, vertical_dst_col_base, row_index,
      scalar_store);
}

template <typename ScalarType>
static kleidicv_error_t rotate_ccw(Rectangle rect,
                                   Rows<const ScalarType> src_rows,
                                   Rows<ScalarType> dst_rows) {
  // 90-degree counter-clockwise rotate mapping.
  auto vector_dst_row_from_src_col = [width = rect.width()](size_t hindex,
                                                            size_t lanes) {
    return width - hindex - lanes;
  };
  auto vector_dst_col_from_src_row = [](size_t vindex, size_t) {
    return vindex;
  };
  auto scalar_dst_row_from_src_col = [width = rect.width()](
                                         size_t, size_t final_hindex) {
    return width - final_hindex;
  };
  auto scalar_dst_col_from_src_row = [](size_t vindex, size_t) {
    return vindex;
  };
  auto vertical_dst_col_base = [](size_t vindex) { return vindex; };
  auto row_index = [](size_t index, size_t lanes) { return lanes - index - 1; };
  auto scalar_store = [width = rect.width()](
                          Rows<ScalarType> &rows, size_t col, size_t row,
                          size_t, const ScalarType value, size_t row_width) {
    // dst[width - j - 1][i] = src[i][j]
    rows.at(row_width - col - 1)[row] = value;
  };
  return rotate_transpose_rect<false>(
      rect, src_rows, dst_rows, vector_dst_row_from_src_col,
      vector_dst_col_from_src_row, scalar_dst_row_from_src_col,
      scalar_dst_col_from_src_row, vertical_dst_col_base, row_index,
      scalar_store);
}

template <typename T>
static kleidicv_error_t rotate(const void *src_void, size_t src_stride,
                               size_t src_width, size_t src_height,
                               void *dst_void, size_t dst_stride, int angle) {
  MAKE_POINTER_CHECK_ALIGNMENT(const T, src, src_void);
  MAKE_POINTER_CHECK_ALIGNMENT(T, dst, dst_void);
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, src_width);
  CHECK_IMAGE_SIZE(src_width, src_height);

  Rectangle rect{src_width, src_height};
  Rows<T> dst_rows{dst, dst_stride};
  Rows<const T> src_rows{src, src_stride};

  if (angle == 90) {
    return rotate_cw(rect, src_rows, dst_rows);
  }
  return rotate_ccw(rect, src_rows, dst_rows);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rotate(const void *src, size_t src_stride, size_t src_width,
                        size_t src_height, void *dst, size_t dst_stride,
                        int angle, size_t pixel_size) {
  if (!rotate_is_implemented(src, dst, angle, pixel_size)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  switch (pixel_size) {
    case sizeof(uint8_t):
      return rotate<uint8_t>(src, src_stride, src_width, src_height, dst,
                             dst_stride, angle);
    case sizeof(uint16_t):
      return rotate<uint16_t>(src, src_stride, src_width, src_height, dst,
                              dst_stride, angle);
    case sizeof(uint32_t):
      return rotate<uint32_t>(src, src_stride, src_width, src_height, dst,
                              dst_stride, angle);
    case sizeof(uint64_t):
      return rotate<uint64_t>(src, src_stride, src_width, src_height, dst,
                              dst_stride, angle);
    // GCOVR_EXCL_START
    default:
      assert(!"pixel size not implemented");
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
      // GCOVR_EXCL_STOP
  }
}

}  // namespace kleidicv::neon
