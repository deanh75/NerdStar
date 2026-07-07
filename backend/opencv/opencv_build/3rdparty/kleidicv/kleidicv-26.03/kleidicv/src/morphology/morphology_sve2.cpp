// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "morphology_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t dilate(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t channels, size_t kernel_width, size_t kernel_height,
    size_t anchor_x, size_t anchor_y, kleidicv_border_type_t border_type,
    const uint8_t *border_value, size_t iterations) {
  return dilate_sc<T, MorphologyWorkspace::CopyDataMemcpy<T> >(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, anchor_x, anchor_y, border_type, border_value, iterations);
}

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
erode(const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
      size_t height, size_t channels, size_t kernel_width, size_t kernel_height,
      size_t anchor_x, size_t anchor_y, kleidicv_border_type_t border_type,
      const uint8_t *border_value, size_t iterations) {
  return erode_sc<T, MorphologyWorkspace::CopyDataMemcpy<T> >(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, anchor_x, anchor_y, border_type, border_value, iterations);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, type)                        \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<type>(         \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,  \
      size_t width, size_t height, size_t channels, size_t kernel_width, \
      size_t kernel_height, size_t anchor_x, size_t anchor_y,            \
      kleidicv_border_type_t border_type, const uint8_t *border_value,   \
      size_t iterations)

KLEIDICV_INSTANTIATE_TEMPLATE(dilate, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(erode, uint8_t);

}  // namespace kleidicv::sve2
