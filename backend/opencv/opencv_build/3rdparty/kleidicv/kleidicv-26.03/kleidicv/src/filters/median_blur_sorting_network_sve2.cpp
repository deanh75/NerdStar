// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/filters/median_blur.h"
#include "median_blur_sorting_network_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t median_blur_sorting_network_stripe(
    const T* src, size_t src_stride, T* dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type) {
  return median_blur_sorting_network_stripe_sc(
      src, src_stride, dst, dst_stride, width, height, y_begin, y_end, channels,
      kernel_width, kernel_height, border_type);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                    \
  median_blur_sorting_network_stripe<type>(                             \
      const type* src, size_t src_stride, type* dst, size_t dst_stride, \
      size_t width, size_t height, size_t y_begin, size_t y_end,        \
      size_t channels, size_t kernel_width, size_t kernel_height,       \
      FixedBorderType border_type)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::sve2
