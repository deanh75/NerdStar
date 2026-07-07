// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/resize/resize_linear.h"
#include "resize_linear_generic_sc.h"
#include "resize_linear_sc.h"

namespace kleidicv::sme2 {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
kleidicv_resize_2x2_stripe_u8(const uint8_t *src, size_t src_stride,
                              size_t src_width, size_t src_height,
                              size_t y_begin, size_t y_end, uint8_t *dst,
                              size_t dst_stride) {
  return kleidicv_resize_2x2_u8_sc(src, src_stride, src_width, src_height,
                                   y_begin, y_end, dst, dst_stride);
}

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
kleidicv_resize_4x4_stripe_u8(const uint8_t *src, size_t src_stride,
                              size_t src_width, size_t src_height,
                              size_t y_begin, size_t y_end, uint8_t *dst,
                              size_t dst_stride) {
  return kleidicv_resize_4x4_u8_sc(src, src_stride, src_width, src_height,
                                   y_begin, y_end, dst, dst_stride);
}

template <int kRatio, int kChannels>
KLEIDICV_LOCALLY_STREAMING kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height) {
  return kleidicv_resize_generic_stripe_u8_sc<kRatio, kChannels>(
      src, src_stride, src_width, src_height, y_begin, y_end, dst, dst_stride,
      dst_width, dst_height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(ratio, channels)               \
  template kleidicv_error_t                                          \
  kleidicv_resize_generic_stripe_u8<ratio, channels>(                \
      const uint8_t *src, size_t src_stride, size_t src_width,       \
      size_t src_height, size_t y_begin, size_t y_end, uint8_t *dst, \
      size_t dst_stride, size_t dst_width, size_t dst_height)

KLEIDICV_INSTANTIATE_TEMPLATE(2, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 3);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 3);

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
kleidicv_resize_linear_stripe_f32(const float *src, size_t src_stride,
                                  size_t src_width, size_t src_height,
                                  size_t y_begin, size_t y_end, float *dst,
                                  size_t dst_stride, size_t dst_width,
                                  size_t dst_height) {
  return kleidicv_resize_linear_stripe_f32_sc(
      src, src_stride, src_width, src_height, y_begin, y_end, dst, dst_stride,
      dst_width, dst_height);
}

}  // namespace kleidicv::sme2
