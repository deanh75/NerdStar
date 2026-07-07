// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv420p_to_rgb_sc.h"

namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t* src, size_t src_stride, uint8_t* dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end) {
  return yuv420p_to_rgb_stripe_u8_sc(src, src_stride, dst, dst_stride, width,
                                     height, color_format, begin, end);
}

}  // namespace kleidicv::sve2
