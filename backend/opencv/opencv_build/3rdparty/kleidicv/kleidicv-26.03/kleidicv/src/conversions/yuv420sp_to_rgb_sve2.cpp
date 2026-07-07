// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv420sp_to_rgb_sc.h"

namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t yuv420sp_to_rgb_u8(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, kleidicv_color_conversion_t color_format) {
  return yuv420sp_to_rgb_u8_sc(src_y, src_y_stride, src_uv, src_uv_stride, dst,
                               dst_stride, width, height, color_format);
}

}  // namespace kleidicv::sve2
