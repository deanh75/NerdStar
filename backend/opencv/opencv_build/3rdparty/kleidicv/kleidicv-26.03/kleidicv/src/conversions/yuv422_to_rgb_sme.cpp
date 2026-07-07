// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "yuv422_to_rgb_sc.h"

namespace kleidicv::sme {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
yuv422_to_rgb_u8(const uint8_t* src, size_t src_stride, uint8_t* dst,
                 size_t dst_stride, size_t width, size_t height,
                 kleidicv_color_conversion_t color_format) {
  return yuv422_to_rgb_u8_sc(src, src_stride, dst, dst_stride, width, height,
                             color_format);
}

}  // namespace kleidicv::sme
