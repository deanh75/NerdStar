// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_YUV_TO_RGB_H
#define KLEIDICV_CONVERSIONS_YUV_TO_RGB_H

#include "kleidicv/kleidicv.h"

extern "C" {
// For internal use only. See instead kleidicv_yuv_to_rgb_u8.
// Converts a stripe (range of rows) of a planar YUV420 image (I420 or YV12)
// to RGB, RGBA, BGR, or BGRA format. All channels are 8-bit wide. The stripe
// is defined by the range [begin, end).
KLEIDICV_API_DECLARATION(kleidicv_yuv420p_to_rgb_stripe_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_color_conversion_t color_format, size_t begin,
                         size_t end);

// For internal use only. See instead kleidicv_yuv_to_rgb_u8.
// Converts an interleaved YUV444 image to RGB, RGBA, BGR, or BGRA format.
// All channels are 8-bit wide.
KLEIDICV_API_DECLARATION(kleidicv_yuv444_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_color_conversion_t color_format);

// For internal use only. See instead kleidicv_yuv_to_rgb_u8.
// Converts an interleaved YUV422 image to RGB, RGBA, BGR, or BGRA format.
// All channels are 8-bit wide.
KLEIDICV_API_DECLARATION(kleidicv_yuv422_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_color_conversion_t color_format);
}
namespace kleidicv {

namespace neon {
kleidicv_error_t yuv420sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                    const uint8_t *src_uv, size_t src_uv_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height,
                                    kleidicv_color_conversion_t color_format);

kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);

kleidicv_error_t yuv444_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t yuv422_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

}  // namespace neon

namespace sve2 {
kleidicv_error_t yuv420sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                    const uint8_t *src_uv, size_t src_uv_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height,
                                    kleidicv_color_conversion_t color_format);

kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);

kleidicv_error_t yuv444_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t yuv422_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

}  // namespace sve2

namespace sme {
kleidicv_error_t yuv420sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                    const uint8_t *src_uv, size_t src_uv_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height,
                                    kleidicv_color_conversion_t color_format);

kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);

kleidicv_error_t yuv444_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t yuv422_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

}  // namespace sme

namespace sme2 {

kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);
}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_YUV_TO_RGB_H
