// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_RGB_TO_YUV_H
#define KLEIDICV_CONVERSIONS_RGB_TO_YUV_H

#include "kleidicv/kleidicv.h"

extern "C" {
// For internal use only. See instead kleidicv_rgb_to_yuv_semiplanar_u8.
// Converts a stripe (range of rows) of an interleaved RGB-family image
// (RGB, RGBA, BGR, or BGRA) to semi-planar YUV420 (nv12 or nv21).
// All channels are 8-bit wide. The stripe is defined by the range [begin, end).
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_yuv420sp_stripe_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *y_dst, size_t y_stride,
                         uint8_t *uv_dst, size_t uv_stride, size_t width,
                         size_t height,
                         kleidicv_color_conversion_t color_format, size_t begin,
                         size_t end);

// For internal use only. See instead kleidicv_rgb_to_yuv_u8.
// Converts a stripe (range of rows) of an interleaved RGB-family image
// (RGB, RGBA, BGR, or BGRA) to planar YUV420 (I420 or YV12).
// All channels are 8-bit wide. The stripe is defined by the range [begin, end).
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_yuv420p_stripe_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_color_conversion_t color_format, size_t begin,
                         size_t end);

// For internal use only. See instead kleidicv_rgb_to_yuv_u8.
// Converts an interleaved RGB-family image (RGB, RGBA, BGR, or BGRA)
// to an interleaved YUV444 image. All channels are 8-bit wide.
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_yuv444_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_color_conversion_t color_format);

// For internal use only. See instead kleidicv_rgb_to_yuv_u8.
// Converts an interleaved RGB-family image (RGB, RGBA, BGR, or BGRA)
// to an interleaved YUV422 image. All channels are 8-bit wide.
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_yuv422_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_color_conversion_t color_format);
}

namespace kleidicv {
namespace neon {

kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t rgb_to_yuv420p_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);

kleidicv_error_t rgb_to_yuv420sp_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    kleidicv_color_conversion_t color_format, size_t begin, size_t end);

kleidicv_error_t rgb_to_yuv422_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);
}  // namespace neon

namespace sve2 {

kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t rgb_to_yuv420p_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);

kleidicv_error_t rgb_to_yuv420sp_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    kleidicv_color_conversion_t color_format, size_t begin, size_t end);

kleidicv_error_t rgb_to_yuv422_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);
}  // namespace sve2

namespace sme {

kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t rgb_to_yuv420p_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);

kleidicv_error_t rgb_to_yuv420sp_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    kleidicv_color_conversion_t color_format, size_t begin, size_t end);

kleidicv_error_t rgb_to_yuv422_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);
}  // namespace sme

namespace sme2 {

kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format);

kleidicv_error_t rgb_to_yuv420p_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end);
}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_RGB_TO_YUV_H
