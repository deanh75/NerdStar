// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "rgb_to_yuv420_sc.h"

namespace kleidicv::sme2 {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_yuv420p_stripe_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         kleidicv_color_conversion_t color_format, size_t begin,
                         size_t end) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, (height * 3 + 1) / 2);
  CHECK_IMAGE_SIZE(width, height);
  uint8_t *uv_dst = dst + dst_stride * height;

  switch (color_format) {
    case KLEIDICV_BGR_TO_YV12: {
      return RGBxorBGRxToYUV420<false, false, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_RGB_TO_YV12: {
      return RGBxorBGRxToYUV420<false, true, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_BGRA_TO_YV12: {
      return RGBxorBGRxToYUV420<true, false, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_RGBA_TO_YV12: {
      return RGBxorBGRxToYUV420<true, true, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_BGR_TO_IYUV: {
      return RGBxorBGRxToYUV420<false, false, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          false, begin, end);
    }

    case KLEIDICV_RGB_TO_IYUV: {
      return RGBxorBGRxToYUV420<false, true, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          false, begin, end);
    }

    case KLEIDICV_BGRA_TO_IYUV: {
      return RGBxorBGRxToYUV420<true, false, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          false, begin, end);
    }

    case KLEIDICV_RGBA_TO_IYUV: {
      return RGBxorBGRxToYUV420<true, true, false>::rgb2yuv420_operation_sc(
          src, src_stride, dst, dst_stride, uv_dst, dst_stride, width, height,
          false, begin, end);
    }

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace kleidicv::sme2
