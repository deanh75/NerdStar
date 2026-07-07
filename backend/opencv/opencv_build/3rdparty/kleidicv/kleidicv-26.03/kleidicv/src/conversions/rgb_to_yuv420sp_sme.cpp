// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "rgb_to_yuv420_sc.h"

namespace kleidicv::sme {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_yuv420sp_stripe_u8(const uint8_t *src, size_t src_stride, uint8_t *y_dst,
                          size_t y_stride, uint8_t *uv_dst, size_t uv_stride,
                          size_t width, size_t height,
                          kleidicv_color_conversion_t color_format,
                          size_t begin, size_t end) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(y_dst, y_stride, height);
  CHECK_POINTER_AND_STRIDE(uv_dst, uv_stride, (height + 1) / 2);
  CHECK_IMAGE_SIZE(width, height);
  switch (color_format) {
    case KLEIDICV_BGR_TO_NV21: {
      return RGBxorBGRxToYUV420<false, false, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_RGB_TO_NV21: {
      return RGBxorBGRxToYUV420<false, true, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_BGRA_TO_NV21: {
      return RGBxorBGRxToYUV420<true, false, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_RGBA_TO_NV21: {
      return RGBxorBGRxToYUV420<true, true, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    }

    case KLEIDICV_BGR_TO_NV12: {
      return RGBxorBGRxToYUV420<false, false, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    }

    case KLEIDICV_RGB_TO_NV12: {
      return RGBxorBGRxToYUV420<false, true, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    }

    case KLEIDICV_BGRA_TO_NV12: {
      return RGBxorBGRxToYUV420<true, false, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    }

    case KLEIDICV_RGBA_TO_NV12: {
      return RGBxorBGRxToYUV420<true, true, true>::rgb2yuv420_operation_sc(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    }

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace kleidicv::sme
