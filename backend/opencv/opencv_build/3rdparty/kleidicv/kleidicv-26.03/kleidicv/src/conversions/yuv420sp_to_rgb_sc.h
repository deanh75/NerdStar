// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_YUV420SP_TO_RGB_SC_H
#define KLEIDICV_YUV420SP_TO_RGB_SC_H

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "yuv420_to_rgb_sc.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <bool BGR, bool kAlpha>
class YUVSpToRGBxOrBGRx final : public YUV420XToRGBxOrBGRx<BGR, kAlpha> {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<uint8_t>;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::yuv420x_to_rgb;

  explicit YUVSpToRGBxOrBGRx(bool v_first) KLEIDICV_STREAMING
      : YUV420XToRGBxOrBGRx<BGR, kAlpha>(v_first) {}

  // Returns the number of channels in the output image.
  static constexpr size_t output_channels() KLEIDICV_STREAMING {
    return kAlpha ? /* RGBA */ 4 : /* RGB */ 3;
  }

  // Processes 2 * 16 bytes (even and odd rows) of the input YUV data, and
  // outputs 2 * 3 (or 4) * 16 bytes of RGB (or RGBA) data per loop iteration.
  KLEIDICV_FORCE_INLINE
  void vector_path(ContextType ctx, const uint8_t *y_row_0,
                   const uint8_t *y_row_1, const uint8_t *uv_row,
                   uint8_t *rgbx_row_0,
                   uint8_t *rgbx_row_1) KLEIDICV_STREAMING {
    auto pg = ctx.predicate();
    // Load channels: y0 and y1 are two adjacent rows.
    svuint8_t y0 = svld1(pg, y_row_0);
    svuint8_t y1 = svld1(pg, y_row_1);
    svuint8_t uv = svld1(pg, uv_row);
    // Widen U and V to 32 bits.
    svint16_t u = svreinterpret_s16(svmovlb(uv));
    svint16_t v = svreinterpret_s16(svmovlt(uv));
    yuv420x_to_rgb(pg, y0, y1, u, v, rgbx_row_0, rgbx_row_1);
  }
};  // end of class YUVSpToRGBxOrBGRx<bool, bool>

using YUVSpToRGB = YUVSpToRGBxOrBGRx<false, false>;
using YUVSpToRGBA = YUVSpToRGBxOrBGRx<false, true>;
using YUVSpToBGR = YUVSpToRGBxOrBGRx<true, false>;
using YUVSpToBGRA = YUVSpToRGBxOrBGRx<true, true>;

template <typename OperationType, typename ScalarType>
KLEIDICV_FORCE_INLINE kleidicv_error_t yuv2rgbx_operation(
    OperationType &operation, const ScalarType *src_y, size_t src_y_stride,
    const ScalarType *src_uv, size_t src_uv_stride, ScalarType *dst,
    size_t dst_stride, size_t width, size_t height) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src_y, src_y_stride, height);
  CHECK_POINTER_AND_STRIDE(src_uv, src_uv_stride, (height + 1) / 2);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  ParallelRows y_rows{src_y, src_y_stride};
  Rows uv_rows{src_uv, src_uv_stride};
  ParallelRows rgbx_rows{dst, dst_stride, operation.output_channels()};

  ForwardingOperation forwarding_operation{operation};
  OperationAdapter operation_adapter{forwarding_operation};
  RemainingPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  ParallelRowsAdapter parallel_rows_adapter{context_adapter};
  RowBasedOperation row_based_operation{parallel_rows_adapter};
  zip_parallel_rows(row_based_operation, rect, y_rows, uv_rows, rgbx_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv420sp_to_rgb_u8_sc(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height,
    kleidicv_color_conversion_t color_format) KLEIDICV_STREAMING {
  switch (color_format) {
    case KLEIDICV_NV21_TO_BGR: {
      YUVSpToBGR operation{true};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV21_TO_RGB: {
      YUVSpToRGB operation{true};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV21_TO_BGRA: {
      YUVSpToBGRA operation{true};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV21_TO_RGBA: {
      YUVSpToRGBA operation{true};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV12_TO_BGR: {
      YUVSpToBGR operation{false};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV12_TO_RGB: {
      YUVSpToRGB operation{false};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV12_TO_BGRA: {
      YUVSpToBGRA operation{false};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    case KLEIDICV_NV12_TO_RGBA: {
      YUVSpToRGBA operation{false};
      return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                                src_uv_stride, dst, dst_stride, width, height);
    }

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_YUV420SP_TO_RGB_SC_H
