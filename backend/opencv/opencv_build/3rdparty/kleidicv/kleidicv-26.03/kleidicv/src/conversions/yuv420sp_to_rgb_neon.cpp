// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "yuv420_to_rgb_neon.h"

namespace kleidicv::neon {
template <bool BGR, bool kAlpha>
class YUVSpToRGBxOrBGRx final : public YUV420XToRGBxOrBGRx<BGR, kAlpha>,
                                public UnrollOnce,
                                public TryToAvoidTailLoop {
 public:
  using VecTraits = neon::VecTraits<uint8_t>;
  using ScalarType = VecTraits::ScalarType;
  using VectorType = VecTraits::VectorType;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::de_interleave_indices_;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::yuv420x_to_rgb;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::v_first_;

  explicit YUVSpToRGBxOrBGRx(bool v_first)
      : YUV420XToRGBxOrBGRx<BGR, kAlpha>(v_first) {}

  // Processes 2 * 16 bytes (even and odd rows) of the input YUV data, and
  // outputs 2 * 3 (or 4) * 16 bytes of RGB (or RGBA) data per loop iteration.
  KLEIDICV_FORCE_INLINE
  void vector_path(VectorType y0, VectorType y1, VectorType uv,
                   ScalarType *rgbx_row_0, ScalarType *rgbx_row_1) {
    // Widen U and V to 32 bits.
    int32x4_t u_l = vqtbl1q_s8(uv, de_interleave_indices_.val[0]);
    int32x4_t u_h = vqtbl1q_s8(uv, de_interleave_indices_.val[1]);

    int32x4_t v_l = vqtbl1q_s8(uv, de_interleave_indices_.val[2]);
    int32x4_t v_h = vqtbl1q_s8(uv, de_interleave_indices_.val[3]);

    yuv420x_to_rgb(y0, y1, u_l, u_h, v_l, v_h, rgbx_row_0, rgbx_row_1);
  }

  // Processes inputs which are not long enough to fit a vector.
  KLEIDICV_FORCE_INLINE
  void scalar_path(size_t length, const ScalarType *y_row_0,
                   const ScalarType *y_row_1, const ScalarType *uv_row,
                   ScalarType *rgbx_row_0, ScalarType *rgbx_row_1) {
    const uint8_t *y_rows[2] = {y_row_0, y_row_1};
    uint8_t *rgbx_rows[2] = {rgbx_row_0, rgbx_row_1};

    int32_t u_m128 = 0, v_m128 = 0;
    for (size_t index = 0; index < length; ++index) {
      disable_loop_vectorization();

      // There is one {U, V} pair for 4 Y values.
      if ((index % 2) == 0) {
        u_m128 = uv_row[0] - 128;
        v_m128 = uv_row[1] - 128;
        uv_row += 2;
        if (v_first_) {
          std::swap(u_m128, v_m128);
        }
      }

      yuv420x_to_rgb(y_rows, index, u_m128, v_m128, rgbx_rows);
    }
  }
};  // end of class YUVSpToRGBxOrBGRx<bool, bool>

using YUVSpToRGB = YUVSpToRGBxOrBGRx<false, false>;
using YUVSpToRGBA = YUVSpToRGBxOrBGRx<false, true>;
using YUVSpToBGR = YUVSpToRGBxOrBGRx<true, false>;
using YUVSpToBGRA = YUVSpToRGBxOrBGRx<true, true>;

template <typename OperationType, typename ScalarType>
kleidicv_error_t yuv2rgbx_operation(
    OperationType &operation, const ScalarType *src_y, size_t src_y_stride,
    const ScalarType *src_uv, size_t src_uv_stride, ScalarType *dst,
    size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src_y, src_y_stride, height);
  CHECK_POINTER_AND_STRIDE(src_uv, src_uv_stride, (height + 1) / 2);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  ParallelRows y_rows{src_y, src_y_stride};
  Rows uv_rows{src_uv, src_uv_stride};
  ParallelRows rgbx_rows{dst, dst_stride, operation.output_channels()};

  RemoveContextAdapter remove_context_adapter{operation};
  OperationAdapter operation_adapter{remove_context_adapter};
  RemainingPathToScalarPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  ParallelRowsAdapter parallel_rows_adapter{context_adapter};
  RowBasedOperation row_based_operation{parallel_rows_adapter};
  zip_parallel_rows(row_based_operation, rect, y_rows, uv_rows, rgbx_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv420sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                    const uint8_t *src_uv, size_t src_uv_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height,
                                    kleidicv_color_conversion_t color_format) {
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

}  // namespace kleidicv::neon
