// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "yuv420_to_rgb_neon.h"

namespace kleidicv::neon {
template <bool BGR, bool kAlpha>
class YUVpToRGBxOrBGRx final : public YUV420XToRGBxOrBGRx<BGR, kAlpha>,
                               public UnrollOnce,
                               public TryToAvoidTailLoop {
 public:
  using VecTraits = neon::VecTraits<uint8_t>;
  using ScalarType = VecTraits::ScalarType;
  using VectorType = VecTraits::VectorType;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::de_interleave_indices_;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::yuv420x_to_rgb;
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::v_first_;

  explicit YUVpToRGBxOrBGRx(bool is_yv12)
      : YUV420XToRGBxOrBGRx<BGR, kAlpha>(is_yv12) {}

  KLEIDICV_FORCE_INLINE
  void vector_path(VectorType &y0, VectorType &y1, VectorType &y2,
                   VectorType &y3, VectorType &u, VectorType &v,
                   ScalarType *rgbx_row_0, ScalarType *rgbx_row_1) {
    // Indices to extract every 4 bytes into 4x 32-bit slots (0xff = ignore)
    // These are needed to expand each group of 4 bytes into a full 32-bit lane
    uint8x16_t index_lo_lo = {0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff,
                              2, 0xff, 0xff, 0xff, 3, 0xff, 0xff, 0xff};

    uint8x16_t index_lo_hi = {4, 0xff, 0xff, 0xff, 5, 0xff, 0xff, 0xff,
                              6, 0xff, 0xff, 0xff, 7, 0xff, 0xff, 0xff};

    uint8x16_t index_hi_lo = {8,  0xff, 0xff, 0xff, 9,  0xff, 0xff, 0xff,
                              10, 0xff, 0xff, 0xff, 11, 0xff, 0xff, 0xff};

    uint8x16_t index_hi_hi = {12, 0xff, 0xff, 0xff, 13, 0xff, 0xff, 0xff,
                              14, 0xff, 0xff, 0xff, 15, 0xff, 0xff, 0xff};

    // Expand each 8-bit channel into 32-bit vectors using table lookup and
    // reinterpret
    int32x4_t u_lo_lo = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_lo_lo));
    int32x4_t u_lo_hi = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_lo_hi));
    int32x4_t u_hi_lo = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_hi_lo));
    int32x4_t u_hi_hi = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_hi_hi));

    int32x4_t v_lo_lo = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_lo_lo));
    int32x4_t v_lo_hi = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_lo_hi));
    int32x4_t v_hi_lo = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_hi_lo));
    int32x4_t v_hi_hi = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_hi_hi));

    constexpr size_t step = kAlpha ? 4 * 16 : 3 * 16;

    yuv420x_to_rgb(y0, y1, u_lo_lo, u_lo_hi, v_lo_lo, v_lo_hi, rgbx_row_0,
                   rgbx_row_1);

    yuv420x_to_rgb(y2, y3, u_hi_lo, u_hi_hi, v_hi_lo, v_hi_hi,
                   rgbx_row_0 + step, rgbx_row_1 + step);
  }

  // Processes inputs which are not long enough to fit a vector.
  void scalar_path(size_t length, const ScalarType *y_row_0,
                   const ScalarType *y_row_1, const ScalarType *u_row,
                   const ScalarType *v_row, ScalarType *rgbx_row_0,
                   ScalarType *rgbx_row_1) {
    const uint8_t *y_rows[2] = {y_row_0, y_row_1};
    uint8_t *rgbx_rows[2] = {rgbx_row_0, rgbx_row_1};

    int32_t u_m128 = 0, v_m128 = 0;
    for (size_t index = 0; index < length; ++index) {
      disable_loop_vectorization();

      // There is one {U, V} pair for 4 Y values.
      if ((index % 2) == 0) {
        u_m128 = u_row[0] - 128;
        v_m128 = v_row[0] - 128;
        u_row += 1;
        v_row += 1;
        if (v_first_) {
          std::swap(u_m128, v_m128);
        }
      }

      yuv420x_to_rgb(y_rows, index, u_m128, v_m128, rgbx_rows);
    }
  }
};  // end of class YUVpToRGBxOrBGRx<bool, bool>

using YUVpToRGB = YUVpToRGBxOrBGRx<false, false>;
using YUVpToRGBA = YUVpToRGBxOrBGRx<false, true>;
using YUVpToBGR = YUVpToRGBxOrBGRx<true, false>;
using YUVpToBGRA = YUVpToRGBxOrBGRx<true, true>;

template <typename OperationType, typename ScalarType>
kleidicv_error_t yuv2rgbx_operation(OperationType &operation,
                                    const ScalarType *src, size_t src_stride,
                                    ScalarType *dst, size_t dst_stride,
                                    size_t width, size_t height, size_t begin,
                                    size_t end) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, (height * 3 + 1) / 2);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  // Pointer to the start of the U plane.
  // Since `src` points to a planar YUV buffer, the Y plane comes first,
  // occupying `src_stride * height` bytes.
  const ScalarType *u = src + src_stride * height;
  // Pointer to the start of the V plane.
  // The V plane follows the U plane. Both U and V planes are
  // subsampled at a 2:1 vertical ratio (i.e., each has height / 2 rows), and
  // are often stored in a single contiguous chroma region in memory. Depending
  // on image height and stride, the starting offset of V may require adjustment
  // to maintain correct alignment. In particular, when the image height is not
  // divisible evenly by 4, the chroma rows may not align perfectly, so a
  // fractional offset (in rows) is applied to calculate the V plane position.
  // The formula used here accounts for this by adjusting based on row parity,
  // assuming consistent memory layout across the Y, U, and V planes.
  const ScalarType *v =
      u + src_stride * (height / 4) + (width / 2) * ((height % 4) / 2);

  // These indices control how U and V row strides are selected across the image
  // height. In planar YUV 4:2:0 format, each chroma row (U/V) corresponds to
  // two luma (Y) rows. However, when the image height is not divisible by 4,
  // the mapping between chroma and luma rows becomes asymmetric. Specifically,
  // when `height % 4 == 2`, the start of the V plane is offset by one chroma
  // row relative to U.
  //
  // This results in U and V rows being interleaved with a phase difference,
  // which must be accounted for during row-wise traversal. To handle this,
  // `u_index` and `v_index` are used to alternate the stride selection
  // independently for U and V across the loop.
  //
  // This mechanism ensures that memory access patterns remain correct,
  // especially in layouts where U and V share a contiguous buffer with
  // alternating strides. Offsetting `v_index` allows the traversal logic to
  // maintain correct alignment and prevents misaligned or incorrect reads from
  // the chroma buffer.
  size_t u_index = 0;
  size_t v_index = height % 4 == 2 ? 1 : 0;

  // Compute the actual row range in the Y plane (full resolution).
  // Since each UV row maps to 2 Y rows, we double the begin/end indices.
  size_t row_begin = begin * 2;
  size_t row_end = std::min<size_t>(height, end * 2);
  size_t row_uv = begin;

  // UV stepping pattern: first half of row, then padded second half.
  // Needed to match row strides between chroma and luma components.
  size_t uv_strides[2] = {width / 2, src_stride - width / 2};

  // Calculate starting pointers for Y, U, and V planes at the given stripe
  // start.
  const ScalarType *y0 = src + row_begin * src_stride;
  u = u + (row_uv / 2) * src_stride;
  v = v + (row_uv / 2) * src_stride;

  if (row_uv % 2 == 1) {
    u += uv_strides[(u_index++) & 1];
    v += uv_strides[(v_index++) & 1];
  }

  size_t dcn = operation.output_channels();
  for (size_t h = row_begin; h < row_end; h += 2) {
    ScalarType *row0 = dst + dst_stride * h;
    ScalarType *row1 = dst + dst_stride * (h + 1);
    const ScalarType *y1 = y0 + src_stride;

    // Guard for odd-height images.
    // If the last row in the stripe is unpaired (odd number of rows),
    // reuse the previous row pointers to avoid out-of-bounds access.
    if (KLEIDICV_UNLIKELY(h == (row_end - 1))) {
      row1 = row0;
      y1 = y0;
    }

    LoopUnroll2 loop{width, kVectorLength};

    loop.unroll_twice([&](size_t index) {
      uint8x16_t u_vec = vld1q_u8(u + index / 2);
      uint8x16_t v_vec = vld1q_u8(v + index / 2);
      uint8x16_t y0_vec = vld1q_u8(y0 + index);
      uint8x16_t y1_vec = vld1q_u8(y1 + index);
      uint8x16_t y2_vec = vld1q_u8(y0 + index + kVectorLength);
      uint8x16_t y3_vec = vld1q_u8(y1 + index + kVectorLength);

      operation.vector_path(y0_vec, y1_vec, y2_vec, y3_vec, u_vec, v_vec,
                            &row0[index * dcn], &row1[index * dcn]);
    });

    loop.remaining([&](size_t index, size_t length) {
      operation.scalar_path(length - index, y0 + index, y1 + index,
                            u + index / 2, v + index / 2, &row0[index * dcn],
                            &row1[index * dcn]);
    });

    y0 += src_stride * 2;
    u += uv_strides[(u_index++) & 1];
    v += uv_strides[(v_index++) & 1];
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end) {
  switch (color_format) {
    case KLEIDICV_YV12_TO_BGR: {
      YUVpToBGR operation{true};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_YV12_TO_RGB: {
      YUVpToRGB operation{true};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_YV12_TO_BGRA: {
      YUVpToBGRA operation{true};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_YV12_TO_RGBA: {
      YUVpToRGBA operation{true};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_IYUV_TO_BGR: {
      YUVpToBGR operation{false};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_IYUV_TO_RGB: {
      YUVpToRGB operation{false};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_IYUV_TO_BGRA: {
      YUVpToBGRA operation{false};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    case KLEIDICV_IYUV_TO_RGBA: {
      YUVpToRGBA operation{false};
      return yuv2rgbx_operation(operation, src, src_stride, dst, dst_stride,
                                width, height, begin, end);
    }

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace kleidicv::neon
