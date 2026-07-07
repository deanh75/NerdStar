// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_YUV420P_TO_RGB_SC_H
#define KLEIDICV_YUV420P_TO_RGB_SC_H

#include <algorithm>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "yuv420_to_rgb_sc.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <bool BGR, bool kAlpha>
class YUVpToRGBxOrBGRx final : public YUV420XToRGBxOrBGRx<BGR, kAlpha> {
 public:
  using YUV420XToRGBxOrBGRx<BGR, kAlpha>::yuv420x_to_rgb;

  explicit YUVpToRGBxOrBGRx(bool v_first) KLEIDICV_STREAMING
      : YUV420XToRGBxOrBGRx<BGR, kAlpha>(v_first) {}

  // Returns the number of channels in the output image.
  static constexpr size_t output_channels() KLEIDICV_STREAMING {
    return kAlpha ? /* RGBA */ 4 : /* RGB */ 3;
  }

  // Processes 2 * 16 bytes (even and odd rows) of the input YUV data, and
  // outputs 2 * 3 (or 4) * 16 bytes of RGB (or RGBA) data per loop iteration.
  KLEIDICV_FORCE_INLINE
  void vector_path(svbool_t &pg, svuint8_t &y0, svuint8_t &y1, svint16_t &u,
                   svint16_t &v, uint8_t *rgbx_row_0,
                   uint8_t *rgbx_row_1) const KLEIDICV_STREAMING {
    yuv420x_to_rgb(pg, y0, y1, u, v, rgbx_row_0, rgbx_row_1);
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
                                    size_t end) KLEIDICV_STREAMING {
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
  const size_t kVectorLength = svcntb();
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

    LoopUnroll2 loop{width, svcntb()};

    struct VectorPath2x {
      const ScalarType *y0, *y1, *u, *v;
      ScalarType *row0, *row1;
      const size_t kVectorLength, dcn;
      OperationType operation;
      KLEIDICV_FORCE_INLINE
      void operator()(size_t index) const KLEIDICV_STREAMING {
        svbool_t pg = svptrue_b8();
        svuint8_t u8_vec = svld1(pg, u + index / 2);
        svint16_t u_vec_lo = svreinterpret_s16_u16(svunpklo_u16(u8_vec));
        svint16_t u_vec_hi = svreinterpret_s16_u16(svunpkhi_u16(u8_vec));

        svuint8_t v8_vec = svld1(pg, v + index / 2);
        svint16_t v_vec_lo = svreinterpret_s16_u16(svunpklo_u16(v8_vec));
        svint16_t v_vec_hi = svreinterpret_s16_u16(svunpkhi_u16(v8_vec));

#if KLEIDICV_TARGET_SME2
        // assume the predicate is full true
        svcount_t pg_counter = svptrue_c8();
        svuint8x2_t y_even = svld1_x2(pg_counter, y0 + index);
        svuint8x2_t y_odd = svld1_x2(pg_counter, y1 + index);
        svuint8_t y0_vec = svget2(y_even, 0);
        svuint8_t y1_vec = svget2(y_odd, 0);
        svuint8_t y2_vec = svget2(y_even, 1);
        svuint8_t y3_vec = svget2(y_odd, 1);
#else
        svuint8_t y0_vec = svld1(pg, y0 + index);
        svuint8_t y1_vec = svld1(pg, y1 + index);
        svuint8_t y2_vec = svld1(pg, y0 + index + kVectorLength);
        svuint8_t y3_vec = svld1(pg, y1 + index + kVectorLength);
#endif  // KLEIDICV_TARGET_SME2

        operation.vector_path(pg, y0_vec, y1_vec, u_vec_lo, v_vec_lo,
                              &row0[index * dcn], &row1[index * dcn]);

        operation.vector_path(pg, y2_vec, y3_vec, u_vec_hi, v_vec_hi,
                              &row0[(index + kVectorLength) * dcn],
                              &row1[(index + kVectorLength) * dcn]);
      }
    };
    loop.unroll_twice(
        VectorPath2x{y0, y1, u, v, row0, row1, kVectorLength, dcn, operation});

    struct VectorPath1x {
      const ScalarType *y0, *y1, *u, *v;
      ScalarType *row0, *row1;
      const size_t dcn;
      OperationType operation;
      KLEIDICV_FORCE_INLINE
      void operator()(size_t index) const KLEIDICV_STREAMING {
        svbool_t pg = svptrue_b8();
        svbool_t pg_half = svwhilelt_b8(0UL, svcntb() / 2);

        svuint8_t u8_vec = svld1(pg_half, u + index / 2);
        svint16_t u_vec_lo = svreinterpret_s16_u16(svunpklo_u16(u8_vec));

        svuint8_t v8_vec = svld1(pg_half, v + index / 2);
        svint16_t v_vec_lo = svreinterpret_s16_u16(svunpklo_u16(v8_vec));

        svuint8_t y0_vec = svld1(pg, y0 + index);
        svuint8_t y1_vec = svld1(pg, y1 + index);

        operation.vector_path(pg, y0_vec, y1_vec, u_vec_lo, v_vec_lo,
                              &row0[index * dcn], &row1[index * dcn]);
      }
    };
    loop.unroll_once(VectorPath1x{y0, y1, u, v, row0, row1, dcn, operation});

    struct RemainingPath {
      const ScalarType *y0, *y1, *u, *v;
      ScalarType *row0, *row1;
      const size_t dcn;
      OperationType operation;
      KLEIDICV_FORCE_INLINE
      void operator()(size_t index, size_t length) const KLEIDICV_STREAMING {
        svbool_t pg = svwhilelt_b8_u64(index, length);
        svbool_t pg_half = svwhilelt_b8_u64((index + 1) / 2, (length + 1) >> 1);

        svuint8_t u8_vec = svld1(pg_half, u + index / 2);
        svint16_t u_vec_lo = svreinterpret_s16_u16(svunpklo_u16(u8_vec));

        svuint8_t v8_vec = svld1(pg_half, v + index / 2);
        svint16_t v_vec_lo = svreinterpret_s16_u16(svunpklo_u16(v8_vec));

        svuint8_t y0_vec = svld1(pg, y0 + index);
        svuint8_t y1_vec = svld1(pg, y1 + index);

        operation.vector_path(pg, y0_vec, y1_vec, u_vec_lo, v_vec_lo,
                              &row0[index * dcn], &row1[index * dcn]);
      }
    };
    loop.remaining(RemainingPath{y0, y1, u, v, row0, row1, dcn, operation});

    y0 += src_stride * 2;
    u += uv_strides[(u_index++) & 1];
    v += uv_strides[(v_index++) & 1];
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv420p_to_rgb_stripe_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end) KLEIDICV_STREAMING {
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

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_YUV420P_TO_RGB_SC_H
