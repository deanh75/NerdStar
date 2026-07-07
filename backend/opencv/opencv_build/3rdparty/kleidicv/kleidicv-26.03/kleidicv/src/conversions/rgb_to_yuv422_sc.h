// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RGB_TO_YUV422_SC_H
#define KLEIDICV_RGB_TO_YUV422_SC_H

#include <utility>

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

static const int kWeightScale = 14;

static const int16_t KR2Y422Weight =
    4211;  // 0.299077 * (236 - 16) / 256 * 16384
static const int16_t KG2Y422Weight =
    8258;  // 0.586506 * (236 - 16) / 256 * 16384
static const int16_t KB2Y422Weight =
    1606;  // 0.114062 * (236 - 16) / 256 * 16384

static const int16_t KR2U422Weight = -1212;  // -0.148 * 8192
static const int16_t KG2U422Weight = -2384;  // -0.291 * 8192
static const int16_t KB2U422Weight = 3596;   //  0.439 * 8192
static const int16_t KG2V422Weight = -3015;  // -0.368 * 8192
static const int16_t KB2V422Weight = -582;   // -0.071 * 8192

template <size_t b_idx, size_t u_idx, size_t y_idx, size_t scn>
class RGBxOrBGRxToYUV422 {
 public:
  static constexpr size_t r_idx = 2 - b_idx;
  static constexpr size_t v_idx = (u_idx + 2) % 4;
  using ArrayOfTwo_svuint8 = ScalableVectorArray1D<svuint8_t, 2>;
  using ArrayOfTwo_svint16 = ScalableVectorArray1D<svint16_t, 2>;

  static kleidicv_error_t rgbx2yuv422_operation(
      const uint8_t* src, size_t src_stride, uint8_t* dst, size_t dst_stride,
      size_t width, size_t height) KLEIDICV_STREAMING {
    // Destination channel count (dcn = 2) because YUV422 is interleaved with
    // two channels per pixel on average: one luma (Y) and one shared
    // chroma (U or V). Thus, dcn is set to 2 for this color format.
    constexpr size_t dcn = 2;
    auto kVectorLength = svcntb();

    // Loop through rows along the image height.
    for (size_t h = 0; h < height; h++, src += src_stride) {
      // Keep track of the current output row being written.
      Columns<uint8_t> dst_row{dst + dst_stride * h, dcn};
      LoopUnroll2 loop{width, kVectorLength};

      // Unroll by 2: convert two RGB pixels to one YVYU pair [Y0, V0, Y1, U0].
      // Compute Y0 and Y1 per pixel; compute V0/U0 once for the pair (shared
      // chroma); then pack as [Y0, V0, Y1, U0] each iteration for higher
      // throughput.
      loop.unroll_twice([&](size_t index) KLEIDICV_STREAMING {
        svbool_t pg = svptrue_b8();
        svuint8_t r0, g0, b0, r1, g1, b1;
        ArrayOfTwo_svuint8 r = {{std::ref(r0), std::ref(r1)}};
        ArrayOfTwo_svuint8 g = {{std::ref(g0), std::ref(g1)}};
        ArrayOfTwo_svuint8 b = {{std::ref(b0), std::ref(b1)}};
        if constexpr (scn == 4) {
          svuint8x4_t rgbx_0 = svld4(pg, src + index * scn);
          svuint8x4_t rgbx_1 =
              svld4(pg, src + index * scn + kVectorLength * scn);

          r(0) = svget4(rgbx_0, r_idx), g(0) = svget4(rgbx_0, 1),
          b(0) = svget4(rgbx_0, b_idx);
          r(1) = svget4(rgbx_1, r_idx), g(1) = svget4(rgbx_1, 1),
          b(1) = svget4(rgbx_1, b_idx);
        } else {
          svuint8x3_t rgbx_0 = svld3(pg, src + index * scn);
          svuint8x3_t rgbx_1 =
              svld3(pg, src + index * scn + kVectorLength * scn);

          r(0) = svget3(rgbx_0, r_idx), g(0) = svget3(rgbx_0, 1),
          b(0) = svget3(rgbx_0, b_idx);
          r(1) = svget3(rgbx_1, r_idx), g(1) = svget3(rgbx_1, 1),
          b(1) = svget3(rgbx_1, b_idx);
        }

        rgb_to_yuv422(r, g, b, dst_row.ptr_at(static_cast<ptrdiff_t>(index)),
                      pg);
      });

      loop.remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
        svbool_t pg1 = svwhilelt_b8_u64(index, length);
        svbool_t pg2 = svwhilelt_b8_u64(index + kVectorLength, length);
        svbool_t pg3 = svwhilelt_b8_u64(0, (length - index) / 2);
        svuint8_t r0, g0, b0, r1, g1, b1;
        ArrayOfTwo_svuint8 r = {{std::ref(r0), std::ref(r1)}};
        ArrayOfTwo_svuint8 g = {{std::ref(g0), std::ref(g1)}};
        ArrayOfTwo_svuint8 b = {{std::ref(b0), std::ref(b1)}};
        if constexpr (scn == 4) {
          svuint8x4_t rgbx_0 = svld4(pg1, src + index * scn);
          svuint8x4_t rgbx_1 =
              svld4(pg2, src + index * scn + kVectorLength * scn);

          r(0) = svget4(rgbx_0, r_idx), g(0) = svget4(rgbx_0, 1),
          b(0) = svget4(rgbx_0, b_idx);
          r(1) = svget4(rgbx_1, r_idx), g(1) = svget4(rgbx_1, 1),
          b(1) = svget4(rgbx_1, b_idx);
        } else {
          svuint8x3_t rgbx_0 = svld3(pg1, src + index * scn);
          svuint8x3_t rgbx_1 =
              svld3(pg2, src + index * scn + kVectorLength * scn);

          r(0) = svget3(rgbx_0, r_idx), g(0) = svget3(rgbx_0, 1),
          b(0) = svget3(rgbx_0, b_idx);
          r(1) = svget3(rgbx_1, r_idx), g(1) = svget3(rgbx_1, 1),
          b(1) = svget3(rgbx_1, b_idx);
        }

        rgb_to_yuv422(r, g, b, dst_row.ptr_at(static_cast<ptrdiff_t>(index)),
                      pg3);
      });
    }
    return KLEIDICV_OK;
  }

 private:
  static inline void rgb_to_yuv422(ArrayOfTwo_svuint8 r, ArrayOfTwo_svuint8 g,
                                   ArrayOfTwo_svuint8 b, uint8_t* dst_ptr,
                                   svbool_t pg) KLEIDICV_STREAMING {
    int y_base = (1 << (kWeightScale - 1)) + (1 << kWeightScale) * 16;
    int uv_bias = (1 << (kWeightScale - 1)) + (1 << (kWeightScale - 1)) * 256;

    // get the even element to calculate y0
    svint16_t r0_even, g0_even, b0_even, r1_even, g1_even, b1_even;
    ArrayOfTwo_svint16 r_even = {{std::ref(r0_even), std::ref(r1_even)}};
    ArrayOfTwo_svint16 g_even = {{std::ref(g0_even), std::ref(g1_even)}};
    ArrayOfTwo_svint16 b_even = {{std::ref(b0_even), std::ref(b1_even)}};

    r_even(0) = svreinterpret_s16(svmovlb(r(0)));
    g_even(0) = svreinterpret_s16(svmovlb(g(0)));
    b_even(0) = svreinterpret_s16(svmovlb(b(0)));
    r_even(1) = svreinterpret_s16(svmovlb(r(1)));
    g_even(1) = svreinterpret_s16(svmovlb(g(1)));
    b_even(1) = svreinterpret_s16(svmovlb(b(1)));

    svuint8_t y1 =
        compute_weighted_channel_422(r_even, g_even, b_even, KR2Y422Weight,
                                     KG2Y422Weight, KB2Y422Weight, y_base);

    // get the odd element to calculate y0
    svint16_t r0_odd, g0_odd, b0_odd, r1_odd, g1_odd, b1_odd;
    ArrayOfTwo_svint16 r_odd = {{std::ref(r0_odd), std::ref(r1_odd)}};
    ArrayOfTwo_svint16 g_odd = {{std::ref(g0_odd), std::ref(g1_odd)}};
    ArrayOfTwo_svint16 b_odd = {{std::ref(b0_odd), std::ref(b1_odd)}};
    r_odd(0) = svreinterpret_s16(svmovlt(r(0)));
    g_odd(0) = svreinterpret_s16(svmovlt(g(0)));
    b_odd(0) = svreinterpret_s16(svmovlt(b(0)));
    r_odd(1) = svreinterpret_s16(svmovlt(r(1)));
    g_odd(1) = svreinterpret_s16(svmovlt(g(1)));
    b_odd(1) = svreinterpret_s16(svmovlt(b(1)));

    svuint8_t y2 =
        compute_weighted_channel_422(r_odd, g_odd, b_odd, KR2Y422Weight,
                                     KG2Y422Weight, KB2Y422Weight, y_base);

    svint16_t r0_avg, r1_avg, g0_avg, g1_avg, b0_avg, b1_avg;
    ArrayOfTwo_svint16 r_avg = {{std::ref(r0_avg), std::ref(r1_avg)}};
    ArrayOfTwo_svint16 g_avg = {{std::ref(g0_avg), std::ref(g1_avg)}};
    ArrayOfTwo_svint16 b_avg = {{std::ref(b0_avg), std::ref(b1_avg)}};
    r_avg(0) = svadd_x(svptrue_b16(), r_even(0), r_odd(0));
    r_avg(1) = svadd_x(svptrue_b16(), r_even(1), r_odd(1));
    g_avg(0) = svadd_x(svptrue_b16(), g_even(0), g_odd(0));
    g_avg(1) = svadd_x(svptrue_b16(), g_even(1), g_odd(1));
    b_avg(0) = svadd_x(svptrue_b16(), b_even(0), b_odd(0));
    b_avg(1) = svadd_x(svptrue_b16(), b_even(1), b_odd(1));

    svuint8_t u =
        compute_weighted_channel_422(r_avg, g_avg, b_avg, KR2U422Weight,
                                     KG2U422Weight, KB2U422Weight, uv_bias);
    svuint8_t v =
        compute_weighted_channel_422(r_avg, g_avg, b_avg, KB2U422Weight,
                                     KG2V422Weight, KB2V422Weight, uv_bias);

    svuint8x4_t yuv422 = svcreate4(svdup_n_u8(0xFF), svdup_n_u8(0xFF),
                                   svdup_n_u8(0xFF), svdup_n_u8(0xFF));
    yuv422 = svset4(yuv422, u_idx, u);
    yuv422 = svset4(yuv422, v_idx, v);
    yuv422 = svset4(yuv422, y_idx, y1);
    yuv422 = svset4(yuv422, y_idx + 2, y2);

    svst4_u8(pg, dst_ptr, yuv422);
  }

  static svuint8_t normalize_and_pack(svint32_t vec_0, svint32_t vec_1,
                                      svint32_t vec_2,
                                      svint32_t vec_3) KLEIDICV_STREAMING {
    svuint16_t y_b = svshrnb_n_u32(svreinterpret_u32(vec_0), kWeightScale);
    y_b = svshrnt_n_u32(y_b, svreinterpret_u32(vec_1), kWeightScale);
    svuint16_t y_t = svshrnb_n_u32(svreinterpret_u32(vec_2), kWeightScale);
    y_t = svshrnt_n_u32(y_t, svreinterpret_u32(vec_3), kWeightScale);

    return svuzp1_u8(svreinterpret_u8(y_b), svreinterpret_u8(y_t));
  }

  // Common helper: apply RGB weights into 4x s32 accumulators and pack to u8.
  static inline svuint8_t compute_weighted_channel_422(
      ArrayOfTwo_svint16 r, ArrayOfTwo_svint16 g, ArrayOfTwo_svint16 b,
      int16_t r_coeff, int16_t g_coeff, int16_t b_coeff,
      int fixed) KLEIDICV_STREAMING {
    svint32_t bias = svdup_s32(fixed);
    svint32_t acc_lo_lo = bias;
    svint32_t acc_lo_hi = bias;
    svint32_t acc_hi_lo = bias;
    svint32_t acc_hi_hi = bias;

    // R contributions
    acc_lo_lo = svmlalb_n_s32(acc_lo_lo, r(0), r_coeff);
    acc_lo_hi = svmlalt_n_s32(acc_lo_hi, r(0), r_coeff);
    acc_hi_lo = svmlalb_n_s32(acc_hi_lo, r(1), r_coeff);
    acc_hi_hi = svmlalt_n_s32(acc_hi_hi, r(1), r_coeff);

    // G contributions
    acc_lo_lo = svmlalb_n_s32(acc_lo_lo, g(0), g_coeff);
    acc_lo_hi = svmlalt_n_s32(acc_lo_hi, g(0), g_coeff);
    acc_hi_lo = svmlalb_n_s32(acc_hi_lo, g(1), g_coeff);
    acc_hi_hi = svmlalt_n_s32(acc_hi_hi, g(1), g_coeff);

    // B contributions
    acc_lo_lo = svmlalb_n_s32(acc_lo_lo, b(0), b_coeff);
    acc_lo_hi = svmlalt_n_s32(acc_lo_hi, b(0), b_coeff);
    acc_hi_lo = svmlalb_n_s32(acc_hi_lo, b(1), b_coeff);
    acc_hi_hi = svmlalt_n_s32(acc_hi_hi, b(1), b_coeff);

    return normalize_and_pack(acc_lo_lo, acc_lo_hi, acc_hi_lo, acc_hi_hi);
  }
};

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgb_to_yuv422_u8_sc(
    const uint8_t* src, size_t src_stride, uint8_t* dst, size_t dst_stride,
    size_t width, size_t height,
    kleidicv_color_conversion_t color_format) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  // YUV422 packs pixels in pairs: (Y0, U, Y1, V).
  // Therefore, the image width must be at least 2 and always even.
  if (width < 2 || (width % 2) != 0) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  switch (color_format) {
    case KLEIDICV_BGR_TO_YUYV:
      return RGBxOrBGRxToYUV422<0, 1, 0, 3>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_BGR_TO_UYVY:
      return RGBxOrBGRxToYUV422<0, 0, 1, 3>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_BGR_TO_YVYU:
      return RGBxOrBGRxToYUV422<0, 3, 0, 3>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_RGB_TO_YUYV:
      return RGBxOrBGRxToYUV422<2, 1, 0, 3>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_RGB_TO_UYVY:
      return RGBxOrBGRxToYUV422<2, 0, 1, 3>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_RGB_TO_YVYU:
      return RGBxOrBGRxToYUV422<2, 3, 0, 3>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_BGRA_TO_YUYV:
      return RGBxOrBGRxToYUV422<0, 1, 0, 4>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_BGRA_TO_UYVY:
      return RGBxOrBGRxToYUV422<0, 0, 1, 4>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_BGRA_TO_YVYU:
      return RGBxOrBGRxToYUV422<0, 3, 0, 4>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_RGBA_TO_YUYV:
      return RGBxOrBGRxToYUV422<2, 1, 0, 4>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_RGBA_TO_UYVY:
      return RGBxOrBGRxToYUV422<2, 0, 1, 4>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_RGBA_TO_YVYU:
      return RGBxOrBGRxToYUV422<2, 3, 0, 4>::rgbx2yuv422_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
      break;
  }

  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RGB_TO_YUV422_SC_H
