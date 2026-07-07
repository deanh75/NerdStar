// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <utility>

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

static const int kWeightScale = 14;

static const int KR2Y422Weight = 4211;  // 0.299077 * (236 - 16) / 256 * 16384
static const int KG2Y422Weight = 8258;  // 0.586506 * (236 - 16) / 256 * 16384
static const int KB2Y422Weight = 1606;  // 0.114062 * (236 - 16) / 256 * 16384

static const int KR2U422Weight = -1212;  // -0.148 * 8192
static const int KG2U422Weight = -2384;  // -0.291 * 8192
static const int KB2U422Weight = 3596;   //  0.439 * 8192
static const int KG2V422Weight = -3015;  // -0.368 * 8192
static const int KB2V422Weight = -582;   // -0.071 * 8192

template <size_t b_idx, size_t u_idx, size_t y_idx, size_t scn>
class RGBxOrBGRxToYUV422 {
 public:
  static constexpr size_t r_idx = 2 - b_idx;
  static constexpr size_t v_idx = (u_idx + 2) % 4;

  static kleidicv_error_t rgbx2yuv422_operation(const uint8_t* src,
                                                size_t src_stride, uint8_t* dst,
                                                size_t dst_stride, size_t width,
                                                size_t height) {
    // Destination channel count (dcn = 2) because YUV422 is interleaved with
    // two channels per pixel on average: one luma (Y) and one shared
    // chroma (U or V). Thus, dcn is set to 2 for this color format.
    constexpr size_t dcn = 2;

    // Loop through rows along the image height.
    for (size_t h = 0; h < height; h++, src += src_stride) {
      // Keep track of the current output row being written.
      Columns<uint8_t> dst_row{dst + dst_stride * h, dcn};
      LoopUnroll2 loop{width, kVectorLength};

      // Unroll by 2: convert two RGB pixels to one YVYU pair [Y0, V0, Y1, U0].
      // Compute Y0 and Y1 per pixel; compute V0/U0 once for the pair (shared
      // chroma); then pack as [Y0, V0, Y1, U0] each iteration for higher
      // throughput.
      loop.unroll_twice([&](size_t index) {
        uint8x16_t r0, g0, b0, r1, g1, b1;
        if constexpr (scn == 4) {
          uint8x16x4_t rgbx_0 = vld4q_u8(src + index * scn);
          uint8x16x4_t rgbx_1 =
              vld4q_u8(src + index * scn + kVectorLength * scn);

          r0 = rgbx_0.val[r_idx], g0 = rgbx_0.val[1], b0 = rgbx_0.val[b_idx];
          r1 = rgbx_1.val[r_idx], g1 = rgbx_1.val[1], b1 = rgbx_1.val[b_idx];
        } else {
          uint8x16x3_t rgbx_0 = vld3q_u8(src + index * scn);
          uint8x16x3_t rgbx_1 =
              vld3q_u8(src + index * scn + kVectorLength * scn);

          r0 = rgbx_0.val[r_idx], g0 = rgbx_0.val[1], b0 = rgbx_0.val[b_idx];
          r1 = rgbx_1.val[r_idx], g1 = rgbx_1.val[1], b1 = rgbx_1.val[b_idx];
        }

        rgb_to_yuv422(r0, g0, b0, r1, g1, b1,
                      dst_row.ptr_at(static_cast<ptrdiff_t>(index)));
      });

      loop.remaining([&](size_t index, size_t length) {
        for (; index < length; index += 2) {
          const uint8_t r1 = src[index * scn + r_idx],
                        g1 = src[index * scn + 1],
                        b1 = src[index * scn + b_idx];

          const uint8_t r2 = src[index * scn + scn + r_idx],
                        g2 = src[index * scn + scn + 1],
                        b2 = src[index * scn + scn + b_idx];

          rgb_to_yuv422(r1, g1, b1, r2, g2, b2,
                        dst_row.ptr_at(static_cast<ptrdiff_t>(index)));
        }
      });
    }
    return KLEIDICV_OK;
  }

 private:
  static uint8_t compute_y(const uint8_t r, const uint8_t g, const uint8_t b) {
    int y_ = r * KR2Y422Weight + g * KG2Y422Weight + b * KB2Y422Weight +
             (1 << kWeightScale) * 16;
    return saturating_cast<int, uint8_t>(((1 << (kWeightScale - 1)) + y_) >>
                                         kWeightScale);
  }

  static std::pair<uint8_t, uint8_t> compute_uv(
      const uint8_t r1, const uint8_t g1, const uint8_t b1, const uint8_t r2,
      const uint8_t g2, const uint8_t b2) {
    int sr = r1 + r2, sg = g1 + g2, sb = b1 + b2;

    int u_ = sr * KR2U422Weight + sg * KG2U422Weight + sb * KB2U422Weight +
             (1 << (kWeightScale - 1)) * 256;
    uint8_t u = saturating_cast<int, uint8_t>(
        ((1 << (kWeightScale - 1)) + u_) >> kWeightScale);

    int v_ = sr * KB2U422Weight + sg * KG2V422Weight + sb * KB2V422Weight +
             (1 << (kWeightScale - 1)) * 256;
    uint8_t v = saturating_cast<int, uint8_t>(
        ((1 << (kWeightScale - 1)) + v_) >> kWeightScale);
    return {u, v};
  }

  KLEIDICV_FORCE_INLINE
  static void rgb_to_yuv422(const uint8_t r0, const uint8_t g0,
                            const uint8_t b0, const uint8_t r1,
                            const uint8_t g1, const uint8_t b1,
                            uint8_t* dst_ptr) {
    auto y0 = compute_y(r0, g0, b0);
    auto y1 = compute_y(r1, g1, b1);
    auto [u, v] = compute_uv(r0, g0, b0, r1, g1, b1);

    dst_ptr[y_idx] = y0;
    dst_ptr[y_idx + 2] = y1;
    dst_ptr[u_idx] = u;
    dst_ptr[v_idx] = v;
  }

  static inline void rgb_to_yuv422(const uint8x16_t r0, const uint8x16_t g0,
                                   const uint8x16_t b0, const uint8x16_t r1,
                                   const uint8x16_t g1, const uint8x16_t b1,
                                   uint8_t* dst_ptr) {
    int y_base = (1 << (kWeightScale - 1)) + (1 << kWeightScale) * 16;
    int uv_bias = (1 << (kWeightScale - 1)) + (1 << (kWeightScale - 1)) * 256;

    int16x8_t r_even[2];
    int16x8_t g_even[2];
    int16x8_t b_even[2];

    const uint8x16_t index_even = {0, 0xff, 2,  0xff, 4,  0xff, 6,  0xff,
                                   8, 0xff, 10, 0xff, 12, 0xff, 14, 0xff};

    const uint8x16_t index_odd = {1, 0xff, 3,  0xff, 5,  0xff, 7,  0xff,
                                  9, 0xff, 11, 0xff, 13, 0xff, 15, 0xff};

    r_even[0] = vreinterpretq_s16_u8(vqtbl1q_u8(r0, index_even));
    r_even[1] = vreinterpretq_s16_u8(vqtbl1q_u8(r1, index_even));

    g_even[0] = vreinterpretq_s16_u8(vqtbl1q_u8(g0, index_even));
    g_even[1] = vreinterpretq_s16_u8(vqtbl1q_u8(g1, index_even));

    b_even[0] = vreinterpretq_s16_u8(vqtbl1q_u8(b0, index_even));
    b_even[1] = vreinterpretq_s16_u8(vqtbl1q_u8(b1, index_even));

    uint8x16_t y1 =
        compute_weighted_channel_422(r_even, g_even, b_even, KR2Y422Weight,
                                     KG2Y422Weight, KB2Y422Weight, y_base);

    int16x8_t r_odd[2];
    int16x8_t g_odd[2];
    int16x8_t b_odd[2];

    r_odd[0] = vreinterpretq_s16_u8(vqtbl1q_u8(r0, index_odd));
    r_odd[1] = vreinterpretq_s16_u8(vqtbl1q_u8(r1, index_odd));

    g_odd[0] = vreinterpretq_s16_u8(vqtbl1q_u8(g0, index_odd));
    g_odd[1] = vreinterpretq_s16_u8(vqtbl1q_u8(g1, index_odd));

    b_odd[0] = vreinterpretq_s16_u8(vqtbl1q_u8(b0, index_odd));
    b_odd[1] = vreinterpretq_s16_u8(vqtbl1q_u8(b1, index_odd));

    uint8x16_t y2 =
        compute_weighted_channel_422(r_odd, g_odd, b_odd, KR2Y422Weight,
                                     KG2Y422Weight, KB2Y422Weight, y_base);

    int16x8_t r_avg[2];
    int16x8_t g_avg[2];
    int16x8_t b_avg[2];

    r_avg[0] = vaddq_s16(r_even[0], r_odd[0]);
    r_avg[1] = vaddq_s16(r_even[1], r_odd[1]);
    g_avg[0] = vaddq_s16(g_even[0], g_odd[0]);
    g_avg[1] = vaddq_s16(g_even[1], g_odd[1]);
    b_avg[0] = vaddq_s16(b_even[0], b_odd[0]);
    b_avg[1] = vaddq_s16(b_even[1], b_odd[1]);

    uint8x16_t u =
        compute_weighted_channel_422(r_avg, g_avg, b_avg, KR2U422Weight,
                                     KG2U422Weight, KB2U422Weight, uv_bias);
    uint8x16_t v =
        compute_weighted_channel_422(r_avg, g_avg, b_avg, KB2U422Weight,
                                     KG2V422Weight, KB2V422Weight, uv_bias);

    uint8x16x4_t yuv422;

    yuv422.val[u_idx] = u;
    yuv422.val[v_idx] = v;
    yuv422.val[y_idx] = y1, yuv422.val[y_idx + 2] = y2;

    vst4q_u8(dst_ptr, yuv422);
  }

  static inline uint8x16_t compute_weighted_channel_422(
      int16x8_t r[2], int16x8_t g[2], int16x8_t b[2], int16_t r_coeff,
      int16_t g_coeff, int16_t b_coeff, int fixed) {
    int32x4_t bias = vdupq_n_s32(fixed);
    int32x4_t acc_lo[2] = {bias, bias};
    int32x4_t acc_hi[2] = {bias, bias};

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 2; i++) {
      // R contributions
      acc_lo[i] = vmlal_n_s16(bias, vget_low(r[i]), r_coeff);
      // G contributions
      acc_lo[i] = vmlal_n_s16(acc_lo[i], vget_low(g[i]), g_coeff);
      // B contributions
      int32x4_t tmp = vmull_n_s16(vget_low(b[i]), b_coeff);
      acc_lo[i] = vaddq(acc_lo[i], tmp);
    }

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 2; i++) {
      // R contributions
      acc_hi[i] = vmlal_high_n_s16(bias, r[i], r_coeff);
      // G contributions
      acc_hi[i] = vmlal_high_n_s16(acc_hi[i], g[i], g_coeff);
      // B contributions
      int32x4_t tmp = vmull_high_n_s16(b[i], b_coeff);
      acc_hi[i] = vaddq(acc_hi[i], tmp);
    }

    return normalize_and_pack(acc_lo[0], acc_hi[0], acc_lo[1], acc_hi[1]);
  }

  KLEIDICV_FORCE_INLINE
  static uint8x16_t normalize_and_pack(int32x4_t vec_0, int32x4_t vec_1,
                                       int32x4_t vec_2, int32x4_t vec_3) {
    uint8x16_t index = {0,  2,  4,  6,  8,  10, 12, 14,
                        16, 18, 20, 22, 24, 26, 28, 30};
    int16x4_t tmp_lo_lo = vqshrun_n_s32(vec_0, kWeightScale);
    int16x8_t tmp_lo_hi = vqshrun_high_n_s32(tmp_lo_lo, vec_1, kWeightScale);
    int16x4_t tmp_hi_lo = vqshrun_n_s32(vec_2, kWeightScale);
    int16x8_t tmp_hi_hi = vqshrun_high_n_s32(tmp_hi_lo, vec_3, kWeightScale);

    uint8x16x2_t tmp;
    tmp.val[0] = vreinterpretq_u8(tmp_lo_hi);  // 0, 1, 2, 3, 4, 5, 6, 7
    tmp.val[1] = vreinterpretq_u8(tmp_hi_hi);  // 8, 6, 10, 11, 12, 13, 14,

    uint8x16_t output = vqtbl2q_u8(tmp, index);

    return output;
  }
};

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_yuv422_u8(const uint8_t* src, size_t src_stride,
                                  uint8_t* dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format) {
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

}  // namespace kleidicv::neon
