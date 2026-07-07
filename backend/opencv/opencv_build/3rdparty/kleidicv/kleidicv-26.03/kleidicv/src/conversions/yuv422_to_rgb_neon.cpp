// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "yuv42x_coefficients.h"

namespace kleidicv::neon {
template <size_t b_idx, size_t u_chroma_idx, size_t y_idx, size_t dcn>
class YUV422ToRGBxOrBGRx {
 public:
  // Byte offsets for chroma samples inside a 4-byte YUV422 tuple (Y0 U Y1 V).
  static constexpr size_t u_idx = u_chroma_idx;
  static constexpr size_t v_idx = (u_idx + 2) % 4;
  // Source channel count (scn = 2) because YUV422 is interleaved with
  // two channels per pixel on average: one luma (Y) and one shared
  // chroma (U or V).
  static constexpr size_t scn = 2;

  static kleidicv_error_t yuv2rgbx_operation(const uint8_t* src,
                                             size_t src_stride, uint8_t* dst,
                                             size_t dst_stride, size_t width,
                                             size_t height) {
    Rows<uint8_t> dst_rows{dst, dst_stride, dcn};

    for (size_t y = 0; y < height; y++, src += src_stride) {
      LoopUnroll2 loop{width, kVectorLength};

      // Use loop.unroll_twice to process two pixels per iteration.
      // In YUV422, two pixels are interleaved as (Y0, U0, Y1, V0).
      // These four values produce two RGBx output pixels. By unrolling,
      // we handle both pixels together in a single iteration, improving
      // overall efficiency for that loop body.
      struct UnrollTwiceFunctor {
        const uint8_t* src_row;
        Rows<uint8_t>& dst_rows;

        KLEIDICV_FORCE_INLINE void operator()(size_t index) const {
          // Deinterleave the YUV422 data into separate channels.
          // vld4q_u8() loads 16 groups of 4 bytes: (Y0, U0, Y1, V0).
          // Because we unroll twice, we must process two pixels at once.
          // Each pixel contributes two components (Y + chroma), so 4 vectors
          // are required: Y0, Y1, U, and V. This is why we perform 4 loads
          // instead of 2 - they directly correspond to the unrolled iteration.
          uint8x16x4_t yuv422 = vld4q_u8(src_row + index * scn);
          uint8x16_t y_even_lanes = yuv422.val[y_idx];
          uint8x16_t y_odd_lanes = yuv422.val[y_idx + scn];
          uint8x16_t u = yuv422.val[u_idx];
          uint8x16_t v = yuv422.val[v_idx];
          // Convert two output vectors in one go (loop unrolled twice).
          // The second destination pointer is advanced by kVectorLength * dcn:
          //   - kVectorLength: number of pixels produced per vector
          //   - dcn: destination channels per pixel (3 for RGB, 4 for RGBA)
          // Because we emit two RGBx vectors per iteration, the second write
          // starts exactly kVectorLength * dcn bytes after the first.
          yuv422_to_rgb(
              y_even_lanes, y_odd_lanes, u, v,
              dst_rows.as_columns().ptr_at(static_cast<ptrdiff_t>(index)),
              dst_rows.as_columns().ptr_at(
                  static_cast<ptrdiff_t>(index + kVectorLength)));
        }
      };
      loop.unroll_twice(UnrollTwiceFunctor{src, dst_rows});

      // Scalar loop over YUV422 pixels.
      struct RemainingFunctor {
        const uint8_t* src_row;
        Rows<uint8_t>& dst_rows;

        KLEIDICV_FORCE_INLINE void operator()(size_t index,
                                              size_t length) const {
          for (; index < length; index += 2) {
            const uint8_t u = src_row[(index * scn) + u_idx];
            const uint8_t v = src_row[(index * scn) + v_idx];
            const uint8_t y0 = src_row[(index * scn) + y_idx];
            const uint8_t y1 = src_row[(index * scn) + y_idx + scn];

            const int32_t u_m128 = static_cast<int32_t>(u) - 128;
            const int32_t v_m128 = static_cast<int32_t>(v) - 128;

            const uint8_t y_rows[2] = {y0, y1};
            uint8_t* rgbx_rows[2] = {
                dst_rows.as_columns().ptr_at(static_cast<ptrdiff_t>(index)),
                dst_rows.as_columns().ptr_at(
                    static_cast<ptrdiff_t>(index + 1))};

            yuv422_to_rgb(y_rows, u_m128, v_m128, rgbx_rows);
          }
        }
      };
      loop.remaining(RemainingFunctor{src, dst_rows});

      ++dst_rows;
    }

    return KLEIDICV_OK;
  }

 private:
  KLEIDICV_FORCE_INLINE
  static uint8x16_t normalize_and_pack(int32x4_t vec_0, int32x4_t vec_1,
                                       int32x4_t vec_2, int32x4_t vec_3) {
    int16x4_t tmp_lo_lo = vqshrun_n_s32(vec_0, kWeightScale - 8);
    int16x8_t tmp_lo_hi =
        vqshrun_high_n_s32(tmp_lo_lo, vec_1, kWeightScale - 8);
    int16x4_t tmp_hi_lo = vqshrun_n_s32(vec_2, kWeightScale - 8);
    int16x8_t tmp_hi_hi =
        vqshrun_high_n_s32(tmp_hi_lo, vec_3, kWeightScale - 8);
    uint8x16_t output =
        vtrn2q_u8(vreinterpretq_u8(tmp_lo_hi), vreinterpretq_u8(tmp_hi_hi));
    return output;
  }
  // Convert two blocks of YUV422 (deinterleaved) data into RGBx color format.
  // Each block contains 16 Y values (y_even_lanes, y_odd_lanes) plus shared U
  // and V values. The function computes R, G, B channels, normalizes, and
  // stores results either as RGB (3 channels) or RGBA (4 channels).
  KLEIDICV_FORCE_INLINE
  static void yuv422_to_rgb(const uint8x16_t& y_even_lanes,
                            const uint8x16_t& y_odd_lanes, const uint8x16_t& u,
                            const uint8x16_t& v, uint8_t* rgbx0,
                            uint8_t* rgbx1) {
    // --- Preprocess Y channel ---
    // Subtract 16 from luma (Y') with saturation and widen later to 32 bits.
    uint8x16_t y_even_lanes_m16 = vqsubq_u8(y_even_lanes, vdupq_n_u8(16));
    uint8x16_t y_odd_lanes_m16 = vqsubq_u8(y_odd_lanes, vdupq_n_u8(16));

    // --- Zero-extend (8 to 32) via table lookups ---
    // The masks feed vqtbl1q_u8 so each lookup pulls 4 bytes out of a 16-lane
    // u8 vector and places the selected byte as the least-significant byte of a
    // 32-bit lane while zeroing the remaining three bytes.
    // vqtbl1q_u8 inserts 0 for indices >= 16 (e.g., 0xFF), letting us build
    // [x,0,0,0] groups that we reinterpret as int32x4_t to get u8 to s32 lanes
    // in one step.
    const uint8x16_t index_0 = {0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff,
                                2, 0xff, 0xff, 0xff, 3, 0xff, 0xff, 0xff};
    const uint8x16_t index_1 = {4, 0xff, 0xff, 0xff, 5, 0xff, 0xff, 0xff,
                                6, 0xff, 0xff, 0xff, 7, 0xff, 0xff, 0xff};
    const uint8x16_t index_2 = {8,  0xff, 0xff, 0xff, 9,  0xff, 0xff, 0xff,
                                10, 0xff, 0xff, 0xff, 11, 0xff, 0xff, 0xff};
    const uint8x16_t index_3 = {12, 0xff, 0xff, 0xff, 13, 0xff, 0xff, 0xff,
                                14, 0xff, 0xff, 0xff, 15, 0xff, 0xff, 0xff};

    // Expand Y values into 32-bit lanes for later arithmetic.
    // Note: "even" and "odd" describe the pixel position in the YUV422 packing,
    // not the Y component itself.
    //
    // In YUV422, pixels are stored as (Y0, U0, Y1, V0).
    // - The "even" vectors collect Y0, Y2, Y4, ... - these generate the
    //   even-positioned RGB outputs.
    // - The "odd" vectors collect Y1, Y3, Y5, ... - these generate the
    //   odd-positioned RGB outputs.
    int32x4_t y_even_lo_lo =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_even_lanes_m16, index_0));
    int32x4_t y_even_lo_hi =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_even_lanes_m16, index_1));
    int32x4_t y_even_hi_lo =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_even_lanes_m16, index_2));
    int32x4_t y_even_hi_hi =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_even_lanes_m16, index_3));
    int32x4_t y_odd_lo_lo =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_odd_lanes_m16, index_0));
    int32x4_t y_odd_lo_hi =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_odd_lanes_m16, index_1));
    int32x4_t y_odd_hi_lo =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_odd_lanes_m16, index_2));
    int32x4_t y_odd_hi_hi =
        vreinterpretq_s32_u8(vqtbl1q_u8(y_odd_lanes_m16, index_3));

    // Expand U and V into 32-bit lanes (shared chroma).
    // In YUV422, each U and V value is shared by a pair of pixels:
    //   (Y_even, U, Y_odd, V)
    // Therefore, the same U and V vectors are used when computing both
    // the "even" and "odd" RGB outputs.
    int32x4_t u_lo_lo = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_0));
    int32x4_t u_lo_hi = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_1));
    int32x4_t u_hi_lo = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_2));
    int32x4_t u_hi_hi = vreinterpretq_s32_u8(vqtbl1q_u8(u, index_3));
    int32x4_t v_lo_lo = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_0));
    int32x4_t v_lo_hi = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_1));
    int32x4_t v_hi_lo = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_2));
    int32x4_t v_hi_hi = vreinterpretq_s32_u8(vqtbl1q_u8(v, index_3));

    // Scale the Y (luma) values by the fixed coefficient kYWeight.
    // This produces the weighted luma contribution (Y') that forms the
    // base term for all R, G, and B channel calculations in the
    // YUV to RGB conversion.
    y_even_lo_lo = vmulq_n_s32(y_even_lo_lo, kYWeight);
    y_even_lo_hi = vmulq_n_s32(y_even_lo_hi, kYWeight);
    y_even_hi_lo = vmulq_n_s32(y_even_hi_lo, kYWeight);
    y_even_hi_hi = vmulq_n_s32(y_even_hi_hi, kYWeight);
    y_odd_lo_lo = vmulq_n_s32(y_odd_lo_lo, kYWeight);
    y_odd_lo_hi = vmulq_n_s32(y_odd_lo_hi, kYWeight);
    y_odd_hi_lo = vmulq_n_s32(y_odd_hi_lo, kYWeight);
    y_odd_hi_hi = vmulq_n_s32(y_odd_hi_hi, kYWeight);

    // Precompute constant base offsets for R, G, and B channels.
    // These include the rounding term (1 << (kWeightScale - 1)) and the
    // bias correction for centering U and V around 128.
    // This ensures that chroma values (U,V) are properly zero-based before
    // applying their respective weighting factors in the YUV to RGB formulas.
    int32x4_t r_base_{vdupq_n_s32((1 << (kWeightScale - 1)) -
                                  128 * kUVWeights[kRVWeightIndex])};
    int32x4_t g_base_{vdupq_n_s32(
        (1 << (kWeightScale - 1)) -
        128 * (kUVWeights[kGUWeightIndex] + kUVWeights[kGVWeightIndex]))};
    int32x4_t b_base_{vdupq_n_s32((1 << (kWeightScale - 1)) -
                                  128 * kUVWeights[kBUWeightIndex])};

    // --- Compute the Red channel ---
    // Formula: R = Y + (kRV * V) + bias
    // - Start with r_base_ (rounding + bias correction for V centered at 128).
    // - Multiply V by kUVWeights[kRVWeightIndex] and add the result to r_base_.
    // - Reuse the same V contribution for both even and odd pixels, since
    //   chroma is shared in YUV422.
    // - Finally, add the weighted Y values (even and odd) to produce
    //   the full R channel before normalization and packing to 8 bits.
    int32x4_t r_even_lo_lo =
        vmlaq_n_s32(r_base_, v_lo_lo, kUVWeights[kRVWeightIndex]);
    int32x4_t r_even_lo_hi =
        vmlaq_n_s32(r_base_, v_lo_hi, kUVWeights[kRVWeightIndex]);
    int32x4_t r_even_hi_lo =
        vmlaq_n_s32(r_base_, v_hi_lo, kUVWeights[kRVWeightIndex]);
    int32x4_t r_even_hi_hi =
        vmlaq_n_s32(r_base_, v_hi_hi, kUVWeights[kRVWeightIndex]);

    // Odd pixels reuse the same chroma base, so compute them before the even
    // registers are updated with their Y contribution.
    int32x4_t r_odd_lo_lo = vaddq_s32(r_even_lo_lo, y_odd_lo_lo);
    int32x4_t r_odd_lo_hi = vaddq_s32(r_even_lo_hi, y_odd_lo_hi);
    int32x4_t r_odd_hi_lo = vaddq_s32(r_even_hi_lo, y_odd_hi_lo);
    int32x4_t r_odd_hi_hi = vaddq_s32(r_even_hi_hi, y_odd_hi_hi);

    r_even_lo_lo = vaddq_s32(r_even_lo_lo, y_even_lo_lo);
    r_even_lo_hi = vaddq_s32(r_even_lo_hi, y_even_lo_hi);
    r_even_hi_lo = vaddq_s32(r_even_hi_lo, y_even_hi_lo);
    r_even_hi_hi = vaddq_s32(r_even_hi_hi, y_even_hi_hi);

    // Re-interleave and pack the Red channel to u8.
    // We computed R in four 4-lane chunks split by pixel parity:
    //   r_even_lo_lo (even pixels 0..3),   r_even_lo_hi (even 4..7)
    //   r_odd_lo_lo  (odd  pixels 0..3),   r_odd_lo_hi  (odd  4..7)
    // normalize_and_pack(...) saturates, shifts, narrows s32 to u8 *and*
    // interleaves even/odd so the output is in raster order:
    //   [R0, R1, R2, R3, ...] (i.e., even0, odd0, even1, odd1, ...).
    // r0 packs the first 16 R samples; r1 packs the next 16, which come from
    // the *_hi_* groups.
    uint8x16_t r0 = normalize_and_pack(r_even_lo_lo, r_even_lo_hi, r_odd_lo_lo,
                                       r_odd_lo_hi);
    uint8x16_t r1 = normalize_and_pack(r_even_hi_lo, r_even_hi_hi, r_odd_hi_lo,
                                       r_odd_hi_hi);

    // --- Compute the Green channel ---
    // Formula: G = Y + (kGU * U + kGV * V) + bias, reusing the shared U and V
    // samples for both pixels in each YUV422 pair. normalize_and_pack(...)
    // narrows back to u8 and interleaves even/odd results into raster order.
    int32x4_t g_even_lo_lo =
        vmlaq_n_s32(g_base_, u_lo_lo, kUVWeights[kGUWeightIndex]);
    int32x4_t g_even_lo_hi =
        vmlaq_n_s32(g_base_, u_lo_hi, kUVWeights[kGUWeightIndex]);
    int32x4_t g_even_hi_lo =
        vmlaq_n_s32(g_base_, u_hi_lo, kUVWeights[kGUWeightIndex]);
    int32x4_t g_even_hi_hi =
        vmlaq_n_s32(g_base_, u_hi_hi, kUVWeights[kGUWeightIndex]);

    g_even_lo_lo =
        vmlaq_n_s32(g_even_lo_lo, v_lo_lo, kUVWeights[kGVWeightIndex]);
    g_even_lo_hi =
        vmlaq_n_s32(g_even_lo_hi, v_lo_hi, kUVWeights[kGVWeightIndex]);
    g_even_hi_lo =
        vmlaq_n_s32(g_even_hi_lo, v_hi_lo, kUVWeights[kGVWeightIndex]);
    g_even_hi_hi =
        vmlaq_n_s32(g_even_hi_hi, v_hi_hi, kUVWeights[kGVWeightIndex]);

    // Same rationale as for Red: capture odd pixels before the even lanes add
    // Y.
    int32x4_t g_odd_lo_lo = vaddq_s32(g_even_lo_lo, y_odd_lo_lo);
    int32x4_t g_odd_lo_hi = vaddq_s32(g_even_lo_hi, y_odd_lo_hi);
    int32x4_t g_odd_hi_lo = vaddq_s32(g_even_hi_lo, y_odd_hi_lo);
    int32x4_t g_odd_hi_hi = vaddq_s32(g_even_hi_hi, y_odd_hi_hi);

    g_even_lo_lo = vaddq_s32(g_even_lo_lo, y_even_lo_lo);
    g_even_lo_hi = vaddq_s32(g_even_lo_hi, y_even_lo_hi);
    g_even_hi_lo = vaddq_s32(g_even_hi_lo, y_even_hi_lo);
    g_even_hi_hi = vaddq_s32(g_even_hi_hi, y_even_hi_hi);

    uint8x16_t g0 = normalize_and_pack(g_even_lo_lo, g_even_lo_hi, g_odd_lo_lo,
                                       g_odd_lo_hi);
    uint8x16_t g1 = normalize_and_pack(g_even_hi_lo, g_even_hi_hi, g_odd_hi_lo,
                                       g_odd_hi_hi);

    // --- Compute the Blue channel ---
    // Formula: B = Y + (kBU * U) + bias, sharing the same U samples across the
    // even/odd pair before normalize_and_pack(...) interleaves the outputs.
    int32x4_t b_even_lo_lo =
        vmlaq_n_s32(b_base_, u_lo_lo, kUVWeights[kBUWeightIndex]);
    int32x4_t b_even_lo_hi =
        vmlaq_n_s32(b_base_, u_lo_hi, kUVWeights[kBUWeightIndex]);
    int32x4_t b_even_hi_lo =
        vmlaq_n_s32(b_base_, u_hi_lo, kUVWeights[kBUWeightIndex]);
    int32x4_t b_even_hi_hi =
        vmlaq_n_s32(b_base_, u_hi_hi, kUVWeights[kBUWeightIndex]);

    // Blue follows the same ordering so odd lanes are finalized before evens.
    int32x4_t b_odd_lo_lo = vaddq_s32(b_even_lo_lo, y_odd_lo_lo);
    int32x4_t b_odd_lo_hi = vaddq_s32(b_even_lo_hi, y_odd_lo_hi);
    int32x4_t b_odd_hi_lo = vaddq_s32(b_even_hi_lo, y_odd_hi_lo);
    int32x4_t b_odd_hi_hi = vaddq_s32(b_even_hi_hi, y_odd_hi_hi);

    b_even_lo_lo = vaddq_s32(b_even_lo_lo, y_even_lo_lo);
    b_even_lo_hi = vaddq_s32(b_even_lo_hi, y_even_lo_hi);
    b_even_hi_lo = vaddq_s32(b_even_hi_lo, y_even_hi_lo);
    b_even_hi_hi = vaddq_s32(b_even_hi_hi, y_even_hi_hi);

    uint8x16_t b0 = normalize_and_pack(b_even_lo_lo, b_even_lo_hi, b_odd_lo_lo,
                                       b_odd_lo_hi);
    uint8x16_t b1 = normalize_and_pack(b_even_hi_lo, b_even_hi_hi, b_odd_hi_lo,
                                       b_odd_hi_hi);

    if constexpr (dcn > 3) {
      uint8x16x4_t rgba0, rgba1;
      // Red channel
      rgba0.val[2 - b_idx] = r0;
      rgba1.val[2 - b_idx] = r1;
      // Green channel
      rgba0.val[1] = g0;
      rgba1.val[1] = g1;
      // Blue channel
      rgba0.val[b_idx] = b0;
      rgba1.val[b_idx] = b1;
      // Alpha channel
      rgba0.val[3] = vdupq_n_u8(0xFF);
      rgba1.val[3] = vdupq_n_u8(0xFF);
      // Store RGB pixels to memory.
      vst4q_u8(rgbx0, rgba0);
      vst4q_u8(rgbx1, rgba1);

    } else {
      uint8x16x3_t rgba0, rgba1;
      // Red channel
      rgba0.val[2 - b_idx] = r0;
      rgba1.val[2 - b_idx] = r1;
      // Green channel
      rgba0.val[1] = g0;
      rgba1.val[1] = g1;
      // Blue channel
      rgba0.val[b_idx] = b0;
      rgba1.val[b_idx] = b1;
      // Store RGB pixels to memory.
      vst3q_u8(rgbx0, rgba0);
      vst3q_u8(rgbx1, rgba1);
    }
  }

  KLEIDICV_FORCE_INLINE
  static void yuv422_to_rgb(const uint8_t y_rows[2], int32_t u_m128,
                            int32_t v_m128, uint8_t* rgbx_rows[2]) {
    int32_t r_sub_y =
        kUVWeights[kRVWeightIndex] * v_m128 + (1 << (kWeightScale - 1));
    int32_t g_sub_y = kUVWeights[kGUWeightIndex] * u_m128 +
                      kUVWeights[kGVWeightIndex] * v_m128 +
                      (1 << (kWeightScale - 1));
    int32_t b_sub_y =
        kUVWeights[kBUWeightIndex] * u_m128 + (1 << (kWeightScale - 1));

    for (size_t selector = 0; selector < 2; ++selector) {
      int32_t y = kYWeight * std::max(y_rows[selector] - 16, 0);
      int32_t r = y + r_sub_y;
      int32_t g = y + g_sub_y;
      int32_t b = y + b_sub_y;

      r >>= kWeightScale;
      g >>= kWeightScale;
      b >>= kWeightScale;

      rgbx_rows[selector][2 - b_idx] = saturating_cast<int32_t, uint8_t>(r);
      rgbx_rows[selector][1] = saturating_cast<int32_t, uint8_t>(g);
      rgbx_rows[selector][b_idx] = saturating_cast<int32_t, uint8_t>(b);

      if constexpr (dcn > 3) {
        rgbx_rows[selector][3] = 0xFF;
      }

      rgbx_rows[selector] += dcn;
    }
  }
};

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv422_to_rgb_u8(const uint8_t* src, size_t src_stride,
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
    case KLEIDICV_YUYV_TO_BGR:
      return YUV422ToRGBxOrBGRx<0, 1, 0, 3>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_BGR:
      return YUV422ToRGBxOrBGRx<0, 0, 1, 3>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_BGR:
      return YUV422ToRGBxOrBGRx<0, 3, 0, 3>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YUYV_TO_RGB:
      return YUV422ToRGBxOrBGRx<2, 1, 0, 3>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_RGB:
      return YUV422ToRGBxOrBGRx<2, 0, 1, 3>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_RGB:
      return YUV422ToRGBxOrBGRx<2, 3, 0, 3>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YUYV_TO_BGRA:
      return YUV422ToRGBxOrBGRx<0, 1, 0, 4>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_BGRA:
      return YUV422ToRGBxOrBGRx<0, 0, 1, 4>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_BGRA:
      return YUV422ToRGBxOrBGRx<0, 3, 0, 4>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YUYV_TO_RGBA:
      return YUV422ToRGBxOrBGRx<2, 1, 0, 4>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_RGBA:
      return YUV422ToRGBxOrBGRx<2, 0, 1, 4>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_RGBA:
      return YUV422ToRGBxOrBGRx<2, 3, 0, 4>::yuv2rgbx_operation(
          src, src_stride, dst, dst_stride, width, height);
      break;
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
      break;
  }
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace kleidicv::neon
