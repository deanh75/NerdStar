// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_YUV422_TO_RGB_SC_H
#define KLEIDICV_YUV422_TO_RGB_SC_H

#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "yuv42x_coefficients.h"

namespace KLEIDICV_TARGET_NAMESPACE {
template <size_t b_idx, size_t u_chroma_idx, size_t y_idx, size_t dcn,
          bool use_unpack_path>
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
                                             size_t height) KLEIDICV_STREAMING {
    // Keep track of the current output row being written.
    Rows<uint8_t> dst_rows{dst, dst_stride, dcn};
    auto kVectorLength = svcntb();

    // Loop through rows along the image height.
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
        size_t kVectorLength;

        KLEIDICV_FORCE_INLINE void operator()(size_t index) const
            KLEIDICV_STREAMING {
          svbool_t pg = svptrue_b8();

          // Deinterleave the YUV422 data into separate channels.
          // svld4() loads 16 groups of 4 bytes: (Y0, U0, Y1, V0).
          // Because we unroll twice, we must process two pixels at once.
          // Each pixel contributes two components (Y + chroma), so 4 vectors
          // are required: Y0, Y1, U, and V. This is why we perform 4 loads
          // instead of 2 - they directly correspond to the unrolled iteration.
          svuint8x4_t yuv422 = svld4(pg, src_row + index * scn);
          svuint8_t y_even_lanes = svget4(yuv422, y_idx);
          svuint8_t y_odd_lanes = svget4(yuv422, y_idx + scn);
          svuint8_t u = svget4(yuv422, u_idx);
          svuint8_t v = svget4(yuv422, v_idx);

          // Convert two output vectors in one go (loop unrolled twice).
          // The second destination pointer is advanced by kVectorLength * dcn:
          //   - kVectorLength: number of pixels produced per vector
          //   - dcn: destination channels per pixel (3 for RGB, 4 for RGBA)
          // Because we emit two RGBx vectors per iteration, the second write
          // starts exactly kVectorLength * dcn bytes after the first.
          yuv422_to_rgb(
              pg, y_even_lanes, y_odd_lanes, u, v,
              dst_rows.as_columns().ptr_at(static_cast<ptrdiff_t>(index)),
              dst_rows.as_columns().ptr_at(
                  static_cast<ptrdiff_t>(index + kVectorLength)),
              pg, pg);
        }
      };

      loop.unroll_twice(UnrollTwiceFunctor{src, dst_rows, kVectorLength});

      struct RemainingFunctor {
        const uint8_t* src_row;
        Rows<uint8_t>& dst_rows;
        size_t kVectorLength;

        KLEIDICV_FORCE_INLINE void operator()(size_t index, size_t length) const
            KLEIDICV_STREAMING {
          svbool_t pg = svwhilelt_b8_u64(index, length);
          svbool_t pg_st1 = svwhilelt_b8_u64(index, length);
          svbool_t pg_st2 = svwhilelt_b8_u64(index + kVectorLength, length);

          svuint8x4_t yuv422 = svld4(svwhilelt_b8_u64(0, (length - index) / 2),
                                     src_row + index * scn);

          svuint8_t y_even_lanes = svget4(yuv422, y_idx);
          svuint8_t y_odd_lanes = svget4(yuv422, y_idx + scn);
          svuint8_t u = svget4(yuv422, u_idx);
          svuint8_t v = svget4(yuv422, v_idx);

          // Convert two output vectors in one go (loop unrolled twice).
          // The second destination pointer is advanced by kVectorLength * dcn:
          //   - kVectorLength: number of pixels produced per vector
          //   - dcn: destination channels per pixel (3 for RGB, 4 for RGBA)
          // Because we emit two RGBx vectors per iteration, the second write
          // starts exactly kVectorLength * dcn bytes after the first.
          yuv422_to_rgb(
              pg, y_even_lanes, y_odd_lanes, u, v,
              dst_rows.as_columns().ptr_at(static_cast<ptrdiff_t>(index)),
              dst_rows.as_columns().ptr_at(
                  static_cast<ptrdiff_t>(index + kVectorLength)),
              pg_st1, pg_st2);
        }
      };

      loop.remaining(RemainingFunctor{src, dst_rows, kVectorLength});

      ++dst_rows;
    }
    return KLEIDICV_OK;
  }

 private:
  // Convert two blocks of YUV422 (deinterleaved) data into RGBx color format.
  // Each block contains 16 Y values (y0, y1) plus shared U and V values.
  // The function computes R, G, B channels, normalizes, and stores
  // results either as RGB (3 channels) or RGBA (4 channels).
  KLEIDICV_FORCE_INLINE
  static void yuv422_to_rgb(svbool_t& pg, const svuint8_t& y_even_lanes,
                            const svuint8_t& y_odd_lanes, const svuint8_t& u,
                            const svuint8_t& v, uint8_t* rgbx0, uint8_t* rgbx1,
                            svbool_t& pg_st1,
                            svbool_t& pg_st2) KLEIDICV_STREAMING {
    // --- Preprocess Y channel ---
    // Subtract 16 from luma (Y') with saturation and widen later to 32 bits.
    svuint8_t y_even_lanes_m16 = svqsub(y_even_lanes, static_cast<uint8_t>(16));
    svuint8_t y_odd_lanes_m16 = svqsub(y_odd_lanes, static_cast<uint8_t>(16));

    // Expand Y values into 32-bit lanes for later arithmetic.
    // Note: "even" and "odd" refer to the pixel position in the YUV422 packing,
    // not the Y component itself.
    //
    // In YUV422, pixels are stored as (Y0, U0, Y1, V0).
    // - The "even" vectors collect Y0, Y2, Y4, ... - these generate the
    //   even-positioned RGB outputs.
    // - The "odd" vectors collect Y1, Y3, Y5, ... - these generate the
    //   odd-positioned RGB outputs.
    //
    // How it works here:
    // 1. Split Y into low/high halves using svmovlb()/svmovlt().
    // 2. Widen each half to 32-bit lanes with svunpklo()/svunpkhi().
    //
    // Why this may look unusual:
    // - At first glance, you might expect "_lo_hi" to come from *_hi, but
    //   that would require extra moves and shuffles.
    // - Current scheme uses only 2x svmov + 4x svunpk per group, which is
    //   efficient since the pipeline can issue two svunpk in parallel.
    // - A more "intuitive" pairing would need 4x svmov + 2x svunpk, which is
    //   slower because svmov has less bundling freedom.
    // - Using svmovlb/svmovlt also aligns lanes so later narrowing can run
    //   without additional shuffles, improving overall performance.
    svint32_t y_even_lo_lo{}, y_even_lo_hi{}, y_even_hi_lo{}, y_even_hi_hi{};
    svint32_t y_odd_lo_lo{}, y_odd_lo_hi{}, y_odd_hi_lo{}, y_odd_hi_hi{};
    // Expand U and V into 32-bit lanes (shared chroma).
    // In YUV422, each U and V value is shared by a pair of pixels:
    //   (Y_even, U, Y_odd, V)
    // Therefore, the same U and V vectors are used when computing both
    // the "even" and "odd" RGB outputs.
    svint32_t v_lo_lo{}, v_lo_hi{}, v_hi_lo{}, v_hi_hi{};
    svint32_t u_lo_lo{}, u_lo_hi{}, u_hi_lo{}, u_hi_hi{};
    if constexpr (use_unpack_path) {
      svuint16_t y_even_lo = svmovlb(y_even_lanes_m16);
      svuint16_t y_even_hi = svmovlt(y_even_lanes_m16);
      svuint16_t y_odd_lo = svmovlb(y_odd_lanes_m16);
      svuint16_t y_odd_hi = svmovlt(y_odd_lanes_m16);
      y_even_lo_lo = svreinterpret_s32(svunpklo(y_even_lo));
      y_even_lo_hi = svreinterpret_s32(svunpklo(y_even_hi));
      y_even_hi_lo = svreinterpret_s32(svunpkhi(y_even_lo));
      y_even_hi_hi = svreinterpret_s32(svunpkhi(y_even_hi));
      y_odd_lo_lo = svreinterpret_s32(svunpklo(y_odd_lo));
      y_odd_lo_hi = svreinterpret_s32(svunpklo(y_odd_hi));
      y_odd_hi_lo = svreinterpret_s32(svunpkhi(y_odd_lo));
      y_odd_hi_hi = svreinterpret_s32(svunpkhi(y_odd_hi));

      svuint16_t v_lo = svmovlb(v);
      svuint16_t v_hi = svmovlt(v);
      svuint16_t u_lo = svmovlb(u);
      svuint16_t u_hi = svmovlt(u);
      v_lo_lo = svreinterpret_s32(svunpklo(v_lo));
      v_lo_hi = svreinterpret_s32(svunpklo(v_hi));
      v_hi_lo = svreinterpret_s32(svunpkhi(v_lo));
      v_hi_hi = svreinterpret_s32(svunpkhi(v_hi));
      u_lo_lo = svreinterpret_s32(svunpklo(u_lo));
      u_lo_hi = svreinterpret_s32(svunpklo(u_hi));
      u_hi_lo = svreinterpret_s32(svunpkhi(u_lo));
      u_hi_hi = svreinterpret_s32(svunpkhi(u_hi));
    } else {
      svuint8_t index0 = svreinterpret_u8_u32(
          svindex_u32(0xFFFFFF00, 0x0002));  // 0, 2, 4, 6, ...
      svuint8_t index1 = svreinterpret_u8_u32(
          svindex_u32(0xFFFFFF00 + svcnth(), 0x0002));  // 8, 10, 12, 14, ...
      svuint8_t index2 = svreinterpret_u8_u32(
          svindex_u32(0xFFFFFF01, 0x0002));  // 1, 3, 5, 7, ...
      svuint8_t index3 = svreinterpret_u8_u32(
          svindex_u32(0xFFFFFF01 + svcnth(), 0x0002));  // 9, 11, 13, 15, ...

      y_even_lo_lo = svreinterpret_s32(svtbl_u8(y_even_lanes_m16, index0));
      y_even_lo_hi = svreinterpret_s32(svtbl_u8(y_even_lanes_m16, index2));
      y_even_hi_lo = svreinterpret_s32(svtbl_u8(y_even_lanes_m16, index1));
      y_even_hi_hi = svreinterpret_s32(svtbl_u8(y_even_lanes_m16, index3));
      y_odd_lo_lo = svreinterpret_s32(svtbl_u8(y_odd_lanes_m16, index0));
      y_odd_lo_hi = svreinterpret_s32(svtbl_u8(y_odd_lanes_m16, index2));
      y_odd_hi_lo = svreinterpret_s32(svtbl_u8(y_odd_lanes_m16, index1));
      y_odd_hi_hi = svreinterpret_s32(svtbl_u8(y_odd_lanes_m16, index3));

      v_lo_lo = svreinterpret_s32(svtbl_u8(v, index0));
      v_lo_hi = svreinterpret_s32(svtbl_u8(v, index2));
      v_hi_lo = svreinterpret_s32(svtbl_u8(v, index1));
      v_hi_hi = svreinterpret_s32(svtbl_u8(v, index3));
      u_lo_lo = svreinterpret_s32(svtbl_u8(u, index0));
      u_lo_hi = svreinterpret_s32(svtbl_u8(u, index2));
      u_hi_lo = svreinterpret_s32(svtbl_u8(u, index1));
      u_hi_hi = svreinterpret_s32(svtbl_u8(u, index3));
    }

    // Scale the Y (luma) values by the fixed coefficient kYWeight.
    // This produces the weighted luma contribution (Y') that forms the
    // base term for all R, G, and B channel calculations in the
    // YUV to RGB conversion.
    y_even_lo_lo = svmul_x(pg, y_even_lo_lo, kYWeight);
    y_even_lo_hi = svmul_x(pg, y_even_lo_hi, kYWeight);
    y_even_hi_lo = svmul_x(pg, y_even_hi_lo, kYWeight);
    y_even_hi_hi = svmul_x(pg, y_even_hi_hi, kYWeight);
    y_odd_lo_lo = svmul_x(pg, y_odd_lo_lo, kYWeight);
    y_odd_lo_hi = svmul_x(pg, y_odd_lo_hi, kYWeight);
    y_odd_hi_lo = svmul_x(pg, y_odd_hi_lo, kYWeight);
    y_odd_hi_hi = svmul_x(pg, y_odd_hi_hi, kYWeight);

    // Precompute constant base offsets for R, G, and B channels.
    // These include the rounding term (1 << (kWeightScale - 1)) and the
    // bias correction for centering U and V around 128.
    // This ensures that chroma values (U,V) are properly zero-based before
    // applying their respective weighting factors in the YUV to RGB formulas.
    constexpr int32_t kOffset = 1 << (kWeightScale - 1);
    svint32_t r_base = svdup_s32(kOffset - 128 * kUVWeights[kRVWeightIndex]);
    svint32_t g_base =
        svdup_s32(kOffset - 128 * (kUVWeights[1] + kUVWeights[2]));
    svint32_t b_base = svdup_s32(kOffset - 128 * kUVWeights[3]);

    // --- Compute the Red channel ---
    // Formula: R = Y + (kRV * V) + bias
    // - Start with r_base_ (rounding + bias correction for V centered at 128).
    // - Multiply V by kUVWeights[kRVWeightIndex] and add the result to r_base_.
    // - Reuse the same V contribution for both even and odd pixels, since
    //   chroma is shared in YUV422.
    // - Finally, add the weighted Y values (even and odd) to produce
    //   the full R channel before normalization and packing to 8 bits.
    svint32_t r_even_lo_lo =
        svmla_x(pg, r_base, v_lo_lo, kUVWeights[kRVWeightIndex]);
    svint32_t r_even_lo_hi =
        svmla_x(pg, r_base, v_lo_hi, kUVWeights[kRVWeightIndex]);
    svint32_t r_even_hi_lo =
        svmla_x(pg, r_base, v_hi_lo, kUVWeights[kRVWeightIndex]);
    svint32_t r_even_hi_hi =
        svmla_x(pg, r_base, v_hi_hi, kUVWeights[kRVWeightIndex]);

    // Re-interleave and pack the Red channel to u8.
    // We computed R in four 4-lane chunks split by pixel parity:
    //   r_even_lo_lo (even pixels 0..3),   r_even_lo_hi (even 4..7)
    //   r_even_hi_lo (even pixels 8..11),  r_even_hi_hi (even 12..15)
    // The same chroma sums drive the odd pixels, so we reuse the r_even_*
    // vectors when adding the odd Y lanes below.
    // normalize_and_pack(...) saturates, shifts, narrows s32 to u8 *and*
    // interleaves even/odd so the output is in raster order:
    //   [R0, R1, R2, R3, ...] (i.e., even0, odd0, even1, odd1, ...).
    // r0 packs the first 16 R samples; r1 packs the next 16, which come from
    // the *_hi_* groups.
    svint16_t r0_even = svaddhnb(r_even_lo_lo, y_even_lo_lo);
    r0_even = svaddhnt(r0_even, r_even_lo_hi, y_even_lo_hi);
    r0_even = svsra(svdup_n_s16(0), r0_even, kWeightScale - 16);
    svint16_t r0_odd = svaddhnb(r_even_lo_lo, y_odd_lo_lo);
    r0_odd = svaddhnt(r0_odd, r_even_lo_hi, y_odd_lo_hi);
    r0_odd = svsra(svdup_n_s16(0), r0_odd, kWeightScale - 16);
    svuint8_t r0 = svqxtunt(svqxtunb(r0_even), r0_odd);

    svint16_t r1_even = svaddhnb(r_even_hi_lo, y_even_hi_lo);
    r1_even = svaddhnt(r1_even, r_even_hi_hi, y_even_hi_hi);
    r1_even = svsra(svdup_n_s16(0), r1_even, kWeightScale - 16);
    svint16_t r1_odd = svaddhnb(r_even_hi_lo, y_odd_hi_lo);
    r1_odd = svaddhnt(r1_odd, r_even_hi_hi, y_odd_hi_hi);
    r1_odd = svsra(svdup_n_s16(0), r1_odd, kWeightScale - 16);
    svuint8_t r1 = svqxtunt(svqxtunb(r1_even), r1_odd);

    // --- Compute the Green channel ---
    // Formula: G = Y + (kGU * U + kGV * V) + bias, reusing the shared chroma
    // samples for both pixels in each YUV422 pair before normalize/pack
    // interleaves them back into raster order.
    svint32_t g_even_lo_lo =
        svmla_x(pg, g_base, u_lo_lo, kUVWeights[kGUWeightIndex]);
    svint32_t g_even_lo_hi =
        svmla_x(pg, g_base, u_lo_hi, kUVWeights[kGUWeightIndex]);
    svint32_t g_even_hi_lo =
        svmla_x(pg, g_base, u_hi_lo, kUVWeights[kGUWeightIndex]);
    svint32_t g_even_hi_hi =
        svmla_x(pg, g_base, u_hi_hi, kUVWeights[kGUWeightIndex]);
    g_even_lo_lo =
        svmla_x(pg, g_even_lo_lo, v_lo_lo, kUVWeights[kGVWeightIndex]);
    g_even_lo_hi =
        svmla_x(pg, g_even_lo_hi, v_lo_hi, kUVWeights[kGVWeightIndex]);
    g_even_hi_lo =
        svmla_x(pg, g_even_hi_lo, v_hi_lo, kUVWeights[kGVWeightIndex]);
    g_even_hi_hi =
        svmla_x(pg, g_even_hi_hi, v_hi_hi, kUVWeights[kGVWeightIndex]);

    svint16_t g0_even = svaddhnb(g_even_lo_lo, y_even_lo_lo);
    g0_even = svaddhnt(g0_even, g_even_lo_hi, y_even_lo_hi);
    g0_even = svsra(svdup_n_s16(0), g0_even, kWeightScale - 16);
    // Same rationale as for Red: reuse the g_even_* chroma base when adding
    // the odd Y lanes, avoiding redundant temporaries.
    svint16_t g0_odd = svaddhnb(g_even_lo_lo, y_odd_lo_lo);
    g0_odd = svaddhnt(g0_odd, g_even_lo_hi, y_odd_lo_hi);
    g0_odd = svsra(svdup_n_s16(0), g0_odd, kWeightScale - 16);
    svuint8_t g0 = svqxtunt(svqxtunb(g0_even), g0_odd);

    svint16_t g1_even = svaddhnb(g_even_hi_lo, y_even_hi_lo);
    g1_even = svaddhnt(g1_even, g_even_hi_hi, y_even_hi_hi);
    g1_even = svsra(svdup_n_s16(0), g1_even, kWeightScale - 16);
    svint16_t g1_odd = svaddhnb(g_even_hi_lo, y_odd_hi_lo);
    g1_odd = svaddhnt(g1_odd, g_even_hi_hi, y_odd_hi_hi);
    g1_odd = svsra(svdup_n_s16(0), g1_odd, kWeightScale - 16);
    svuint8_t g1 = svqxtunt(svqxtunb(g1_even), g1_odd);

    // --- Compute the Blue channel ---
    // Formula: B = Y + (kBU * U) + bias, sharing the U samples across each
    // even/odd pair before normalize/pack interleaves the results.
    svint32_t b_even_lo_lo =
        svmla_x(pg, b_base, u_lo_lo, kUVWeights[kBUWeightIndex]);
    svint32_t b_even_lo_hi =
        svmla_x(pg, b_base, u_lo_hi, kUVWeights[kBUWeightIndex]);
    svint32_t b_even_hi_lo =
        svmla_x(pg, b_base, u_hi_lo, kUVWeights[kBUWeightIndex]);
    svint32_t b_even_hi_hi =
        svmla_x(pg, b_base, u_hi_hi, kUVWeights[kBUWeightIndex]);

    svint16_t b0_even = svaddhnb(b_even_lo_lo, y_even_lo_lo);
    b0_even = svaddhnt(b0_even, b_even_lo_hi, y_even_lo_hi);
    b0_even = svsra(svdup_n_s16(0), b0_even, kWeightScale - 16);
    // Blue follows the same pattern, so reuse the b_even_* vectors for odd Y.
    svint16_t b0_odd = svaddhnb(b_even_lo_lo, y_odd_lo_lo);
    b0_odd = svaddhnt(b0_odd, b_even_lo_hi, y_odd_lo_hi);
    b0_odd = svsra(svdup_n_s16(0), b0_odd, kWeightScale - 16);
    svuint8_t b0 = svqxtunt(svqxtunb(b0_even), b0_odd);

    svint16_t b1_even = svaddhnb(b_even_hi_lo, y_even_hi_lo);
    b1_even = svaddhnt(b1_even, b_even_hi_hi, y_even_hi_hi);
    b1_even = svsra(svdup_n_s16(0), b1_even, kWeightScale - 16);
    svint16_t b1_odd = svaddhnb(b_even_hi_lo, y_odd_hi_lo);
    b1_odd = svaddhnt(b1_odd, b_even_hi_hi, y_odd_hi_hi);
    b1_odd = svsra(svdup_n_s16(0), b1_odd, kWeightScale - 16);
    svuint8_t b1 = svqxtunt(svqxtunb(b1_even), b1_odd);

    if constexpr (dcn > 3) {
      svuint8x4_t rgba0 =
          svcreate4(b_idx ? r0 : b0, g0, b_idx ? b0 : r0, svdup_n_u8(0xFF));
      svuint8x4_t rgba1 =
          svcreate4(b_idx ? r1 : b1, g1, b_idx ? b1 : r1, svdup_n_u8(0xFF));
      // Store RGBA pixels to memory.
      svst4_u8(pg_st1, rgbx0, rgba0);
      svst4_u8(pg_st2, rgbx1, rgba1);
    } else {
      svuint8x3_t rgb0 = svcreate3(b_idx ? r0 : b0, g0, b_idx ? b0 : r0);
      svuint8x3_t rgb1 = svcreate3(b_idx ? r1 : b1, g1, b_idx ? b1 : r1);
      // Store RGB pixels to memory.
      svst3(pg_st1, rgbx0, rgb0);
      svst3(pg_st2, rgbx1, rgb1);
    }
  }
};

template <size_t b_idx, size_t u_chroma_idx, size_t y_idx, size_t dcn>
kleidicv_error_t dispatch_yuv422_to_rgb(bool use_unpack_path,
                                        const uint8_t* src, size_t src_stride,
                                        uint8_t* dst, size_t dst_stride,
                                        size_t width, size_t height) {
  if (use_unpack_path) {
    return YUV422ToRGBxOrBGRx<b_idx, u_chroma_idx, y_idx, dcn,
                              true>::yuv2rgbx_operation(src, src_stride, dst,
                                                        dst_stride, width,
                                                        height);
  }
  return YUV422ToRGBxOrBGRx<b_idx, u_chroma_idx, y_idx, dcn,
                            false>::yuv2rgbx_operation(src, src_stride, dst,
                                                       dst_stride, width,
                                                       height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv422_to_rgb_u8_sc(
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
  bool use_unpack_path = KLEIDICV_UNLIKELY(svcntb() >= 256);
  switch (color_format) {
    case KLEIDICV_YUYV_TO_BGR:
      return dispatch_yuv422_to_rgb<0, 1, 0, 3>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_BGR:
      return dispatch_yuv422_to_rgb<0, 0, 1, 3>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_BGR:
      return dispatch_yuv422_to_rgb<0, 3, 0, 3>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YUYV_TO_RGB:
      return dispatch_yuv422_to_rgb<2, 1, 0, 3>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_RGB:
      return dispatch_yuv422_to_rgb<2, 0, 1, 3>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_RGB:
      return dispatch_yuv422_to_rgb<2, 3, 0, 3>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YUYV_TO_BGRA:
      return dispatch_yuv422_to_rgb<0, 1, 0, 4>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_BGRA:
      return dispatch_yuv422_to_rgb<0, 0, 1, 4>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_BGRA:
      return dispatch_yuv422_to_rgb<0, 3, 0, 4>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YUYV_TO_RGBA:
      return dispatch_yuv422_to_rgb<2, 1, 0, 4>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_UYVY_TO_RGBA:
      return dispatch_yuv422_to_rgb<2, 0, 1, 4>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    case KLEIDICV_YVYU_TO_RGBA:
      return dispatch_yuv422_to_rgb<2, 3, 0, 4>(
          use_unpack_path, src, src_stride, dst, dst_stride, width, height);
      break;
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
      break;
  }
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_YUV422_TO_RGB_SC_H
