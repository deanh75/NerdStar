// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RGB_TO_YUV420_SC_H
#define KLEIDICV_RGB_TO_YUV420_SC_H

#include <algorithm>

#include "kleidicv/sve2.h"
#include "yuv42x_coefficients.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <bool kAlpha, bool RGB, bool kInterleave>
class RGBxorBGRxToYUV420 {
 public:
  using ArrayOfFour_svuint32 = ScalableVectorArray1D<svuint32_t, 4>;
  using ArrayOfFour_svint32 = ScalableVectorArray1D<svint32_t, 4>;
  using ArrayOfTwo_svint32 = ScalableVectorArray1D<svint32_t, 2>;

  static kleidicv_error_t rgb2yuv420_operation_sc(
      const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
      uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
      bool v_first, size_t begin, size_t end) KLEIDICV_STREAMING {
    size_t row_begin = begin * 2;
    size_t row_end = std::min<size_t>(height, end * 2);
    const uint8_t *src_row = nullptr;
    uint8_t *y_row = nullptr;
    uint8_t *u_row = nullptr;
    uint8_t *v_row = nullptr;
    for (size_t h = row_begin; h < row_end; h++) {
      src_row = src + src_stride * h;
      y_row = y_dst + y_stride * h;
      bool evenRow = (h & 1) == 0;
      if (evenRow) {
        if constexpr (kInterleave) {
          u_row = uv_dst + uv_stride * (h / 2);
        } else {
          u_row =
              uv_dst + uv_stride * (h / 4) + ((h / 2) % 2) * ((width + 1) / 2);
          // Pointer to the start of the V plane.
          // The V plane follows the U plane. Both U and V planes are
          // subsampled at a 2:1 vertical ratio (i.e., each has height / 2
          // rows), and are often stored in a single contiguous chroma region in
          // memory. Depending on image height and stride, the starting offset
          // of V may require adjustment to maintain correct alignment. In
          // particular, the chroma rows may not align perfectly, so a
          // fractional offset (in rows) is applied to calculate the V plane
          // position. The formula used here accounts for this by adjusting
          // based on row parity, assuming consistent memory layout across the
          // Y, U, and V planes.
          v_row = uv_dst + uv_stride * ((h + height + 1) / 4) +
                  (((h + height + 1) / 2) % 2) * ((width + 1) / 2);
        }
      }

      const size_t kVectorLength = svcntb();
      LoopUnroll2 loop{width, kVectorLength};

      loop.unroll_twice([&](size_t index) KLEIDICV_STREAMING {
        svbool_t pg = svptrue_b8();

        vector_path_2x(src_row, y_row, u_row, v_row, v_first, index, evenRow,
                       pg, pg, pg);
      });

      loop.remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
        svbool_t pg = svwhilelt_b8_u64(index, length);
        svbool_t pg_half = svwhilelt_b8_u64((index + 1) / 2, (length + 1) / 2);
        while (svptest_first(svptrue_b8(), pg)) {
          vector_path(src_row, y_row, u_row, v_row, v_first, index, evenRow, pg,
                      pg_half);
          index += kVectorLength;
          pg = svwhilelt_b8_u64(index, length);
          pg_half = svwhilelt_b8_u64((index + 1) / 2, (length + 1) / 2);
        }
      });
    }
    return KLEIDICV_OK;
  }

 private:
  KLEIDICV_FORCE_INLINE
  static void vector_path_2x(const uint8_t *src_row, uint8_t *y_row,
                             uint8_t *u_row, uint8_t *v_row, bool v_first,
                             const size_t index, const bool evenRow,
                             const svbool_t pg0, const svbool_t pg1,
                             const svbool_t pg_half) KLEIDICV_STREAMING {
    svuint32_t r0_0, r0_1, r0_2, r0_3, g0_0, g0_1, g0_2, g0_3, b0_0, b0_1, b0_2,
        b0_3, r1_0, r1_1, r1_2, r1_3, g1_0, g1_1, g1_2, g1_3, b1_0, b1_1, b1_2,
        b1_3;

    ArrayOfFour_svuint32 r0 = {
        {std::ref(r0_0), std::ref(r0_1), std::ref(r0_2), std::ref(r0_3)}};
    ArrayOfFour_svuint32 g0 = {
        {std::ref(g0_0), std::ref(g0_1), std::ref(g0_2), std::ref(g0_3)}};
    ArrayOfFour_svuint32 b0 = {
        {std::ref(b0_0), std::ref(b0_1), std::ref(b0_2), std::ref(b0_3)}};
    ArrayOfFour_svuint32 r1 = {
        {std::ref(r1_0), std::ref(r1_1), std::ref(r1_2), std::ref(r1_3)}};
    ArrayOfFour_svuint32 g1 = {
        {std::ref(g1_0), std::ref(g1_1), std::ref(g1_2), std::ref(g1_3)}};
    ArrayOfFour_svuint32 b1 = {
        {std::ref(b1_0), std::ref(b1_1), std::ref(b1_2), std::ref(b1_3)}};

    load_rgb_2x(r0, g0, b0, r1, g1, b1, src_row, scn * index, pg0, pg1);

    svuint8_t y0 = rgb_to_y(r0, g0, b0);

    svuint8_t y1 = rgb_to_y(r1, g1, b1);

#if KLEIDICV_TARGET_SME2
    // assume the predicates are full true
    svcount_t pg_counter = svptrue_c8();
    svst1(pg_counter, y_row + index, svcreate2(y0, y1));
#else
    svst1(pg0, y_row + index, y0);
    svst1(pg1, y_row + index + svcntb(), y1);
#endif  // KLEIDICV_TARGET_SME2

    if (evenRow) {
      svuint8_t u, v;
      svint32_t r_even_0 = svreinterpret_s32(r0_0);
      svint32_t r_even_1 = svreinterpret_s32(r0_1);
      svint32_t r_even_2 = svreinterpret_s32(r1_0);
      svint32_t r_even_3 = svreinterpret_s32(r1_1);
      svint32_t g_even_0 = svreinterpret_s32(g0_0);
      svint32_t g_even_1 = svreinterpret_s32(g0_1);
      svint32_t g_even_2 = svreinterpret_s32(g1_0);
      svint32_t g_even_3 = svreinterpret_s32(g1_1);
      svint32_t b_even_0 = svreinterpret_s32(b0_0);
      svint32_t b_even_1 = svreinterpret_s32(b0_1);
      svint32_t b_even_2 = svreinterpret_s32(b1_0);
      svint32_t b_even_3 = svreinterpret_s32(b1_1);

      ArrayOfFour_svint32 r_even = {{std::ref(r_even_0), std::ref(r_even_1),
                                     std::ref(r_even_2), std::ref(r_even_3)}};
      ArrayOfFour_svint32 g_even = {{std::ref(g_even_0), std::ref(g_even_1),
                                     std::ref(g_even_2), std::ref(g_even_3)}};
      ArrayOfFour_svint32 b_even = {{std::ref(b_even_0), std::ref(b_even_1),
                                     std::ref(b_even_2), std::ref(b_even_3)}};

      rgb_to_uv_2x(r_even, g_even, b_even, u, v);

      if (v_first) {
        swap_scalable(u, v);
      }

      if constexpr (kInterleave) {
        svuint8x2_t uv = svcreate2(u, v);
        svst2_u8(pg_half, u_row + index, uv);
      } else {
        svst1(pg_half, u_row + index / 2, u);
        svst1(pg_half, v_row + index / 2, v);
      }
    }
  }

  static void vector_path(const uint8_t *src_row, uint8_t *y_row,
                          uint8_t *u_row, uint8_t *v_row, bool v_first,
                          const size_t index, const bool evenRow,
                          const svbool_t pg0,
                          const svbool_t pg_half) KLEIDICV_STREAMING {
    svuint32_t r0_0, r0_1, r0_2, r0_3, g0_0, g0_1, g0_2, g0_3, b0_0, b0_1, b0_2,
        b0_3;

    ArrayOfFour_svuint32 r0 = {
        {std::ref(r0_0), std::ref(r0_1), std::ref(r0_2), std::ref(r0_3)}};
    ArrayOfFour_svuint32 g0 = {
        {std::ref(g0_0), std::ref(g0_1), std::ref(g0_2), std::ref(g0_3)}};
    ArrayOfFour_svuint32 b0 = {
        {std::ref(b0_0), std::ref(b0_1), std::ref(b0_2), std::ref(b0_3)}};

    load_rgb(r0, g0, b0, src_row, scn * index, pg0);

    svuint8_t y0 = rgb_to_y(r0, g0, b0);

    svst1(pg0, y_row + index, y0);

    if (evenRow) {
      svuint8_t u, v;
      svint32_t r_even_0 = svreinterpret_s32(r0_0);
      svint32_t r_even_1 = svreinterpret_s32(r0_1);
      svint32_t g_even_0 = svreinterpret_s32(g0_0);
      svint32_t g_even_1 = svreinterpret_s32(g0_1);
      svint32_t b_even_0 = svreinterpret_s32(b0_0);
      svint32_t b_even_1 = svreinterpret_s32(b0_1);

      ArrayOfTwo_svint32 r_even = {{std::ref(r_even_0), std::ref(r_even_1)}};
      ArrayOfTwo_svint32 g_even = {{std::ref(g_even_0), std::ref(g_even_1)}};
      ArrayOfTwo_svint32 b_even = {{std::ref(b_even_0), std::ref(b_even_1)}};

      rgb_to_uv(r_even, g_even, b_even, u, v);

      if (v_first) {
        swap_scalable(u, v);
      }

      if constexpr (kInterleave) {
        svuint8x2_t uv = svcreate2(u, v);
        svst2_u8(pg_half, u_row + index, uv);
      } else {
        svst1(pg_half, u_row + index / 2, u);
        svst1(pg_half, v_row + index / 2, v);
      }
    }
  }

  static svuint8_t rgb_to_y(ArrayOfFour_svuint32 r, ArrayOfFour_svuint32 g,
                            ArrayOfFour_svuint32 b) KLEIDICV_STREAMING {
    const uint32_t kShifted16 = (16 << kWeightScale);
    const uint32_t kHalfShift = (1 << (kWeightScale - 1));

    svbool_t pg = svptrue_b32();

    // Y = kR*R + kG*G + kB*B + rounding bias
    svuint32_t bias = svdup_u32(kHalfShift + kShifted16);
    svuint32_t y_0 = bias;
    svuint32_t y_1 = bias;
    svuint32_t y_2 = bias;
    svuint32_t y_3 = bias;

    ArrayOfFour_svuint32 y = {
        {std::ref(y_0), std::ref(y_1), std::ref(y_2), std::ref(y_3)}};

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
      y(i) = svmla_n_u32_x(pg, y(i), r(i), kRYWeight);
      y(i) = svmla_n_u32_x(pg, y(i), g(i), kGYWeight);
      y(i) = svmla_n_u32_x(pg, y(i), b(i), kBYWeight);
    }

    svuint16_t y_b = svshrnb_n_u32(y(0), kWeightScale - 8);
    y_b = svshrnt_n_u32(y_b, y(2), kWeightScale - 8);  // 0, 1, 2, 3, 4, 5, 6, 7
    svuint16_t y_t = svshrnb_n_u32(y(1), kWeightScale - 8);
    y_t = svshrnt_n_u32(y_t, y(3),
                        kWeightScale - 8);  // 8, 9, 10, 11, 12, 13, 14, 15

    return svuzp2_u8(svreinterpret_u8(y_b), svreinterpret_u8(y_t));
  }

  static svuint8_t compute_u_or_v_2x(ArrayOfFour_svint32 r,
                                     ArrayOfFour_svint32 g,
                                     ArrayOfFour_svint32 b, const int r_coeff,
                                     const int g_coeff,
                                     const int b_coeff) KLEIDICV_STREAMING {
    svbool_t pg = svptrue_b32();
    const int kHalfShift = (1 << (kWeightScale - 1));
    const int kShifted128 = (128 << kWeightScale);
    svint32_t bias = svdup_s32(kHalfShift + kShifted128);
    svint32_t uv0 = bias;
    svint32_t uv1 = bias;
    svint32_t uv2 = bias;
    svint32_t uv3 = bias;

    ArrayOfFour_svint32 uv = {
        {std::ref(uv0), std::ref(uv1), std::ref(uv2), std::ref(uv3)}};

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
      uv(i) = svmla_n_s32_x(pg, uv(i), r(i), r_coeff);
      uv(i) = svmla_n_s32_x(pg, uv(i), g(i), g_coeff);
      uv(i) = svmla_n_s32_x(pg, uv(i), b(i), b_coeff);
    }

    svint16_t uv_b =
        svuzp2_s16(svreinterpret_s16(uv(0)), svreinterpret_s16(uv(1)));
    svint16_t uv_t =
        svuzp2_s16(svreinterpret_s16(uv(2)), svreinterpret_s16(uv(3)));

    uv_b = svasr_n_s16_x(pg, uv_b, kWeightScale - 16);
    uv_t = svasr_n_s16_x(pg, uv_t, kWeightScale - 16);

    return svuzp1_u8(svreinterpret_u8(uv_b), svreinterpret_u8(uv_t));
  }

  static void rgb_to_uv_2x(ArrayOfFour_svint32 r, ArrayOfFour_svint32 g,
                           ArrayOfFour_svint32 b, svuint8_t &u,
                           svuint8_t &v) KLEIDICV_STREAMING {
    // ---------------- U (Cb) Component ----------------
    // U = R * kRU + G * kGU + B * kBU + bias
    u = compute_u_or_v_2x(r, g, b, kRUWeight, kGUWeight, kBUWeight);

    // ---------------- V (Cr) Component ----------------
    // V = R * kBU + G * kGV + B * kBV + bias
    v = compute_u_or_v_2x(r, g, b, kBUWeight, kGVWeight, kBVWeight);
  }

  static svuint8_t compute_u_or_v(ArrayOfTwo_svint32 r, ArrayOfTwo_svint32 g,
                                  ArrayOfTwo_svint32 b, const int r_coeff,
                                  const int g_coeff,
                                  const int b_coeff) KLEIDICV_STREAMING {
    svbool_t pg = svptrue_b32();
    const int kHalfShift = (1 << (kWeightScale - 1));
    const int kShifted128 = (128 << kWeightScale);

    svint32_t bias = svdup_s32(kHalfShift + kShifted128);
    svint32_t uv0 = bias;
    svint32_t uv1 = bias;

    ArrayOfTwo_svint32 uv = {{std::ref(uv0), std::ref(uv1)}};

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 2; i++) {
      uv(i) = svmla_n_s32_x(pg, uv(i), r(i), r_coeff);
      uv(i) = svmla_n_s32_x(pg, uv(i), g(i), g_coeff);
      uv(i) = svmla_n_s32_x(pg, uv(i), b(i), b_coeff);
    }

    svint16_t output =
        svuzp2_s16(svreinterpret_s16(uv(0)), svreinterpret_s16(uv(1)));

    output = svasr_n_s16_x(pg, output, kWeightScale - 16);

    return svuzp1_u8(svreinterpret_u8(output), svreinterpret_u8(output));
  }

  static void rgb_to_uv(ArrayOfTwo_svint32 r, ArrayOfTwo_svint32 g,
                        ArrayOfTwo_svint32 b, svuint8_t &u,
                        svuint8_t &v) KLEIDICV_STREAMING {
    // ---------------- U (Cb) Component ----------------
    // U = R * kRU + G * kGU + B * kBU + bias
    u = compute_u_or_v(r, g, b, kRUWeight, kGUWeight, kBUWeight);

    // ---------------- V (Cr) Component ----------------
    // V = R * kBU + G * kGV + B * kBV + bias
    v = compute_u_or_v(r, g, b, kBUWeight, kGVWeight, kBVWeight);
  }

  static void load_rgb(ArrayOfFour_svuint32 &r, ArrayOfFour_svuint32 &g,
                       ArrayOfFour_svuint32 &b, const uint8_t *src_row,
                       const size_t w, const svbool_t &pg0) KLEIDICV_STREAMING {
    svuint8_t b0, g0, r0;
    if constexpr (kAlpha) {
      // 4-channel input (RGBA or BGRA)
      svuint8x4_t vsrc0 = svld4(pg0, src_row + w);

      b0 = svget4(vsrc0, b_index);
      g0 = svget4(vsrc0, g_index);
      r0 = svget4(vsrc0, r_index);

    } else {
      // 3-channel input (RGB or BGR)
      svuint8x3_t vsrc0 = svld3(pg0, src_row + w);

      b0 = svget3(vsrc0, b_index);
      g0 = svget3(vsrc0, g_index);
      r0 = svget3(vsrc0, r_index);
    }
    svuint16_t r0_lo = svmovlb(r0);
    svuint16_t r0_hi = svmovlt(r0);
    r(0) = svunpklo(r0_lo);  // 0, 2, 4, 6
    r(1) = svunpkhi(r0_lo);  // 8, 10, 12, 14
    r(2) = svunpklo(r0_hi);  // 1, 3, 5, 7
    r(3) = svunpkhi(r0_hi);  // 9, 11, 13, 15

    svuint16_t g0_lo = svmovlb(g0);
    svuint16_t g0_hi = svmovlt(g0);
    g(0) = svunpklo(g0_lo);
    g(1) = svunpkhi(g0_lo);
    g(2) = svunpklo(g0_hi);
    g(3) = svunpkhi(g0_hi);

    svuint16_t b0_lo = svmovlb(b0);
    svuint16_t b0_hi = svmovlt(b0);
    b(0) = svunpklo(b0_lo);
    b(1) = svunpkhi(b0_lo);
    b(2) = svunpklo(b0_hi);
    b(3) = svunpkhi(b0_hi);
  }

  static void load_rgb_2x(ArrayOfFour_svuint32 &r0, ArrayOfFour_svuint32 &g0,
                          ArrayOfFour_svuint32 &b0, ArrayOfFour_svuint32 &r1,
                          ArrayOfFour_svuint32 &g1, ArrayOfFour_svuint32 &b1,
                          const uint8_t *src_row, const size_t w,
                          const svbool_t pg0,
                          const svbool_t pg1) KLEIDICV_STREAMING {
    const size_t kVectorLength = svcntb();
    load_rgb(r0, g0, b0, src_row, w, pg0);

    load_rgb(r1, g1, b1, src_row, w + scn * kVectorLength, pg1);
  }

  static constexpr int b_index = RGB ? 2 : 0;
  static constexpr int g_index = 1;
  static constexpr int r_index = RGB ? 0 : 2;
  static constexpr size_t scn = kAlpha ? 4 : 3;
};

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RGB_TO_YUV420_SC_H
