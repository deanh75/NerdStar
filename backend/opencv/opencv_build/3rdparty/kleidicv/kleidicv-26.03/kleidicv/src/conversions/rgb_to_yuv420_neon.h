// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RGB_TO_YUV420_H
#define KLEIDICV_RGB_TO_YUV420_H

#include <algorithm>
#include <utility>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "yuv42x_coefficients.h"

namespace kleidicv::neon {

template <bool kAlpha, bool RGB, bool kInterleave>
class RGBxorBGRxToYUV420 {
 public:
  static kleidicv_error_t rgb2yuv420_operation(
      const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
      uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
      bool v_first, size_t begin, size_t end) {
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
          // rows) and stored in a single contiguous chroma region.
          // Depending on image height and stride, the starting offset
          // of V may require adjustment, so a
          // fractional offset (in rows) is applied to calculate the V plane
          // position.
          v_row = uv_dst + uv_stride * ((h + height + 1) / 4) +
                  (((h + height + 1) / 2) % 2) * ((width + 1) / 2);
        }
      }

      LoopUnroll2<TryToAvoidTailLoop> loop{width, kVectorLength};
      loop.unroll_twice([&](size_t index) {
        vector_path_2x(src_row, y_row, u_row, v_row, v_first, index, evenRow);
      });

      loop.tail([&](size_t index) {
        scalar_path(src_row, y_row, u_row, v_row, v_first, index, width,
                    evenRow);
      });
    }

    return KLEIDICV_OK;
  }

 private:
  KLEIDICV_FORCE_INLINE
  static void vector_path_2x(const uint8_t *src_row, uint8_t *y_row,
                             uint8_t *u_row, uint8_t *v_row, const bool v_first,
                             const size_t index, const bool evenRow) {
    uint32x4_t r0[4], g0[4], b0[4], r1[4], g1[4], b1[4];

    load_rgb_2x(r0, g0, b0, r1, g1, b1, src_row, index);

    uint8x16_t y0 = rgb_to_y(r0, g0, b0);

    uint8x16_t y1 = rgb_to_y(r1, g1, b1);

    vst1q_u8(y_row + index, y0);
    vst1q_u8(y_row + index + kVectorLength, y1);

    // U and V are subsampled by a factor of 2 in both horizontal and vertical
    // directions for YUV420 format. Therefore, we only compute U and V from
    // even rows and even columns. When the input RGB image has an odd width or
    // height, the chroma (U and V) dimensions are rounded up. For example, if
    // the height is 9, Y will be 9 rows, but U and V will be 5 rows (9 / 2
    // = 4.5 -> rounded up). The same rounding is applied for width.
    if (evenRow) {
      uint8x16x2_t uv;
      int32x4_t r_even[4] = {r0[0], r0[2], r1[0], r1[2]};
      int32x4_t g_even[4] = {g0[0], g0[2], g1[0], g1[2]};
      int32x4_t b_even[4] = {b0[0], b0[2], b1[0], b1[2]};
      rgb_to_uv_2x(r_even, g_even, b_even, uv.val[0], uv.val[1]);

      if (v_first) {
        std::swap(uv.val[0], uv.val[1]);
      }

      if constexpr (kInterleave) {
        vst2q_u8(u_row + index, uv);
      } else {
        vst1q_u8(u_row + index / 2, uv.val[0]);
        vst1q_u8(v_row + index / 2, uv.val[1]);
      }
    }
  }

  static void scalar_path(const uint8_t *src_row, uint8_t *y_row,
                          uint8_t *u_row, uint8_t *v_row, const bool v_first,
                          size_t index, const size_t length,
                          const bool evenRow) {
    const size_t u_index_ = v_first;
    const size_t v_index_ = !v_first;

    for (; index < length; index += 1) {
      uint8_t b0{}, g0{}, r0{};
      bool evenCol = (index & 1) == 0;
      b0 = src_row[index * scn + b_index_];
      g0 = src_row[index * scn + g_index_];
      r0 = src_row[index * scn + r_index_];

      uint8_t y0 = rgb_to_y(r0, g0, b0);
      y_row[index] = y0;

      // U and V are subsampled by a factor of 2 in both horizontal and vertical
      // directions
      // for YUV420 format. Therefore, we only compute U and V from even rows
      // and even columns. When the input RGB image has an odd width or height,
      // the chroma (U and V) dimensions are rounded up. For example, if the
      // height is 9, Y will be 9 rows, but U and V will be 5 rows (9 / 2 = 4.5
      // -> rounded up). The same rounding is applied for width.
      if (evenRow && evenCol) {
        uint8_t uv[2] = {0, 0};
        rgb_to_uv(r0, g0, b0, uv);
        if constexpr (kInterleave) {
          u_row[index] = uv[u_index_];
          u_row[index + 1] = uv[v_index_];
        } else {
          u_row[(index + 1) / 2] = uv[u_index_];
          v_row[(index + 1) / 2] = uv[v_index_];
        }
      }
    }
  }

  static uint8_t rgb_to_y(uint8_t r, uint8_t g, uint8_t b) {
    const int kShifted16 = (16 << kWeightScale);
    const int kHalfShift = (1 << (kWeightScale - 1));
    int yy =
        kRYWeight * r + kGYWeight * g + kBYWeight * b + kHalfShift + kShifted16;

    return std::clamp(yy >> kWeightScale, 0, 0xff);
  }

  static uint8x16_t rgb_to_y(const uint32x4_t r[4], const uint32x4_t g[4],
                             const uint32x4_t b[4]) {
    const int kShifted16 = (16 << kWeightScale);
    const int kHalfShift = (1 << (kWeightScale - 1));

    // Y = kR*R + kG*G + kB*B + rounding bias
    uint32x4_t v_kRYWeight = vdupq_n_u32(kRYWeight);
    uint32x4_t v_kGYWeight = vdupq_n_u32(kGYWeight);
    uint32x4_t v_kBYWeight = vdupq_n_u32(kBYWeight);
    uint32x4_t y[4];

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
      y[i] = vdupq_n_u32(kHalfShift + kShifted16);
      y[i] = vmlaq_u32(y[i], r[i], v_kRYWeight);
      y[i] = vmlaq_u32(y[i], g[i], v_kGYWeight);
      y[i] = vmlaq_u32(y[i], b[i], v_kBYWeight);
    }

    return normalize_and_pack_y(y);
  }

  static void rgb_to_uv(uint8_t r, uint8_t g, uint8_t b, uint8_t uv[2]) {
    const int kHalfShift = (1 << (kWeightScale - 1));
    const int kShifted128 = (128 << kWeightScale);
    int uu = kRUWeight * r + kGUWeight * g + kBUWeight * b + kHalfShift +
             kShifted128;
    int vv = kBUWeight * r + kGVWeight * g + kBVWeight * b + kHalfShift +
             kShifted128;

    uv[0] = std::clamp(uu >> kWeightScale, 0, 0xff);
    uv[1] = std::clamp(vv >> kWeightScale, 0, 0xff);
  }

  static uint8x16_t compute_u_or_v_2x(const int32x4_t r[4],
                                      const int32x4_t g[4],
                                      const int32x4_t b[4], const int r_coeff,
                                      const int g_coeff, const int b_coeff) {
    // Constants for U/V calculation
    const int kHalfShift = (1 << (kWeightScale - 1));
    const int kShifted128 = (128 << kWeightScale);

    int32x4_t v_r_coeff = vdupq_n_s32(r_coeff);
    int32x4_t v_g_coeff = vdupq_n_s32(g_coeff);
    int32x4_t v_b_coeff = vdupq_n_s32(b_coeff);
    int32x4_t uv[4];

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
      uv[i] = vdupq_n_s32(kHalfShift + kShifted128);
      uv[i] = vmlaq_s32(uv[i], r[i], v_r_coeff);
      uv[i] = vmlaq_s32(uv[i], g[i], v_g_coeff);
      uv[i] = vmlaq_s32(uv[i], b[i], v_b_coeff);
    }

    return normalize_and_pack_u_or_v(uv);
  }

  static void rgb_to_uv_2x(const int32x4_t r[4], const int32x4_t g[4],
                           const int32x4_t b[4], uint8x16_t &u, uint8x16_t &v) {
    // ---------------- U (Cb) Component ----------------
    // U = R * kRU + G * kGU + B * kBU + bias
    u = compute_u_or_v_2x(r, g, b, kRUWeight, kGUWeight, kBUWeight);

    // ---------------- V (Cr) Component ----------------
    // V = R * kBU + G * kGV + B * kBV + bias
    v = compute_u_or_v_2x(r, g, b, kBUWeight, kGVWeight, kBVWeight);
  }

  static uint8x16_t normalize_and_pack_y(uint32x4_t vec[4]) {
    // The y_index table selects the correct output order after normalization.
    // When we load and separate the RGB values for UV calculation, we
    // deinterleave them into even and odd components. As a result, the
    // processed values are stored in two separate vectors. During
    // normalization, we need to interleave them again to produce the final
    // contiguous output, and this index pattern achieves that.
    uint8x16_t y_index = {1, 17, 3,  19, 5,  21, 7,  23,
                          9, 25, 11, 27, 13, 29, 15, 31};

    // Normalize down by right-shifting the fixed-point result
    // vshrn_n can only shift by an immediate value between 1 and 16.
    // Since kWeightScale is 20, we use (kWeightScale - 8) to shift down to 12
    // bits. This ensures that the most relevant 8-bit result lies in the second
    // byte of each 16-bit element. As a result, the lookup tables  are
    // constructed with only odd indices to extract the second byte from each
    // element.
    uint16x4_t tmp_lo_lo = vshrn_n_u32(vec[0], kWeightScale - 8);
    uint16x8_t tmp_lo_hi =
        vshrn_high_n_u32(tmp_lo_lo, vec[2], kWeightScale - 8);
    uint16x4_t tmp_hi_lo = vshrn_n_u32(vec[1], kWeightScale - 8);
    uint16x8_t tmp_hi_hi =
        vshrn_high_n_u32(tmp_hi_lo, vec[3], kWeightScale - 8);

    uint8x16x2_t tmp;
    tmp.val[0] = vreinterpretq_u8(tmp_lo_hi);  // 0, 2, 4, 6, 8, 10, 12, 14
    tmp.val[1] = vreinterpretq_u8(tmp_hi_hi);  // 1, 3, 5, 7, 9, 11, 13, 15

    uint8x16_t output = vqtbl2q_u8(tmp, y_index);

    return output;
  }

  static uint8x16_t normalize_and_pack_u_or_v(int32x4_t vec[4]) {
    // The uv_index table is used to finalize the order of U and V values.
    // Unlike the Y component, we don't need to interleave even and odd elements
    // manually. This is because the first vector already contains even-indexed
    // values from the lower RGB block, and the second vector contains
    // even-indexed values from the higher RGB block. As a result, the values
    // are already sorted in the correct order for output.
    uint8x16_t uv_index = {1,  3,  5,  7,  9,  11, 13, 15,
                           17, 19, 21, 23, 25, 27, 29, 31};

    // Normalize down by right-shifting the fixed-point result
    // vshrn_n can only shift by an immediate value between 1 and 16.
    // Since kWeightScale is 20, we use (kWeightScale - 8) to shift down to 12
    // bits. This ensures that the most relevant 8-bit result lies in the second
    // byte of each 16-bit element. As a result, the lookup tables  are
    // constructed with only odd indices to extract the second byte from each
    // element.
    int16x4_t tmp_lo_lo = vshrn_n_s32(vec[0], kWeightScale - 8);
    int16x8_t tmp_lo_hi = vshrn_high_n_s32(tmp_lo_lo, vec[1], kWeightScale - 8);
    int16x4_t tmp_hi_lo = vshrn_n_s32(vec[2], kWeightScale - 8);
    int16x8_t tmp_hi_hi = vshrn_high_n_s32(tmp_hi_lo, vec[3], kWeightScale - 8);

    uint8x16x2_t tmp;
    tmp.val[0] = vreinterpretq_u8(
        tmp_lo_hi);  // 0, 2, 4, 6, 8, 10, 12, 14 for the first vector
    tmp.val[1] = vreinterpretq_u8(
        tmp_hi_hi);  // 0, 2, 4, 6, 8, 10, 12, 14 for the second vector
    uint8x16_t output = vqtbl2q_u8(tmp, uv_index);

    return output;
  }

  static void load_rgb_2x(uint32x4_t r0[4], uint32x4_t g0[4], uint32x4_t b0[4],
                          uint32x4_t r1[4], uint32x4_t g1[4], uint32x4_t b1[4],
                          const uint8_t *src_row, const size_t index) {
    uint8x16_t tmp_b0, tmp_b1, tmp_g0, tmp_g1, tmp_r0, tmp_r1;
    // Load 32 pixels: two vectors of interleaved channels

    if constexpr (kAlpha) {
      // 4-channel input (RGBA or BGRA)
      uint8x16x4_t vsrc0 = vld4q_u8(src_row + scn * index);
      uint8x16x4_t vsrc1 =
          vld4q_u8(src_row + scn * index + scn * kVectorLength);

      tmp_b0 = vsrc0.val[b_index_];
      tmp_g0 = vsrc0.val[g_index_];
      tmp_r0 = vsrc0.val[r_index_];

      tmp_b1 = vsrc1.val[b_index_];
      tmp_g1 = vsrc1.val[g_index_];
      tmp_r1 = vsrc1.val[r_index_];
    } else {
      // 3-channel input (RGB or BGR)
      uint8x16x3_t vsrc0 = vld3q_u8(src_row + scn * index);
      uint8x16x3_t vsrc1 =
          vld3q_u8(src_row + scn * index + scn * kVectorLength);

      tmp_b0 = vsrc0.val[b_index_];
      tmp_g0 = vsrc0.val[g_index_];
      tmp_r0 = vsrc0.val[r_index_];

      tmp_b1 = vsrc1.val[b_index_];
      tmp_g1 = vsrc1.val[g_index_];
      tmp_r1 = vsrc1.val[r_index_];
    }
    // After loading the vector, we extend the channels and separate even and
    // odd elements. This separation is important for UV calculation, as only
    // the even-indexed values are used.
    uint8x16_t indices[4] = {
        0,    0xff, 0xff, 0xff, 2,    0xff, 0xff, 0xff, 4,    0xff, 0xff,
        0xff, 6,    0xff, 0xff, 0xff, 1,    0xff, 0xff, 0xff, 3,    0xff,
        0xff, 0xff, 5,    0xff, 0xff, 0xff, 7,    0xff, 0xff, 0xff, 8,
        0xff, 0xff, 0xff, 10,   0xff, 0xff, 0xff, 12,   0xff, 0xff, 0xff,
        14,   0xff, 0xff, 0xff, 9,    0xff, 0xff, 0xff, 11,   0xff, 0xff,
        0xff, 13,   0xff, 0xff, 0xff, 15,   0xff, 0xff, 0xff};

    // Expand each 8-bit channel into 32-bit vectors using table lookup and
    // reinterpret
    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
      r0[i] = vreinterpretq_u32_u8(vqtbl1q_u8(tmp_r0, indices[i]));
      g0[i] = vreinterpretq_u32_u8(vqtbl1q_u8(tmp_g0, indices[i]));
      b0[i] = vreinterpretq_u32_u8(vqtbl1q_u8(tmp_b0, indices[i]));
      r1[i] = vreinterpretq_u32_u8(vqtbl1q_u8(tmp_r1, indices[i]));
      g1[i] = vreinterpretq_u32_u8(vqtbl1q_u8(tmp_g1, indices[i]));
      b1[i] = vreinterpretq_u32_u8(vqtbl1q_u8(tmp_b1, indices[i]));
    }
  }

  static constexpr size_t r_index_ = RGB ? 0 : 2;
  static constexpr size_t g_index_ = 1;
  static constexpr size_t b_index_ = RGB ? 2 : 0;
  static constexpr size_t scn = kAlpha ? 4 : 3;
};

}  // namespace kleidicv::neon

#endif  // KLEIDICV_RGB_TO_YUV420_H
