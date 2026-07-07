// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/analysis/standalone_lucas_kanade_alg.h"
#include "kleidicv/neon.h"
#include "standalone_lucas_kanade_alg_common.h"

namespace kleidicv::neon {

// NEON implementation of the Lucas-Kanade primitives consumed by
// LucasKanadeLevelTracker.
class StandaloneLucasKanadeAlg {
 public:
  KLEIDICV_FORCE_INLINE
  static void sample_patch_and_gradients(
      int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
      ptrdiff_t prev_data_stride, const int16_t *scharr_data,
      ptrdiff_t scharr_stride_elements, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_scharr_xx, float &sum_scharr_xy, float &sum_scharr_yy) {
    // Bilinear-resample precomputed Scharr derivatives into a window at the
    // same subpixel position, writing Ix/Iy pairs to `scharr_window` and
    // accumulating Ix^2, Ix*Iy, Iy^2 for the structure tensor used later in
    // the LK solve.
    sum_scharr_xx = 0;
    sum_scharr_xy = 0;
    sum_scharr_yy = 0;

    int32x4_t sum_scharr_xx_v = vdupq_n_s32(0),
              sum_scharr_xy_v = vdupq_n_s32(0),
              sum_scharr_yy_v = vdupq_n_s32(0);

    const int window_width_times_channels = window_width * channels;
    const int16x4_t leftover_mask =
        make_leftover_accumulate_mask_s16(window_width_times_channels);
    const bool use_vector_tail = window_width_times_channels > 4;
    const int16x8_t coeff_tl_v8 = vdupq_n_s16(coeff_tl);
    const int16x8_t coeff_tr_v8 = vdupq_n_s16(coeff_tr);
    const int16x8_t coeff_bl_v8 = vdupq_n_s16(coeff_bl);
    const int16x8_t coeff_br_v8 = vdupq_n_s16(coeff_br);
    const int16x4_t coeff_tl_v4 = vdup_n_s16(coeff_tl);
    const int16x4_t coeff_tr_v4 = vdup_n_s16(coeff_tr);
    const int16x4_t coeff_bl_v4 = vdup_n_s16(coeff_bl);
    const int16x4_t coeff_br_v4 = vdup_n_s16(coeff_br);

    for (int y = 0; y < window_height; y++) {
      const uint8_t *const prev_row0 =
          prev_data + (y + window_corner_y) * prev_data_stride +
          static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *const prev_row1 = prev_row0 + prev_data_stride;
      int16_t *const window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;

      const int16_t *const scharr_row0 =
          scharr_data + (y + window_corner_y) * scharr_stride_elements +
          static_cast<ptrdiff_t>(window_corner_x) * channels * 2L;
      const int16_t *const scharr_row1 = scharr_row0 + scharr_stride_elements;
      int16_t *const scharr_window_row =
          scharr_window +
          static_cast<ptrdiff_t>(y) * window_width * channels * 2L;

      LoopUnroll2 loop{static_cast<size_t>(window_width_times_channels), 4};

      loop.unroll_twice([&](size_t index) {
        store_interpolated_8_pixels(
            window_row, prev_row0, prev_row1, static_cast<int>(index), channels,
            coeff_tl_v8, coeff_tr_v8, coeff_bl_v8, coeff_br_v8);
        accumulate_scharr8(scharr_row0, scharr_row1, scharr_window_row,
                           static_cast<int>(index), channels, coeff_tl_v8,
                           coeff_tr_v8, coeff_bl_v8, coeff_br_v8,
                           sum_scharr_xx_v, sum_scharr_xy_v, sum_scharr_yy_v);
      });

      loop.unroll_once([&](size_t index) {
        store_interpolated_4_pixels(
            window_row, prev_row0, prev_row1, static_cast<int>(index), channels,
            coeff_tl_v4, coeff_tr_v4, coeff_bl_v4, coeff_br_v4);
        accumulate_scharr4(scharr_row0, scharr_row1, scharr_window_row,
                           static_cast<int>(index), channels, coeff_tl_v4,
                           coeff_tr_v4, coeff_bl_v4, coeff_br_v4,
                           sum_scharr_xx_v, sum_scharr_xy_v, sum_scharr_yy_v);
      });

      loop.remaining([&](size_t index, size_t end) {
        if (use_vector_tail) {
          accumulate_scharr4_tail(scharr_row0, scharr_row1, scharr_window_row,
                                  window_width_times_channels, channels,
                                  coeff_tl_v4, coeff_tr_v4, coeff_bl_v4,
                                  coeff_br_v4, leftover_mask, sum_scharr_xx_v,
                                  sum_scharr_xy_v, sum_scharr_yy_v);
          store_interpolated_4_pixels(
              window_row, prev_row0, prev_row1, window_width_times_channels - 4,
              channels, coeff_tl_v4, coeff_tr_v4, coeff_bl_v4, coeff_br_v4);

        } else {
          for (size_t x = index; x < end; ++x) {
            accumulate_scharr_scalar(
                scharr_window_row, scharr_row0, scharr_row1,
                static_cast<int>(x), channels, coeff_tl, coeff_tr, coeff_bl,
                coeff_br, sum_scharr_xx, sum_scharr_xy, sum_scharr_yy);
            store_interpolated_pixel(window_row, prev_row0, prev_row1,
                                     static_cast<int>(x), channels, coeff_tl,
                                     coeff_tr, coeff_bl, coeff_br);
          }
        }
      });
    }

    sum_scharr_xx += vaddvq_f32(vcvtq_f32_s32(sum_scharr_xx_v));
    sum_scharr_xy += vaddvq_f32(vcvtq_f32_s32(sum_scharr_xy_v));
    sum_scharr_yy += vaddvq_f32(vcvtq_f32_s32(sum_scharr_yy_v));
    sum_scharr_xx *= kLucasKanadeAlgFixedPointDescale;
    sum_scharr_xy *= kLucasKanadeAlgFixedPointDescale;
    sum_scharr_yy *= kLucasKanadeAlgFixedPointDescale;
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_mismatch_vector(
      const uint8_t *next_data, ptrdiff_t next_stride, const int16_t *window,
      const int16_t *scharr_window, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_diff_scharr_x, float &sum_diff_scharr_y) {
    sum_diff_scharr_x = 0;
    sum_diff_scharr_y = 0;

    int32x4_t sum_diff_scharr_x_v = vdupq_n_s32(0),
              sum_diff_scharr_y_v = vdupq_n_s32(0);

    const int16x4_t coeff_tl_v = vdup_n_s16(coeff_tl);
    const int16x4_t coeff_tr_v = vdup_n_s16(coeff_tr);
    const int16x4_t coeff_bl_v = vdup_n_s16(coeff_bl);
    const int16x4_t coeff_br_v = vdup_n_s16(coeff_br);
    const int16x8_t coeff_tl_v8 = vdupq_n_s16(coeff_tl);
    const int16x8_t coeff_tr_v8 = vdupq_n_s16(coeff_tr);
    const int16x8_t coeff_bl_v8 = vdupq_n_s16(coeff_bl);
    const int16x8_t coeff_br_v8 = vdupq_n_s16(coeff_br);

    const int window_width_times_channels = window_width * channels;
    const bool use_vector_tail = window_width_times_channels > 4;
    const int32x4_t leftover_mask =
        make_leftover_accumulate_mask_s32(window_width_times_channels);

    for (int y = 0; y < window_height; y++) {
      const uint8_t *row0 = next_data + (y + window_corner_y) * next_stride +
                            static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *row1 = row0 + next_stride;
      const int16_t *window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;
      const int16_t *scharr_window_row =
          scharr_window +
          static_cast<ptrdiff_t>(y) * window_width * channels * 2L;

      LoopUnroll2 loop{static_cast<size_t>(window_width_times_channels), 4};

      loop.unroll_twice([&](size_t index) {
        accumulate_diff_scharr8(row0, row1, window_row, scharr_window_row,
                                static_cast<int>(index), channels, coeff_tl_v8,
                                coeff_tr_v8, coeff_bl_v8, coeff_br_v8,
                                sum_diff_scharr_x_v, sum_diff_scharr_y_v);
      });

      loop.unroll_once([&](size_t index) {
        accumulate_diff_scharr4(row0, row1, window_row, scharr_window_row,
                                static_cast<int>(index), channels, coeff_tl_v,
                                coeff_tr_v, coeff_bl_v, coeff_br_v,
                                sum_diff_scharr_x_v, sum_diff_scharr_y_v);
      });

      loop.remaining([&](size_t index, size_t end) {
        if (use_vector_tail) {
          accumulate_diff_scharr4_tail(
              row0, row1, window_row, scharr_window_row,
              window_width_times_channels, channels, coeff_tl_v, coeff_tr_v,
              coeff_bl_v, coeff_br_v, leftover_mask, sum_diff_scharr_x_v,
              sum_diff_scharr_y_v);
        } else {
          for (size_t x = index; x < end; ++x) {
            accumulate_diff_scharr_scalar(
                row0, row1, window_row, scharr_window_row, static_cast<int>(x),
                channels, coeff_tl, coeff_tr, coeff_bl, coeff_br,
                sum_diff_scharr_x, sum_diff_scharr_y);
          }
        }
      });
    }

    sum_diff_scharr_x += vaddvq_f32(vcvtq_f32_s32(sum_diff_scharr_x_v));
    sum_diff_scharr_y += vaddvq_f32(vcvtq_f32_s32(sum_diff_scharr_y_v));
    sum_diff_scharr_x *= kLucasKanadeAlgFixedPointDescale;
    sum_diff_scharr_y *= kLucasKanadeAlgFixedPointDescale;
  }

 private:
  // Helper routines for loading/interpolating patches and accumulating LK
  // matrices in vector-friendly chunks.
  // Like SVE's svld1ub_s16 but for Neon.
  // Loads 4 bytes and widens them to int16.
  KLEIDICV_FORCE_INLINE
  static int16x4_t vld1ub_s16(const uint8_t *ptr) {
    uint32_t a = 0;
    memcpy(&a, ptr, sizeof(a));
    uint32x2_t b = vset_lane_u32(a, vdup_n_s32(0), 0);
    uint8x8_t c = vreinterpret_u8_u32(b);
    return vget_low_s16(vmovl_u8(c));
  }

  KLEIDICV_FORCE_INLINE
  static int16x4_t make_leftover_accumulate_mask_s16(int width) {
    int count4 = ((width - 1) & 3) + 1;
    return int16x4_t{static_cast<int16_t>(count4 >= 4 ? 0xFFFF : 0),
                     static_cast<int16_t>(count4 >= 3 ? 0xFFFF : 0),
                     static_cast<int16_t>(count4 >= 2 ? 0xFFFF : 0),
                     static_cast<int16_t>(count4 >= 1 ? 0xFFFF : 0)};
  }

  KLEIDICV_FORCE_INLINE
  static int32x4_t make_leftover_accumulate_mask_s32(int width) {
    int count4 = (width & 3) | ((width & 7) == 4 ? 4 : 0);
    return int32x4_t{static_cast<int32_t>(count4 >= 4 ? -1 : 0),
                     static_cast<int32_t>(count4 >= 3 ? -1 : 0),
                     static_cast<int32_t>(count4 >= 2 ? -1 : 0),
                     static_cast<int32_t>(count4 >= 1 ? -1 : 0)};
  }

  KLEIDICV_FORCE_INLINE
  static void store_interpolated_pixel(int16_t *window_row,
                                       const uint8_t *prev_row0,
                                       const uint8_t *prev_row1, int x,
                                       int channels, int16_t coeff_tl,
                                       int16_t coeff_tr, int16_t coeff_bl,
                                       int16_t coeff_br) {
    window_row[x] =
        static_cast<int16_t>(round_fixed_point<kLucasKanadeAlgFractionBits - 5>(
            prev_row0[x] * coeff_tl + prev_row0[x + channels] * coeff_tr +
            prev_row1[x] * coeff_bl + prev_row1[x + channels] * coeff_br));
  }

  KLEIDICV_FORCE_INLINE
  static void store_interpolated_4_pixels(
      int16_t *window_row, const uint8_t *prev_row0, const uint8_t *prev_row1,
      int x, int channels, int16x4_t coeff_tl, int16x4_t coeff_tr,
      int16x4_t coeff_bl, int16x4_t coeff_br) {
    int16x4_t prev_tl = vld1ub_s16(prev_row0 + x);
    int16x4_t prev_tr = vld1ub_s16(prev_row0 + x + channels);
    int16x4_t prev_bl = vld1ub_s16(prev_row1 + x);
    int16x4_t prev_br = vld1ub_s16(prev_row1 + x + channels);

    int16x4_t prev = lerp<kLucasKanadeAlgFractionBits - 5>(
        prev_tl, prev_tr, prev_bl, prev_br, coeff_tl, coeff_tr, coeff_bl,
        coeff_br);
    vst1_s16(window_row + x, prev);
  }

  KLEIDICV_FORCE_INLINE
  static void store_interpolated_8_pixels(
      int16_t *window_row, const uint8_t *prev_row0, const uint8_t *prev_row1,
      int x, int channels, int16x8_t coeff_tl, int16x8_t coeff_tr,
      int16x8_t coeff_bl, int16x8_t coeff_br) {
    int16x8_t prev_tl = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prev_row0 + x)));
    int16x8_t prev_tr =
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prev_row0 + x + channels)));
    int16x8_t prev_bl = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prev_row1 + x)));
    int16x8_t prev_br =
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prev_row1 + x + channels)));

    int16x8_t prev = lerp<kLucasKanadeAlgFractionBits - 5>(
        prev_tl, prev_tr, prev_bl, prev_br, coeff_tl, coeff_tr, coeff_bl,
        coeff_br);

    vst1q_s16(window_row + x, prev);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_scharr_scalar(int16_t *scharr_window_row,
                                       const int16_t *row0, const int16_t *row1,
                                       int x, int channels, int16_t coeff_tl,
                                       int16_t coeff_tr, int16_t coeff_bl,
                                       int16_t coeff_br, float &sum_scharr_xx,
                                       float &sum_scharr_xy,
                                       float &sum_scharr_yy) {
    int scharr_x = round_fixed_point<kLucasKanadeAlgFractionBits>(
        row0[x * 2L] * coeff_tl + row0[(x + channels) * 2L] * coeff_tr +
        row1[x * 2L] * coeff_bl + row1[(x + channels) * 2L] * coeff_br);

    int scharr_y = round_fixed_point<kLucasKanadeAlgFractionBits>(
        row0[x * 2L + 1] * coeff_tl + row0[(x + channels) * 2L + 1] * coeff_tr +
        row1[x * 2L + 1] * coeff_bl + row1[(x + channels) * 2L + 1] * coeff_br);

    scharr_window_row[x * 2L] = static_cast<int16_t>(scharr_x);
    scharr_window_row[x * 2L + 1] = static_cast<int16_t>(scharr_y);

    sum_scharr_xx += static_cast<float>(scharr_x * scharr_x);
    sum_scharr_xy += static_cast<float>(scharr_x * scharr_y);
    sum_scharr_yy += static_cast<float>(scharr_y * scharr_y);
  }

  KLEIDICV_FORCE_INLINE
  static int16x8x2_t compute_scharr8(const int16_t *row0, const int16_t *row1,
                                     int16_t *scharr_window_row, int x,
                                     int channels, int16x8_t coeff_tl,
                                     int16x8_t coeff_tr, int16x8_t coeff_bl,
                                     int16x8_t coeff_br) {
    int16x8x2_t scharr_tl = vld2q_s16(row0 + x * 2L);
    int16x8x2_t scharr_tr = vld2q_s16(row0 + (x + channels) * 2L);
    int16x8x2_t scharr_bl = vld2q_s16(row1 + x * 2L);
    int16x8x2_t scharr_br = vld2q_s16(row1 + (x + channels) * 2L);

    int16x8_t scharr_x = lerp<kLucasKanadeAlgFractionBits>(
        scharr_tl.val[0], scharr_tr.val[0], scharr_bl.val[0], scharr_br.val[0],
        coeff_tl, coeff_tr, coeff_bl, coeff_br);

    int16x8_t scharr_y = lerp<kLucasKanadeAlgFractionBits>(
        scharr_tl.val[1], scharr_tr.val[1], scharr_bl.val[1], scharr_br.val[1],
        coeff_tl, coeff_tr, coeff_bl, coeff_br);

    int16x8x2_t scharr_xy{scharr_x, scharr_y};
    vst2q_s16(scharr_window_row + x * 2L, scharr_xy);
    return scharr_xy;
  }

  KLEIDICV_FORCE_INLINE
  static int16x4x2_t compute_scharr4(const int16_t *row0, const int16_t *row1,
                                     int16_t *scharr_window_row, int x,
                                     int channels, int16x4_t coeff_tl,
                                     int16x4_t coeff_tr, int16x4_t coeff_bl,
                                     int16x4_t coeff_br) {
    int16x4x2_t scharr_tl = vld2_s16(row0 + x * 2L);
    int16x4x2_t scharr_tr = vld2_s16(row0 + (x + channels) * 2L);
    int16x4x2_t scharr_bl = vld2_s16(row1 + x * 2L);
    int16x4x2_t scharr_br = vld2_s16(row1 + (x + channels) * 2L);

    int16x4_t scharr_x = lerp<kLucasKanadeAlgFractionBits>(
        scharr_tl.val[0], scharr_tr.val[0], scharr_bl.val[0], scharr_br.val[0],
        coeff_tl, coeff_tr, coeff_bl, coeff_br);

    int16x4_t scharr_y = lerp<kLucasKanadeAlgFractionBits>(
        scharr_tl.val[1], scharr_tr.val[1], scharr_bl.val[1], scharr_br.val[1],
        coeff_tl, coeff_tr, coeff_bl, coeff_br);

    int16x4x2_t scharr_xy{scharr_x, scharr_y};
    vst2_s16(scharr_window_row + x * 2L, scharr_xy);
    return scharr_xy;
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_scharr_vectors(int16x4_t scharr_x, int16x4_t scharr_y,
                                        int32x4_t &sum_scharr_xx_v,
                                        int32x4_t &sum_scharr_xy_v,
                                        int32x4_t &sum_scharr_yy_v) {
    sum_scharr_xx_v = vmlal_s16(sum_scharr_xx_v, scharr_x, scharr_x);
    sum_scharr_xy_v = vmlal_s16(sum_scharr_xy_v, scharr_x, scharr_y);
    sum_scharr_yy_v = vmlal_s16(sum_scharr_yy_v, scharr_y, scharr_y);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_scharr_vectors(int16x8_t scharr_x, int16x8_t scharr_y,
                                        int32x4_t &sum_scharr_xx_v,
                                        int32x4_t &sum_scharr_xy_v,
                                        int32x4_t &sum_scharr_yy_v) {
    accumulate_scharr_vectors(vget_low(scharr_x), vget_low(scharr_y),
                              sum_scharr_xx_v, sum_scharr_xy_v,
                              sum_scharr_yy_v);

    sum_scharr_xx_v = vmlal_high_s16(sum_scharr_xx_v, scharr_x, scharr_x);
    sum_scharr_xy_v = vmlal_high_s16(sum_scharr_xy_v, scharr_x, scharr_y);
    sum_scharr_yy_v = vmlal_high_s16(sum_scharr_yy_v, scharr_y, scharr_y);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_scharr8(const int16_t *row0, const int16_t *row1,
                                 int16_t *scharr_window_row, int x,
                                 int channels, int16x8_t coeff_tl,
                                 int16x8_t coeff_tr, int16x8_t coeff_bl,
                                 int16x8_t coeff_br, int32x4_t &sum_scharr_xx_v,
                                 int32x4_t &sum_scharr_xy_v,
                                 int32x4_t &sum_scharr_yy_v) {
    int16x8x2_t scharr_xy =
        compute_scharr8(row0, row1, scharr_window_row, x, channels, coeff_tl,
                        coeff_tr, coeff_bl, coeff_br);

    accumulate_scharr_vectors(scharr_xy.val[0], scharr_xy.val[1],
                              sum_scharr_xx_v, sum_scharr_xy_v,
                              sum_scharr_yy_v);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_scharr4(const int16_t *row0, const int16_t *row1,
                                 int16_t *scharr_window_row, int x,
                                 int channels, int16x4_t coeff_tl,
                                 int16x4_t coeff_tr, int16x4_t coeff_bl,
                                 int16x4_t coeff_br, int32x4_t &sum_scharr_xx_v,
                                 int32x4_t &sum_scharr_xy_v,
                                 int32x4_t &sum_scharr_yy_v) {
    int16x4x2_t scharr_xy =
        compute_scharr4(row0, row1, scharr_window_row, x, channels, coeff_tl,
                        coeff_tr, coeff_bl, coeff_br);
    accumulate_scharr_vectors(scharr_xy.val[0], scharr_xy.val[1],
                              sum_scharr_xx_v, sum_scharr_xy_v,
                              sum_scharr_yy_v);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_scharr4_tail(
      const int16_t *row0, const int16_t *row1, int16_t *scharr_window_row,
      int window_width_times_channels, int channels, int16x4_t coeff_tl,
      int16x4_t coeff_tr, int16x4_t coeff_bl, int16x4_t coeff_br,
      int16x4_t leftover_mask, int32x4_t &sum_scharr_xx_v,
      int32x4_t &sum_scharr_xy_v, int32x4_t &sum_scharr_yy_v) {
    int16x4x2_t scharr_xy = compute_scharr4(
        row0, row1, scharr_window_row, window_width_times_channels - 4,
        channels, coeff_tl, coeff_tr, coeff_bl, coeff_br);
    int16x4_t masked_x = vand_s16(scharr_xy.val[0], leftover_mask);
    int16x4_t masked_y = vand_s16(scharr_xy.val[1], leftover_mask);
    accumulate_scharr_vectors(masked_x, masked_y, sum_scharr_xx_v,
                              sum_scharr_xy_v, sum_scharr_yy_v);
  }

  KLEIDICV_FORCE_INLINE
  static void compute_diff_and_scharr4(
      const uint8_t *row0, const uint8_t *row1, const int16_t *window_row,
      const int16_t *scharr_window_row, int x, int channels,
      int16x4_t coeff_tl_v, int16x4_t coeff_tr_v, int16x4_t coeff_bl_v,
      int16x4_t coeff_br_v, int16x4_t &diff, int16x4_t &scharr_x,
      int16x4_t &scharr_y) {
    int16x4_t tl = vld1ub_s16(row0 + x);
    int16x4_t tr = vld1ub_s16(row0 + x + channels);
    int16x4_t bl = vld1ub_s16(row1 + x);
    int16x4_t br = vld1ub_s16(row1 + x + channels);

    int16x4_t next = lerp<kLucasKanadeAlgFractionBits - 5>(
        tl, tr, bl, br, coeff_tl_v, coeff_tr_v, coeff_bl_v, coeff_br_v);

    diff = vsub_s16(next, vld1_s16(window_row + x));

    int16x4x2_t scharr_xy = vld2_s16(scharr_window_row + x * 2L);
    scharr_x = scharr_xy.val[0];
    scharr_y = scharr_xy.val[1];
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_diff_scharr4(
      const uint8_t *row0, const uint8_t *row1, const int16_t *window_row,
      const int16_t *scharr_window_row, int x, int channels,
      int16x4_t coeff_tl_v, int16x4_t coeff_tr_v, int16x4_t coeff_bl_v,
      int16x4_t coeff_br_v, int32x4_t &sum_diff_scharr_x_v,
      int32x4_t &sum_diff_scharr_y_v) {
    int16x4_t diff;
    int16x4_t scharr_x;
    int16x4_t scharr_y;
    compute_diff_and_scharr4(row0, row1, window_row, scharr_window_row, x,
                             channels, coeff_tl_v, coeff_tr_v, coeff_bl_v,
                             coeff_br_v, diff, scharr_x, scharr_y);

    sum_diff_scharr_x_v = vmlal_s16(sum_diff_scharr_x_v, scharr_x, diff);
    sum_diff_scharr_y_v = vmlal_s16(sum_diff_scharr_y_v, scharr_y, diff);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_diff_scharr4_tail(
      const uint8_t *row0, const uint8_t *row1, const int16_t *window_row,
      const int16_t *scharr_window_row, int window_width_times_channels,
      int channels, int16x4_t coeff_tl_v, int16x4_t coeff_tr_v,
      int16x4_t coeff_bl_v, int16x4_t coeff_br_v, int32x4_t leftover_mask,
      int32x4_t &sum_diff_scharr_x_v, int32x4_t &sum_diff_scharr_y_v) {
    int x = window_width_times_channels - 4;
    int16x4_t diff;
    int16x4_t scharr_x;
    int16x4_t scharr_y;
    compute_diff_and_scharr4(row0, row1, window_row, scharr_window_row, x,
                             channels, coeff_tl_v, coeff_tr_v, coeff_bl_v,
                             coeff_br_v, diff, scharr_x, scharr_y);

    int32x4_t diff_scharr_x = vmull_s16(scharr_x, diff);
    int32x4_t diff_scharr_y = vmull_s16(scharr_y, diff);
    int32x4_t masked_x = vandq_s32(diff_scharr_x, leftover_mask);
    int32x4_t masked_y = vandq_s32(diff_scharr_y, leftover_mask);
    sum_diff_scharr_x_v = vaddq_s32(sum_diff_scharr_x_v, masked_x);
    sum_diff_scharr_y_v = vaddq_s32(sum_diff_scharr_y_v, masked_y);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_diff_scharr_scalar(
      const uint8_t *row0, const uint8_t *row1, const int16_t *window_row,
      const int16_t *scharr_window_row, int x, int channels, int16_t coeff_tl,
      int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_diff_scharr_x, float &sum_diff_scharr_y) {
    int next = round_fixed_point<kLucasKanadeAlgFractionBits - 5>(
        row0[x] * coeff_tl + row0[x + channels] * coeff_tr +
        row1[x] * coeff_bl + row1[x + channels] * coeff_br);
    int diff = next - window_row[x];
    sum_diff_scharr_x += static_cast<float>(diff * scharr_window_row[x * 2L]);
    sum_diff_scharr_y +=
        static_cast<float>(diff * scharr_window_row[x * 2L + 1]);
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_diff_scharr8(
      const uint8_t *row0, const uint8_t *row1, const int16_t *window_row,
      const int16_t *scharr_window_row, int x, int channels, int16x8_t coeff_tl,
      int16x8_t coeff_tr, int16x8_t coeff_bl, int16x8_t coeff_br,
      int32x4_t &sum_diff_scharr_x_v, int32x4_t &sum_diff_scharr_y_v) {
    uint8x8_t tl = vld1_u8(row0 + x);
    uint8x8_t tr = vld1_u8(row0 + x + channels);
    uint8x8_t bl = vld1_u8(row1 + x);
    uint8x8_t br = vld1_u8(row1 + x + channels);

    int16x8_t tl_wide = vreinterpretq_s16_u16(vmovl_u8(tl));
    int16x8_t tr_wide = vreinterpretq_s16_u16(vmovl_u8(tr));
    int16x8_t bl_wide = vreinterpretq_s16_u16(vmovl_u8(bl));
    int16x8_t br_wide = vreinterpretq_s16_u16(vmovl_u8(br));

    int16x8_t next = lerp<kLucasKanadeAlgFractionBits - 5>(
        tl_wide, tr_wide, bl_wide, br_wide, coeff_tl, coeff_tr, coeff_bl,
        coeff_br);

    int16x8_t diff = vsubq_s16(next, vld1q_s16(window_row + x));

    int16x8x2_t scharr_xy = vld2q_s16(scharr_window_row + x * 2L);
    int16x8_t scharr_x = scharr_xy.val[0];
    int16x8_t scharr_y = scharr_xy.val[1];

    sum_diff_scharr_x_v = vmlal_s16(sum_diff_scharr_x_v, vget_low_s16(scharr_x),
                                    vget_low_s16(diff));
    sum_diff_scharr_x_v = vmlal_high_s16(sum_diff_scharr_x_v, scharr_x, diff);

    sum_diff_scharr_y_v = vmlal_s16(sum_diff_scharr_y_v, vget_low_s16(scharr_y),
                                    vget_low_s16(diff));
    sum_diff_scharr_y_v = vmlal_high_s16(sum_diff_scharr_y_v, scharr_y, diff);
  }

  template <int shift>
  KLEIDICV_FORCE_INLINE static int16x4_t lerp(int16x4_t tl, int16x4_t tr,
                                              int16x4_t bl, int16x4_t br,
                                              int16x4_t coeff_tl,
                                              int16x4_t coeff_tr,
                                              int16x4_t coeff_bl,
                                              int16x4_t coeff_br) {
    int32x4_t accumulator = vmull_s16(tl, coeff_tl);
    accumulator = vmlal_s16(accumulator, tr, coeff_tr);
    accumulator = vmlal_s16(accumulator, bl, coeff_bl);
    accumulator = vmlal_s16(accumulator, br, coeff_br);
    return vqrshrn_n_s32(accumulator, shift);
  }

  template <int shift>
  KLEIDICV_FORCE_INLINE static int16x8_t lerp(int16x8_t tl, int16x8_t tr,
                                              int16x8_t bl, int16x8_t br,
                                              int16x8_t coeff_tl,
                                              int16x8_t coeff_tr,
                                              int16x8_t coeff_bl,
                                              int16x8_t coeff_br) {
    int16x4_t accumulator_lo =
        lerp<shift>(vget_low(tl), vget_low(tr), vget_low(bl), vget_low(br),
                    vget_low(coeff_tl), vget_low(coeff_tr), vget_low(coeff_bl),
                    vget_low(coeff_br));

    int32x4_t accumulator_hi = vmull_high_s16(tl, coeff_tl);
    accumulator_hi = vmlal_high_s16(accumulator_hi, tr, coeff_tr);
    accumulator_hi = vmlal_high_s16(accumulator_hi, bl, coeff_bl);
    accumulator_hi = vmlal_high_s16(accumulator_hi, br, coeff_br);

    return vqrshrn_high_n_s32(accumulator_lo, accumulator_hi, shift);
  }
};

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold) {
  if (kleidicv_error_t validation_error =
          validate_standalone_lucas_kanade_alg_u8_args(
              prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
              next_data, next_data_stride, width, height, channels, prev_points,
              next_points, point_count, window_width, window_height)) {
    return validation_error;
  }

  auto window_buffer_or_error =
      LucasKanadePatchBuffer<>::create(window_width, window_height, channels);
  if (!std::holds_alternative<LucasKanadePatchBuffer<>>(
          window_buffer_or_error)) {
    return std::get<kleidicv_error_t>(window_buffer_or_error);
  }
  auto &window_buffer =
      std::get<LucasKanadePatchBuffer<>>(window_buffer_or_error);
  return LucasKanadeLevelTracker<StandaloneLucasKanadeAlg>::compute(
      window_buffer.window(), window_buffer.deriv_window(), prev_data,
      prev_data_stride, prev_deriv_data, prev_deriv_stride, next_data,
      next_data_stride, width, height, channels, prev_points, next_points,
      point_count, status, err, window_width, window_height, termination_count,
      termination_epsilon, get_min_eigen_vals, min_eigen_vals_threshold);
}

}  // namespace kleidicv::neon
