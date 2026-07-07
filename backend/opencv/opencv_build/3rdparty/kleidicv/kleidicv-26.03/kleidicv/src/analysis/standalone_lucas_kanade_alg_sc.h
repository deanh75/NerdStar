// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_SC_H
#define KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_SC_H

#include <functional>
#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "standalone_lucas_kanade_alg_common.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Scalable vector implementation of the Lucas-Kanade primitives consumed by
// LucasKanadeLevelTracker.
struct StandaloneLucasKanadeAlg {
  struct Lerp_coefficients {
    int16_t coeff_tl;
    int16_t coeff_tr;
    int16_t coeff_bl;
    int16_t coeff_br;

    static KLEIDICV_FORCE_INLINE Lerp_coefficients
    create(int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl,
           int16_t coeff_br) KLEIDICV_STREAMING {
      return {coeff_tl, coeff_tr, coeff_bl, coeff_br};
    }
  };

  template <uint64_t kShift>
  static KLEIDICV_FORCE_INLINE svint16_t
  lerp(svint16_t tl, svint16_t tr, svint16_t bl, svint16_t br,
       const Lerp_coefficients &coeffs) KLEIDICV_STREAMING {
    svint32_t b = svmullb_n_s32(tl, coeffs.coeff_tl);
    svint32_t t = svmullt_n_s32(tl, coeffs.coeff_tl);

    b = svmlalb_n_s32(b, tr, coeffs.coeff_tr);
    t = svmlalt_n_s32(t, tr, coeffs.coeff_tr);

    b = svmlalb_n_s32(b, bl, coeffs.coeff_bl);
    t = svmlalt_n_s32(t, bl, coeffs.coeff_bl);

    b = svmlalb_n_s32(b, br, coeffs.coeff_br);
    t = svmlalt_n_s32(t, br, coeffs.coeff_br);

    svint16_t result = svqrshrnb_n_s32(b, kShift);
    result = svqrshrnt_n_s32(result, t, kShift);
    return result;
  }

  static KLEIDICV_FORCE_INLINE void sample_patch_and_gradients(
      int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
      ptrdiff_t prev_data_stride, const int16_t *scharr_data,
      ptrdiff_t scharr_stride_elements, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_scharr_xx, float &sum_scharr_xy,
      float &sum_scharr_yy) KLEIDICV_STREAMING {
    const Lerp_coefficients lerp_coeffs =
        Lerp_coefficients::create(coeff_tl, coeff_tr, coeff_bl, coeff_br);

    svint32_t sum_scharr_xx_v = svdup_n_s32(0),
              sum_scharr_xy_v = svdup_n_s32(0),
              sum_scharr_yy_v = svdup_n_s32(0);

    for (int y = 0; y < window_height; y++) {
      const int16_t *const scharr_row0 =
          scharr_data + (y + window_corner_y) * scharr_stride_elements +
          window_corner_x * 2L * channels;
      const int16_t *const scharr_row1 = scharr_row0 + scharr_stride_elements;
      int16_t *const scharr_window_row =
          scharr_window +
          static_cast<ptrdiff_t>(y) * window_width * channels * 2L;

      const uint8_t *const prev_row0 =
          prev_data + (y + window_corner_y) * prev_data_stride +
          static_cast<ptrdiff_t>(window_corner_x) * channels;

      const uint8_t *const prev_row1 = prev_row0 + prev_data_stride;

      int16_t *const window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;

      const size_t window_width_times_channels =
          static_cast<size_t>(window_width) * static_cast<size_t>(channels);

      LoopUnroll2 loop{window_width_times_channels, svcnth()};

      auto calc_scharr_x_scharr_y = [&](size_t index,
                                        svbool_t pg16) KLEIDICV_STREAMING {
        svint16_t prev_tl = svld1ub_s16(pg16, prev_row0 + index);
        svint16_t prev_tr = svld1ub_s16(pg16, prev_row0 + index + channels);
        svint16_t prev_bl = svld1ub_s16(pg16, prev_row1 + index);
        svint16_t prev_br = svld1ub_s16(pg16, prev_row1 + index + channels);

        svint16_t prev = lerp<kLucasKanadeAlgFractionBits - 5>(
            prev_tl, prev_tr, prev_bl, prev_br, lerp_coeffs);
        svst1_s16(pg16, window_row + index, prev);

        svint16x2_t scharr_tl = svld2_s16(pg16, scharr_row0 + index * 2L);
        svint16x2_t scharr_tr =
            svld2_s16(pg16, scharr_row0 + (index + channels) * 2L);
        svint16x2_t scharr_bl = svld2_s16(pg16, scharr_row1 + index * 2L);
        svint16x2_t scharr_br =
            svld2_s16(pg16, scharr_row1 + (index + channels) * 2L);

        svint16_t scharr_x = lerp<kLucasKanadeAlgFractionBits>(
            svget2(scharr_tl, 0), svget2(scharr_tr, 0), svget2(scharr_bl, 0),
            svget2(scharr_br, 0), lerp_coeffs);

        svint16_t scharr_y = lerp<kLucasKanadeAlgFractionBits>(
            svget2(scharr_tl, 1), svget2(scharr_tr, 1), svget2(scharr_bl, 1),
            svget2(scharr_br, 1), lerp_coeffs);

        svst2_s16(pg16, scharr_window_row + index * 2L,
                  svcreate2_s16(scharr_x, scharr_y));

#if KLEIDICV_TARGET_SME2
        sum_scharr_xx_v = svdot_s32_s16(sum_scharr_xx_v, scharr_x, scharr_x);
        sum_scharr_xy_v = svdot_s32_s16(sum_scharr_xy_v, scharr_x, scharr_y);
        sum_scharr_yy_v = svdot_s32_s16(sum_scharr_yy_v, scharr_y, scharr_y);
#else
        sum_scharr_xx_v = svmlalb_s32(sum_scharr_xx_v, scharr_x, scharr_x);
        sum_scharr_xx_v = svmlalt_s32(sum_scharr_xx_v, scharr_x, scharr_x);

        sum_scharr_xy_v = svmlalb_s32(sum_scharr_xy_v, scharr_x, scharr_y);
        sum_scharr_xy_v = svmlalt_s32(sum_scharr_xy_v, scharr_x, scharr_y);

        sum_scharr_yy_v = svmlalb_s32(sum_scharr_yy_v, scharr_y, scharr_y);
        sum_scharr_yy_v = svmlalt_s32(sum_scharr_yy_v, scharr_y, scharr_y);
#endif
      };

      loop.unroll_once([&](size_t index) KLEIDICV_STREAMING {
        svbool_t pg16 = svptrue_b16();
        calc_scharr_x_scharr_y(index, pg16);
      });

      loop.remaining([&](size_t index, size_t end) KLEIDICV_STREAMING {
        svbool_t pg16 = svwhilelt_b16_u64(index, end);
        calc_scharr_x_scharr_y(index, pg16);
      });
    }
    sum_scharr_xx =
        svaddv_f32(svptrue_b32(),
                   svcvt_f32_s32_x(svptrue_b32(), sum_scharr_xx_v)) *
        kLucasKanadeAlgFixedPointDescale;
    sum_scharr_xy =
        svaddv_f32(svptrue_b32(),
                   svcvt_f32_s32_x(svptrue_b32(), sum_scharr_xy_v)) *
        kLucasKanadeAlgFixedPointDescale;
    sum_scharr_yy =
        svaddv_f32(svptrue_b32(),
                   svcvt_f32_s32_x(svptrue_b32(), sum_scharr_yy_v)) *
        kLucasKanadeAlgFixedPointDescale;
  }

  // Accumulates mismatch vector (sum diff*Ix, sum diff*Iy) between the next
  // frame and the sampled window from the previous frame.
  static KLEIDICV_FORCE_INLINE void accumulate_mismatch_vector(
      const uint8_t *next_data, ptrdiff_t next_stride, const int16_t *window,
      const int16_t *scharr_window, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_diff_scharr_x, float &sum_diff_scharr_y) KLEIDICV_STREAMING {
    const Lerp_coefficients lerp_coeffs =
        Lerp_coefficients::create(coeff_tl, coeff_tr, coeff_bl, coeff_br);
    svint32_t sum_diff_scharr_x_v = svdup_n_s32(0),
              sum_diff_scharr_y_v = svdup_n_s32(0);

    for (int y = 0; y < window_height; y++) {
      const uint8_t *next_row0 =
          next_data + (y + window_corner_y) * next_stride +
          static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *next_row1 = next_row0 + next_stride;
      const int16_t *window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;
      const int16_t *scharr_window_row =
          scharr_window +
          static_cast<ptrdiff_t>(y) * window_width * channels * 2L;

      const size_t window_width_times_channels =
          static_cast<size_t>(window_width) * static_cast<size_t>(channels);
      LoopUnroll2 loop{window_width_times_channels, svcnth()};

      auto accumulate_diff_row = [&](size_t index,
                                     svbool_t pg16) KLEIDICV_STREAMING {
        svint16_t tl = svld1ub_s16(pg16, next_row0 + index);
        svint16_t tr = svld1ub_s16(pg16, next_row0 + index + channels);
        svint16_t bl = svld1ub_s16(pg16, next_row1 + index);
        svint16_t br = svld1ub_s16(pg16, next_row1 + index + channels);
        svint16_t diff =
            lerp<kLucasKanadeAlgFractionBits - 5>(tl, tr, bl, br, lerp_coeffs);
        svint16_t win = svld1_s16(pg16, window_row + index);
        diff = svsub_s16_x(pg16, diff, win);

        svint16x2_t scharr_xy = svld2_s16(pg16, scharr_window_row + index * 2L);
        svint16_t scharr_x = svget2(scharr_xy, 0);
        svint16_t scharr_y = svget2(scharr_xy, 1);

#if KLEIDICV_TARGET_SME2
        sum_diff_scharr_x_v =
            svdot_s32_s16(sum_diff_scharr_x_v, scharr_x, diff);
        sum_diff_scharr_y_v =
            svdot_s32_s16(sum_diff_scharr_y_v, scharr_y, diff);
#else
        sum_diff_scharr_x_v = svmlalb_s32(sum_diff_scharr_x_v, scharr_x, diff);
        sum_diff_scharr_x_v = svmlalt_s32(sum_diff_scharr_x_v, scharr_x, diff);

        sum_diff_scharr_y_v = svmlalb_s32(sum_diff_scharr_y_v, scharr_y, diff);
        sum_diff_scharr_y_v = svmlalt_s32(sum_diff_scharr_y_v, scharr_y, diff);

#endif
      };

      loop.unroll_once([&](size_t index) KLEIDICV_STREAMING {
        svbool_t pg16 = svptrue_b16();
        accumulate_diff_row(index, pg16);
      });

      loop.remaining([&](size_t index, size_t end) KLEIDICV_STREAMING {
        svbool_t pg16 = svwhilelt_b16_u64(index, end);
        accumulate_diff_row(index, pg16);
      });
    }
    sum_diff_scharr_x =
        svaddv_f32(svptrue_b32(),
                   svcvt_f32_s32_x(svptrue_b32(), sum_diff_scharr_x_v)) *
        kLucasKanadeAlgFixedPointDescale;
    sum_diff_scharr_y =
        svaddv_f32(svptrue_b32(),
                   svcvt_f32_s32_x(svptrue_b32(), sum_diff_scharr_y_v)) *
        kLucasKanadeAlgFixedPointDescale;
  }
};

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_SC_H
