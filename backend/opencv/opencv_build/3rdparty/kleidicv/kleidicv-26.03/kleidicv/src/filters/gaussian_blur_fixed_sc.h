// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_GAUSSIAN_BLUR_SC_H
#define KLEIDICV_GAUSSIAN_BLUR_SC_H

#include <array>
#include <cassert>

#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/filters/separable_filter_15x15_sc.h"
#include "kleidicv/filters/separable_filter_21x21_sc.h"
#include "kleidicv/filters/separable_filter_3x3_sc.h"
#include "kleidicv/filters/separable_filter_5x5_sc.h"
#include "kleidicv/filters/separable_filter_7x7_sc.h"
#include "kleidicv/filters/separable_filter_9x9_sc.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/workspace/separable.h"

#if KLEIDICV_TARGET_SME || KLEIDICV_TARGET_SME2
#include <arm_sme.h>
#endif

namespace KLEIDICV_TARGET_NAMESPACE {

// Primary template for Gaussian Blur filters.
template <typename ScalarType, size_t KernelSize, bool IsBinomial>
class GaussianBlur;

// Template for 3x3 Gaussian Blur binomial filters.
//
//             [ 1, 2, 1 ]          [ 1 ]
//  F = 1/16 * [ 2, 4, 2 ] = 1/16 * [ 2 ] * [ 1, 2, 1 ]
//             [ 1, 2, 1 ]          [ 1 ]
template <>
class GaussianBlur<uint8_t, 3, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<svuint8_t> src[3],
                            BufferType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_0_2_b = svaddlb_u16(src[0], src[2]);
    svuint16_t acc_0_2_t = svaddlt_u16(src[0], src[2]);

    svuint16_t acc_1_b = svshllb_n_u16(src[1], 1);
    svuint16_t acc_1_t = svshllt_n_u16(src[1], 1);

    svuint16_t acc_u16_b = svadd_u16_x(pg, acc_0_2_b, acc_1_b);
    svuint16_t acc_u16_t = svadd_u16_x(pg, acc_0_2_t, acc_1_t);

    svuint16x2_t interleaved = svcreate2(acc_u16_b, acc_u16_t);
    svst2(pg, &dst[0], interleaved);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/16 * [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<svuint16_t> src[3],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_0_2 = svhadd_u16_x(pg, src[0], src[2]);

    svuint16_t acc = svadd_u16_x(pg, acc_0_2, src[1]);
    acc = svrshr_x(pg, acc, 3);

    svst1b(pg, &dst[0], acc);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/16 * [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    auto acc = src[0] + 2 * src[1] + src[2];
    dst[0] = rounding_shift_right(acc, 4);
  }
};  // end of class GaussianBlur<uint8_t, 3, true>

// Template for 5x5 Gaussian Blur binomial filters.
//
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//  F = 1/256 * [ 6, 24, 36, 24, 6 ] = 1/256 * [ 6 ] * [ 1,  4,  6,  4, 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
template <>
class GaussianBlur<uint8_t, 5, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<svuint8_t> src[5],
                            BufferType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_0_4_b = svaddlb_u16(src[0], src[4]);
    svuint16_t acc_0_4_t = svaddlt_u16(src[0], src[4]);
    svuint16_t acc_1_3_b = svaddlb_u16(src[1], src[3]);
    svuint16_t acc_1_3_t = svaddlt_u16(src[1], src[3]);

    svuint16_t acc_u16_b = svmlalb_n_u16(acc_0_4_b, src[2], 6);
    svuint16_t acc_u16_t = svmlalt_n_u16(acc_0_4_t, src[2], 6);
    acc_u16_b = svmla_n_u16_x(pg, acc_u16_b, acc_1_3_b, 4);
    acc_u16_t = svmla_n_u16_x(pg, acc_u16_t, acc_1_3_t, 4);

    svuint16x2_t interleaved = svcreate2(acc_u16_b, acc_u16_t);
    svst2(pg, &dst[0], interleaved);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<svuint16_t> src[5],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_0_4 = svadd_x(pg, src[0], src[4]);
    svuint16_t acc_1_3 = svadd_x(pg, src[1], src[3]);
    svuint16_t acc = svmla_n_u16_x(pg, acc_0_4, src[2], 6);
    acc = svmla_n_u16_x(pg, acc, acc_1_3, 4);
    acc = svrshr_x(pg, acc, 8);
    svst1b(pg, &dst[0], acc);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    auto acc = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
    dst[0] = rounding_shift_right(acc, 8);
  }
};  // end of class GaussianBlur<uint8_t, 5, true>

// Template for 7x7 Gaussian Blur binomial filters.
//
//               [  4,  14,  28,  36,  28,  14,  4 ]
//               [ 14,  49,  98, 126,  98,  49, 14 ]
//               [ 28,  98, 196, 252, 196,  98, 28 ]
//  F = 1/4096 * [ 36, 126, 252, 324, 252, 126, 36 ] =
//               [ 28,  98, 196, 252, 196,  98, 28 ]
//               [ 14,  49,  98, 126,  98,  49, 14 ]
//               [  4,  14,  28,  36,  28,  14,  4 ]
//
//               [  2 ]
//               [  7 ]
//               [ 14 ]
//  = 1/4096  *  [ 18 ] * [ 2, 7, 14, 18, 14, 7, 2 ]
//               [ 14 ]
//               [  7 ]
//               [  2 ]
template <>
class GaussianBlur<uint8_t, 7, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //     * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<svuint8_t> src[7],
                            BufferType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_0_6_b = svaddlb_u16(src[0], src[6]);
    svuint16_t acc_0_6_t = svaddlt_u16(src[0], src[6]);

    svuint16_t acc_1_5_b = svaddlb_u16(src[1], src[5]);
    svuint16_t acc_1_5_t = svaddlt_u16(src[1], src[5]);

    svuint16_t acc_2_4_b = svaddlb_u16(src[2], src[4]);
    svuint16_t acc_2_4_t = svaddlt_u16(src[2], src[4]);

    svuint16_t acc_3_b = svmovlb_u16(src[3]);
    svuint16_t acc_3_t = svmovlt_u16(src[3]);

    svuint16_t acc_0_2_4_6_b = svmla_n_u16_x(pg, acc_0_6_b, acc_2_4_b, 7);
    svuint16_t acc_0_2_4_6_t = svmla_n_u16_x(pg, acc_0_6_t, acc_2_4_t, 7);

    svuint16_t acc_0_2_3_4_6_b = svmla_n_u16_x(pg, acc_0_2_4_6_b, acc_3_b, 9);
    svuint16_t acc_0_2_3_4_6_t = svmla_n_u16_x(pg, acc_0_2_4_6_t, acc_3_t, 9);
    acc_0_2_3_4_6_b = svlsl_n_u16_x(pg, acc_0_2_3_4_6_b, 1);
    acc_0_2_3_4_6_t = svlsl_n_u16_x(pg, acc_0_2_3_4_6_t, 1);

    svuint16_t acc_0_1_2_3_4_5_6_b =
        svmla_n_u16_x(pg, acc_0_2_3_4_6_b, acc_1_5_b, 7);
    svuint16_t acc_0_1_2_3_4_5_6_t =
        svmla_n_u16_x(pg, acc_0_2_3_4_6_t, acc_1_5_t, 7);

    svuint16x2_t interleaved =
        svcreate2(acc_0_1_2_3_4_5_6_b, acc_0_1_2_3_4_5_6_t);
    svst2(pg, &dst[0], interleaved);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/4096 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //              * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<svuint16_t> src[7],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    svuint32_t acc_0_6_b = svaddlb_u32(src[0], src[6]);
    svuint32_t acc_0_6_t = svaddlt_u32(src[0], src[6]);

    svuint32_t acc_1_5_b = svaddlb_u32(src[1], src[5]);
    svuint32_t acc_1_5_t = svaddlt_u32(src[1], src[5]);

    svuint16_t acc_2_4 = svadd_u16_x(pg, src[2], src[4]);

    svuint32_t acc_0_2_4_6_b = svmlalb_n_u32(acc_0_6_b, acc_2_4, 7);
    svuint32_t acc_0_2_4_6_t = svmlalt_n_u32(acc_0_6_t, acc_2_4, 7);

    svuint32_t acc_0_2_3_4_6_b = svmlalb_n_u32(acc_0_2_4_6_b, src[3], 9);
    svuint32_t acc_0_2_3_4_6_t = svmlalt_n_u32(acc_0_2_4_6_t, src[3], 9);

    acc_0_2_3_4_6_b = svlsl_n_u32_x(pg, acc_0_2_3_4_6_b, 1);
    acc_0_2_3_4_6_t = svlsl_n_u32_x(pg, acc_0_2_3_4_6_t, 1);

    svuint32_t acc_0_1_2_3_4_5_6_b =
        svmla_n_u32_x(pg, acc_0_2_3_4_6_b, acc_1_5_b, 7);
    svuint32_t acc_0_1_2_3_4_5_6_t =
        svmla_n_u32_x(pg, acc_0_2_3_4_6_t, acc_1_5_t, 7);

    svuint16_t acc_0_1_2_3_4_5_6_u16_b =
        svrshrnb_n_u32(acc_0_1_2_3_4_5_6_b, 12);
    svuint16_t acc_0_1_2_3_4_5_6_u16 =
        svrshrnt_n_u32(acc_0_1_2_3_4_5_6_u16_b, acc_0_1_2_3_4_5_6_t, 12);

    svst1b(pg, &dst[0], acc_0_1_2_3_4_5_6_u16);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/4096 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //              * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void horizontal_scalar_path(const BufferType src[7],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    uint32_t acc = src[0] * 2 + src[1] * 7 + src[2] * 14 + src[3] * 18 +
                   src[4] * 14 + src[5] * 7 + src[6] * 2;
    dst[0] = rounding_shift_right(acc, 12);
  }
};  // end of class GaussianBlur<uint8_t, 7, true>

// Template for 9x9 Gaussian Blur binomial filters.
//
//                [  16,   52,  120,  204,  240,  204,  120,   52,  16 ]
//                [  52,  169,  390,  663,  780,  663,  390,  169,  52 ]
//                [ 120,  390,  900, 1530, 1800, 1530,  900,  390, 120 ]
//  F = 1/65536 * [ 204,  663, 1530, 2601, 3060, 2601, 1530,  663, 204 ] =
//                [ 240,  780, 1800, 3060, 3600, 3060, 1800,  780, 240 ]
//                [ 204,  663, 1530, 2601, 3060, 2601, 1530,  663, 204 ]
//                [ 120,  390,  900, 1530, 1800, 1530,  900,  390, 120 ]
//                [  52,  169,  390,  663,  780,  663,  390,  169,  52 ]
//                [  16,   52,  120,  204,  240,  204,  120,   52,  16 ]
//
//                [  4 ]
//                [ 13 ]
//                [ 30 ]
//  = 1/65536 *   [ 51 ] * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]
//                [ 60 ]
//                [ 51 ]
//                [ 30 ]
//                [ 13 ]
//                [  4 ]
template <>
class GaussianBlur<uint8_t, 9, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //     * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<svuint8_t> src[9],
                            BufferType *dst) const KLEIDICV_STREAMING {
    // Lane-level split after widening: *_lo/*_hi are lower/upper lanes.
    svuint16_t acc_0_8_lo = svaddlb_u16(src[0], src[8]);
    svuint16_t acc_0_8_hi = svaddlt_u16(src[0], src[8]);

    svuint16_t acc_1_7_lo = svaddlb_u16(src[1], src[7]);
    svuint16_t acc_1_7_hi = svaddlt_u16(src[1], src[7]);

    svuint16_t acc_2_6_lo = svaddlb_u16(src[2], src[6]);
    svuint16_t acc_2_6_hi = svaddlt_u16(src[2], src[6]);

    svuint16_t acc_3_5_lo = svaddlb_u16(src[3], src[5]);
    svuint16_t acc_3_5_hi = svaddlt_u16(src[3], src[5]);

    svuint16_t acc_4_lo = svmovlb_u16(src[4]);
    svuint16_t acc_4_hi = svmovlt_u16(src[4]);

    // Window-level grouping: *_tap_even/*_tap_odd are even/odd taps
    // (0, 2, 4, 6, 8 vs 1, 3, 5, 7).
    svuint16_t acc_lo_tap_even = svlsl_n_u16_x(pg, acc_0_8_lo, 2);
    svuint16_t acc_hi_tap_even = svlsl_n_u16_x(pg, acc_0_8_hi, 2);
    acc_lo_tap_even = svmla_n_u16_x(pg, acc_lo_tap_even, acc_2_6_lo, 30);
    acc_hi_tap_even = svmla_n_u16_x(pg, acc_hi_tap_even, acc_2_6_hi, 30);
    acc_lo_tap_even = svmla_n_u16_x(pg, acc_lo_tap_even, acc_4_lo, 60);
    acc_hi_tap_even = svmla_n_u16_x(pg, acc_hi_tap_even, acc_4_hi, 60);

    svuint16_t acc_lo_tap_odd = svmul_n_u16_x(pg, acc_1_7_lo, 13);
    svuint16_t acc_hi_tap_odd = svmul_n_u16_x(pg, acc_1_7_hi, 13);
    acc_lo_tap_odd = svmla_n_u16_x(pg, acc_lo_tap_odd, acc_3_5_lo, 51);
    acc_hi_tap_odd = svmla_n_u16_x(pg, acc_hi_tap_odd, acc_3_5_hi, 51);

    svuint16_t acc_lo = svadd_u16_x(pg, acc_lo_tap_even, acc_lo_tap_odd);
    svuint16_t acc_hi = svadd_u16_x(pg, acc_hi_tap_even, acc_hi_tap_odd);

    svuint16x2_t interleaved = svcreate2(acc_lo, acc_hi);
    svst2(pg, &dst[0], interleaved);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/65536 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //               * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<svuint16_t> src[9],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    // Lane-level split after widening: *_lo/*_hi are lower/upper lanes.
    svuint32_t acc_0_8_lo = svaddlb_u32(src[0], src[8]);
    svuint32_t acc_0_8_hi = svaddlt_u32(src[0], src[8]);

    svuint32_t acc_1_7_lo = svaddlb_u32(src[1], src[7]);
    svuint32_t acc_1_7_hi = svaddlt_u32(src[1], src[7]);

    svuint32_t acc_2_6_lo = svaddlb_u32(src[2], src[6]);
    svuint32_t acc_2_6_hi = svaddlt_u32(src[2], src[6]);

    svuint32_t acc_3_5_lo = svaddlb_u32(src[3], src[5]);
    svuint32_t acc_3_5_hi = svaddlt_u32(src[3], src[5]);

    svuint32_t acc_4_lo = svmovlb_u32(src[4]);
    svuint32_t acc_4_hi = svmovlt_u32(src[4]);

    // Window-level grouping: *_tap_even/*_tap_odd are even/odd taps
    // (0, 2, 4, 6, 8 vs 1, 3, 5, 7).
    svuint32_t acc_lo_tap_even = svlsl_n_u32_x(pg, acc_0_8_lo, 2);
    svuint32_t acc_hi_tap_even = svlsl_n_u32_x(pg, acc_0_8_hi, 2);
    acc_lo_tap_even = svmla_n_u32_x(pg, acc_lo_tap_even, acc_2_6_lo, 30);
    acc_hi_tap_even = svmla_n_u32_x(pg, acc_hi_tap_even, acc_2_6_hi, 30);
    acc_lo_tap_even = svmla_n_u32_x(pg, acc_lo_tap_even, acc_4_lo, 60);
    acc_hi_tap_even = svmla_n_u32_x(pg, acc_hi_tap_even, acc_4_hi, 60);

    svuint32_t acc_lo_tap_odd = svmul_n_u32_x(pg, acc_1_7_lo, 13);
    svuint32_t acc_hi_tap_odd = svmul_n_u32_x(pg, acc_1_7_hi, 13);
    acc_lo_tap_odd = svmla_n_u32_x(pg, acc_lo_tap_odd, acc_3_5_lo, 51);
    acc_hi_tap_odd = svmla_n_u32_x(pg, acc_hi_tap_odd, acc_3_5_hi, 51);

    svuint32_t acc_lo = svadd_u32_x(pg, acc_lo_tap_even, acc_lo_tap_odd);
    svuint32_t acc_hi = svadd_u32_x(pg, acc_hi_tap_even, acc_hi_tap_odd);

    svuint16_t acc_u16_lo = svrshrnb_n_u32(acc_lo, 16);
    svuint16_t acc_u16 = svrshrnt_n_u32(acc_u16_lo, acc_hi, 16);

    svst1b(pg, &dst[0], acc_u16);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/65536 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //               * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void horizontal_scalar_path(const BufferType src[9],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    uint32_t acc = src[0] * 4 + src[1] * 13 + src[2] * 30 + src[3] * 51 +
                   src[4] * 60 + src[5] * 51 + src[6] * 30 + src[7] * 13 +
                   src[8] * 4;
    dst[0] = rounding_shift_right(acc, 16);
  }
};  // end of class GaussianBlur<uint8_t, 9, true>

// CustomSigma variant
template <size_t KernelSize>
class GaussianBlur<uint8_t, KernelSize, false> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint8_t;
  using DestinationType = uint8_t;
  using SourceVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;

  static constexpr size_t kHalfKernelSize = get_half_kernel_size(KernelSize);

  explicit GaussianBlur(const uint8_t *half_kernel)
      : half_kernel_(half_kernel) {}

  void vertical_vector_path(
      svbool_t pg, std::reference_wrapper<SourceVectorType> src[KernelSize],
      BufferType *dst) const KLEIDICV_STREAMING {
    common_vector_path(pg, src, dst);
  }

  void vertical_scalar_path(const SourceType src[KernelSize],
                            BufferType *dst) const KLEIDICV_STREAMING {
    uint32_t acc = src[kHalfKernelSize - 1] * half_kernel_[kHalfKernelSize - 1];

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 0; i < kHalfKernelSize - 1; i++) {
      acc += (src[i] + src[KernelSize - i - 1]) * half_kernel_[i];
    }

    dst[0] = static_cast<BufferType>(rounding_shift_right(acc, 8));
  }

  void horizontal_vector_path(
      svbool_t pg, std::reference_wrapper<SourceVectorType> src[KernelSize],
      BufferType *dst) const KLEIDICV_STREAMING {
    common_vector_path(pg, src, dst);
  }

  void horizontal_scalar_path(const BufferType src[KernelSize],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    vertical_scalar_path(src, dst);
  }

 private:
  void common_vector_path(
      svbool_t pg, std::reference_wrapper<SourceVectorType> src[KernelSize],
      BufferType *dst) const KLEIDICV_STREAMING {
    svbool_t pg16_all = svptrue_b16();
    svuint16_t acc_b = svmullb_n_u16(src[kHalfKernelSize - 1],
                                     half_kernel_[kHalfKernelSize - 1]);
    svuint16_t acc_t = svmullt_n_u16(src[kHalfKernelSize - 1],
                                     half_kernel_[kHalfKernelSize - 1]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 0; i < kHalfKernelSize - 1; i++) {
      const size_t j = KernelSize - i - 1;
      svuint16_t vec_b = svaddlb_u16(src[i], src[j]);
      svuint16_t vec_t = svaddlt_u16(src[i], src[j]);

      acc_b = svmla_n_u16_x(pg16_all, acc_b, vec_b, half_kernel_[i]);
      acc_t = svmla_n_u16_x(pg16_all, acc_t, vec_t, half_kernel_[i]);
    }

    // Rounding before narrowing
    acc_b = svqadd_n_u16(acc_b, 128);
    acc_t = svqadd_n_u16(acc_t, 128);
    // Keep only the highest 8 bits
    svuint8_t result =
        svtrn2_u8(svreinterpret_u8_u16(acc_b), svreinterpret_u8_u16(acc_t));
    svst1(pg, &dst[0], result);
  }

  const uint8_t *half_kernel_;
};  // end of class GaussianBlur<uint8_t, KernelSize, false>

template <size_t KernelSize, bool IsBinomial, typename ScalarType>
static kleidicv_error_t gaussian_blur_fixed_kernel_size(
    const ScalarType *src, size_t src_stride, ScalarType *dst,
    size_t dst_stride, Rectangle &rect, size_t y_begin, size_t y_end,
    size_t channels, float sigma,
    FixedBorderType border_type) KLEIDICV_STREAMING {
  using GaussianBlurFilter = GaussianBlur<ScalarType, KernelSize, IsBinomial>;
  constexpr size_t intermediate_size{
      sizeof(typename GaussianBlurFilter::BufferType)};

  auto workspace_variant =
      SeparableFilterWorkspace::create(rect, channels, intermediate_size);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<SeparableFilterWorkspace>(&workspace_variant);

  Rows<const ScalarType> src_rows{src, src_stride, channels};
  Rows<ScalarType> dst_rows{dst, dst_stride, channels};

  if constexpr (IsBinomial) {
    GaussianBlurFilter blur;
    SeparableFilter<GaussianBlurFilter, KernelSize> filter{blur};
    workspace.process(y_begin, y_end, src_rows, dst_rows, border_type, filter);

    return KLEIDICV_OK;
  } else {
    constexpr size_t kHalfKernelSize = get_half_kernel_size(KernelSize);
    uint8_t half_kernel[128];
    bool success =
        generate_gaussian_half_kernel(half_kernel, kHalfKernelSize, sigma);
    if (success) {
      GaussianBlurFilter blur(half_kernel);
      SeparableFilter<GaussianBlurFilter, KernelSize> filter{blur};
      workspace.process(y_begin, y_end, src_rows, dst_rows, border_type,
                        filter);
    } else {
      // Sigma is too small that the middle point would get all the weight
      // => it's just a copy.
      for (size_t row = y_begin; row < y_end; ++row) {
#if KLEIDICV_TARGET_SME && defined(__ANDROID__)
        __arm_sc_memcpy(
            static_cast<void *>(&dst_rows.at(row)[0]),
            static_cast<const void *>(&src_rows.at(row)[0]),
            rect.width() * sizeof(ScalarType) * dst_rows.channels());
#else
        std::memcpy(static_cast<void *>(&dst_rows.at(row)[0]),
                    static_cast<const void *>(&src_rows.at(row)[0]),
                    rect.width() * sizeof(ScalarType) * dst_rows.channels());
#endif
      }
    }
    return KLEIDICV_OK;
  }
}

template <bool IsBinomial, typename ScalarType>
static kleidicv_error_t gaussian_blur(
    size_t kernel_size, const ScalarType *src, size_t src_stride,
    ScalarType *dst, size_t dst_stride, Rectangle &rect, size_t y_begin,
    size_t y_end, size_t channels, float sigma,
    FixedBorderType border_type) KLEIDICV_STREAMING {
  switch (kernel_size) {
    case 3:
      return gaussian_blur_fixed_kernel_size<3, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type);
    case 5:
      return gaussian_blur_fixed_kernel_size<5, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type);
    case 7:
      return gaussian_blur_fixed_kernel_size<7, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type);
    case 9:
      return gaussian_blur_fixed_kernel_size<9, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type);
    case 15:
      // 15x15 does not have a binomial variant
      return gaussian_blur_fixed_kernel_size<15, false>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type);
    case 21:
      // 21x21 does not have a binomial variant
      return gaussian_blur_fixed_kernel_size<21, false>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type);
      // gaussian_blur_is_implemented checked the kernel size already.
    // GCOVR_EXCL_START
    default:
      assert(!"kernel size not implemented");
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
      // GCOVR_EXCL_STOP
  }
}

static kleidicv_error_t gaussian_blur_fixed_stripe_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t /*kernel_height*/, float sigma_x,
    float /*sigma_y*/, FixedBorderType fixed_border_type) KLEIDICV_STREAMING {
  if (auto result =
          gaussian_blur_checks(src, src_stride, dst, dst_stride, width, height);
      result != KLEIDICV_OK) {
    return result;
  }

  Rectangle rect{width, height};

  if (sigma_x == 0.0) {
    return gaussian_blur<true>(kernel_width, src, src_stride, dst, dst_stride,
                               rect, y_begin, y_end, channels, sigma_x,
                               fixed_border_type);
  }

  return gaussian_blur<false>(kernel_width, src, src_stride, dst, dst_stride,
                              rect, y_begin, y_end, channels, sigma_x,
                              fixed_border_type);
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_GAUSSIAN_BLUR_SC_H
