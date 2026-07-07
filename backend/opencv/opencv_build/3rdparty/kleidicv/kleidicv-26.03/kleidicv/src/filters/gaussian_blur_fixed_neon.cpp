// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstddef>

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/filters/separable_filter_15x15_neon.h"
#include "kleidicv/filters/separable_filter_21x21_neon.h"
#include "kleidicv/filters/separable_filter_3x3_neon.h"
#include "kleidicv/filters/separable_filter_5x5_neon.h"
#include "kleidicv/filters/separable_filter_7x7_neon.h"
#include "kleidicv/filters/separable_filter_9x9_neon.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/neon.h"
#include "kleidicv/workspace/border_types.h"
#include "kleidicv/workspace/separable.h"

namespace kleidicv::neon {

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
  using ScalarType = uint8_t;
  using SourceType = ScalarType;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using BufferType = double_element_width_t<ScalarType>;
  using BufferVectorType = typename VecTraits<BufferType>::VectorType;
  using DestinationType = ScalarType;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_vector_path(SourceVectorType src[3], BufferType *dst) const {
    // acc_0_2 = src[0] + src[2]
    BufferVectorType acc_0_2_l = vaddl(vget_low(src[0]), vget_low(src[2]));
    BufferVectorType acc_0_2_h = vaddl(vget_high(src[0]), vget_high(src[2]));
    // acc_1 = src[1] + src[1]
    BufferVectorType acc_1_l = vshll_n<1>(vget_low(src[1]));
    BufferVectorType acc_1_h = vshll_n<1>(vget_high(src[1]));
    // acc = acc_0_2 + acc_1
    BufferVectorType acc_l = vaddq(acc_0_2_l, acc_1_l);
    BufferVectorType acc_h = vaddq(acc_0_2_h, acc_1_h);

    VecTraits<BufferType>::store_consecutive(acc_l, acc_h, &dst[0]);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_scalar_path(const SourceType src[3], BufferType *dst) const {
    dst[0] = src[0] + 2 * src[1] + src[2];
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/16 * [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_vector_path(BufferVectorType src[3],
                              DestinationType *dst) const {
    BufferVectorType acc_wide = vaddq(src[0], src[2]);
    acc_wide = vaddq(acc_wide, vshlq_n<1>(src[1]));
    auto acc_narrow = vrshrn_n<4>(acc_wide);
    vst1(&dst[0], acc_narrow);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/16 * [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const {
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

  GaussianBlur()
      : const_6_u8_half_{vdup_n_u8(6)},
        const_6_u16_{vdupq_n_u16(6)},
        const_4_u16_{vdupq_n_u16(4)} {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_vector_path(uint8x16_t src[5], BufferType *dst) const {
    uint16x8_t acc_0_4_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[4]));
    uint16x8_t acc_0_4_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[4]));
    uint16x8_t acc_1_3_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[3]));
    uint16x8_t acc_1_3_h = vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[3]));
    uint16x8_t acc_l =
        vmlal_u8(acc_0_4_l, vget_low_u8(src[2]), const_6_u8_half_);
    uint16x8_t acc_h =
        vmlal_u8(acc_0_4_h, vget_high_u8(src[2]), const_6_u8_half_);
    acc_l = vmlaq_u16(acc_l, acc_1_3_l, const_4_u16_);
    acc_h = vmlaq_u16(acc_h, acc_1_3_h, const_4_u16_);
    vst1q(&dst[0], acc_l);
    vst1q(&dst[8], acc_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_scalar_path(const SourceType src[5], BufferType *dst) const {
    dst[0] = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_vector_path(uint16x8_t src[5], DestinationType *dst) const {
    uint16x8_t acc_0_4 = vaddq_u16(src[0], src[4]);
    uint16x8_t acc_1_3 = vaddq_u16(src[1], src[3]);
    uint16x8_t acc_u16 = vmlaq_u16(acc_0_4, src[2], const_6_u16_);
    acc_u16 = vmlaq_u16(acc_u16, acc_1_3, const_4_u16_);
    uint8x8_t acc_u8 = vrshrn_n_u16(acc_u16, 8);
    vst1(&dst[0], acc_u8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const {
    auto acc = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
    dst[0] = rounding_shift_right(acc, 8);
  }

 private:
  uint8x8_t const_6_u8_half_;
  uint16x8_t const_6_u16_;
  uint16x8_t const_4_u16_;
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

  GaussianBlur()
      : const_7_u16_{vdupq_n_u16(7)},
        const_7_u32_{vdupq_n_u32(7)},
        const_9_u16_{vdupq_n_u16(9)} {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //     * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void vertical_vector_path(uint8x16_t src[7], BufferType *dst) const {
    uint16x8_t acc_0_6_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[6]));
    uint16x8_t acc_0_6_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[6]));

    uint16x8_t acc_1_5_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[5]));
    uint16x8_t acc_1_5_h = vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[5]));

    uint16x8_t acc_2_4_l = vaddl_u8(vget_low_u8(src[2]), vget_low_u8(src[4]));
    uint16x8_t acc_2_4_h = vaddl_u8(vget_high_u8(src[2]), vget_high_u8(src[4]));

    uint16x8_t acc_3_l = vmovl_u8(vget_low_u8(src[3]));
    uint16x8_t acc_3_h = vmovl_u8(vget_high_u8(src[3]));

    uint16x8_t acc_0_2_4_6_l = vmlaq_u16(acc_0_6_l, acc_2_4_l, const_7_u16_);
    uint16x8_t acc_0_2_4_6_h = vmlaq_u16(acc_0_6_h, acc_2_4_h, const_7_u16_);

    uint16x8_t acc_0_2_3_4_6_l =
        vmlaq_u16(acc_0_2_4_6_l, acc_3_l, const_9_u16_);
    uint16x8_t acc_0_2_3_4_6_h =
        vmlaq_u16(acc_0_2_4_6_h, acc_3_h, const_9_u16_);

    acc_0_2_3_4_6_l = vshlq_n_u16(acc_0_2_3_4_6_l, 1);
    acc_0_2_3_4_6_h = vshlq_n_u16(acc_0_2_3_4_6_h, 1);

    uint16x8_t acc_0_1_2_3_4_5_6_l =
        vmlaq_u16(acc_0_2_3_4_6_l, acc_1_5_l, const_7_u16_);
    uint16x8_t acc_0_1_2_3_4_5_6_h =
        vmlaq_u16(acc_0_2_3_4_6_h, acc_1_5_h, const_7_u16_);

    vst1q(&dst[0], acc_0_1_2_3_4_5_6_l);
    vst1q(&dst[8], acc_0_1_2_3_4_5_6_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //     * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void vertical_scalar_path(const SourceType src[7], BufferType *dst) const {
    uint16_t acc = src[0] * 2 + src[1] * 7 + src[2] * 14 + src[3] * 18 +
                   src[4] * 14 + src[5] * 7 + src[6] * 2;
    dst[0] = acc;
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/4096 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //              * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void horizontal_vector_path(uint16x8_t src[7], DestinationType *dst) const {
    uint32x4_t acc_0_6_l =
        vaddl_u16(vget_low_u16(src[0]), vget_low_u16(src[6]));
    uint32x4_t acc_0_6_h =
        vaddl_u16(vget_high_u16(src[0]), vget_high_u16(src[6]));

    uint32x4_t acc_1_5_l =
        vaddl_u16(vget_low_u16(src[1]), vget_low_u16(src[5]));
    uint32x4_t acc_1_5_h =
        vaddl_u16(vget_high_u16(src[1]), vget_high_u16(src[5]));

    uint16x8_t acc_2_4 = vaddq_u16(src[2], src[4]);

    uint32x4_t acc_0_2_4_6_l =
        vmlal_u16(acc_0_6_l, vget_low_u16(acc_2_4), vget_low_u16(const_7_u16_));
    uint32x4_t acc_0_2_4_6_h = vmlal_u16(acc_0_6_h, vget_high_u16(acc_2_4),
                                         vget_high_u16(const_7_u16_));

    uint32x4_t acc_0_2_3_4_6_l = vmlal_u16(acc_0_2_4_6_l, vget_low_u16(src[3]),
                                           vget_low_u16(const_9_u16_));
    uint32x4_t acc_0_2_3_4_6_h = vmlal_u16(acc_0_2_4_6_h, vget_high_u16(src[3]),
                                           vget_high_u16(const_9_u16_));

    acc_0_2_3_4_6_l = vshlq_n_u32(acc_0_2_3_4_6_l, 1);
    acc_0_2_3_4_6_h = vshlq_n_u32(acc_0_2_3_4_6_h, 1);

    uint32x4_t acc_0_1_2_3_4_5_6_l =
        vmlaq_u32(acc_0_2_3_4_6_l, acc_1_5_l, const_7_u32_);
    uint32x4_t acc_0_1_2_3_4_5_6_h =
        vmlaq_u32(acc_0_2_3_4_6_h, acc_1_5_h, const_7_u32_);

    uint16x4_t acc_0_1_2_3_4_5_6_u16_l = vrshrn_n_u32(acc_0_1_2_3_4_5_6_l, 12);
    uint16x4_t acc_0_1_2_3_4_5_6_u16_h = vrshrn_n_u32(acc_0_1_2_3_4_5_6_h, 12);

    uint16x8_t acc_0_1_2_3_4_5_6_u16 =
        vcombine_u16(acc_0_1_2_3_4_5_6_u16_l, acc_0_1_2_3_4_5_6_u16_h);
    uint8x8_t acc_0_1_2_3_4_5_6_u8 = vmovn_u16(acc_0_1_2_3_4_5_6_u16);

    vst1(&dst[0], acc_0_1_2_3_4_5_6_u8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/4096 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //              * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void horizontal_scalar_path(const BufferType src[7],
                              DestinationType *dst) const {
    uint32_t acc = src[0] * 2 + src[1] * 7 + src[2] * 14 + src[3] * 18 +
                   src[4] * 14 + src[5] * 7 + src[6] * 2;
    dst[0] = static_cast<DestinationType>(rounding_shift_right(acc, 12));
  }

 private:
  uint16x8_t const_7_u16_;
  uint32x4_t const_7_u32_;
  uint16x8_t const_9_u16_;
};  // end of class GaussianBlur<uint8_t, 7, true>

// Template for 9x9 Gaussian Blur binomial filters.
//
//               [  16,   52,  120,  204,  240,  204,  120,   52,  16 ]
//               [  52,  169,  390,  663,  780,  663,  390,  169,  52 ]
//               [ 120,  390,  900, 1530, 1800, 1530,  900,  390, 120 ]
//  F = 1/65536 * [ 204,  663, 1530, 2601, 3060, 2601, 1530,  663, 204 ] =
//               [ 240,  780, 1800, 3060, 3600, 3060, 1800,  780, 240 ]
//               [ 204,  663, 1530, 2601, 3060, 2601, 1530,  663, 204 ]
//               [ 120,  390,  900, 1530, 1800, 1530,  900,  390, 120 ]
//               [  52,  169,  390,  663,  780,  663,  390,  169,  52 ]
//               [  16,   52,  120,  204,  240,  204,  120,   52,  16 ]
//
//               [  4 ]
//               [ 13 ]
//               [ 30 ]
//  = 1/65536 *  [ 51 ] * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]
//               [ 60 ]
//               [ 51 ]
//               [ 30 ]
//               [ 13 ]
//               [  4 ]
template <>
class GaussianBlur<uint8_t, 9, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  GaussianBlur()
      : const_13_u16_{vdupq_n_u16(13)},
        const_30_u16_{vdupq_n_u16(30)},
        const_51_u16_{vdupq_n_u16(51)},
        const_60_u16_{vdupq_n_u16(60)},
        const_13_u32_{vdupq_n_u32(13)},
        const_30_u32_{vdupq_n_u32(30)},
        const_51_u32_{vdupq_n_u32(51)},
        const_60_u32_{vdupq_n_u32(60)} {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //     * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void vertical_vector_path(uint8x16_t src[9], BufferType *dst) const {
    uint16x8_t acc_0_8_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[8]));
    uint16x8_t acc_0_8_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[8]));

    uint16x8_t acc_1_7_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[7]));
    uint16x8_t acc_1_7_h = vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[7]));

    uint16x8_t acc_2_6_l = vaddl_u8(vget_low_u8(src[2]), vget_low_u8(src[6]));
    uint16x8_t acc_2_6_h = vaddl_u8(vget_high_u8(src[2]), vget_high_u8(src[6]));

    uint16x8_t acc_3_5_l = vaddl_u8(vget_low_u8(src[3]), vget_low_u8(src[5]));
    uint16x8_t acc_3_5_h = vaddl_u8(vget_high_u8(src[3]), vget_high_u8(src[5]));

    uint16x8_t acc_4_l = vmovl_u8(vget_low_u8(src[4]));
    uint16x8_t acc_4_h = vmovl_u8(vget_high_u8(src[4]));

    // Split the work into two independent accumulators.
    uint16x8_t acc_l_even = vshlq_n_u16(acc_0_8_l, 2);
    uint16x8_t acc_h_even = vshlq_n_u16(acc_0_8_h, 2);
    uint16x8_t acc_l_odd = vmulq_u16(acc_1_7_l, const_13_u16_);
    uint16x8_t acc_h_odd = vmulq_u16(acc_1_7_h, const_13_u16_);

    acc_l_even = vmlaq_u16(acc_l_even, acc_2_6_l, const_30_u16_);
    acc_h_even = vmlaq_u16(acc_h_even, acc_2_6_h, const_30_u16_);
    acc_l_odd = vmlaq_u16(acc_l_odd, acc_3_5_l, const_51_u16_);
    acc_h_odd = vmlaq_u16(acc_h_odd, acc_3_5_h, const_51_u16_);
    acc_l_even = vmlaq_u16(acc_l_even, acc_4_l, const_60_u16_);
    acc_h_even = vmlaq_u16(acc_h_even, acc_4_h, const_60_u16_);

    uint16x8_t acc_l = vaddq_u16(acc_l_even, acc_l_odd);
    uint16x8_t acc_h = vaddq_u16(acc_h_even, acc_h_odd);

    vst1q(&dst[0], acc_l);
    vst1q(&dst[8], acc_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //     * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void vertical_scalar_path(const SourceType src[9], BufferType *dst) const {
    uint16_t acc = src[0] * 4 + src[1] * 13 + src[2] * 30 + src[3] * 51 +
                   src[4] * 60 + src[5] * 51 + src[6] * 30 + src[7] * 13 +
                   src[8] * 4;
    dst[0] = acc;
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/65536 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //               * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void horizontal_vector_path(uint16x8_t src[9], DestinationType *dst) const {
    uint32x4_t acc_0_8_l =
        vaddl_u16(vget_low_u16(src[0]), vget_low_u16(src[8]));
    uint32x4_t acc_0_8_h =
        vaddl_u16(vget_high_u16(src[0]), vget_high_u16(src[8]));

    uint32x4_t acc_1_7_l =
        vaddl_u16(vget_low_u16(src[1]), vget_low_u16(src[7]));
    uint32x4_t acc_1_7_h =
        vaddl_u16(vget_high_u16(src[1]), vget_high_u16(src[7]));

    uint32x4_t acc_2_6_l =
        vaddl_u16(vget_low_u16(src[2]), vget_low_u16(src[6]));
    uint32x4_t acc_2_6_h =
        vaddl_u16(vget_high_u16(src[2]), vget_high_u16(src[6]));

    uint32x4_t acc_3_5_l =
        vaddl_u16(vget_low_u16(src[3]), vget_low_u16(src[5]));
    uint32x4_t acc_3_5_h =
        vaddl_u16(vget_high_u16(src[3]), vget_high_u16(src[5]));

    uint32x4_t acc_4_l = vmovl_u16(vget_low_u16(src[4]));
    uint32x4_t acc_4_h = vmovl_u16(vget_high_u16(src[4]));

    // Split the work into two independent accumulators.
    uint32x4_t acc_l_even = vshlq_n_u32(acc_0_8_l, 2);
    uint32x4_t acc_h_even = vshlq_n_u32(acc_0_8_h, 2);
    uint32x4_t acc_l_odd = vmulq_u32(acc_1_7_l, const_13_u32_);
    uint32x4_t acc_h_odd = vmulq_u32(acc_1_7_h, const_13_u32_);

    acc_l_even = vmlaq_u32(acc_l_even, acc_2_6_l, const_30_u32_);
    acc_h_even = vmlaq_u32(acc_h_even, acc_2_6_h, const_30_u32_);
    acc_l_odd = vmlaq_u32(acc_l_odd, acc_3_5_l, const_51_u32_);
    acc_h_odd = vmlaq_u32(acc_h_odd, acc_3_5_h, const_51_u32_);
    acc_l_even = vmlaq_u32(acc_l_even, acc_4_l, const_60_u32_);
    acc_h_even = vmlaq_u32(acc_h_even, acc_4_h, const_60_u32_);

    uint32x4_t acc_l = vaddq_u32(acc_l_even, acc_l_odd);
    uint32x4_t acc_h = vaddq_u32(acc_h_even, acc_h_odd);

    uint16x4_t acc_u16_l = vrshrn_n_u32(acc_l, 16);
    uint16x4_t acc_u16_h = vrshrn_n_u32(acc_h, 16);
    uint16x8_t acc_u16 = vcombine_u16(acc_u16_l, acc_u16_h);
    uint8x8_t acc_u8 = vmovn_u16(acc_u16);

    vst1(&dst[0], acc_u8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/65536 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6, SRC7, SRC8 ] *
  //               * [ 4, 13, 30, 51, 60, 51, 30, 13, 4 ]T
  void horizontal_scalar_path(const BufferType src[9],
                              DestinationType *dst) const {
    uint32_t acc = src[0] * 4 + src[1] * 13 + src[2] * 30 + src[3] * 51 +
                   src[4] * 60 + src[5] * 51 + src[6] * 30 + src[7] * 13 +
                   src[8] * 4;
    dst[0] = static_cast<DestinationType>(rounding_shift_right(acc, 16));
  }

 private:
  uint16x8_t const_13_u16_;
  uint16x8_t const_30_u16_;
  uint16x8_t const_51_u16_;
  uint16x8_t const_60_u16_;
  uint32x4_t const_13_u32_;
  uint32x4_t const_30_u32_;
  uint32x4_t const_51_u32_;
  uint32x4_t const_60_u32_;
};  // end of class GaussianBlur<uint8_t, 9, true>

template <size_t KernelSize>
class GaussianBlur<uint8_t, KernelSize, false> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint8_t;
  using DestinationType = uint8_t;

  static constexpr size_t kHalfKernelSize = get_half_kernel_size(KernelSize);

  explicit GaussianBlur(const uint8_t *half_kernel)
      : half_kernel_(half_kernel) {}

  void vertical_vector_path(uint8x16_t src[KernelSize], BufferType *dst) const {
    common_vector_path(src, dst);
  }

  void vertical_scalar_path(const SourceType src[KernelSize],
                            BufferType *dst) const {
    uint16_t acc = src[kHalfKernelSize - 1] * half_kernel_[kHalfKernelSize - 1];

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 0; i < kHalfKernelSize - 1; i++) {
      acc += (src[i] + src[KernelSize - i - 1]) * half_kernel_[i];
    }

    dst[0] = static_cast<DestinationType>(rounding_shift_right(acc, 8));
  }

  void horizontal_vector_path(uint8x16_t src[KernelSize],
                              DestinationType *dst) const {
    common_vector_path(src, dst);
  }

  void horizontal_scalar_path(const BufferType src[KernelSize],
                              DestinationType *dst) const {
    vertical_scalar_path(src, dst);
  }

 private:
  void common_vector_path(uint8x16_t src[KernelSize], BufferType *dst) const {
    uint8x8_t half_kernel_mid = vdup_n_u8(half_kernel_[kHalfKernelSize - 1]);
    uint16x8_t acc_l =
        vmlal_u8(vdupq_n_u16(128), vget_low_u8(src[kHalfKernelSize - 1]),
                 half_kernel_mid);
    uint16x8_t acc_h =
        vmlal_u8(vdupq_n_u16(128), vget_high_u8(src[kHalfKernelSize - 1]),
                 half_kernel_mid);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 0; i < kHalfKernelSize - 1; i++) {
      const size_t j = KernelSize - i - 1;
      uint16x8_t vec_l = vaddl_u8(vget_low_u8(src[i]), vget_low_u8(src[j]));
      uint16x8_t vec_h = vaddl_high_u8(src[i], src[j]);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);

      acc_l = vmlaq_u16(acc_l, vec_l, coeff);
      acc_h = vmlaq_u16(acc_h, vec_h, coeff);
    }

    // Keep only the highest 8 bits
    uint8x16_t result =
        vuzp2q_u8(vreinterpretq_u8_u16(acc_l), vreinterpretq_u8_u16(acc_h));
    neon::VecTraits<uint8_t>::store(result, &dst[0]);
  }

  const uint8_t *half_kernel_;
};  // end of class GaussianBlur<uint8_t, KernelSize, false>

template <size_t KernelSize, bool IsBinomial, typename ScalarType>
static kleidicv_error_t gaussian_blur_fixed_kernel_size(
    const ScalarType *src, size_t src_stride, ScalarType *dst,
    size_t dst_stride, Rectangle &rect, size_t y_begin, size_t y_end,
    size_t channels, float sigma, FixedBorderType border_type) {
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
        std::memcpy(static_cast<void *>(&dst_rows.at(row)[0]),
                    static_cast<const void *>(&src_rows.at(row)[0]),
                    rect.width() * sizeof(ScalarType) * dst_rows.channels());
      }
    }
    return KLEIDICV_OK;
  }
}

template <bool IsBinomial, typename ScalarType>
static kleidicv_error_t gaussian_blur_fixed(
    size_t kernel_size, const ScalarType *src, size_t src_stride,
    ScalarType *dst, size_t dst_stride, Rectangle &rect, size_t y_begin,
    size_t y_end, size_t channels, float sigma, FixedBorderType border_type) {
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

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gaussian_blur_fixed_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t /*kernel_height*/, float sigma_x,
    float /*sigma_y*/, FixedBorderType fixed_border_type) {
  if (auto result =
          gaussian_blur_checks(src, src_stride, dst, dst_stride, width, height);
      result != KLEIDICV_OK) {
    return result;
  }

  Rectangle rect{width, height};

  if (sigma_x == 0.0) {
    return gaussian_blur_fixed<true>(kernel_width, src, src_stride, dst,
                                     dst_stride, rect, y_begin, y_end, channels,
                                     sigma_x, fixed_border_type);
  }

  return gaussian_blur_fixed<false>(kernel_width, src, src_stride, dst,
                                    dst_stride, rect, y_begin, y_end, channels,
                                    sigma_x, fixed_border_type);
}

}  // namespace kleidicv::neon
