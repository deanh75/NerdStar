// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/blur_and_downsample.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/utils.h"
#include "kleidicv/workspace/blur_and_downsample_ws.h"
#include "kleidicv/workspace/border_5x5.h"

namespace kleidicv::neon {

// Applies Gaussian Blur binomial filter to even rows and columns
//
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//  F = 1/256 * [ 6, 24, 36, 24, 6 ] = 1/256 * [ 6 ] * [ 1,  4,  6,  4, 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
class BlurAndDownsample {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits = typename neon::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo5x5<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  BlurAndDownsample()
      : const_6_u8_half_{vdup_n_u8(6)},
        const_6_u16_{vdupq_n_u16(6)},
        const_4_u16_{vdupq_n_u16(4)} {}

  static constexpr size_t margin = 2UL;

  void process_vertical(size_t width, Rows<const SourceType> src_rows,
                        Rows<BufferType> dst_rows,
                        BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_twice([&](ptrdiff_t index) {
      const auto *src_0 = &src_rows.at(border_offsets.c0())[index];
      const auto *src_1 = &src_rows.at(border_offsets.c1())[index];
      const auto *src_2 = &src_rows.at(border_offsets.c2())[index];
      const auto *src_3 = &src_rows.at(border_offsets.c3())[index];
      const auto *src_4 = &src_rows.at(border_offsets.c4())[index];

      SourceVectorType src_a[5], src_b[5];
      src_a[0] = vld1q(&src_0[0]);
      src_b[0] = vld1q(&src_0[SourceVecTraits::num_lanes()]);
      src_a[1] = vld1q(&src_1[0]);
      src_b[1] = vld1q(&src_1[SourceVecTraits::num_lanes()]);
      src_a[2] = vld1q(&src_2[0]);
      src_b[2] = vld1q(&src_2[SourceVecTraits::num_lanes()]);
      src_a[3] = vld1q(&src_3[0]);
      src_b[3] = vld1q(&src_3[SourceVecTraits::num_lanes()]);
      src_a[4] = vld1q(&src_4[0]);
      src_b[4] = vld1q(&src_4[SourceVecTraits::num_lanes()]);
      vertical_vector_path(src_a, &dst_rows[index]);
      vertical_vector_path(
          src_b, &dst_rows[index + static_cast<ptrdiff_t>(
                                       SourceVecTraits::num_lanes())]);
    });

    loop.unroll_once([&](ptrdiff_t index) {
      SourceVectorType src[5];
      src[0] = vld1q(&src_rows.at(border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(border_offsets.c2())[index]);
      src[3] = vld1q(&src_rows.at(border_offsets.c3())[index]);
      src[4] = vld1q(&src_rows.at(border_offsets.c4())[index]);
      vertical_vector_path(src, &dst_rows[index]);
    });

    loop.tail([&](ptrdiff_t index) {
      SourceType src[5];
      src[0] = src_rows.at(border_offsets.c0())[index];
      src[1] = src_rows.at(border_offsets.c1())[index];
      src[2] = src_rows.at(border_offsets.c2())[index];
      src[3] = src_rows.at(border_offsets.c3())[index];
      src[4] = src_rows.at(border_offsets.c4())[index];
      vertical_scalar_path(src, &dst_rows[index]);
    });
  }

  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const {
    switch (src_rows.channels()) {
      case 1:
        process_horizontal<1>(width, src_rows, dst_rows, border_offsets);
        break;
      case 2:
        process_horizontal<2>(width, src_rows, dst_rows, border_offsets);
        break;
      case 3:
        process_horizontal_3channels(width, src_rows, dst_rows, border_offsets);
        break;
      default /* channel == 4 */:
        process_horizontal<4>(width, src_rows, dst_rows, border_offsets);
        break;
    }
  }

  void process_horizontal_borders(Rows<const BufferType> src_rows,
                                  Rows<DestinationType> dst_rows,
                                  BorderOffsets border_offsets) const {
    const ptrdiff_t channels = static_cast<ptrdiff_t>(src_rows.channels());
    for (ptrdiff_t channel = 0; channel < channels; ++channel) {
      disable_loop_vectorization();
      process_horizontal_scalar(src_rows, dst_rows, border_offsets, channel,
                                channel);
    }
  }

 private:
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
  uint8x8_t horizontal_vector_path(uint16x8_t src[5]) const {
    uint16x8_t acc_0_4 = vaddq_u16(src[0], src[4]);
    uint16x8_t acc_1_3 = vaddq_u16(src[1], src[3]);
    uint16x8_t acc_u16 = vmlaq_u16(acc_0_4, src[2], const_6_u16_);
    acc_u16 = vmlaq_u16(acc_u16, acc_1_3, const_4_u16_);
    return vrshrn_n_u16(acc_u16, 8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void process_horizontal_scalar(Rows<const BufferType> src_rows,
                                 Rows<DestinationType> dst_rows,
                                 BorderOffsets border_offsets, ptrdiff_t index,
                                 ptrdiff_t dst_index) const {
    BufferType src[5];
    src[0] = src_rows.at(0, border_offsets.c0())[index];
    src[1] = src_rows.at(0, border_offsets.c1())[index];
    src[2] = src_rows.at(0, border_offsets.c2())[index];
    src[3] = src_rows.at(0, border_offsets.c3())[index];
    src[4] = src_rows.at(0, border_offsets.c4())[index];

    auto acc = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
    dst_rows[dst_index] = rounding_shift_right(acc, 8);
  }

  template <ptrdiff_t Channels>
  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](ptrdiff_t index) {
      const auto *src_0 = &src_rows.at(0, border_offsets.c0())[index];
      const auto *src_1 = &src_rows.at(0, border_offsets.c1())[index];
      const auto *src_2 = &src_rows.at(0, border_offsets.c2())[index];
      const auto *src_3 = &src_rows.at(0, border_offsets.c3())[index];
      const auto *src_4 = &src_rows.at(0, border_offsets.c4())[index];

      BufferVectorType src_a[5], src_b[5];
      src_a[0] = vld1q(&src_0[0]);
      src_b[0] = vld1q(&src_0[BufferVecTraits::num_lanes()]);
      src_a[1] = vld1q(&src_1[0]);
      src_b[1] = vld1q(&src_1[BufferVecTraits::num_lanes()]);
      src_a[2] = vld1q(&src_2[0]);
      src_b[2] = vld1q(&src_2[BufferVecTraits::num_lanes()]);
      src_a[3] = vld1q(&src_3[0]);
      src_b[3] = vld1q(&src_3[BufferVecTraits::num_lanes()]);
      src_a[4] = vld1q(&src_4[0]);
      src_b[4] = vld1q(&src_4[BufferVecTraits::num_lanes()]);

      uint8x8_t res_a = horizontal_vector_path(src_a);
      uint8x8_t res_b = horizontal_vector_path(src_b);

      // Only store even indices.
      if constexpr (Channels == 1) {
        vst1(&dst_rows[index / 2], vuzp1_u8(res_a, res_b));
      } else if constexpr (Channels == 2) {
        vst1(&dst_rows[index / 2],
             vreinterpret_u8_u16(vuzp1_u16(vreinterpret_u16_u8(res_a),
                                           vreinterpret_u16_u8(res_b))));
      } else {
        static_assert(Channels == 4);
        vst1(&dst_rows[index / 2],
             vreinterpret_u8_u32(vuzp1_u32(vreinterpret_u32_u8(res_a),
                                           vreinterpret_u32_u8(res_b))));
      }
    });

    loop.remaining([&](ptrdiff_t index, size_t max_index) {
      ptrdiff_t pixel = index / Channels;
      pixel = align_up(pixel, static_cast<ptrdiff_t>(2));
      index = pixel * Channels;
      while (index < static_cast<ptrdiff_t>(max_index)) {
        for (ptrdiff_t channel = 0; channel < Channels; ++channel) {
          process_horizontal_scalar(src_rows, dst_rows, border_offsets,
                                    index + channel, index / 2 + channel);
        }
        index += 2 * Channels;
      }
    });
  }

  void process_horizontal_3channels(size_t width,
                                    Rows<const BufferType> src_rows,
                                    Rows<DestinationType> dst_rows,
                                    BorderOffsets border_offsets) const {
    constexpr ptrdiff_t channels = 3;
    const ptrdiff_t vec_stride =
        static_cast<ptrdiff_t>(BufferVecTraits::num_lanes()) * channels;
    LoopUnroll2<TryToAvoidTailLoop> loop{width, BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](ptrdiff_t column) {
      const auto *src_0 =
          &src_rows.at(0, border_offsets.c0())[column * channels];
      const auto *src_1 =
          &src_rows.at(0, border_offsets.c1())[column * channels];
      const auto *src_2 =
          &src_rows.at(0, border_offsets.c2())[column * channels];
      const auto *src_3 =
          &src_rows.at(0, border_offsets.c3())[column * channels];
      const auto *src_4 =
          &src_rows.at(0, border_offsets.c4())[column * channels];

      uint16x8x3_t src0_a = vld3q_u16(&src_0[0]);
      uint16x8x3_t src1_a = vld3q_u16(&src_1[0]);
      uint16x8x3_t src2_a = vld3q_u16(&src_2[0]);
      uint16x8x3_t src3_a = vld3q_u16(&src_3[0]);
      uint16x8x3_t src4_a = vld3q_u16(&src_4[0]);

      uint16x8x3_t src0_b = vld3q_u16(&src_0[vec_stride]);
      uint16x8x3_t src1_b = vld3q_u16(&src_1[vec_stride]);
      uint16x8x3_t src2_b = vld3q_u16(&src_2[vec_stride]);
      uint16x8x3_t src3_b = vld3q_u16(&src_3[vec_stride]);
      uint16x8x3_t src4_b = vld3q_u16(&src_4[vec_stride]);

      uint16x8_t ch0_a[5] = {src0_a.val[0], src1_a.val[0], src2_a.val[0],
                             src3_a.val[0], src4_a.val[0]};
      uint16x8_t ch0_b[5] = {src0_b.val[0], src1_b.val[0], src2_b.val[0],
                             src3_b.val[0], src4_b.val[0]};
      uint16x8_t ch1_a[5] = {src0_a.val[1], src1_a.val[1], src2_a.val[1],
                             src3_a.val[1], src4_a.val[1]};
      uint16x8_t ch1_b[5] = {src0_b.val[1], src1_b.val[1], src2_b.val[1],
                             src3_b.val[1], src4_b.val[1]};
      uint16x8_t ch2_a[5] = {src0_a.val[2], src1_a.val[2], src2_a.val[2],
                             src3_a.val[2], src4_a.val[2]};
      uint16x8_t ch2_b[5] = {src0_b.val[2], src1_b.val[2], src2_b.val[2],
                             src3_b.val[2], src4_b.val[2]};

      uint8x8_t res0_a = horizontal_vector_path(ch0_a);
      uint8x8_t res0_b = horizontal_vector_path(ch0_b);
      uint8x8_t res1_a = horizontal_vector_path(ch1_a);
      uint8x8_t res1_b = horizontal_vector_path(ch1_b);
      uint8x8_t res2_a = horizontal_vector_path(ch2_a);
      uint8x8_t res2_b = horizontal_vector_path(ch2_b);

      uint8x8_t out0 = vuzp1_u8(res0_a, res0_b);
      uint8x8_t out1 = vuzp1_u8(res1_a, res1_b);
      uint8x8_t out2 = vuzp1_u8(res2_a, res2_b);

      uint8x8x3_t interleaved{out0, out1, out2};
      vst3_u8(&dst_rows.at(0, column / 2)[0], interleaved);
    });

    loop.remaining([&](ptrdiff_t column, size_t max_column) {
      column = align_up(column, 2);
      while (column < static_cast<ptrdiff_t>(max_column)) {
        Rows<const BufferType> src_row = src_rows.at(0, column);
        Rows<DestinationType> dst_row = dst_rows.at(0, column / 2);
        for (ptrdiff_t channel = 0; channel < channels; ++channel) {
          process_horizontal_scalar(src_row, dst_row, border_offsets, channel,
                                    channel);
        }
        column += 2;
      }
    });
  }

  uint8x8_t const_6_u8_half_;
  uint16x8_t const_6_u16_;
  uint16x8_t const_4_u16_;
};  // end of class BlurAndDownsample

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t kleidicv_blur_and_downsample_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, (src_height + 1) / 2);
  CHECK_IMAGE_SIZE(src_width, src_height);

  Rectangle rect{src_width, src_height};
  constexpr size_t intermediate_size{
      sizeof(typename BlurAndDownsample::BufferType)};

  auto workspace_variant = BlurAndDownsampleFilterWorkspace::create(
      rect, channels, intermediate_size);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace =
      *std::get_if<BlurAndDownsampleFilterWorkspace>(&workspace_variant);

  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<uint8_t> dst_rows{dst, dst_stride, channels};
  workspace.process(y_begin, y_end, src_rows, dst_rows, fixed_border_type,
                    BlurAndDownsample{});

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
