// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SEPARABLE_FILTER_21X21_NEON_H
#define KLEIDICV_SEPARABLE_FILTER_21X21_NEON_H

#include "kleidicv/neon.h"
#include "kleidicv/workspace/border_21x21.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Template for drivers of separable NxM filters.
template <typename FilterType, const size_t S>
class SeparableFilter;

// Driver for a separable 21x21 filter.
template <typename FilterType>
class SeparableFilter<FilterType, 21UL> {
 public:
  using SourceType = typename FilterType::SourceType;
  using BufferType = typename FilterType::BufferType;
  using DestinationType = typename FilterType::DestinationType;
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits = typename neon::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo21x21<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  explicit SeparableFilter(FilterType filter) : filter_{filter} {}

  static constexpr size_t margin = 10UL;

  void process_vertical(size_t width, Rows<const SourceType> src_rows,
                        Rows<BufferType> dst_rows,
                        BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) {
      SourceVectorType src[21];
      src[0] = vld1q(&src_rows.at(border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(border_offsets.c2())[index]);
      src[3] = vld1q(&src_rows.at(border_offsets.c3())[index]);
      src[4] = vld1q(&src_rows.at(border_offsets.c4())[index]);
      src[5] = vld1q(&src_rows.at(border_offsets.c5())[index]);
      src[6] = vld1q(&src_rows.at(border_offsets.c6())[index]);
      src[7] = vld1q(&src_rows.at(border_offsets.c7())[index]);
      src[8] = vld1q(&src_rows.at(border_offsets.c8())[index]);
      src[9] = vld1q(&src_rows.at(border_offsets.c9())[index]);
      src[10] = vld1q(&src_rows.at(border_offsets.c10())[index]);
      src[11] = vld1q(&src_rows.at(border_offsets.c11())[index]);
      src[12] = vld1q(&src_rows.at(border_offsets.c12())[index]);
      src[13] = vld1q(&src_rows.at(border_offsets.c13())[index]);
      src[14] = vld1q(&src_rows.at(border_offsets.c14())[index]);
      src[15] = vld1q(&src_rows.at(border_offsets.c15())[index]);
      src[16] = vld1q(&src_rows.at(border_offsets.c16())[index]);
      src[17] = vld1q(&src_rows.at(border_offsets.c17())[index]);
      src[18] = vld1q(&src_rows.at(border_offsets.c18())[index]);
      src[19] = vld1q(&src_rows.at(border_offsets.c19())[index]);
      src[20] = vld1q(&src_rows.at(border_offsets.c20())[index]);
      filter_.vertical_vector_path(src, &dst_rows[index]);
    });

    // No tail path needed in NEON, because TryToAvoidTailPath works for any
    // supported size (i.e. the minimum size is kernel_size - 1, which is 20,
    // and the NEON vector length is 16 which is smaller than that).
  }

  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](size_t index) {
      auto src_0 = &src_rows.at(0, border_offsets.c0())[index];
      auto src_1 = &src_rows.at(0, border_offsets.c1())[index];
      auto src_2 = &src_rows.at(0, border_offsets.c2())[index];
      auto src_3 = &src_rows.at(0, border_offsets.c3())[index];
      auto src_4 = &src_rows.at(0, border_offsets.c4())[index];
      auto src_5 = &src_rows.at(0, border_offsets.c5())[index];
      auto src_6 = &src_rows.at(0, border_offsets.c6())[index];
      auto src_7 = &src_rows.at(0, border_offsets.c7())[index];
      auto src_8 = &src_rows.at(0, border_offsets.c8())[index];
      auto src_9 = &src_rows.at(0, border_offsets.c9())[index];
      auto src_10 = &src_rows.at(0, border_offsets.c10())[index];
      auto src_11 = &src_rows.at(0, border_offsets.c11())[index];
      auto src_12 = &src_rows.at(0, border_offsets.c12())[index];
      auto src_13 = &src_rows.at(0, border_offsets.c13())[index];
      auto src_14 = &src_rows.at(0, border_offsets.c14())[index];
      auto src_15 = &src_rows.at(0, border_offsets.c15())[index];
      auto src_16 = &src_rows.at(0, border_offsets.c16())[index];
      auto src_17 = &src_rows.at(0, border_offsets.c17())[index];
      auto src_18 = &src_rows.at(0, border_offsets.c18())[index];
      auto src_19 = &src_rows.at(0, border_offsets.c19())[index];
      auto src_20 = &src_rows.at(0, border_offsets.c20())[index];

      BufferVectorType src_a[21], src_b[21];
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
      src_a[5] = vld1q(&src_5[0]);
      src_b[5] = vld1q(&src_5[BufferVecTraits::num_lanes()]);
      src_a[6] = vld1q(&src_6[0]);
      src_b[6] = vld1q(&src_6[BufferVecTraits::num_lanes()]);
      src_a[7] = vld1q(&src_7[0]);
      src_b[7] = vld1q(&src_7[BufferVecTraits::num_lanes()]);
      src_a[8] = vld1q(&src_8[0]);
      src_b[8] = vld1q(&src_8[BufferVecTraits::num_lanes()]);
      src_a[9] = vld1q(&src_9[0]);
      src_b[9] = vld1q(&src_9[BufferVecTraits::num_lanes()]);
      src_a[10] = vld1q(&src_10[0]);
      src_b[10] = vld1q(&src_10[BufferVecTraits::num_lanes()]);
      src_a[11] = vld1q(&src_11[0]);
      src_b[11] = vld1q(&src_11[BufferVecTraits::num_lanes()]);
      src_a[12] = vld1q(&src_12[0]);
      src_b[12] = vld1q(&src_12[BufferVecTraits::num_lanes()]);
      src_a[13] = vld1q(&src_13[0]);
      src_b[13] = vld1q(&src_13[BufferVecTraits::num_lanes()]);
      src_a[14] = vld1q(&src_14[0]);
      src_b[14] = vld1q(&src_14[BufferVecTraits::num_lanes()]);
      src_a[15] = vld1q(&src_15[0]);
      src_b[15] = vld1q(&src_15[BufferVecTraits::num_lanes()]);
      src_a[16] = vld1q(&src_16[0]);
      src_b[16] = vld1q(&src_16[BufferVecTraits::num_lanes()]);
      src_a[17] = vld1q(&src_17[0]);
      src_b[17] = vld1q(&src_17[BufferVecTraits::num_lanes()]);
      src_a[18] = vld1q(&src_18[0]);
      src_b[18] = vld1q(&src_18[BufferVecTraits::num_lanes()]);
      src_a[19] = vld1q(&src_19[0]);
      src_b[19] = vld1q(&src_19[BufferVecTraits::num_lanes()]);
      src_a[20] = vld1q(&src_20[0]);
      src_b[20] = vld1q(&src_20[BufferVecTraits::num_lanes()]);

      filter_.horizontal_vector_path(src_a, &dst_rows[index]);
      filter_.horizontal_vector_path(
          src_b, &dst_rows[index + BufferVecTraits::num_lanes()]);
    });

    loop.unroll_once([&](size_t index) {
      BufferVectorType src[21];
      src[0] = vld1q(&src_rows.at(0, border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(0, border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(0, border_offsets.c2())[index]);
      src[3] = vld1q(&src_rows.at(0, border_offsets.c3())[index]);
      src[4] = vld1q(&src_rows.at(0, border_offsets.c4())[index]);
      src[5] = vld1q(&src_rows.at(0, border_offsets.c5())[index]);
      src[6] = vld1q(&src_rows.at(0, border_offsets.c6())[index]);
      src[7] = vld1q(&src_rows.at(0, border_offsets.c7())[index]);
      src[8] = vld1q(&src_rows.at(0, border_offsets.c8())[index]);
      src[9] = vld1q(&src_rows.at(0, border_offsets.c9())[index]);
      src[10] = vld1q(&src_rows.at(0, border_offsets.c10())[index]);
      src[11] = vld1q(&src_rows.at(0, border_offsets.c11())[index]);
      src[12] = vld1q(&src_rows.at(0, border_offsets.c12())[index]);
      src[13] = vld1q(&src_rows.at(0, border_offsets.c13())[index]);
      src[14] = vld1q(&src_rows.at(0, border_offsets.c14())[index]);
      src[15] = vld1q(&src_rows.at(0, border_offsets.c15())[index]);
      src[16] = vld1q(&src_rows.at(0, border_offsets.c16())[index]);
      src[17] = vld1q(&src_rows.at(0, border_offsets.c17())[index]);
      src[18] = vld1q(&src_rows.at(0, border_offsets.c18())[index]);
      src[19] = vld1q(&src_rows.at(0, border_offsets.c19())[index]);
      src[20] = vld1q(&src_rows.at(0, border_offsets.c20())[index]);
      filter_.horizontal_vector_path(src, &dst_rows[index]);
    });

    loop.tail([&](size_t index) {
      process_horizontal_scalar(src_rows, dst_rows, border_offsets, index);
    });
  }

  void process_horizontal_borders(Rows<const BufferType> src_rows,
                                  Rows<DestinationType> dst_rows,
                                  BorderOffsets border_offsets) const {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      disable_loop_vectorization();
      process_horizontal_scalar(src_rows, dst_rows, border_offsets, index);
    }
  }

 private:
  void process_horizontal_scalar(Rows<const BufferType> src_rows,
                                 Rows<DestinationType> dst_rows,
                                 BorderOffsets border_offsets,
                                 size_t index) const {
    BufferType src[21];
    src[0] = src_rows.at(0, border_offsets.c0())[index];
    src[1] = src_rows.at(0, border_offsets.c1())[index];
    src[2] = src_rows.at(0, border_offsets.c2())[index];
    src[3] = src_rows.at(0, border_offsets.c3())[index];
    src[4] = src_rows.at(0, border_offsets.c4())[index];
    src[5] = src_rows.at(0, border_offsets.c5())[index];
    src[6] = src_rows.at(0, border_offsets.c6())[index];
    src[7] = src_rows.at(0, border_offsets.c7())[index];
    src[8] = src_rows.at(0, border_offsets.c8())[index];
    src[9] = src_rows.at(0, border_offsets.c9())[index];
    src[10] = src_rows.at(0, border_offsets.c10())[index];
    src[11] = src_rows.at(0, border_offsets.c11())[index];
    src[12] = src_rows.at(0, border_offsets.c12())[index];
    src[13] = src_rows.at(0, border_offsets.c13())[index];
    src[14] = src_rows.at(0, border_offsets.c14())[index];
    src[15] = src_rows.at(0, border_offsets.c15())[index];
    src[16] = src_rows.at(0, border_offsets.c16())[index];
    src[17] = src_rows.at(0, border_offsets.c17())[index];
    src[18] = src_rows.at(0, border_offsets.c18())[index];
    src[19] = src_rows.at(0, border_offsets.c19())[index];
    src[20] = src_rows.at(0, border_offsets.c20())[index];
    filter_.horizontal_scalar_path(src, &dst_rows[index]);
  }

  FilterType filter_;
};  // end of class SeparableFilter<FilterType, 21UL>

// Shorthand for 21x21 separable filters driver type.
template <class FilterType>
using SeparableFilter21x21 = SeparableFilter<FilterType, 21UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SEPARABLE_FILTER_21X21_NEON_H
