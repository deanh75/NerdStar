// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RESIZE_SC_H
#define KLEIDICV_RESIZE_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

/// Resizes source data by averaging 4 elements to one.
/// In-place operation is not supported.
///
/// Only even source dimensions are supported.
/// Only single-channel images are supported.
/// The source is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Example of 2x2 to 1x1 conversion:
/// ```
/// | a | b | --> | (a+b+c+d)/4 |
/// | c | d |
/// ```

namespace KLEIDICV_TARGET_NAMESPACE {

static inline svuint8_t resize_parallel_vectors(
    svbool_t pg, svuint8_t top_row, svuint8_t bottom_row) KLEIDICV_STREAMING {
  svuint16_t result_before_averaging_b = svaddlb(top_row, bottom_row);
  svuint16_t result_before_averaging_t = svaddlt(top_row, bottom_row);
  svuint16_t result_before_averaging =
      svadd_x(pg, result_before_averaging_b, result_before_averaging_t);
  return svrshrnb(result_before_averaging, 2);
}

static inline void parallel_rows_vectors_path_2x(
    svbool_t pg, Rows<const uint8_t> src_rows,
    Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
  svcount_t pg_counter = svptrue_c8();
  auto src_top = svld1_x2(pg_counter, &src_rows.at(0)[0]);
  auto src_bottom = svld1_x2(pg_counter, &src_rows.at(1)[0]);
  svuint8_t top_row_0 = svget2(src_top, 0);
  svuint8_t top_row_1 = svget2(src_top, 1);
  svuint8_t bottom_row_0 = svget2(src_bottom, 0);
  svuint8_t bottom_row_1 = svget2(src_bottom, 1);
#else
  svuint8_t top_row_0 = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t bottom_row_0 = svld1(pg, &src_rows.at(1)[0]);
  svuint8_t top_row_1 = svld1_vnum(pg, &src_rows.at(0)[0], 1);
  svuint8_t bottom_row_1 = svld1_vnum(pg, &src_rows.at(1)[0], 1);
#endif  // KLEIDICV_TARGET_SME2
  svuint16_t sum0b = svaddlb(top_row_0, bottom_row_0);
  svuint16_t sum0t = svaddlt(top_row_0, bottom_row_0);
  svuint16_t sum1b = svaddlb(top_row_1, bottom_row_1);
  svuint16_t sum1t = svaddlt(top_row_1, bottom_row_1);
  svuint8_t res0 = svrshrnb(svadd_x(pg, sum0b, sum0t), 2);
  svuint8_t res1 = svrshrnb(svadd_x(pg, sum1b, sum1t), 2);
  svuint8_t result = svuzp1(res0, res1);
  svst1(pg, &dst_rows[0], result);
}

static inline void parallel_rows_vectors_path(
    svbool_t pg, Rows<const uint8_t> src_rows,
    Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svuint8_t top_line = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t bottom_line = svld1(pg, &src_rows.at(1)[0]);
  svuint8_t result = resize_parallel_vectors(pg, top_line, bottom_line);
  svst1b(pg, &dst_rows[0], svreinterpret_u16_u8(result));
}

template <typename ScalarType>
static inline void process_parallel_rows(
    Rows<const ScalarType> src_rows, size_t src_width,
    Rows<ScalarType> dst_rows) KLEIDICV_STREAMING {
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  const size_t size_mask = ~static_cast<size_t>(1U);

  // Process rows up to the last even pixel index.
  LoopUnroll2{src_width & size_mask, VecTraits::num_lanes()}
      // Process double vector chunks.
      .unroll_twice([&](size_t index) KLEIDICV_STREAMING {
        auto pg = VecTraits::svptrue();
        parallel_rows_vectors_path_2x(pg, src_rows.at(0, index),
                                      dst_rows.at(0, index / 2));
      })
      .unroll_once([&](size_t index) KLEIDICV_STREAMING {
        auto pg = VecTraits::svptrue();
        parallel_rows_vectors_path(pg, src_rows.at(0, index),
                                   dst_rows.at(0, index / 2));
      })
      // Process the remaining chunk of the row.
      .remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
        auto pg = VecTraits::svwhilelt(index, length);
        parallel_rows_vectors_path(pg, src_rows.at(0, index),
                                   dst_rows.at(0, index / 2));
      });
}

static inline svuint8_t resize_single_row(svbool_t pg,
                                          svuint8_t row) KLEIDICV_STREAMING {
  return svrshrnb(svadalp_x(pg, svdup_u16(0), row), 1);
}

KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t resize_to_quarter_u8_sc(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride) KLEIDICV_STREAMING {
  Rows<const uint8_t> src_rows{src, src_stride, /* channels*/ 1};
  Rows<uint8_t> dst_rows{dst, dst_stride, /* channels*/ 1};
  LoopUnroll2 loop{src_height, /* Process two rows */ 2};

  // Process two rows at once.
  loop.unroll_once([&](size_t)  // NOLINT(readability/casting)
                   KLEIDICV_STREAMING {
                     process_parallel_rows(src_rows, src_width, dst_rows);
                     src_rows += 2;
                     ++dst_rows;
                   });
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RESIZE_SC_H
