// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MORPHOLOGY_SC_H
#define KLEIDICV_MORPHOLOGY_SC_H

#include <cstdio>

#include "kleidicv/morphology/morphology.h"
#include "kleidicv/morphology/workspace.h"
#include "kleidicv/sve2.h"
#include "kleidicv/types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename T>
class CopyDataSVE2 {
  class CopyOperation final : public UnrollTwice {
   public:
    using ContextType = Context;
    using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<T>;
    using VectorType = typename VecTraits::VectorType;

    VectorType vector_path(ContextType, VectorType src) KLEIDICV_STREAMING {
      return src;
    }
  };  // end of class CopyOperation

 public:
  void operator()(Rows<const T> src_rows, Rows<T> dst_rows,
                  size_t length) const KLEIDICV_STREAMING {
    // 'apply_operation_by_rows' can only handle one channel well
    // so width must be multiplied in order to copy all the data
    Rectangle rect{length * dst_rows.channels(), std::size_t{1}};
    Rows<const T> src_1ch{&src_rows[0], src_rows.stride(), 1};
    Rows<T> dst_1ch{&dst_rows[0], dst_rows.stride(), 1};
    CopyOperation op{};
    apply_operation_by_rows(op, rect, src_1ch, dst_1ch);
  }
};

template <typename ScalarType, typename O>
class VerticalOp final {
 public:
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using Vector4Type = typename VecTraits::Vector4Type;
  using Vector2Type = typename VecTraits::Vector2Type;

  VerticalOp(Rectangle rect, Rectangle kernel) KLEIDICV_STREAMING
      : rect_(rect),
        kernel_(kernel) {}

  void process_rows(IndirectRows<ScalarType> src_rows,
                    Rows<ScalarType> dst_rows) KLEIDICV_STREAMING {
    if (KLEIDICV_UNLIKELY(kernel_.height()) == 1) {
      CopyRows<ScalarType>::copy_rows(rect_, src_rows, dst_rows);
      return;
    }

    // Iterate across the rows from top to bottom. This implementation can
    // handle two rows at once.
    for (size_t height = 0; height < rect_.height(); height += 2) {
      // Iterate across the columns from left to right.
      LoopUnroll2 loop{rect_.width() * src_rows.channels(),
                       VecTraits::num_lanes()};
      // clang-format off
      loop.unroll_four_times([&](size_t index) KLEIDICV_STREAMING {
            vector_path_4x(src_rows, dst_rows, index, height);
          })
          .unroll_twice([&](size_t index) KLEIDICV_STREAMING {
            vector_path_2x(src_rows, dst_rows, index, height);
          })
          .remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
            svbool_t pg = VecTraits::svwhilelt(index, length);
            while (svptest_first(VecTraits::svptrue(), pg)) {
              vector_path(pg, src_rows, dst_rows, index, height);
              index += VecTraits::num_lanes();
              pg = VecTraits::svwhilelt(index, length);
            }
          });
      // clang-format on
      src_rows += 2;
      dst_rows += 2;
    }
  }

 private:
  void vector_path_4x(IndirectRows<ScalarType> src_rows,
                      Rows<ScalarType> dst_rows, const size_t index,
                      const size_t height) KLEIDICV_STREAMING {
    const ScalarType *src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    svcount_t p_counter = VecTraits::svptrue_c();
    Vector4Type v = svld1_x4(p_counter, &src_row[0]);
    auto first_row0 = svget4(v, 0);
    auto first_row1 = svget4(v, 1);
    auto first_row2 = svget4(v, 2);
    auto first_row3 = svget4(v, 3);
#else
    auto first_row0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto first_row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
    auto first_row2 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 2);
    auto first_row3 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 3);
#endif  // KLEIDICV_TARGET_SME2
    ++src_rows;

    src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    v = svld1_x4(p_counter, &src_row[0]);
    auto acc0 = svget4(v, 0);
    auto acc1 = svget4(v, 1);
    auto acc2 = svget4(v, 2);
    auto acc3 = svget4(v, 3);
#else
    auto acc0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto acc1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
    auto acc2 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 2);
    auto acc3 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 3);
#endif  // KLEIDICV_TARGET_SME2
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) KLEIDICV_STREAMING {
      const ScalarType *src_row0 = &src_rows.at(0)[index];
      const ScalarType *src_row1 = &src_rows.at(1)[index];
#if KLEIDICV_TARGET_SME2
      v = svld1_x4(p_counter, src_row0);
      auto row00 = svget4(v, 0);
      auto row01 = svget4(v, 1);
      auto row02 = svget4(v, 2);
      auto row03 = svget4(v, 3);
      v = svld1_x4(p_counter, src_row1);
      auto row10 = svget4(v, 0);
      auto row11 = svget4(v, 1);
      auto row12 = svget4(v, 2);
      auto row13 = svget4(v, 3);
#else
      auto row00 = svld1(VecTraits::svptrue(), src_row0);
      auto row01 = svld1_vnum(VecTraits::svptrue(), src_row0, 1);
      auto row02 = svld1_vnum(VecTraits::svptrue(), src_row0, 2);
      auto row03 = svld1_vnum(VecTraits::svptrue(), src_row0, 3);
      auto row10 = svld1(VecTraits::svptrue(), src_row1);
      auto row11 = svld1_vnum(VecTraits::svptrue(), src_row1, 1);
      auto row12 = svld1_vnum(VecTraits::svptrue(), src_row1, 2);
      auto row13 = svld1_vnum(VecTraits::svptrue(), src_row1, 3);
#endif  // KLEIDICV_TARGET_SME2
      acc0 = O::operation(VecTraits::svptrue(), acc0,
                          O::operation(VecTraits::svptrue(), row00, row10));
      acc1 = O::operation(VecTraits::svptrue(), acc1,
                          O::operation(VecTraits::svptrue(), row01, row11));
      acc2 = O::operation(VecTraits::svptrue(), acc2,
                          O::operation(VecTraits::svptrue(), row02, row12));
      acc3 = O::operation(VecTraits::svptrue(), acc3,
                          O::operation(VecTraits::svptrue(), row03, row13));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */)  // NOLINT(readability/casting)
              KLEIDICV_STREAMING {
                const ScalarType *src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
                v = svld1_x4(p_counter, &src_row[0]);
                auto row0 = svget4(v, 0);
                auto row1 = svget4(v, 1);
                auto row2 = svget4(v, 2);
                auto row3 = svget4(v, 3);
#else
                auto row0 = svld1(VecTraits::svptrue(), &src_row[0]);
                auto row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
                auto row2 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 2);
                auto row3 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 3);
#endif  // KLEIDICV_TARGET_SME2
                acc0 = O::operation(VecTraits::svptrue(), acc0, row0);
                acc1 = O::operation(VecTraits::svptrue(), acc1, row1);
                acc2 = O::operation(VecTraits::svptrue(), acc2, row2);
                acc3 = O::operation(VecTraits::svptrue(), acc3, row3);
                ++src_rows;
              });

    // Save partial results which do not contain the first row.
    auto partial_acc0 = acc0;
    auto partial_acc1 = acc1;
    auto partial_acc2 = acc2;
    auto partial_acc3 = acc3;

    // Take the first row into account.
    acc0 = O::operation(VecTraits::svptrue(), acc0, first_row0);
    acc1 = O::operation(VecTraits::svptrue(), acc1, first_row1);
    acc2 = O::operation(VecTraits::svptrue(), acc2, first_row2);
    acc3 = O::operation(VecTraits::svptrue(), acc3, first_row3);

    // Store the results.
    ScalarType *dst_row = &dst_rows[index];
#if KLEIDICV_TARGET_SME2
    Vector4Type res4 = svcreate4(acc0, acc1, acc2, acc3);
    svst1(p_counter, &dst_row[0], res4);
#else
    svst1(VecTraits::svptrue(), &dst_row[0], acc0);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 1, acc1);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 2, acc2);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 3, acc3);
#endif  // KLEIDICV_TARGET_SME2

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    v = svld1_x4(p_counter, &src_row[0]);
    auto next_row0 = svget4(v, 0);
    auto next_row1 = svget4(v, 1);
    auto next_row2 = svget4(v, 2);
    auto next_row3 = svget4(v, 3);
#else
    auto next_row0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto next_row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
    auto next_row2 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 2);
    auto next_row3 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 3);
#endif  // KLEIDICV_TARGET_SME2

    acc0 = O::operation(VecTraits::svptrue(), partial_acc0, next_row0);
    acc1 = O::operation(VecTraits::svptrue(), partial_acc1, next_row1);
    acc2 = O::operation(VecTraits::svptrue(), partial_acc2, next_row2);
    acc3 = O::operation(VecTraits::svptrue(), partial_acc3, next_row3);

    // Store the results.
    dst_row = &dst_rows[index];
#if KLEIDICV_TARGET_SME2
    res4 = svcreate4(acc0, acc1, acc2, acc3);
    svst1(p_counter, &dst_row[0], res4);
#else
    svst1(VecTraits::svptrue(), &dst_row[0], acc0);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 1, acc1);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 2, acc2);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 3, acc3);
#endif  // KLEIDICV_TARGET_SME2
  }

  void vector_path_2x(IndirectRows<ScalarType> src_rows,
                      Rows<ScalarType> dst_rows, const size_t index,
                      const size_t height) KLEIDICV_STREAMING {
    const ScalarType *src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    svcount_t p_counter = VecTraits::svptrue_c();
    Vector2Type v = svld1_x2(p_counter, &src_row[0]);
    auto first_row0 = svget2(v, 0);
    auto first_row1 = svget2(v, 1);
#else
    auto first_row0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto first_row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
#endif  // KLEIDICV_TARGET_SME2

    ++src_rows;

    src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    v = svld1_x2(p_counter, &src_row[0]);
    auto acc0 = svget2(v, 0);
    auto acc1 = svget2(v, 1);
#else
    auto acc0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto acc1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
#endif  // KLEIDICV_TARGET_SME2
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) KLEIDICV_STREAMING {
      const ScalarType *src_row0 = &src_rows.at(0)[index];
      const ScalarType *src_row1 = &src_rows.at(1)[index];
#if KLEIDICV_TARGET_SME2
      Vector2Type v0 = svld1_x2(p_counter, src_row0);
      Vector2Type v1 = svld1_x2(p_counter, src_row1);
      auto row00 = svget2(v0, 0);
      auto row01 = svget2(v0, 1);
      auto row10 = svget2(v1, 0);
      auto row11 = svget2(v1, 1);
#else
      auto row00 = svld1(VecTraits::svptrue(), src_row0);
      auto row01 = svld1_vnum(VecTraits::svptrue(), src_row0, 1);
      auto row10 = svld1(VecTraits::svptrue(), src_row1);
      auto row11 = svld1_vnum(VecTraits::svptrue(), src_row1, 1);
#endif  // KLEIDICV_TARGET_SME2
      acc0 = O::operation(VecTraits::svptrue(), acc0,
                          O::operation(VecTraits::svptrue(), row00, row10));
      acc1 = O::operation(VecTraits::svptrue(), acc1,
                          O::operation(VecTraits::svptrue(), row01, row11));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */)  // NOLINT(readability/casting)
              KLEIDICV_STREAMING {
                const ScalarType *src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
                v = svld1_x2(p_counter, &src_row[0]);
                auto row0 = svget2(v, 0);
                auto row1 = svget2(v, 1);
#else
                auto row0 = svld1(VecTraits::svptrue(), &src_row[0]);
                auto row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
#endif  // KLEIDICV_TARGET_SME2
                acc0 = O::operation(VecTraits::svptrue(), acc0, row0);
                acc1 = O::operation(VecTraits::svptrue(), acc1, row1);
                ++src_rows;
              });

    // Save partial results which do not contain the first row.
    auto partial_acc0 = acc0;
    auto partial_acc1 = acc1;

    // Take the first row into account.
    acc0 = O::operation(VecTraits::svptrue(), acc0, first_row0);
    acc1 = O::operation(VecTraits::svptrue(), acc1, first_row1);

    // Store the results.
    ScalarType *dst_row = &dst_rows[index];
#if KLEIDICV_TARGET_SME2
    Vector2Type res2 = svcreate2(acc0, acc1);
    svst1(p_counter, &dst_row[0], res2);
#else
    svst1(VecTraits::svptrue(), &dst_row[0], acc0);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 1, acc1);
#endif  // KLEIDICV_TARGET_SME2

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    v = svld1_x2(p_counter, &src_row[0]);
    auto next_row0 = svget2(v, 0);
    auto next_row1 = svget2(v, 1);
#else
    auto next_row0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto next_row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
#endif  // KLEIDICV_TARGET_SME2

    acc0 = O::operation(VecTraits::svptrue(), partial_acc0, next_row0);
    acc1 = O::operation(VecTraits::svptrue(), partial_acc1, next_row1);

    dst_row = &dst_rows[index];
#if KLEIDICV_TARGET_SME2
    res2 = svcreate2(acc0, acc1);
    svst1(p_counter, &dst_row[0], res2);
#else
    svst1(VecTraits::svptrue(), &dst_row[0], acc0);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 1, acc1);
#endif  // KLEIDICV_TARGET_SME2
  }

  void vector_path(svbool_t pg, IndirectRows<ScalarType> src_rows,
                   Rows<ScalarType> dst_rows, const size_t index,
                   const size_t height) KLEIDICV_STREAMING {
    auto first_row = svld1(pg, &src_rows[index]);
    ++src_rows;

    auto acc = svld1(pg, &src_rows[index]);
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) KLEIDICV_STREAMING {
      auto row0 = svld1(pg, &src_rows.at(0)[index]);
      auto row1 = svld1(pg, &src_rows.at(1)[index]);
      acc = O::operation(pg, acc, O::operation(pg, row0, row1));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */)  // NOLINT(readability/casting)
              KLEIDICV_STREAMING {
                auto row = svld1(pg, &src_rows[index]);
                acc = O::operation(pg, acc, row);
                ++src_rows;
              });

    // Save partial result which does not contain the first row.
    auto partial_acc = acc;

    // Take the first row into account.
    acc = O::operation(pg, acc, first_row);

    // Store the results.
    svst1(pg, &dst_rows[index], acc);

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    auto next_row = svld1(pg, &src_rows[index]);
    acc = O::operation(pg, partial_acc, next_row);
    svst1(pg, &dst_rows[index], acc);
  }

  Rectangle rect_;
  Rectangle kernel_;
};  // end of class VerticalOp<ScalarType, )>

template <typename ScalarType, typename O>
class HorizontalOp final {
 public:
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using Vector4Type = typename VecTraits::Vector4Type;
  using Vector2Type = typename VecTraits::Vector2Type;

  HorizontalOp(Rectangle rect, Rectangle kernel) KLEIDICV_STREAMING
      : rect_(rect),
        kernel_(kernel) {}

  void process_rows(Rows<const ScalarType> src_rows,
                    Rows<ScalarType> dst_rows) KLEIDICV_STREAMING {
    // Iterate across the rows from top to bottom.
    for (size_t height = 0; height < rect_.height(); ++height) {
      // Iterate across the columns from left to right.
      LoopUnroll2 loop{rect_.width() * src_rows.channels(),
                       VecTraits::num_lanes()};
      // clang-format off
      loop.unroll_four_times([&](size_t index) KLEIDICV_STREAMING {
            vector_path_4x(src_rows, dst_rows, index);
          })
          .unroll_twice([&](size_t index) KLEIDICV_STREAMING {
            vector_path_2x(src_rows, dst_rows, index);
          })
          .remaining([&](size_t index, size_t length) KLEIDICV_STREAMING {
            svbool_t pg = VecTraits::svwhilelt(index, length);
            while (svptest_first(VecTraits::svptrue(), pg)) {
              vector_path(pg, src_rows, dst_rows, index);
              index += VecTraits::num_lanes();
              pg = VecTraits::svwhilelt(index, length);
            }
          });
      // clang-format on
      ++src_rows;
      ++dst_rows;
    }
  }

 private:
  void vector_path_4x(Rows<const ScalarType> src_rows,
                      Rows<ScalarType> dst_rows,
                      const size_t index) KLEIDICV_STREAMING {
    const auto *src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    svcount_t p_counter = VecTraits::svptrue_c();
    Vector4Type v = svld1_x4(p_counter, &src_row[0]);
    auto acc0 = svget4(v, 0);
    auto acc1 = svget4(v, 1);
    auto acc2 = svget4(v, 2);
    auto acc3 = svget4(v, 3);
#else
    auto acc0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto acc1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
    auto acc2 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 2);
    auto acc3 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 3);
#endif  // KLEIDICV_TARGET_SME2

    for (size_t width = 1; width < kernel_.width(); ++width) {
      src_row = &src_rows[index + width * src_rows.channels()];
#if KLEIDICV_TARGET_SME2
      Vector4Type v = svld1_x4(p_counter, &src_row[0]);
      auto row0 = svget4(v, 0);
      auto row1 = svget4(v, 1);
      auto row2 = svget4(v, 2);
      auto row3 = svget4(v, 3);
#else
      auto row0 = svld1(VecTraits::svptrue(), &src_row[0]);
      auto row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
      auto row2 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 2);
      auto row3 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 3);
#endif  // KLEIDICV_TARGET_SME2
      acc0 = O::operation(VecTraits::svptrue(), acc0, row0);
      acc1 = O::operation(VecTraits::svptrue(), acc1, row1);
      acc2 = O::operation(VecTraits::svptrue(), acc2, row2);
      acc3 = O::operation(VecTraits::svptrue(), acc3, row3);
    }

    auto dst_row = &dst_rows[index];
#if KLEIDICV_TARGET_SME2
    Vector4Type res4 = svcreate4(acc0, acc1, acc2, acc3);
    svst1(p_counter, &dst_row[0], res4);
#else
    svst1(VecTraits::svptrue(), &dst_row[0], acc0);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 1, acc1);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 2, acc2);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 3, acc3);
#endif  // KLEIDICV_TARGET_SME2
  }

  void vector_path_2x(Rows<const ScalarType> src_rows,
                      Rows<ScalarType> dst_rows,
                      const size_t index) KLEIDICV_STREAMING {
    const auto *src_row = &src_rows[index];
#if KLEIDICV_TARGET_SME2
    svcount_t p_counter = VecTraits::svptrue_c();
    Vector2Type v = svld1_x2(p_counter, &src_row[0]);
    auto acc0 = svget2(v, 0);
    auto acc1 = svget2(v, 1);
#else
    auto acc0 = svld1(VecTraits::svptrue(), &src_row[0]);
    auto acc1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
#endif  // KLEIDICV_TARGET_SME2

    for (size_t width = 1; width < kernel_.width(); ++width) {
      src_row = &src_rows[index + width * src_rows.channels()];
#if KLEIDICV_TARGET_SME2
      v = svld1_x2(p_counter, &src_row[0]);
      auto row0 = svget2(v, 0);
      auto row1 = svget2(v, 1);
#else
      auto row0 = svld1(VecTraits::svptrue(), &src_row[0]);
      auto row1 = svld1_vnum(VecTraits::svptrue(), &src_row[0], 1);
#endif  // KLEIDICV_TARGET_SME2
      acc0 = O::operation(VecTraits::svptrue(), acc0, row0);
      acc1 = O::operation(VecTraits::svptrue(), acc1, row1);
    }

    auto dst_row = &dst_rows[index];
#if KLEIDICV_TARGET_SME2
    Vector2Type res2 = svcreate2(acc0, acc1);
    svst1(p_counter, &dst_row[0], res2);
#else
    svst1(VecTraits::svptrue(), &dst_row[0], acc0);
    svst1_vnum(VecTraits::svptrue(), &dst_row[0], 1, acc1);
#endif  // KLEIDICV_TARGET_SME2
  }

  void vector_path(svbool_t pg, Rows<const ScalarType> src_rows,
                   Rows<ScalarType> dst_rows,
                   const size_t index) KLEIDICV_STREAMING {
    auto acc = svld1(pg, &src_rows[index]);

    for (size_t width = 1; width < kernel_.width(); ++width) {
      const auto *src_row = &src_rows[index + width * src_rows.channels()];
      acc = O::operation(pg, acc, svld1(pg, &src_row[0]));
    }

    svst1(pg, &dst_rows[index], acc);
  }

  Rectangle rect_;
  Rectangle kernel_;
};  // end of class HorizontalOp<ScalarType>

template <typename ScalarType>
class Min final {
 public:
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  static VectorType operation(svbool_t pg, VectorType lhs,
                              VectorType rhs) KLEIDICV_STREAMING {
    return svmin_x(pg, lhs, rhs);
  }
};  // end of class Min<ScalarType>

template <typename ScalarType>
class Max final {
 public:
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  static VectorType operation(svbool_t pg, VectorType lhs,
                              VectorType rhs) KLEIDICV_STREAMING {
    return svmax_x(pg, lhs, rhs);
  }
};  // end of class Max<ScalarType>

template <typename T>
using VerticalMin = VerticalOp<T, Min<T>>;
template <typename T>
using VerticalMax = VerticalOp<T, Max<T>>;

template <typename T>
using HorizontalMin = HorizontalOp<T, Min<T>>;
template <typename T>
using HorizontalMax = HorizontalOp<T, Max<T>>;

template <typename ScalarType, typename CopyDataOperation>
class DilateOperation final {
 public:
  using SourceType = ScalarType;
  using BufferType = ScalarType;
  using DestinationType = ScalarType;
  using CopyData = CopyDataOperation;

  explicit DilateOperation(Rectangle kernel) KLEIDICV_STREAMING
      : kernel_{kernel} {}

  void process_horizontal(Rectangle rect, Rows<const SourceType> src_rows,
                          Rows<BufferType> dst_rows) KLEIDICV_STREAMING {
    HorizontalMax<ScalarType>{rect, kernel_}.process_rows(src_rows, dst_rows);
  }

  void process_vertical(Rectangle rect, IndirectRows<BufferType> src_rows,
                        Rows<DestinationType> dst_rows) KLEIDICV_STREAMING {
    VerticalMax<ScalarType>{rect, kernel_}.process_rows(src_rows, dst_rows);
  }

 private:
  Rectangle kernel_;
};  // end of class DilateOperation<ScalarType>

template <typename T, typename CopyOperation>
static kleidicv_error_t dilate_sc(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t channels, size_t kernel_width, size_t kernel_height,
    size_t anchor_x, size_t anchor_y, kleidicv_border_type_t border_type,
    const uint8_t *border_value, size_t iterations) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  CHECK_IMAGE_SIZE(kernel_width, kernel_height);
  auto morphology_border_type =
      MorphologyWorkspace::get_border_type(border_type);
  if (!morphology_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  if (!morphology_is_implemented(width, height, kernel_width, kernel_height,
                                 channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};
  Rectangle kernel_rect{kernel_width, kernel_height};
  Point anchor{anchor_x, anchor_y};

  auto workspace_variant = MorphologyWorkspace::create(
      kernel_rect, anchor, *morphology_border_type, border_value, channels,
      sizeof(uint8_t), rect);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<MorphologyWorkspace>(&workspace_variant);

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  Rows<const T> current_src_rows = src_rows;
  Rows<T> current_dst_rows = dst_rows;
  for (size_t i = 0; i < iterations; ++i) {
    DilateOperation<T, CopyOperation> operation{kernel_rect};
    workspace.process(current_src_rows, current_dst_rows, operation);
    // Update source for the next iteration.
    current_src_rows = dst_rows;
  }
  return KLEIDICV_OK;
}

// Helper structure for erode.
template <typename ScalarType, typename CopyDataOperation>
class ErodeOperation final {
 public:
  using SourceType = ScalarType;
  using BufferType = ScalarType;
  using DestinationType = ScalarType;
  using CopyData = CopyDataOperation;

  explicit ErodeOperation(Rectangle kernel) KLEIDICV_STREAMING
      : kernel_{kernel} {}

  void process_horizontal(Rectangle rect, Rows<const SourceType> src_rows,
                          Rows<BufferType> dst_rows) KLEIDICV_STREAMING {
    HorizontalMin<ScalarType>{rect, kernel_}.process_rows(src_rows, dst_rows);
  }

  void process_vertical(Rectangle rect, IndirectRows<BufferType> src_rows,
                        Rows<DestinationType> dst_rows) KLEIDICV_STREAMING {
    VerticalMin<ScalarType>{rect, kernel_}.process_rows(src_rows, dst_rows);
  }

 private:
  Rectangle kernel_;
};  // end of class ErodeOperation<ScalarType>

template <typename T, typename CopyOperation>
static kleidicv_error_t erode_sc(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t channels, size_t kernel_width, size_t kernel_height,
    size_t anchor_x, size_t anchor_y, kleidicv_border_type_t border_type,
    const uint8_t *border_value, size_t iterations) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  CHECK_IMAGE_SIZE(kernel_width, kernel_height);
  auto morphology_border_type =
      MorphologyWorkspace::get_border_type(border_type);
  if (!morphology_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  if (!morphology_is_implemented(width, height, kernel_width, kernel_height,
                                 channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};
  Rectangle kernel_rect{kernel_width, kernel_height};
  Point anchor{anchor_x, anchor_y};

  auto workspace_variant = MorphologyWorkspace::create(
      kernel_rect, anchor, *morphology_border_type, border_value, channels,
      sizeof(uint8_t), rect);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<MorphologyWorkspace>(&workspace_variant);

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  Rows<const T> current_src_rows = src_rows;
  Rows<T> current_dst_rows = dst_rows;
  for (size_t i = 0; i < iterations; ++i) {
    ErodeOperation<T, CopyOperation> operation{kernel_rect};
    workspace.process(current_src_rows, current_dst_rows, operation);
    // Update source for the next iteration.
    current_src_rows = dst_rows;
  }
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MORPHOLOGY_SC_H
