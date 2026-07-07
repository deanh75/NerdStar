// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SUM_SC_H
#define KLEIDICV_SUM_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType, typename ScalarTypeInternal>
class Sum;

template <>
class Sum<float, double> final : public UnrollTwice {
 public:
  using ScalarType = float;
  using ScalarTypeInternal = double;
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using VecTraitsInternal =
      KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarTypeInternal>;
  using VectorTypeInternal = typename VecTraitsInternal::VectorType;

  explicit Sum(VectorTypeInternal &accumulator) KLEIDICV_STREAMING
      : accumulator_{accumulator} {
    accumulator_ = VecTraitsInternal::svdup(0);
  }

  void vector_path(ContextType ctx, VectorType src) KLEIDICV_STREAMING {
    VectorTypeInternal src_widened_evens =
        svcvt_f64_f32_x(VecTraits::svptrue(), src);
    VectorTypeInternal src_widened_odds =
        svcvtlt_f64_f32_x(VecTraits::svptrue(), src);
    accumulator_ =
        svadd_m(ctx.predicate(), accumulator_,
                svadd_m(ctx.predicate(), src_widened_evens, src_widened_odds));
  }

  ScalarType get_sum() const KLEIDICV_STREAMING {
    ScalarTypeInternal accumulator_final[VecTraitsInternal::max_num_lanes()] = {
        0};
    svst1(VecTraitsInternal::svptrue(), accumulator_final, accumulator_);

    ScalarTypeInternal sum = 0;
    for (size_t i = 0; i != VecTraitsInternal::num_lanes(); ++i) {
      sum += accumulator_final[i];
    }
    return static_cast<ScalarType>(sum);
  }

 private:
  VectorTypeInternal &accumulator_;
};

template <typename T, typename TInternal>
kleidicv_error_t sum_sc(const T *src, size_t src_stride, size_t width,
                        size_t height, T *sum) KLEIDICV_STREAMING {
  using VecTraitsInternal = KLEIDICV_TARGET_NAMESPACE::VecTraits<TInternal>;
  using VectorTypeInternal = typename VecTraitsInternal::VectorType;

  CHECK_POINTERS(sum);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};

  VectorTypeInternal accumulator;
  Sum<T, TInternal> operation{accumulator};

  apply_operation_by_rows(operation, rect, src_rows);

  *sum = operation.get_sum();

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SUM_SC_H
