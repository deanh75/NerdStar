// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MULTIPLY_SC_H
#define MULTIPLY_SC_H

#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class SaturatingMultiply final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  explicit SaturatingMultiply(double scale = 1.0) : scale_{scale} {};

  VectorType vector_path(ContextType ctx, VectorType src_a,
                         VectorType src_b) KLEIDICV_STREAMING {
    VectorType result;
    (void)ctx;

    // multiply-widen even-indexed elements
    auto bottom_part = svmullb(src_a, src_b);
    // multiply-widen odd-indexed elements
    auto top_part = svmullt(src_a, src_b);
    // saturating-narrow even-indexed
    auto narrow_bottom = svqxtnb(bottom_part);
    // saturaning-narrow odd-indexed and merge with even-indexed
    result = svqxtnt(narrow_bottom, top_part);

    /* TODO: figure out the way to multiply by double or some
      fixed supported scale.
      */

    return result;
  }

 private:
  double scale_;
};

template <typename T>
kleidicv_error_t saturating_multiply_sc(const T *src_a, size_t src_a_stride,
                                        const T *src_b, size_t src_b_stride,
                                        T *dst, size_t dst_stride, size_t width,
                                        size_t height,
                                        double scale) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  (void)scale;  // TODO: figure out the way to process the scale.
  SaturatingMultiply<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // MULTIPLY_SC_H
