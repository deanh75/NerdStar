// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef SUB_SC_H
#define SUB_SC_H

#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class SaturatingSub final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(ContextType ctx, VectorType src_a,
                         VectorType src_b) KLEIDICV_STREAMING {
    return svqsub_m(ctx.predicate(), src_a, src_b);
  }
};  // end of class SaturatingSub<ScalarType>

template <typename T>
kleidicv_error_t saturating_sub_sc(const T *src_a, size_t src_a_stride,
                                   const T *src_b, size_t src_b_stride, T *dst,
                                   size_t dst_stride, size_t width,
                                   size_t height) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  SaturatingSub<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // SUB_SC_H
