// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/utils.h"

namespace kleidicv::neon {

template <typename ScalarType, typename ScalarTypeInternal>
class Sum;

template <>
class Sum<float, double> final : public UnrollTwice {
 public:
  using ScalarType = float;
  using ScalarTypeInternal = double;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using VecTraitsInternal =
      KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarTypeInternal>;
  using VectorTypeInternal = typename VecTraitsInternal::VectorType;

  VectorTypeInternal vector_sum;
  ScalarTypeInternal scalar_sum;

  Sum() : vector_sum(VectorTypeInternal{0}), scalar_sum(0) {}

  void vector_path(VectorType src) {
    VectorTypeInternal src_low = vcvt_f64(vget_low(src));
    VectorTypeInternal src_high = vcvt_f64(vget_high(src));
    vector_sum = vaddq(vector_sum, vaddq(src_low, src_high));
  }

  void scalar_path(ScalarType src) {
    scalar_sum += static_cast<ScalarTypeInternal>(src);
  }

  ScalarType get_sum() const {
    ScalarTypeInternal sum = vaddvq(vector_sum) + scalar_sum;
    return static_cast<ScalarType>(sum);
  }
};

template <typename T, typename TInternal>
kleidicv_error_t sum(const T *src, size_t src_stride, size_t width,
                     size_t height, T *sum) {
  CHECK_POINTERS(sum);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Sum<T, TInternal> operation;
  apply_operation_by_rows(operation, rect, src_rows);

  *sum = operation.get_sum();

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type, type_internal)                     \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sum<type, type_internal>( \
      const type *src, size_t src_stride, size_t width, size_t height,         \
      type *sum)

KLEIDICV_INSTANTIATE_TEMPLATE(float, double);

}  // namespace kleidicv::neon
