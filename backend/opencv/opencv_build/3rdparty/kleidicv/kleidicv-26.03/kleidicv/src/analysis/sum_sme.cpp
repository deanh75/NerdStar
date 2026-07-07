// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "sum_sc.h"

namespace kleidicv::sme {

template <typename T, typename TInternal>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
sum(const T *src, size_t src_stride, size_t width, size_t height, T *sum) {
  return sum_sc<T, TInternal>(src, src_stride, width, height, sum);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type, type_internal)                     \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sum<type, type_internal>( \
      const type *src, size_t src_stride, size_t width, size_t height,         \
      type *sum)

KLEIDICV_INSTANTIATE_TEMPLATE(float, double);

}  // namespace kleidicv::sme
