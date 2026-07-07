// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "sub_sc.h"

namespace kleidicv::sme2 {

template <typename T>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
saturating_sub(const T *src_a, size_t src_a_stride, const T *src_b,
               size_t src_b_stride, T *dst, size_t dst_stride, size_t width,
               size_t height) {
  return saturating_sub_sc<T>(src_a, src_a_stride, src_b, src_b_stride, dst,
                              dst_stride, width, height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t saturating_sub<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,           \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,     \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int64_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint64_t);

}  // namespace kleidicv::sme2
