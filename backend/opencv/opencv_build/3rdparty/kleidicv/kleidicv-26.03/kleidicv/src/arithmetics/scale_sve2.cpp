// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "scale_sc.h"

namespace kleidicv::sve2 {

template <typename T, typename U>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t scale(const T* src, size_t src_stride,
                                                U* dst, size_t dst_stride,
                                                size_t width, size_t height,
                                                double scale, double shift) {
  return scale_sc<T, U>(src, src_stride, dst, dst_stride, width, height, scale,
                        shift);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(src_type, dst_type)                   \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                        \
  scale<src_type, dst_type>(const src_type* src, size_t src_stride,         \
                            dst_type* dst, size_t dst_stride, size_t width, \
                            size_t height, double scale, double shift)

KLEIDICV_INSTANTIATE_TEMPLATE(float, float);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t, float16_t);

}  // namespace kleidicv::sve2
