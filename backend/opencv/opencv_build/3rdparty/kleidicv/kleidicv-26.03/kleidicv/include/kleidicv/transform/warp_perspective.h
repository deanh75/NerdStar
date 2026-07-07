// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TRANSFORM_WARP_PERSPECTIVE_H
#define KLEIDICV_TRANSFORM_WARP_PERSPECTIVE_H

#include <limits>
#include <type_traits>

#include "kleidicv/ctypes.h"

extern "C" {
// For internal use only. See kleidicv_warp_perspective_u8 instead.
// Calculates a stripe of the `dst` image that is a transformed part of `src`.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_warp_perspective_stripe_u8,
                         const uint8_t *src, size_t src_stride,
                         size_t src_width, size_t src_height, uint8_t *dst,
                         size_t dst_stride, size_t dst_width, size_t dst_height,
                         size_t y_begin, size_t y_end,
                         const float transformation[9], size_t channels,
                         kleidicv_interpolation_type_t interpolation,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value);
}
namespace kleidicv {

template <typename T>
inline bool warp_perspective_is_implemented(
    size_t dst_width, size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type) KLEIDICV_STREAMING {
  if constexpr (std::is_same<T, uint8_t>::value) {
    return (dst_width >= 8 &&
            (interpolation == KLEIDICV_INTERPOLATION_NEAREST ||
             interpolation == KLEIDICV_INTERPOLATION_LINEAR) &&
            (border_type == KLEIDICV_BORDER_TYPE_REPLICATE ||
             border_type == KLEIDICV_BORDER_TYPE_CONSTANT) &&
            channels == 1);
  } else {
    return false;
  }
}

namespace neon {

template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const T *border_value);
}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const T *border_value);
}  // namespace sve2

}  // namespace kleidicv

#endif  // KLEIDICV_TRANSFORM_WARP_PERSPECTIVE_H
