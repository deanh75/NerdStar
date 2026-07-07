// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TRANSFORM_ROTATE_H
#define KLEIDICV_TRANSFORM_ROTATE_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/ctypes.h"

namespace kleidicv {

inline bool rotate_is_implemented(const void *src, void *dst, int angle,
                                  size_t pixel_size) {
  if (angle != 90 && angle != -90 && angle != 270) {
    return false;
  }
  if (src == dst) {
    // Do not support inplace rotate at the moment
    return false;
  }
  switch (pixel_size) {
    case sizeof(uint8_t):
    case sizeof(uint16_t):
    case sizeof(uint32_t):
    case sizeof(uint64_t):
      return true;
    default:
      return false;
  }
}

namespace neon {

kleidicv_error_t rotate(const void *src, size_t src_stride, size_t width,
                        size_t height, void *dst, size_t dst_stride, int angle,
                        size_t pixel_size);
}  // namespace neon

}  // namespace kleidicv

#endif  // KLEIDICV_TRANSFORM_ROTATE_H
