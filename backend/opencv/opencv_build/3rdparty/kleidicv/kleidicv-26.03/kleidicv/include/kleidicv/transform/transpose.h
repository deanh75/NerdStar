// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TRANSFORM_TRANSPOSE_H
#define KLEIDICV_TRANSFORM_TRANSPOSE_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/ctypes.h"

namespace kleidicv {

namespace neon {
kleidicv_error_t transpose(const void *src, size_t src_stride, void *dst,
                           size_t dst_stride, size_t width, size_t height,
                           size_t pixel_size);
}  // namespace neon

}  // namespace kleidicv

#endif  // KLEIDICV_TRANSFORM_TRANSPOSE_H
