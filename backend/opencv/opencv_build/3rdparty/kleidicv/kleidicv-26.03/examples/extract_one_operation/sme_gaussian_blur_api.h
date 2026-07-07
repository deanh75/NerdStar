// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef SME_GAUSSIAN_BLUR_H
#define SME_GAUSSIAN_BLUR_H

#ifdef __cplusplus
#include <cinttypes>
#include <cstddef>
#else  // __cplusplus
#include "inttypes.h"
#include "stddef.h"
#endif  // __cplusplus

// Copied from kleidicv/include/kleidicv/ctypes.h
typedef enum {
  /// Success.
  KLEIDICV_OK = 0,
  /// Requested operation is not implemented.
  KLEIDICV_ERROR_NOT_IMPLEMENTED,
  /// Null pointer was passed as an argument.
  KLEIDICV_ERROR_NULL_POINTER,
  /// A value was encountered outside the representable or valid range.
  KLEIDICV_ERROR_RANGE,
  /// Could not allocate memory.
  KLEIDICV_ERROR_ALLOCATION,
  /// A value did not meet alignment requirements.
  KLEIDICV_ERROR_ALIGNMENT,
  /// The provided context (like @ref kleidicv_morphology_context_t) is not
  /// compatible with the operation.
  KLEIDICV_ERROR_CONTEXT_MISMATCH,
} kleidicv_error_t;

// Copied from kleidicv/include/kleidicv/ctypes.h
typedef enum {
  /// The border is a constant value.
  KLEIDICV_BORDER_TYPE_CONSTANT,
  /// The border is the value of the first/last element.
  KLEIDICV_BORDER_TYPE_REPLICATE,
  /// The border is the mirrored value of the first/last elements.
  KLEIDICV_BORDER_TYPE_REFLECT,
  /// The border simply acts as a "wrap around" to the beginning/end.
  KLEIDICV_BORDER_TYPE_WRAP,
  /// Like KLEIDICV_BORDER_TYPE_REFLECT, but the first/last elements are
  /// ignored.
  KLEIDICV_BORDER_TYPE_REVERSE,
  /// The border is the "continuation" of the input rows. It is the caller's
  /// responsibility to provide the input data (and an appropriate stride value)
  /// in a way that the rows can be under and over read. E.g. can be used when
  /// executing an operation on a region of a picture.
  KLEIDICV_BORDER_TYPE_TRANSPARENT,
  /// The border is a hard border, there are no additional values to use.
  KLEIDICV_BORDER_TYPE_NONE,
} kleidicv_border_type_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Similar funtionality as of kleidicv_gaussian_blur_u8 but it uses the SME
// backend directly, so less kernel sizes are supported.
kleidicv_error_t sme_gaussian_blur_u8(const uint8_t *src, size_t src_stride,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t width, size_t height,
                                      size_t channels, size_t kernel_width,
                                      size_t kernel_height, float sigma_x,
                                      float sigma_y,
                                      kleidicv_border_type_t border_type);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // SME_GAUSSIAN_BLUR_H
