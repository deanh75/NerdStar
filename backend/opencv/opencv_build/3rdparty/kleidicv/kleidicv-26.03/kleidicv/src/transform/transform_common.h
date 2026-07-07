// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "kleidicv/types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename T>
bool is_image_large(const Rows<T> &rows, size_t height) {
  return rows.stride() * height >= 1ULL << 32;
}

// Convert channels to a template argument.
template <typename ScalarType, bool IsLarge,
          kleidicv_interpolation_type_t Inter, kleidicv_border_type_t Border,
          typename... Args>
void transform_operation(size_t channels, Args &&...args) {
  switch (channels) {
    case 1:
      transform_operation<ScalarType, IsLarge, Inter, Border, 1UL>(
          std::forward<Args>(args)...);
      break;
    case 2:
      transform_operation<ScalarType, IsLarge, Inter, Border, 2UL>(
          std::forward<Args>(args)...);
    default:
      return;
  }
}

// Convert border_type to a template argument.
template <typename ScalarType, bool IsLarge,
          kleidicv_interpolation_type_t Inter, typename... Args>
void transform_operation(kleidicv_border_type_t border_type, Args &&...args) {
  if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
    transform_operation<ScalarType, IsLarge, Inter,
                        KLEIDICV_BORDER_TYPE_REPLICATE>(
        std::forward<Args>(args)...);
  } else {
    transform_operation<ScalarType, IsLarge, Inter,
                        KLEIDICV_BORDER_TYPE_CONSTANT>(
        std::forward<Args>(args)...);
  }
}

// Convert interpolation_type to a template argument.
template <typename ScalarType, bool IsLarge, typename... Args>
void transform_operation(kleidicv_interpolation_type_t interpolation_type,
                         Args &&...args) {
  if (interpolation_type == KLEIDICV_INTERPOLATION_NEAREST) {
    transform_operation<ScalarType, IsLarge, KLEIDICV_INTERPOLATION_NEAREST>(
        std::forward<Args>(args)...);
  } else {
    transform_operation<ScalarType, IsLarge, KLEIDICV_INTERPOLATION_LINEAR>(
        std::forward<Args>(args)...);
  }
}

// Convert is_large to a template argument.
template <typename ScalarType, typename... Args>
void transform_operation(bool is_large, Args &&...args) {
  if (KLEIDICV_UNLIKELY(is_large)) {
    transform_operation<ScalarType, true>(std::forward<Args>(args)...);
  } else {
    transform_operation<ScalarType, false>(std::forward<Args>(args)...);
  }
}

}  // namespace KLEIDICV_TARGET_NAMESPACE
