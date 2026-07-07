// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTERS_GAUSSIAN_BLUR_H
#define KLEIDICV_FILTERS_GAUSSIAN_BLUR_H

#include "kleidicv/config.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"
#include "kleidicv/utils.h"
#include "kleidicv/workspace/border_types.h"
#include "kleidicv/workspace/separable.h"

extern "C" {
// For internal use only. See instead kleidicv_gaussian_blur_u8.
// Blur a horizontal stripe across an image. The stripe is defined by the
// range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_gaussian_blur_fixed_stripe_u8,
                         const uint8_t *src, size_t src_stride, uint8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         float sigma_x, float sigma_y,
                         kleidicv::FixedBorderType border_type);

KLEIDICV_API_DECLARATION(kleidicv_gaussian_blur_arbitrary_stripe_u8,
                         const uint8_t *src, size_t src_stride, uint8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         float sigma_x, float sigma_y,
                         kleidicv::FixedBorderType border_type);
}

namespace kleidicv {

inline bool gaussian_blur_is_implemented(
    size_t width, size_t height, size_t kernel_width, size_t kernel_height,
    float sigma_x, float sigma_y, size_t channels,
    kleidicv::FixedBorderType border_type) {
  if (kernel_width != kernel_height) {
    return false;
  }

  if (kernel_width < 3 || kernel_width > 255) {
    return false;
  }

  if ((kernel_width & 1) != 1) {
    return false;
  }

  if (sigma_x != sigma_y) {
    return false;
  }

  if (width < kernel_width - 1 || height < kernel_width - 1) {
    return false;
  }

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return false;
  }

  if (kernel_width > 9 && kernel_width != 15 && kernel_width != 21) {
    if (border_type != FixedBorderType::REPLICATE) {
      return false;
    }

    size_t margin = kernel_width / 2;
    // Number of 16bit elements in a 128-bit vector
    size_t max_border_length = 8;
    size_t aligned_margin = (margin + max_border_length - 1) /
                            max_border_length * max_border_length;
    if (width < aligned_margin + margin) {
      return false;
    }
  }

  return true;
}

// Does not include checks for whether the operation is implemented.
// This must be done earlier, by gaussian_blur_is_implemented.
template <typename T>
kleidicv_error_t gaussian_blur_checks(const T *src, size_t src_stride, T *dst,
                                      size_t dst_stride, size_t width,
                                      size_t height) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  return KLEIDICV_OK;
}

namespace neon {

kleidicv_error_t gaussian_blur_fixed_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type);

kleidicv_error_t gaussian_blur_arbitrary_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type);

}  // namespace neon

namespace sve2 {

kleidicv_error_t gaussian_blur_fixed_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type);

}  // namespace sve2

namespace sme {

kleidicv_error_t gaussian_blur_fixed_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type);

}  // namespace sme

}  // namespace kleidicv

#endif  // KLEIDICV_FILTERS_GAUSSIAN_BLUR_H
