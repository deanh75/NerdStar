// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/filters/gaussian_blur.h"

extern "C" {

// Implemented based on kleidicv_gaussian_blur_u8 function (placed in
// kleidicv/src/filters/gaussian_blur_api.cpp), but the SME backend is called
// directly. (Original implementation calls the dispatcher to choose between
// backends.)
kleidicv_error_t sme_gaussian_blur_u8(const uint8_t *src, size_t src_stride,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t width, size_t height,
                                      size_t channels, size_t kernel_width,
                                      size_t kernel_height, float sigma_x,
                                      float sigma_y,
                                      kleidicv_border_type_t border_type) {
  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  if (!kleidicv::gaussian_blur_is_implemented(width, height, kernel_width,
                                              kernel_height, sigma_x, sigma_y,
                                              channels, *fixed_border_type)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  if (kernel_width <= 9 || kernel_width == 15 || kernel_width == 21) {
    return kleidicv::sme::gaussian_blur_fixed_stripe_u8(
        src, src_stride, dst, dst_stride, width, height, 0, height, channels,
        kernel_width, kernel_height, sigma_x, sigma_y, *fixed_border_type);
  }

  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // extern "C"
