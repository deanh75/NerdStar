// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API_WITH_SME(
    kleidicv_gaussian_blur_fixed_stripe_u8,
    &kleidicv::neon::gaussian_blur_fixed_stripe_u8,
    KLEIDICV_SVE2_IMPL_IF(kleidicv::sve2::gaussian_blur_fixed_stripe_u8),
    &kleidicv::sme::gaussian_blur_fixed_stripe_u8, nullptr);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(
    kleidicv_gaussian_blur_arbitrary_stripe_u8,
    &kleidicv::neon::gaussian_blur_arbitrary_stripe_u8, nullptr);

namespace kleidicv {

template <bool kUseSME>
kleidicv_error_t gaussian_blur_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, size_t channels,
                                  size_t kernel_width, size_t kernel_height,
                                  float sigma_x, float sigma_y,
                                  kleidicv_border_type_t border_type) {
  auto fixed_border_type = get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  if (!gaussian_blur_is_implemented(width, height, kernel_width, kernel_height,
                                    sigma_x, sigma_y, channels,
                                    *fixed_border_type)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  if (kernel_width <= 9 || kernel_width == 15 || kernel_width == 21) {
    if constexpr (kUseSME) {
      return kleidicv_gaussian_blur_fixed_stripe_u8_sme(
          src, src_stride, dst, dst_stride, width, height, 0, height, channels,
          kernel_width, kernel_height, sigma_x, sigma_y, *fixed_border_type);
    }
    return kleidicv_gaussian_blur_fixed_stripe_u8(
        src, src_stride, dst, dst_stride, width, height, 0, height, channels,
        kernel_width, kernel_height, sigma_x, sigma_y, *fixed_border_type);
  }

  return kleidicv_gaussian_blur_arbitrary_stripe_u8(
      src, src_stride, dst, dst_stride, width, height, 0, height, channels,
      kernel_width, kernel_height, sigma_x, sigma_y, *fixed_border_type);
}

}  // namespace kleidicv

extern "C" {

kleidicv_error_t kleidicv_gaussian_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type) {
  return kleidicv::gaussian_blur_u8<false>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, sigma_x, sigma_y, border_type);
}

kleidicv_error_t kleidicv_gaussian_blur_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type) {
  return kleidicv::gaussian_blur_u8<true>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, sigma_x, sigma_y, border_type);
}

}  // extern "C"
