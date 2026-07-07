// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/filters/median_blur.h"
#include "kleidicv/kleidicv.h"

#define KLEIDICV_DEFINE_C_API(name, type)                              \
  KLEIDICV_MULTIVERSION_C_API_WITH_SME(                                \
      name, &kleidicv::neon::median_blur_sorting_network_stripe<type>, \
      KLEIDICV_SVE2_IMPL_IF(                                           \
          kleidicv::sve2::median_blur_sorting_network_stripe<type>),   \
      &kleidicv::sme::median_blur_sorting_network_stripe<type>, nullptr)

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_s8, int8_t);

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_u8, uint8_t);

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_u16,
                      uint16_t);

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_s16, int16_t);

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_u32,
                      uint32_t);

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_s32, int32_t);

KLEIDICV_DEFINE_C_API(kleidicv_median_blur_sorting_network_stripe_f32, float);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(
    kleidicv_median_blur_small_hist_stripe_u8,
    &kleidicv::neon::median_blur_small_hist_stripe_u8, nullptr);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(
    kleidicv_median_blur_large_hist_stripe_u8,
    &kleidicv::neon::median_blur_large_hist_stripe_u8, nullptr);

namespace kleidicv {

template <auto &StripeFunction, typename T>
kleidicv_error_t median_blur(const T *src, size_t src_stride, T *dst,
                             size_t dst_stride, size_t width, size_t height,
                             size_t channels, size_t kernel_width,
                             size_t kernel_height,
                             kleidicv_border_type_t border_type) {
  auto [checks_result, fixed_border_type] = median_blur_is_implemented(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);

  if (checks_result != KLEIDICV_OK) {
    return checks_result;
  }

  return StripeFunction(src, src_stride, dst, dst_stride, width, height, 0,
                        height, channels, kernel_width, kernel_height,
                        fixed_border_type);
}

template <bool kUseSME>
kleidicv_error_t median_blur_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height, size_t channels,
                                size_t kernel_width, size_t kernel_height,
                                kleidicv_border_type_t border_type) {
  auto [checks_result, fixed_border_type] = median_blur_is_implemented(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);

  if (checks_result != KLEIDICV_OK) {
    return checks_result;
  }

  if (kernel_width <= 7) {
    if constexpr (kUseSME) {
      return kleidicv_median_blur_sorting_network_stripe_u8_sme(
          src, src_stride, dst, dst_stride, width, height, 0, height, channels,
          kernel_width, kernel_height, fixed_border_type);
    }
    return kleidicv_median_blur_sorting_network_stripe_u8(
        src, src_stride, dst, dst_stride, width, height, 0, height, channels,
        kernel_width, kernel_height, fixed_border_type);
  }

  if (kernel_width > 7 && kernel_width <= 15) {
    return kleidicv_median_blur_small_hist_stripe_u8(
        src, src_stride, dst, dst_stride, width, height, 0, height, channels,
        kernel_width, kernel_height, fixed_border_type);
  }

  return kleidicv_median_blur_large_hist_stripe_u8(
      src, src_stride, dst, dst_stride, width, height, 0, height, channels,
      kernel_width, kernel_height, fixed_border_type);
}
}  // namespace kleidicv

extern "C" {

kleidicv_error_t kleidicv_median_blur_s8(const int8_t *src, size_t src_stride,
                                         int8_t *dst, size_t dst_stride,
                                         size_t width, size_t height,
                                         size_t channels, size_t kernel_width,
                                         size_t kernel_height,
                                         kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<kleidicv_median_blur_sorting_network_stripe_s8>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_s8_sme(
    const int8_t *src, size_t src_stride, int8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<
      kleidicv_median_blur_sorting_network_stripe_s8_sme>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_u8(const uint8_t *src, size_t src_stride,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t width, size_t height,
                                         size_t channels, size_t kernel_width,
                                         size_t kernel_height,
                                         kleidicv_border_type_t border_type) {
  return kleidicv::median_blur_u8<false>(src, src_stride, dst, dst_stride,
                                         width, height, channels, kernel_width,
                                         kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur_u8<true>(src, src_stride, dst, dst_stride, width,
                                        height, channels, kernel_width,
                                        kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_s16(const int16_t *src, size_t src_stride,
                                          int16_t *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          size_t channels, size_t kernel_width,
                                          size_t kernel_height,
                                          kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<kleidicv_median_blur_sorting_network_stripe_s16>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_s16_sme(
    const int16_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<
      kleidicv_median_blur_sorting_network_stripe_s16_sme>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_u16(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<kleidicv_median_blur_sorting_network_stripe_u16>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_u16_sme(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<
      kleidicv_median_blur_sorting_network_stripe_u16_sme>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_s32(const int32_t *src, size_t src_stride,
                                          int32_t *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          size_t channels, size_t kernel_width,
                                          size_t kernel_height,
                                          kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<kleidicv_median_blur_sorting_network_stripe_s32>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_s32_sme(
    const int32_t *src, size_t src_stride, int32_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<
      kleidicv_median_blur_sorting_network_stripe_s32_sme>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_u32(
    const uint32_t *src, size_t src_stride, uint32_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<kleidicv_median_blur_sorting_network_stripe_u32>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_u32_sme(
    const uint32_t *src, size_t src_stride, uint32_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<
      kleidicv_median_blur_sorting_network_stripe_u32_sme>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_f32(const float *src, size_t src_stride,
                                          float *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          size_t channels, size_t kernel_width,
                                          size_t kernel_height,
                                          kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<kleidicv_median_blur_sorting_network_stripe_f32>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

kleidicv_error_t kleidicv_median_blur_f32_sme(
    const float *src, size_t src_stride, float *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv::median_blur<
      kleidicv_median_blur_sorting_network_stripe_f32_sme>(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_width,
      kernel_height, border_type);
}

}  // extern "C"
