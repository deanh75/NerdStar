// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/filters/scharr.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API_WITH_SME(
    kleidicv_scharr_interleaved_stripe_s16_u8,
    &kleidicv::neon::kleidicv_scharr_interleaved_stripe_s16_u8,
    KLEIDICV_SVE2_IMPL_IF(
        &kleidicv::sve2::kleidicv_scharr_interleaved_stripe_s16_u8),
    &kleidicv::sme::kleidicv_scharr_interleaved_stripe_s16_u8,
    &kleidicv::sme2::kleidicv_scharr_interleaved_stripe_s16_u8);

namespace kleidicv {

template <auto &StripeFunction>
kleidicv_error_t scharr_interleaved_s16_u8(const uint8_t *src,
                                           size_t src_stride, size_t src_width,
                                           size_t src_height,
                                           size_t src_channels, int16_t *dst,
                                           size_t dst_stride) {
  if (!scharr_interleaved_is_implemented(src_width, src_height, src_channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  // height is decremented by 2 as the result has less rows.
  return StripeFunction(src, src_stride, src_width, src_height, src_channels,
                        dst, dst_stride, 0, src_height - 2);
}

}  // namespace kleidicv

extern "C" {

kleidicv_error_t kleidicv_scharr_interleaved_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride) {
  return kleidicv::scharr_interleaved_s16_u8<
      kleidicv_scharr_interleaved_stripe_s16_u8>(
      src, src_stride, src_width, src_height, src_channels, dst, dst_stride);
}

kleidicv_error_t kleidicv_scharr_interleaved_s16_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride) {
  return kleidicv::scharr_interleaved_s16_u8<
      kleidicv_scharr_interleaved_stripe_s16_u8_sme>(
      src, src_stride, src_width, src_height, src_channels, dst, dst_stride);
}

}  // extern "C"
