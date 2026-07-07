// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTERS_MEDIAN_BLUR_H
#define KLEIDICV_FILTERS_MEDIAN_BLUR_H

#include <utility>

#include "kleidicv/config.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"
#include "kleidicv/workspace/border_types.h"

extern "C" {

// For internal use only. See instead kleidicv_median_blur_s8.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_s8,
                         const int8_t *src, size_t src_stride, int8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_u8.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_u8,
                         const uint8_t *src, size_t src_stride, uint8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_s16.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_s16,
                         const int16_t *src, size_t src_stride, int16_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_u16.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_u16,
                         const uint16_t *src, size_t src_stride, uint16_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_s32.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_s32,
                         const int32_t *src, size_t src_stride, int32_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_u32.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_u32,
                         const uint32_t *src, size_t src_stride, uint32_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_f32.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_sorting_network_stripe_f32,
                         const float *src, size_t src_stride, float *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_u8.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_small_hist_stripe_u8,
                         const uint8_t *src, size_t src_stride, uint8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);

// For internal use only. See instead kleidicv_median_blur_u8.
// Find a median across an image.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_median_blur_large_hist_stripe_u8,
                         const uint8_t *src, size_t src_stride, uint8_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         kleidicv::FixedBorderType border_type);
}

namespace kleidicv {

namespace neon {
template <typename T>
kleidicv_error_t median_blur_sorting_network_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type);

kleidicv_error_t median_blur_small_hist_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type);

kleidicv_error_t median_blur_large_hist_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type);
}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t median_blur_sorting_network_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type);

}  // namespace sve2

namespace sme {

template <typename T>
kleidicv_error_t median_blur_sorting_network_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type);

}  // namespace sme

template <typename T>
inline kleidicv_error_t check_ptrs_strides_imagesizes(const T *src,
                                                      size_t src_stride, T *dst,
                                                      size_t dst_stride,
                                                      size_t width,
                                                      size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  return KLEIDICV_OK;
}

template <typename T>
inline bool is_kernel_size_supported(size_t kernel_width,
                                     size_t kernel_height) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    return (kernel_width == kernel_height) && (kernel_width >= 3) &&
           (kernel_width <= 255) && ((kernel_width % 2) != 0);
  } else {
    return (kernel_width == kernel_height) && (kernel_width >= 3) &&
           (kernel_width <= 7) && ((kernel_width % 2) != 0);
  }
}

template <typename T>
inline std::pair<kleidicv_error_t, FixedBorderType> median_blur_is_implemented(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t channels, size_t kernel_width, size_t kernel_height,
    kleidicv_border_type_t border_type) {
  auto image_check = check_ptrs_strides_imagesizes(src, src_stride, dst,
                                                   dst_stride, width, height);
  if (image_check != KLEIDICV_OK) {
    return std::make_pair(image_check, FixedBorderType{});
  }

  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);

  if ((src != dst) && (channels <= KLEIDICV_MAXIMUM_CHANNEL_COUNT) &&
      (height >= kernel_height - 1) && (width >= kernel_width - 1) &&
      is_kernel_size_supported<T>(kernel_width, kernel_height) &&
      fixed_border_type.has_value()) {
    return std::make_pair(KLEIDICV_OK, *fixed_border_type);
  }

  return std::make_pair(KLEIDICV_ERROR_NOT_IMPLEMENTED, FixedBorderType{});
}

}  // namespace kleidicv

#endif  // KLEIDICV_FILTERS_MEDIAN_BLUR_H
