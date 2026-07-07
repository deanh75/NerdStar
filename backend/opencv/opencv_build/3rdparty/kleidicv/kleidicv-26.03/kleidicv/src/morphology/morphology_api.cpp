// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/morphology/workspace.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        size_t channels, size_t kernel_width,
                        size_t kernel_height, size_t anchor_x, size_t anchor_y,
                        kleidicv_border_type_t border_type,
                        const uint8_t *border_value, size_t iterations);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t kernel_width,
                       size_t kernel_height, size_t anchor_x, size_t anchor_y,
                       kleidicv_border_type_t border_type,
                       const uint8_t *border_value, size_t iterations);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        size_t channels, size_t kernel_width,
                        size_t kernel_height, size_t anchor_x, size_t anchor_y,
                        kleidicv_border_type_t border_type,
                        const uint8_t *border_value, size_t iterations);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t kernel_width,
                       size_t kernel_height, size_t anchor_x, size_t anchor_y,
                       kleidicv_border_type_t border_type,
                       const uint8_t *border_value, size_t iterations);

}  // namespace sve2

namespace sme {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        size_t channels, size_t kernel_width,
                        size_t kernel_height, size_t anchor_x, size_t anchor_y,
                        kleidicv_border_type_t border_type,
                        const uint8_t *border_value, size_t iterations);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t kernel_width,
                       size_t kernel_height, size_t anchor_x, size_t anchor_y,
                       kleidicv_border_type_t border_type,
                       const uint8_t *border_value, size_t iterations);

}  // namespace sme

namespace sme2 {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        size_t channels, size_t kernel_width,
                        size_t kernel_height, size_t anchor_x, size_t anchor_y,
                        kleidicv_border_type_t border_type,
                        const uint8_t *border_value, size_t iterations);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t kernel_width,
                       size_t kernel_height, size_t anchor_x, size_t anchor_y,
                       kleidicv_border_type_t border_type,
                       const uint8_t *border_value, size_t iterations);

}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_C_API(name, tname, type)           \
  KLEIDICV_MULTIVERSION_C_API_WITH_SME(                    \
      name, &kleidicv::neon::tname<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::tname<type>), \
      &kleidicv::sme::tname<type>,                         \
      KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::tname<type>))

KLEIDICV_DEFINE_C_API(kleidicv_dilate_u8, dilate, uint8_t);
KLEIDICV_DEFINE_C_API(kleidicv_erode_u8, erode, uint8_t);
