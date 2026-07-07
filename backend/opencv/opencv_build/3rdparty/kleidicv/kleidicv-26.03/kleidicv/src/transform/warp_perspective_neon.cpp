// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "kleidicv/ctypes.h"
#include "kleidicv/neon.h"
#include "kleidicv/traits.h"
#include "kleidicv/transform/warp_perspective.h"
#include "kleidicv/utils.h"
#include "transform_neon.h"

namespace kleidicv::neon {

// Template for WarpPerspective transformation.
// Destination pixels are filled from the source, by taking pixels using the
// transformed coordinates that are calculated as follows:
//
//                    [ T0, T1, T2 ]   [ x ]
//      (x',y',w') =  [ T3, T4, T5 ] * [ y ]
//                    [ T6, T7, T8 ]   [ 1 ]
//  then
//
//      xt = x' / w'
//      yt = y' / w'
//
//  or putting it together:
//
//      xt = (T0*x + T1*y + T2) / (T6*x + T7*y + T8)
//      yt = (T3*x + T4*y + T5) / (T6*x + T7*y + T8)
//

template <typename ScalarType, bool IsLarge,
          kleidicv_interpolation_type_t Inter, kleidicv_border_type_t Border,
          size_t Channels>
void transform_operation(Rows<const ScalarType> src_rows, size_t src_width,
                         size_t src_height, const float transform[9],
                         const ScalarType *border_values,
                         Rows<ScalarType> dst_rows, size_t dst_width,
                         size_t y_begin, size_t y_end) {
  static constexpr uint32_t first_few_x[] = {0, 1, 2, 3};
  uint32x4_t x0123_ = vld1q_u32(first_few_x);

  uint32x4_t v_src_stride =
      vdupq_n_u32(static_cast<uint32_t>(src_rows.stride()));
  uint32x4_t v_xmax = vdupq_n_u32(static_cast<uint32_t>(src_width - 1));
  uint32x4_t v_ymax = vdupq_n_u32(static_cast<uint32_t>(src_height - 1));
  float32x4_t tx0, ty0, tw0;

  auto calculate_coordinates = [&](uint32_t x) {
    // The next few values can be calculated by adding the corresponding Tn*x
    float32x4_t fx = vcvtq_f32_u32(vaddq_u32(x0123_, vdupq_n_u32(x)));
    float32x4_t tx = vmlaq_n_f32(tx0, fx, transform[0]);
    float32x4_t ty = vmlaq_n_f32(ty0, fx, transform[3]);
    float32x4_t tw = vmlaq_n_f32(tw0, fx, transform[6]);

    // Calculate inverse weight because division is expensive
    float32x4_t iw = vdivq_f32(vdupq_n_f32(1.F), tw);
    // Calculate coordinates into the source image
    float32x4_t xf = vmulq_f32(tx, iw);
    float32x4_t yf = vmulq_f32(ty, iw);
    return FloatVectorPair{xf, yf};
  };

  auto calculate_linear = [&](uint32_t x) {
    float32x4_t a, b, c, d, xfrac, yfrac;
    if constexpr (Border == KLEIDICV_BORDER_TYPE_REPLICATE) {
      load_quad_pixels_replicate<ScalarType, IsLarge>(
          calculate_coordinates(x), v_xmax, v_ymax, v_src_stride, src_rows,
          xfrac, yfrac, a, b, c, d);
    } else {
      static_assert(Border == KLEIDICV_BORDER_TYPE_CONSTANT);
      load_quad_pixels_constant<ScalarType, IsLarge>(
          calculate_coordinates(x), v_xmax, v_ymax, v_src_stride, border_values,
          src_rows, xfrac, yfrac, a, b, c, d);
    }
    return lerp_2d(xfrac, yfrac, a, b, c, d);
  };

  for (size_t y = y_begin; y < y_end; ++y) {
    float dy = static_cast<float>(y);
    Columns<ScalarType> dst = dst_rows.as_columns();
    // Calculate half-transformed values at the first pixel (nominators)
    // tw =  T6*x + T7*y + T8
    // tx = (T0*x + T1*y + T2) / tw
    // ty = (T3*x + T4*y + T5) / tw
    tx0 = vdupq_n_f32(transform[1] * dy + transform[2]);
    ty0 = vdupq_n_f32(transform[4] * dy + transform[5]);
    tw0 = vdupq_n_f32(transform[7] * dy + transform[8]);

    static const size_t kStep = VecTraits<float>::num_lanes();
    LoopUnroll2<TryToAvoidTailLoop> loop{dst_width, kStep};
    if constexpr (Inter == KLEIDICV_INTERPOLATION_NEAREST) {
      if constexpr (Border == KLEIDICV_BORDER_TYPE_REPLICATE) {
        loop.unroll_once([&](size_t x) {
          auto &&[xf, yf] = calculate_coordinates(x);
          transform_pixels_replicate<ScalarType, IsLarge, Channels>(
              xf, yf, v_xmax, v_ymax, v_src_stride, src_rows, dst.at(x));
        });
      } else {
        static_assert(Border == KLEIDICV_BORDER_TYPE_CONSTANT);
        loop.unroll_once([&](size_t x) {
          auto &&[xf, yf] = calculate_coordinates(x);
          transform_pixels_constant<ScalarType, IsLarge, Channels>(
              xf, yf, v_xmax, v_ymax, v_src_stride, src_rows, dst.at(x),
              border_values);
        });
      }
    } else {
      static_assert(Inter == KLEIDICV_INTERPOLATION_LINEAR);
      loop.unroll_four_times([&](size_t _x) {
        uint32_t x = static_cast<uint32_t>(_x);
        ScalarType *p_dst = &dst[static_cast<ptrdiff_t>(_x)];
        uint32x4_t res0 = calculate_linear(x);
        x += kStep;
        uint32x4_t res1 = calculate_linear(x);
        uint16x8_t result16_0 = vuzp1q_u16(res0, res1);
        x += kStep;
        res0 = calculate_linear(x);
        x += kStep;
        res1 = calculate_linear(x);
        uint16x8_t result16_1 = vuzp1q_u16(res0, res1);
        vst1q_u8(p_dst, vuzp1q_u8(result16_0, result16_1));
      });
      loop.unroll_once([&](size_t x) {
        ScalarType *p_dst = &dst[static_cast<ptrdiff_t>(x)];
        uint32x4_t res = calculate_linear(static_cast<uint32_t>(x));
        p_dst[0] = vgetq_lane_u32(res, 0);
        p_dst[1] = vgetq_lane_u32(res, 1);
        p_dst[2] = vgetq_lane_u32(res, 2);
        p_dst[3] = vgetq_lane_u32(res, 3);
      });
    }
    ++dst_rows;
  }
}

template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTERS(transformation);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);

  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  // Calculating in float32_t will only be precise until 24 bits, and
  // multiplication can only be done with 32x32 bits
  // Empty source image is not supported
  if (src_width >= (1ULL << 24) || src_height >= (1ULL << 24) ||
      dst_width >= (1ULL << 24) || dst_height >= (1ULL << 24) ||
      src_stride >= (1ULL << 32) || src_width == 0 || src_height == 0) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};
  dst_rows += y_begin;

  transform_operation<T>(is_image_large(src_rows, src_height), interpolation,
                         border_type, channels, src_rows, src_width, src_height,
                         transformation, border_value, dst_rows, dst_width,
                         y_begin, y_end);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  warp_perspective_stripe<type>(                                               \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t y_begin, size_t y_end, const float transformation[9],             \
      size_t channels, kleidicv_interpolation_type_t interpolation,            \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(uint8_t);

}  // namespace kleidicv::neon
