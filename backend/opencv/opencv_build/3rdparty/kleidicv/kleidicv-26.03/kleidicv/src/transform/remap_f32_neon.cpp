// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "kleidicv/ctypes.h"
#include "kleidicv/neon.h"
#include "kleidicv/transform/remap.h"
#include "transform_neon.h"

namespace kleidicv::neon {

template <typename ScalarType, bool IsLarge, size_t Channels,
          kleidicv_border_type_t Border>
void remap_f32_nearest(uint32x4_t v_xmax, uint32x4_t v_ymax,
                       uint32x4_t v_src_stride, Rows<const ScalarType> src_rows,
                       Columns<ScalarType> dst, size_t dst_width,
                       Columns<const float> mapx, Columns<const float> mapy,
                       const size_t kStep, const ScalarType *border_values) {
  LoopUnroll2<TryToAvoidTailLoop> loop{dst_width, kStep};
  loop.unroll_once([&](size_t x) {
    transform_pixels<ScalarType, IsLarge, Channels, Border>(
        vld1q_f32(&mapx[x]), vld1q_f32(&mapy[x]), v_xmax, v_ymax, v_src_stride,
        src_rows, dst.at(x), border_values);
  });
}

template <typename ScalarType, bool IsLarge, kleidicv_border_type_t Border,
          size_t Channels>
void remap_f32_linear(uint32x4_t v_xmax, uint32x4_t v_ymax,
                      uint32x4_t v_src_stride, Rows<const ScalarType> src_rows,
                      Columns<ScalarType> dst, size_t dst_width,
                      Columns<const float> mapx, Columns<const float> mapy,
                      const size_t kStep, const ScalarType *border_values) {
  auto load_xy = [&](size_t x) {
    return FloatVectorPair{vld1q_f32(&mapx[x]), vld1q_f32(&mapy[x])};
  };

  auto vector_path_1ch = [&](size_t x) {
    float32x4_t a, b, c, d, xfrac, yfrac;
    if constexpr (Border == KLEIDICV_BORDER_TYPE_REPLICATE) {
      load_quad_pixels_replicate<ScalarType, IsLarge>(
          load_xy(x), v_xmax, v_ymax, v_src_stride, src_rows, xfrac, yfrac, a,
          b, c, d);
    } else {
      static_assert(Border == KLEIDICV_BORDER_TYPE_CONSTANT);
      load_quad_pixels_constant<ScalarType, IsLarge>(
          load_xy(x), v_xmax, v_ymax, v_src_stride, border_values, src_rows,
          xfrac, yfrac, a, b, c, d);
    }
    return lerp_2d(xfrac, yfrac, a, b, c, d);
  };

  auto vector_path_2ch = [&](size_t x) {
    float32x4x2_t a, b, c, d;
    float32x4_t xfrac, yfrac;
    if constexpr (Border == KLEIDICV_BORDER_TYPE_REPLICATE) {
      load_quad_pixels_replicate_2ch<ScalarType, IsLarge>(
          load_xy(x), v_xmax, v_ymax, v_src_stride, src_rows, xfrac, yfrac, a,
          b, c, d);
    } else {
      static_assert(Border == KLEIDICV_BORDER_TYPE_CONSTANT);
      load_quad_pixels_constant_2ch<ScalarType, IsLarge>(
          load_xy(x), v_xmax, v_ymax, v_src_stride, border_values, src_rows,
          xfrac, yfrac, a, b, c, d);
    }
    float32x4_t xfrac_low = vzip1q_f32(xfrac, xfrac);
    float32x4_t yfrac_low = vzip1q_f32(yfrac, yfrac);
    float32x4_t xfrac_high = vzip2q_f32(xfrac, xfrac);
    float32x4_t yfrac_high = vzip2q_f32(yfrac, yfrac);

    uint32x4_t result_low =
        lerp_2d(xfrac_low, yfrac_low, a.val[0], b.val[0], c.val[0], d.val[0]);
    uint32x4_t result_high =
        lerp_2d(xfrac_high, yfrac_high, a.val[1], b.val[1], c.val[1], d.val[1]);
    return vuzp1q_u16(result_low, result_high);
  };

  LoopUnroll2<TryToAvoidTailLoop> loop{dst_width, kStep};

  if constexpr (Channels == 1) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      loop.unroll_four_times([&](size_t x) {
        ScalarType *p_dst = &dst[x];
        uint32x4_t res0 = vector_path_1ch(x);
        x += kStep;
        uint32x4_t res1 = vector_path_1ch(x);
        uint16x8_t result16_0 = vuzp1q_u16(res0, res1);

        x += kStep;
        res0 = vector_path_1ch(x);
        x += kStep;
        res1 = vector_path_1ch(x);
        uint16x8_t result16_1 = vuzp1q_u16(res0, res1);

        vst1q_u8(p_dst, vuzp1q_u8(result16_0, result16_1));
      });
      loop.unroll_once([&](size_t x) {
        uint8x16_t result = vreinterpretq_u8_u32(vector_path_1ch(x));
        dst[x] = vgetq_lane_u8(result, 0);
        dst[x + 1] = vgetq_lane_u8(result, 4);
        dst[x + 2] = vgetq_lane_u8(result, 8);
        dst[x + 3] = vgetq_lane_u8(result, 12);
      });
    }
    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      loop.unroll_twice([&](size_t x) {
        ScalarType *p_dst = dst.ptr_at(x);
        uint32x4_t res0 = vector_path_1ch(x);
        x += kStep;
        uint32x4_t res1 = vector_path_1ch(x);
        vst1q_u16(p_dst, vuzp1q_u16(res0, res1));
      });
      loop.unroll_once([&](size_t x) {
        uint32x4_t result = vector_path_1ch(x);
        uint16x4_t result16 = vget_low_u16(vuzp1q_u16(result, result));
        vst1_u16(dst.ptr_at(x), result16);
      });
    }
  }
  if constexpr (Channels == 2) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      loop.unroll_twice([&](size_t x) {
        ScalarType *p_dst = dst.ptr_at(x);
        uint16x8_t result16_0 = vector_path_2ch(x);
        x += kStep;
        uint16x8_t result16_1 = vector_path_2ch(x);
        vst1q_u8(p_dst, vuzp1q_u8(vreinterpretq_u8_u16(result16_0),
                                  vreinterpretq_u8_u16(result16_1)));
      });
      loop.unroll_once([&](size_t x) {
        uint16x8_t result = vector_path_2ch(x);
        vst1_u8(dst.ptr_at(x), vmovn_u16(result));
      });
    }
    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      loop.unroll_once([&](size_t x) {
        uint16x8_t result = vector_path_2ch(x);
        vst1q_u16(dst.ptr_at(x), result);
      });
    }
  }
}

template <typename ScalarType, bool IsLarge,
          kleidicv_interpolation_type_t Inter, kleidicv_border_type_t Border,
          size_t Channels>
void transform_operation(Rows<const ScalarType> src_rows, size_t src_width,
                         size_t src_height, const ScalarType *border_values,
                         Rows<ScalarType> dst_rows, size_t dst_width,
                         size_t y_begin, size_t y_end,
                         Rows<const float> mapx_rows,
                         Rows<const float> mapy_rows) {
  uint32x4_t v_src_element_stride = vdupq_n_u32(
      static_cast<uint32_t>(src_rows.stride() / sizeof(ScalarType)));
  uint32x4_t v_xmax = vdupq_n_u32(static_cast<uint32_t>(src_width - 1));
  uint32x4_t v_ymax = vdupq_n_u32(static_cast<uint32_t>(src_height - 1));
  const size_t kStep = VecTraits<float>::num_lanes();

  for (size_t y = y_begin; y < y_end; ++y) {
    if constexpr (Inter == KLEIDICV_INTERPOLATION_NEAREST) {
      remap_f32_nearest<ScalarType, IsLarge, Channels, Border>(
          v_xmax, v_ymax, v_src_element_stride, src_rows, dst_rows.as_columns(),
          dst_width, mapx_rows.as_columns(), mapy_rows.as_columns(), kStep,
          border_values);
    } else {
      static_assert(Inter == KLEIDICV_INTERPOLATION_LINEAR);
      remap_f32_linear<ScalarType, IsLarge, Border, Channels>(
          v_xmax, v_ymax, v_src_element_stride, src_rows, dst_rows.as_columns(),
          dst_width, mapx_rows.as_columns(), mapy_rows.as_columns(), kStep,
          border_values);
    }
    ++mapx_rows;
    ++mapy_rows;
    ++dst_rows;
  }
}

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t remap_f32(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const float *mapx, size_t mapx_stride,
                           const float *mapy, size_t mapy_stride,
                           kleidicv_interpolation_type_t interpolation,
                           kleidicv_border_type_t border_type,
                           const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapx, mapx_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapy, mapy_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (!remap_f32_is_implemented<T>(src_stride, src_width, src_height, dst_width,
                                   dst_height, border_type, channels,
                                   interpolation)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<const float> mapx_rows{mapx, mapx_stride, 1};
  Rows<const float> mapy_rows{mapy, mapy_stride, 1};
  Rows<T> dst_rows{dst, dst_stride, channels};
  Rectangle rect{dst_width, dst_height};

  transform_operation<T>(is_image_large(src_rows, src_height), interpolation,
                         border_type, channels, src_rows, src_width, src_height,
                         border_value, dst_rows, dst_width, 0, dst_height,
                         mapx_rows, mapy_rows);

  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_f32<type>(          \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const float *mapx, size_t mapx_stride,                  \
      const float *mapy, size_t mapy_stride,                                   \
      kleidicv_interpolation_type_t interpolation,                             \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(uint16_t);

}  // namespace kleidicv::neon
