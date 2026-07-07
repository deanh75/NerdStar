// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kleidicv/sve2.h"
#include "kleidicv/transform/remap.h"
#include "transform_sve2.h"

namespace kleidicv::sve2 {

template <typename ScalarType, bool IsLarge, kleidicv_border_type_t Border,
          size_t Channels>
void remap_f32_nearest(svuint32_t sv_xmax, svuint32_t sv_ymax,
                       svuint32_t sv_src_stride,
                       Rows<const ScalarType> src_rows, svuint32_t sv_border,
                       Columns<ScalarType> dst, size_t kStep, size_t dst_width,
                       Rows<const float> mapx_rows, Rows<const float> mapy_rows,
                       [[maybe_unused]] svuint8_t load_table_2ch) {
  svbool_t pg_all32 = svptrue_b32();
  svbool_t pg_all16 = svptrue_b16();
  auto load_coords = [&](svbool_t pg, size_t xs) {
    auto x = static_cast<ptrdiff_t>(xs);
    return svcreate2(svld1_f32(pg, &mapx_rows.as_columns()[x]),
                     svld1_f32(pg, &mapy_rows.as_columns()[x]));
  };

  auto load_source = [&](svbool_t pg, svuint32_t x, svuint32_t y) {
    if constexpr (Channels == 1) {
      return load_xy<ScalarType, IsLarge>(pg, x, y, sv_src_stride, src_rows);
    }
    if constexpr (Channels == 2) {
      return load_xy_2ch<ScalarType, IsLarge>(pg, x, y, sv_src_stride, src_rows,
                                              load_table_2ch);
    }
  };

  auto get_pixels = [&](svbool_t pg, svuint32x2_t coords) {
    svuint32_t x = svget2(coords, 0);
    svuint32_t y = svget2(coords, 1);
    if constexpr (Border == KLEIDICV_BORDER_TYPE_CONSTANT) {
      svbool_t in_range = svand_b_z(pg, svcmple_u32(pg, x, sv_xmax),
                                    svcmple_u32(pg, y, sv_ymax));
      svuint32_t result = load_source(in_range, x, y);
      // Select between source pixels and border color
      return svsel_u32(in_range, result, sv_border);
    } else {
      static_assert(Border == KLEIDICV_BORDER_TYPE_REPLICATE);
      return load_source(pg, x, y);
    }
  };

  auto calculate_nearest_coordinates = [&](svbool_t pg32, size_t x) {
    svfloat32x2_t coords = load_coords(pg32, x);
    svfloat32_t xf = svget2(coords, 0);
    svfloat32_t yf = svget2(coords, 1);

    svuint32_t xi, yi;
    if constexpr (Border == KLEIDICV_BORDER_TYPE_CONSTANT) {
      // Convert coordinates to integers.
      // Negative numbers will become large positive numbers.
      // Since the source width and height is known to be <=2^24 these large
      // positive numbers will always be treated as outside the source image
      // bounds.
      xi = svreinterpret_u32_s32(
          svcvt_s32_f32_x(pg_all32, svrinta_f32_x(pg_all32, xf)));
      yi = svreinterpret_u32_s32(
          svcvt_s32_f32_x(pg_all32, svrinta_f32_x(pg_all32, yf)));
    } else {
      // Round to the nearest integer, clamp it to within the dimensions of
      // the source image (negative values are already saturated to 0)
      xi = svmin_x(pg_all32,
                   svcvt_u32_f32_x(pg_all32, svadd_n_f32_x(pg_all32, xf, 0.5F)),
                   sv_xmax);
      yi = svmin_x(pg_all32,
                   svcvt_u32_f32_x(pg_all32, svadd_n_f32_x(pg_all32, yf, 0.5F)),
                   sv_ymax);
    }
    return svcreate2(xi, yi);
  };

  LoopUnroll2 loop{dst_width, kStep};

  if constexpr (Channels == 1) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      auto vector_path_generic = [&](size_t x, size_t x_max,
                                     Columns<ScalarType> dst) {
        size_t length = x_max - x;
        svbool_t pg32 = svwhilelt_b32_u64(0ULL, length);
        svuint32_t result =
            get_pixels(pg32, calculate_nearest_coordinates(pg32, x));
        svst1b_u32(pg32, &dst[static_cast<ptrdiff_t>(x)], result);
      };

      loop.unroll_four_times([&](size_t x) {
        ScalarType* p_dst = &dst[static_cast<ptrdiff_t>(x)];
        svuint32_t res32_0 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        x += kStep;
        svuint32_t res32_1 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        svuint16_t result0 = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                        svreinterpret_u16_u32(res32_1));
        x += kStep;
        res32_0 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        x += kStep;
        res32_1 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        svuint16_t result1 = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                        svreinterpret_u16_u32(res32_1));
        svuint8_t result = svuzp1_u8(svreinterpret_u8_u16(result0),
                                     svreinterpret_u8_u16(result1));
        svst1(svptrue_b8(), p_dst, result);
      });
      loop.unroll_once(
          [&](size_t x) { vector_path_generic(x, x + kStep, dst); });
      loop.remaining([&](size_t x, size_t length) {
        vector_path_generic(x, length, dst);
      });
    }

    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      auto vector_path_generic = [&](size_t x, size_t x_max,
                                     Columns<ScalarType> dst) {
        size_t length = x_max - x;
        svbool_t pg32 = svwhilelt_b32(0ULL, length);
        svuint32_t result =
            get_pixels(pg32, calculate_nearest_coordinates(pg32, x));
        svst1h_u32(pg32, &dst[static_cast<ptrdiff_t>(x)], result);
      };

      loop.unroll_twice([&](size_t x) {
        ScalarType* p_dst = &dst[static_cast<ptrdiff_t>(x)];
        svuint32_t res32_0 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        x += kStep;
        svuint32_t res32_1 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        svuint16_t result = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                       svreinterpret_u16_u32(res32_1));
        svst1(svptrue_b16(), p_dst, result);
      });
      loop.unroll_once(
          [&](size_t x) { vector_path_generic(x, x + kStep, dst); });
      loop.remaining([&](size_t x, size_t length) {
        vector_path_generic(x, length, dst);
      });
    }
  }

  if constexpr (Channels == 2) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      auto vector_path_generic = [&](size_t x, size_t x_max,
                                     Columns<ScalarType> dst) {
        size_t length = x_max - x;
        svbool_t pg32 = svwhilelt_b32(0ULL, length);
        svuint32_t result =
            get_pixels(pg32, calculate_nearest_coordinates(pg32, x));
        svbool_t pg16 = svwhilelt_b16_u64(0ULL, 2 * length);
        svst1b_u16(pg16, dst.ptr_at(static_cast<ptrdiff_t>(x)),
                   svreinterpret_u16_u32(result));
      };

      loop.unroll_twice([&](size_t x) {
        ScalarType* p_dst = dst.ptr_at(static_cast<ptrdiff_t>(x));
        svuint32_t result0 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        x += kStep;
        svuint32_t result1 =
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x));
        svuint8_t result = svuzp1_u8(svreinterpret_u8_u32(result0),
                                     svreinterpret_u8_u32(result1));
        svst1(svptrue_b8(), p_dst, result);
      });
      loop.unroll_once(
          [&](size_t x) { vector_path_generic(x, x + kStep, dst); });
      loop.remaining([&](size_t x, size_t length) {
        vector_path_generic(x, length, dst);
      });
    }

    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      loop.unroll_once([&](size_t x) {
        svuint16_t result = svreinterpret_u16_u32(
            get_pixels(pg_all32, calculate_nearest_coordinates(pg_all32, x)));
        svst1_u16(pg_all16, dst.ptr_at(static_cast<ptrdiff_t>(x)), result);
      });
      loop.remaining([&](size_t x, size_t x_max) {
        svbool_t pg32 = svwhilelt_b32_u64(x, x_max);
        svuint16_t result = svreinterpret_u16_u32(
            get_pixels(pg32, calculate_nearest_coordinates(pg32, x)));
        svbool_t pg16 = svwhilelt_b16_u64(2 * x, 2 * x_max);
        svst1_u16(pg16, dst.ptr_at(static_cast<ptrdiff_t>(x)), result);
      });
    }
  }
}

template <typename ScalarType, bool IsLarge, kleidicv_border_type_t Border,
          size_t Channels>
void remap_f32_linear(svuint32_t sv_xmax, svuint32_t sv_ymax,
                      svfloat32_t sv_xmaxf, svfloat32_t sv_ymaxf,
                      svuint32_t sv_src_stride, Rows<const ScalarType> src_rows,
                      svuint32_t sv_border, Columns<ScalarType> dst,
                      size_t kStep, size_t dst_width,
                      Rows<const float> mapx_rows, Rows<const float> mapy_rows,
                      svuint8_t load_table_2ch) {
  auto load_coords = [&](svbool_t pg, size_t xs) {
    auto x = static_cast<ptrdiff_t>(xs);
    return svcreate2(svld1_f32(pg, &mapx_rows.as_columns()[x]),
                     svld1_f32(pg, &mapy_rows.as_columns()[x]));
  };

  auto calculate_linear = [&](svbool_t pg, uint32_t x) {
    if constexpr (Border == KLEIDICV_BORDER_TYPE_REPLICATE) {
      svfloat32x2_t coords = load_coords(pg, x);
      return calculate_linear_replicated_border<ScalarType, IsLarge, Channels>(
          pg, coords, sv_xmaxf, sv_ymaxf, sv_src_stride, src_rows,
          load_table_2ch);
    } else {
      static_assert(Border == KLEIDICV_BORDER_TYPE_CONSTANT);
      svfloat32x2_t coords = load_coords(pg, x);
      return calculate_linear_constant_border<ScalarType, IsLarge, Channels>(
          pg, coords, sv_border, sv_xmax, sv_ymax, sv_src_stride, src_rows,
          load_table_2ch);
    }
  };

  auto store_vector = [](svbool_t pg32, ScalarType* p_dst, svuint32_t result) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      svst1b_u32(pg32, p_dst, result);
    }
    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      svst1h_u32(pg32, p_dst, result);
    }
  };

  svbool_t pg_all32 = svptrue_b32();
  LoopUnroll2 loop{dst_width, kStep};
  if constexpr (Channels == 1) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      loop.unroll_four_times([&](size_t x) {
        ScalarType* p_dst = &dst[static_cast<ptrdiff_t>(x)];
        svuint32_t res0 = calculate_linear(pg_all32, x);
        x += kStep;
        svuint32_t res1 = calculate_linear(pg_all32, x);
        svuint16_t result16_0 = svuzp1_u16(svreinterpret_u16_u32(res0),
                                           svreinterpret_u16_u32(res1));
        x += kStep;
        res0 = calculate_linear(pg_all32, x);
        x += kStep;
        res1 = calculate_linear(pg_all32, x);
        svuint16_t result16_1 = svuzp1_u16(svreinterpret_u16_u32(res0),
                                           svreinterpret_u16_u32(res1));
        svst1_u8(svptrue_b8(), p_dst,
                 svuzp1_u8(svreinterpret_u8_u16(result16_0),
                           svreinterpret_u8_u16(result16_1)));
      });
    }
    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      loop.unroll_twice([&](size_t x) {
        ScalarType* p_dst = &dst[static_cast<ptrdiff_t>(x)];
        svuint32_t res0 = calculate_linear(pg_all32, x);
        x += kStep;
        svuint32_t res1 = calculate_linear(pg_all32, x);
        svuint16_t result16 = svuzp1_u16(svreinterpret_u16_u32(res0),
                                         svreinterpret_u16_u32(res1));
        svst1_u16(svptrue_b16(), p_dst, result16);
      });
    }
    loop.unroll_once([&](size_t x) {
      svuint32_t result = calculate_linear(pg_all32, x);
      store_vector(pg_all32, &dst[static_cast<ptrdiff_t>(x)], result);
    });
    loop.remaining([&](size_t x, size_t x_max) {
      svbool_t pg32 = svwhilelt_b32_u64(x, x_max);
      svuint32_t result = calculate_linear(pg32, x);
      store_vector(pg32, &dst[static_cast<ptrdiff_t>(x)], result);
    });
  }

  if constexpr (Channels == 2) {
    if constexpr (std::is_same<ScalarType, uint8_t>::value) {
      auto vector_path_generic = [&](size_t x, size_t x_max,
                                     Columns<ScalarType> dst) {
        size_t length = x_max - x;
        svbool_t pg32 = svwhilelt_b32(0ULL, length);
        svuint32_t result = calculate_linear(pg32, x);
        svbool_t pg16 = svwhilelt_b16(0ULL, 2 * length);
        svst1b_u16(pg16, dst.ptr_at(static_cast<ptrdiff_t>(x)),
                   svreinterpret_u16_u32(result));
      };

      loop.unroll_twice([&](size_t x) {
        ScalarType* p_dst = dst.ptr_at(static_cast<ptrdiff_t>(x));
        svuint32_t result0 = calculate_linear(pg_all32, x);
        x += kStep;
        svuint32_t result1 = calculate_linear(pg_all32, x);
        svuint8_t result = svuzp1_u8(svreinterpret_u8_u32(result0),
                                     svreinterpret_u8_u32(result1));
        svst1(svptrue_b8(), p_dst, result);
      });
      loop.unroll_once(
          [&](size_t x) { vector_path_generic(x, x + kStep, dst); });
      loop.remaining([&](size_t x, size_t length) {
        vector_path_generic(x, length, dst);
      });
    }
    if constexpr (std::is_same<ScalarType, uint16_t>::value) {
      loop.unroll_once([&](size_t x) {
        svuint16_t result =
            svreinterpret_u16_u32(calculate_linear(pg_all32, x));
        svst1_u16(svptrue_b16(), dst.ptr_at(static_cast<ptrdiff_t>(x)), result);
      });
      loop.remaining([&](size_t x, size_t x_max) {
        svbool_t pg32 = svwhilelt_b32_u64(x, x_max);
        svuint16_t result = svreinterpret_u16_u32(calculate_linear(pg32, x));
        svbool_t pg16 = svwhilelt_b16_u64(2 * x, 2 * x_max);
        svst1_u16(pg16, dst.ptr_at(static_cast<ptrdiff_t>(x)), result);
      });
    }
  }
}

template <typename ScalarType, bool IsLarge,
          kleidicv_interpolation_type_t Inter, kleidicv_border_type_t Border,
          size_t Channels>
void transform_operation(Rows<const ScalarType> src_rows, size_t src_width,
                         size_t src_height, const ScalarType* border_value,
                         Rows<ScalarType> dst_rows, size_t dst_width,
                         size_t y_begin, size_t y_end,
                         Rows<const float> mapx_rows,
                         Rows<const float> mapy_rows) {
  svuint32_t sv_xmax = svdup_n_u32(src_width - 1);
  svuint32_t sv_ymax = svdup_n_u32(src_height - 1);
  svuint32_t sv_src_stride = svdup_n_u32(src_rows.stride());
  svuint32_t sv_border = svdup_n_u32(0);

  if constexpr (Border == KLEIDICV_BORDER_TYPE_CONSTANT) {
    if constexpr (Channels == 1) {
      sv_border = svdup_n_u32(border_value[0]);
    }
    if constexpr (Channels == 2) {
      uint32_t v = static_cast<uint32_t>(border_value[0]) |
                   (static_cast<uint32_t>(border_value[1]) << 16);
      sv_border = svdup_n_u32(v);
    }
  }

  svfloat32_t sv_xmaxf = svdup_n_f32(static_cast<float>(src_width - 1));
  svfloat32_t sv_ymaxf = svdup_n_f32(static_cast<float>(src_height - 1));

  const size_t kStep = VecTraits<float>::num_lanes();

  // Rearrange input for 8bit 2channel:
  // Gather Load 16bits, 2x 8bits for 2 channels:
  // after 32-bit gather load:        ..DC..BA
  // goal is to have 16-bit elements: .D.C.B.A
  svuint8_t load_table_2ch =
      svreinterpret_u8_u32(svindex_u32(0x03010200U, 0x04040404));

  for (size_t y = y_begin; y < y_end; ++y) {
    Columns<ScalarType> dst = dst_rows.as_columns();
    if constexpr (Inter == KLEIDICV_INTERPOLATION_NEAREST) {
      remap_f32_nearest<ScalarType, IsLarge, Border, Channels>(
          sv_xmax, sv_ymax, sv_src_stride, src_rows, sv_border, dst, kStep,
          dst_width, mapx_rows, mapy_rows, load_table_2ch);
    } else {
      static_assert(Inter == KLEIDICV_INTERPOLATION_LINEAR);
      remap_f32_linear<ScalarType, IsLarge, Border, Channels>(
          sv_xmax, sv_ymax, sv_xmaxf, sv_ymaxf, sv_src_stride, src_rows,
          sv_border, dst, kStep, dst_width, mapx_rows, mapy_rows,
          load_table_2ch);
    }
    ++mapx_rows;
    ++mapy_rows;
    ++dst_rows;
  }
}

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t remap_f32(const T* src, size_t src_stride, size_t src_width,
                           size_t src_height, T* dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const float* mapx, size_t mapx_stride,
                           const float* mapy, size_t mapy_stride,
                           kleidicv_interpolation_type_t interpolation,
                           kleidicv_border_type_t border_type,
                           const T* border_value) {
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
      const type* src, size_t src_stride, size_t src_width, size_t src_height, \
      type* dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const float* mapx, size_t mapx_stride,                  \
      const float* mapy, size_t mapy_stride,                                   \
      kleidicv_interpolation_type_t interpolation,                             \
      kleidicv_border_type_t border_type, const type* border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(uint16_t);

}  // namespace kleidicv::sve2
