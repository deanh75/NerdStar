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

template <typename ScalarType>
class RemapS16Replicate {
 public:
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;

  RemapS16Replicate(Rows<const ScalarType> src_rows, size_t src_width,
                    size_t src_height, svuint16_t& v_src_element_stride,
                    MapVectorType& v_x_max, MapVectorType& v_y_max)
      : src_rows_{src_rows},
        v_src_element_stride_{v_src_element_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_element_stride_ = svdup_u16(src_rows.stride() / sizeof(ScalarType));
    v_xmax_ = svdup_s16(static_cast<int16_t>(src_width - 1));
    v_ymax_ = svdup_s16(static_cast<int16_t>(src_height - 1));
  }

  void transform_pixels(svbool_t pg, svuint32_t offsets_b, svbool_t pg_b,
                        svuint32_t offsets_t, svbool_t pg_t,
                        Columns<ScalarType> dst);

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<ScalarType> dst) {
    svuint32_t offsets_b, offsets_t;
    svint16_t svzero = svdup_n_s16(0);
    auto load_offsets = [&](svbool_t pg) {
      MapVector2Type xy = svld2_s16(pg, &mapxy[0]);
      // Clamp coordinates to within the dimensions of the source image
      svuint16_t x = svreinterpret_u16_s16(
          svmax_x(pg, svzero, svmin_x(pg, svget2(xy, 0), v_xmax_)));
      svuint16_t y = svreinterpret_u16_s16(
          svmax_x(pg, svzero, svmin_x(pg, svget2(xy, 1), v_ymax_)));
      // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
      offsets_b = svmlalb_u32(svmovlb_u32(x), y, v_src_element_stride_);
      offsets_t = svmlalt_u32(svmovlt_u32(x), y, v_src_element_stride_);
    };

    svbool_t pg_all16 = MapVecTraits::svptrue();
    svbool_t pg_all32 = svptrue_b32();

    auto gather_load_generic_vector_path = [&](svbool_t pg, ptrdiff_t step) {
      load_offsets(pg);
      svbool_t pg_b = svwhilelt_b32(int64_t{0}, (step + 1) / 2);
      svbool_t pg_t = svwhilelt_b32(int64_t{0}, step / 2);
      transform_pixels(pg, offsets_b, pg_b, offsets_t, pg_t, dst);
      mapxy += step;
      dst += step;
    };

    // NOTE: gather load is not available in streaming mode
    auto gather_load_full_vector_path = [&](ptrdiff_t step) {
      load_offsets(pg_all16);
      transform_pixels(pg_all16, offsets_b, pg_all32, offsets_t, pg_all32, dst);
      mapxy += step;
      dst += step;
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      gather_load_full_vector_path(static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      gather_load_generic_vector_path(pg, static_cast<ptrdiff_t>(length));
    });
  }

 private:
  Rows<const ScalarType> src_rows_;
  svuint16_t& v_src_element_stride_;
  MapVectorType& v_xmax_;
  MapVectorType& v_ymax_;
};  // end of class RemapS16Replicate<ScalarType>

template <>
void RemapS16Replicate<uint8_t>::transform_pixels(
    svbool_t pg, svuint32_t offsets_b, svbool_t pg_b, svuint32_t offsets_t,
    svbool_t pg_t, Columns<uint8_t> dst) {
  // Copy pixels from source
  svuint32_t result_b =
      svld1ub_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
  svuint32_t result_t =
      svld1ub_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);
  svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                 svreinterpret_u16_u32(result_t));

  svst1b_u16(pg, &dst[0], result);
}

template <>
void RemapS16Replicate<uint16_t>::transform_pixels(
    svbool_t pg, svuint32_t offsets_b, svbool_t pg_b, svuint32_t offsets_t,
    svbool_t pg_t, Columns<uint16_t> dst) {
  // Account for the size of the source type when calculating offset
  offsets_b = svlsl_n_u32_x(pg, offsets_b, 1);
  offsets_t = svlsl_n_u32_x(pg, offsets_t, 1);

  // Copy pixels from source
  svuint32_t result_b =
      svld1uh_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
  svuint32_t result_t =
      svld1uh_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);
  svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                 svreinterpret_u16_u32(result_t));

  svst1_u16(pg, &dst[0], result);
}

template <typename ScalarType>
class RemapS16ConstantBorder {
 public:
  RemapS16ConstantBorder(Rows<const ScalarType> src_rows, size_t src_width,
                         size_t src_height, const ScalarType* border_value,
                         svuint16_t& v_src_element_stride, svuint16_t& v_width,
                         svuint16_t& v_height, svuint16_t& v_border)
      : src_rows_{src_rows},
        v_src_element_stride_{v_src_element_stride},
        v_width_{v_width},
        v_height_{v_height},
        v_border_{v_border} {
    v_src_element_stride_ = svdup_u16(src_rows.stride() / sizeof(ScalarType));
    v_width_ = svdup_u16(static_cast<uint16_t>(src_width));
    v_height_ = svdup_u16(static_cast<uint16_t>(src_height));
    v_border_ = svdup_u16(*border_value);
  }

  void transform_pixels(svbool_t pg, svuint32_t offsets_b, svbool_t pg_b,
                        svuint32_t offsets_t, svbool_t pg_t, ScalarType* dst);

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<ScalarType> dst) {
    for (size_t i = 0; i < width; i += svcnth()) {
      svbool_t pg = svwhilelt_b16_u64(i, width);

      svint16x2_t xy = svld2_s16(pg, &mapxy[static_cast<ptrdiff_t>(i * 2)]);
      svuint16_t x = svreinterpret_u16_s16(svget2(xy, 0));
      svuint16_t y = svreinterpret_u16_s16(svget2(xy, 1));

      // Find whether coordinates are within the image dimensions.
      svbool_t in_range = svand_b_z(pg, svcmplt_u16(pg, x, v_width_),
                                    svcmplt_u16(pg, y, v_height_));
      svbool_t pg_b = in_range;
      svbool_t pg_t = svtrn2_b16(in_range, svpfalse());

      // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
      svuint32_t offsets_b =
          svmlalb_u32(svmovlb_u32(x), y, v_src_element_stride_);
      svuint32_t offsets_t =
          svmlalt_u32(svmovlt_u32(x), y, v_src_element_stride_);

      transform_pixels(pg, offsets_b, pg_b, offsets_t, pg_t,
                       &dst[static_cast<ptrdiff_t>(i)]);
    }
  }

 private:
  Rows<const ScalarType> src_rows_;
  svuint16_t& v_src_element_stride_;
  svuint16_t& v_width_;
  svuint16_t& v_height_;
  svuint16_t& v_border_;
};  // end of class RemapS16ConstantBorder<ScalarType>

template <>
void RemapS16ConstantBorder<uint8_t>::transform_pixels(
    svbool_t pg, svuint32_t offsets_b, svbool_t pg_b, svuint32_t offsets_t,
    svbool_t pg_t, uint8_t* dst) {
  // Copy pixels from source
  svuint32_t result_b =
      svld1ub_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
  svuint32_t result_t =
      svld1ub_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);

  svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                 svreinterpret_u16_u32(result_t));

  svuint16_t result_selected = svsel(pg_b, result, v_border_);
  svst1b_u16(pg, dst, result_selected);
}

template <>
void RemapS16ConstantBorder<uint16_t>::transform_pixels(
    svbool_t pg, svuint32_t offsets_b, svbool_t pg_b, svuint32_t offsets_t,
    svbool_t pg_t, uint16_t* dst) {
  // Account for the size of the source type when calculating offset
  offsets_b = svlsl_n_u32_x(pg, offsets_b, 1);
  offsets_t = svlsl_n_u32_x(pg, offsets_t, 1);

  // Copy pixels from source
  svuint32_t result_b =
      svld1uh_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
  svuint32_t result_t =
      svld1uh_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);

  svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                 svreinterpret_u16_u32(result_t));

  svuint16_t result_selected = svsel(pg_b, result, v_border_);
  svst1_u16(pg, dst, result_selected);
}

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t remap_s16(const T* src, size_t src_stride, size_t src_width,
                           size_t src_height, T* dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const int16_t* mapxy, size_t mapxy_stride,
                           kleidicv_border_type_t border_type,
                           const T* border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapxy, mapxy_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (!remap_s16_is_implemented<T>(src_stride, src_width, src_height, dst_width,
                                   border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<const int16_t> mapxy_rows{mapxy, mapxy_stride, 2};
  Rows<T> dst_rows{dst, dst_stride, channels};
  svuint16_t sv_src_element_stride;
  Rectangle rect{dst_width, dst_height};
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT) {
    svuint16_t sv_width, sv_height, sv_border;
    RemapS16ConstantBorder<T> operation{
        src_rows, src_width, src_height, border_value, sv_src_element_stride,
        sv_width, sv_height, sv_border};
    zip_rows(operation, rect, mapxy_rows, dst_rows);
  } else {
    assert(border_type == KLEIDICV_BORDER_TYPE_REPLICATE);
    svint16_t sv_xmax, sv_ymax;
    RemapS16Replicate<T> operation{src_rows,   src_width,
                                   src_height, sv_src_element_stride,
                                   sv_xmax,    sv_ymax};
    zip_rows(operation, rect, mapxy_rows, dst_rows);
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16<type>(          \
      const type* src, size_t src_stride, size_t src_width, size_t src_height, \
      type* dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t* mapxy, size_t mapxy_stride,              \
      kleidicv_border_type_t border_type, const type* border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint16_t);

}  // namespace kleidicv::sve2
