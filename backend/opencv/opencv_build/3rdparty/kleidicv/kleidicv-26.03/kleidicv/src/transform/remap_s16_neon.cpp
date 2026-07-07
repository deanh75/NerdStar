// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <type_traits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/transform/remap.h"

namespace kleidicv::neon {

template <typename ScalarType>
class RemapS16Replicate {
 public:
  using MapVecTraits = neon::VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;

  RemapS16Replicate(Rows<const ScalarType> src_rows, size_t src_width,
                    size_t src_height)
      : src_rows_{src_rows},
        v_src_element_stride_{vdupq_n_u16(
            static_cast<uint16_t>(src_rows_.stride() / sizeof(ScalarType)))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))} {}

  void transform_pixels(uint32x4_t indices_low, uint32x4_t indices_high,
                        Columns<ScalarType> dst) {
    // Copy pixels from source
    dst[0] = src_rows_[vgetq_lane_u32(indices_low, 0)];
    dst[1] = src_rows_[vgetq_lane_u32(indices_low, 1)];
    dst[2] = src_rows_[vgetq_lane_u32(indices_low, 2)];
    dst[3] = src_rows_[vgetq_lane_u32(indices_low, 3)];

    dst[4] = src_rows_[vgetq_lane_u32(indices_high, 0)];
    dst[5] = src_rows_[vgetq_lane_u32(indices_high, 1)];
    dst[6] = src_rows_[vgetq_lane_u32(indices_high, 2)];
    dst[7] = src_rows_[vgetq_lane_u32(indices_high, 3)];
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      MapVector2Type xy = vld2q_s16(&mapxy[0]);
      // Clamp coordinates to within the dimensions of the source image
      uint16x8_t x = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[0], v_xmax_)));
      uint16x8_t y = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[1], v_ymax_)));
      // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
      uint32x4_t indices_low =
          vmlal_u16(vmovl_u16(vget_low_u16(x)), vget_low_u16(y),
                    vget_low_u16(v_src_element_stride_));
      uint32x4_t indices_high =
          vmlal_high_u16(vmovl_high_u16(x), y, v_src_element_stride_);

      transform_pixels(indices_low, indices_high, dst);

      mapxy += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x8_t v_src_element_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
};  // end of class RemapS16Replicate<ScalarType>

template <typename ScalarType>
class RemapS16ConstantBorder {
 public:
  using SrcVecTraits = neon::VecTraits<ScalarType>;
  using SrcVecType = typename SrcVecTraits::VectorType;

  using MapVecTraits = neon::VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;

  RemapS16ConstantBorder(Rows<const ScalarType> src_rows, size_t src_width,
                         size_t src_height, const ScalarType *border_value)
      : src_rows_{src_rows},
        v_src_element_stride_{vdupq_n_u16(
            static_cast<uint16_t>(src_rows_.stride() / sizeof(ScalarType)))},
        v_width_{vdupq_n_u16(static_cast<uint16_t>(src_width))},
        v_height_{vdupq_n_u16(static_cast<uint16_t>(src_height))},
        v_border_{vdupq_n_u16(*border_value)} {}

  void transform_pixels(uint32x4_t indices_low, uint32x4_t indices_high,
                        uint16x8_t in_range, Columns<ScalarType> dst);

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      MapVector2Type xy = vld2q_s16(&mapxy[0]);

      uint16x8_t x = vreinterpretq_u16_s16(xy.val[0]);
      uint16x8_t y = vreinterpretq_u16_s16(xy.val[1]);

      // Find whether coordinates are within the image dimensions.
      // Negative coordinates are interpreted as large values due to the
      // s16->u16 reinterpretation.
      uint16x8_t in_range =
          vandq_u16(vcltq_u16(x, v_width_), vcltq_u16(y, v_height_));

      // Zero out-of-range coordinates.
      x = vandq_u16(in_range, x);
      y = vandq_u16(in_range, y);

      // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
      uint32x4_t indices_low =
          vmlal_u16(vmovl_u16(vget_low_u16(x)), vget_low_u16(y),
                    vget_low_u16(v_src_element_stride_));
      uint32x4_t indices_high =
          vmlal_high_u16(vmovl_high_u16(x), y, v_src_element_stride_);

      transform_pixels(indices_low, indices_high, in_range, dst);

      mapxy += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x8_t v_src_element_stride_;
  uint16x8_t v_width_;
  uint16x8_t v_height_;
  uint16x8_t v_border_;
};  // end of class RemapS16ConstantBorder<ScalarType>

template <>
void RemapS16ConstantBorder<uint8_t>::transform_pixels(uint32x4_t indices_low,
                                                       uint32x4_t indices_high,
                                                       uint16x8_t in_range,
                                                       Columns<uint8_t> dst) {
  uint8x8_t pixels = {
      src_rows_[vgetq_lane_u32(indices_low, 0)],
      src_rows_[vgetq_lane_u32(indices_low, 1)],
      src_rows_[vgetq_lane_u32(indices_low, 2)],
      src_rows_[vgetq_lane_u32(indices_low, 3)],
      src_rows_[vgetq_lane_u32(indices_high, 0)],
      src_rows_[vgetq_lane_u32(indices_high, 1)],
      src_rows_[vgetq_lane_u32(indices_high, 2)],
      src_rows_[vgetq_lane_u32(indices_high, 3)],
  };
  // Select between source pixels and border color
  uint8x8_t pixels_or_border =
      vbsl_u8(vmovn_u16(in_range), pixels, vmovn_u16(v_border_));

  vst1_u8(&dst[0], pixels_or_border);
}

template <>
void RemapS16ConstantBorder<uint16_t>::transform_pixels(uint32x4_t indices_low,
                                                        uint32x4_t indices_high,
                                                        uint16x8_t in_range,
                                                        Columns<uint16_t> dst) {
  uint16x8_t pixels = {
      src_rows_[vgetq_lane_u32(indices_low, 0)],
      src_rows_[vgetq_lane_u32(indices_low, 1)],
      src_rows_[vgetq_lane_u32(indices_low, 2)],
      src_rows_[vgetq_lane_u32(indices_low, 3)],
      src_rows_[vgetq_lane_u32(indices_high, 0)],
      src_rows_[vgetq_lane_u32(indices_high, 1)],
      src_rows_[vgetq_lane_u32(indices_high, 2)],
      src_rows_[vgetq_lane_u32(indices_high, 3)],
  };

  // Select between source pixels and border color
  uint16x8_t pixels_or_border = vbslq_u16(in_range, pixels, v_border_);

  vst1q_u16(&dst[0], pixels_or_border);
}

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t remap_s16(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const int16_t *mapxy, size_t mapxy_stride,
                           kleidicv_border_type_t border_type,
                           [[maybe_unused]] const T *border_value) {
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
  Rectangle rect{dst_width, dst_height};
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT) {
    RemapS16ConstantBorder<T> operation{src_rows, src_width, src_height,
                                        border_value};
    zip_rows(operation, rect, mapxy_rows, dst_rows);
  } else {
    assert(border_type == KLEIDICV_BORDER_TYPE_REPLICATE);
    RemapS16Replicate<T> operation{src_rows, src_width, src_height};
    zip_rows(operation, rect, mapxy_rows, dst_rows);
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16<type>(          \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint16_t);

}  // namespace kleidicv::neon
