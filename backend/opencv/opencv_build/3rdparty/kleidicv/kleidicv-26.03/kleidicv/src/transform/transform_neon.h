// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/neon.h"
#include "kleidicv/types.h"
#include "transform_common.h"

namespace kleidicv::neon {

typedef struct {
  float32x4_t x, y;
} FloatVectorPair;

template <typename ScalarType, bool IsLarge>
float32x4_t inline load_xy(uint32x4_t x, uint32x4_t y, uint32x4_t v_src_stride,
                           Rows<const ScalarType>& src_rows) {
  if constexpr (IsLarge) {
    uint64x2_t indices_low =
        vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                  vget_low_u32(v_src_stride));
    uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                        vget_low_u32(v_src_stride));
    uint64_t acc =
        static_cast<uint64_t>(src_rows[vgetq_lane_u64(indices_low, 0)]) |
        (static_cast<uint64_t>(src_rows[vgetq_lane_u64(indices_low, 1)]) << 32);
    uint64x2_t rawsrc = vdupq_n_u64(acc);
    acc = static_cast<uint64_t>(src_rows[vgetq_lane_u64(indices_high, 0)]) |
          (static_cast<uint64_t>(src_rows[vgetq_lane_u64(indices_high, 1)])
           << 32);
    rawsrc = vsetq_lane_u64(acc, rawsrc, 1);
    return vcvtq_f32_u32(vreinterpretq_u32_u64(rawsrc));
  } else {
    uint32x4_t indices = vmlaq_u32(x, y, v_src_stride);
    uint64_t acc =
        static_cast<uint64_t>(src_rows[vgetq_lane_u32(indices, 0)]) |
        (static_cast<uint64_t>(src_rows[vgetq_lane_u32(indices, 1)]) << 32);
    uint64x2_t rawsrc = vdupq_n_u64(acc);
    acc = static_cast<uint64_t>(src_rows[vgetq_lane_u32(indices, 2)]) |
          (static_cast<uint64_t>(src_rows[vgetq_lane_u32(indices, 3)]) << 32);
    rawsrc = vsetq_lane_u64(acc, rawsrc, 1);
    return vcvtq_f32_u32(vreinterpretq_u32_u64(rawsrc));
  }
}

template <typename ScalarType, bool IsLarge>
float32x4x2_t inline load_xy_2ch(uint32x4_t x, uint32x4_t y,
                                 uint32x4_t v_src_stride,
                                 Rows<const ScalarType>& src_rows) {
  const size_t kBytes = 2 * sizeof(ScalarType);
  ScalarType elements[4 * 2];  // 4 pixels, 2 channels
  // Multiply x with the number of channels (2)
  x = vshlq_n_u32(x, 1);
  if constexpr (IsLarge) {
    uint64x2_t indices_low =
        vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                  vget_low_u32(v_src_stride));
    uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                        vget_low_u32(v_src_stride));
    memcpy(&elements[0], &src_rows[vgetq_lane_u64(indices_low, 0)], kBytes);
    memcpy(&elements[2], &src_rows[vgetq_lane_u64(indices_low, 1)], kBytes);
    memcpy(&elements[4], &src_rows[vgetq_lane_u64(indices_high, 0)], kBytes);
    memcpy(&elements[6], &src_rows[vgetq_lane_u64(indices_high, 1)], kBytes);
  } else {
    uint32x4_t indices = vmlaq_u32(x, y, v_src_stride);
    memcpy(&elements[0], &src_rows[vgetq_lane_u32(indices, 0)], kBytes);
    memcpy(&elements[2], &src_rows[vgetq_lane_u32(indices, 1)], kBytes);
    memcpy(&elements[4], &src_rows[vgetq_lane_u32(indices, 2)], kBytes);
    memcpy(&elements[6], &src_rows[vgetq_lane_u32(indices, 3)], kBytes);
  }
  uint16x8_t pixels16{};
  if constexpr (std::is_same<ScalarType, uint8_t>::value) {
    pixels16 = vmovl_u8(vld1_u8(elements));
  } else if constexpr (std::is_same<ScalarType, uint16_t>::value) {
    pixels16 = vld1q_u16(elements);
  }
  float32x4x2_t result;
  result.val[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixels16)));
  result.val[1] = vcvtq_f32_u32(vmovl_high_u16(pixels16));
  return result;
}

template <typename ScalarType, bool IsLarge>
float32x4_t inline load_xy_or_border(uint32x4_t x, uint32x4_t y,
                                     uint32x4_t in_range,
                                     ScalarType border_value,
                                     uint32x4_t v_src_stride,
                                     Rows<const ScalarType> src_rows) {
  if constexpr (IsLarge) {
    uint64x2_t indices_low =
        vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                  vget_low_u32(v_src_stride));
    uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                        vget_low_u32(v_src_stride));
    uint64_t pixel0 = vgetq_lane_u32(in_range, 0)
                          ? src_rows[vgetq_lane_u64(indices_low, 0)]
                          : border_value;
    uint64_t pixel1 = vgetq_lane_u32(in_range, 1)
                          ? src_rows[vgetq_lane_u64(indices_low, 1)]
                          : border_value;
    uint64_t pixel2 = vgetq_lane_u32(in_range, 2)
                          ? src_rows[vgetq_lane_u64(indices_high, 0)]
                          : border_value;
    uint64_t pixel3 = vgetq_lane_u32(in_range, 3)
                          ? src_rows[vgetq_lane_u64(indices_high, 1)]
                          : border_value;
    uint64x2_t rawsrc = vdupq_n_u64(pixel0 | (pixel1 << 32));
    rawsrc = vdupq_n_u64(pixel0 | (pixel1 << 32));
    rawsrc = vsetq_lane_u64(pixel2 | (pixel3 << 32), rawsrc, 1);
    return vcvtq_f32_u32(vreinterpretq_u32_u64(rawsrc));
  } else {
    uint32x4_t indices = vmlaq_u32(x, y, v_src_stride);
    uint64_t pixel0 = vgetq_lane_u32(in_range, 0)
                          ? src_rows[vgetq_lane_u32(indices, 0)]
                          : border_value;
    uint64_t pixel1 = vgetq_lane_u32(in_range, 1)
                          ? src_rows[vgetq_lane_u32(indices, 1)]
                          : border_value;
    uint64_t pixel2 = vgetq_lane_u32(in_range, 2)
                          ? src_rows[vgetq_lane_u32(indices, 2)]
                          : border_value;
    uint64_t pixel3 = vgetq_lane_u32(in_range, 3)
                          ? src_rows[vgetq_lane_u32(indices, 3)]
                          : border_value;
    uint64x2_t rawsrc = vdupq_n_u64(pixel0 | (pixel1 << 32));
    rawsrc = vdupq_n_u64(pixel0 | (pixel1 << 32));
    rawsrc = vsetq_lane_u64(pixel2 | (pixel3 << 32), rawsrc, 1);
    return vcvtq_f32_u32(vreinterpretq_u32_u64(rawsrc));
  }
}

template <typename ScalarType, bool IsLarge>
float32x4x2_t inline load_xy_or_border_2ch(uint32x4_t x, uint32x4_t y,
                                           uint32x4_t in_range,
                                           const ScalarType* border_values,
                                           uint32x4_t v_src_stride,
                                           Rows<const ScalarType> src_rows) {
  const size_t kBytes = 2 * sizeof(ScalarType);
  const ScalarType *pixel0{}, *pixel1{}, *pixel2{}, *pixel3{};
  ScalarType elements[4 * 2];  // 4 pixels, 2 channels
  // Multiply x with the number of channels
  x = vshlq_n_u32(x, 1);
  if constexpr (IsLarge) {
    uint64x2_t indices_low =
        vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                  vget_low_u32(v_src_stride));
    uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                        vget_low_u32(v_src_stride));
    pixel0 = vgetq_lane_u32(in_range, 0)
                 ? &src_rows[vgetq_lane_u64(indices_low, 0)]
                 : border_values;
    pixel1 = vgetq_lane_u32(in_range, 1)
                 ? &src_rows[vgetq_lane_u64(indices_low, 1)]
                 : border_values;
    pixel2 = vgetq_lane_u32(in_range, 2)
                 ? &src_rows[vgetq_lane_u64(indices_high, 0)]
                 : border_values;
    pixel3 = vgetq_lane_u32(in_range, 3)
                 ? &src_rows[vgetq_lane_u64(indices_high, 1)]
                 : border_values;
  } else {
    uint32x4_t indices = vmlaq_u32(x, y, v_src_stride);
    pixel0 = vgetq_lane_u32(in_range, 0) ? &src_rows[vgetq_lane_u32(indices, 0)]
                                         : border_values;
    pixel1 = vgetq_lane_u32(in_range, 1) ? &src_rows[vgetq_lane_u32(indices, 1)]
                                         : border_values;
    pixel2 = vgetq_lane_u32(in_range, 2) ? &src_rows[vgetq_lane_u32(indices, 2)]
                                         : border_values;
    pixel3 = vgetq_lane_u32(in_range, 3) ? &src_rows[vgetq_lane_u32(indices, 3)]
                                         : border_values;
  }
  memcpy(&elements[0], pixel0, kBytes);
  memcpy(&elements[2], pixel1, kBytes);
  memcpy(&elements[4], pixel2, kBytes);
  memcpy(&elements[6], pixel3, kBytes);
  uint16x8_t pixels16{};
  if constexpr (std::is_same<ScalarType, uint8_t>::value) {
    pixels16 = vmovl_u8(vld1_u8(elements));
  } else if constexpr (std::is_same<ScalarType, uint16_t>::value) {
    pixels16 = vld1q_u16(elements);
  }
  float32x4x2_t result;
  result.val[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixels16)));
  result.val[1] = vcvtq_f32_u32(vmovl_high_u16(pixels16));
  return result;
}

template <typename ScalarType, bool IsLarge>
void load_quad_pixels_replicate(FloatVectorPair xy, uint32x4_t v_xmax,
                                uint32x4_t v_ymax, uint32x4_t v_src_stride,
                                Rows<const ScalarType> src_rows,
                                float32x4_t& xfrac, float32x4_t& yfrac,
                                float32x4_t& a, float32x4_t& b, float32x4_t& c,
                                float32x4_t& d) {
  auto&& [xf, yf] = xy;
  // Truncating convert to int
  uint32x4_t x0 = vminq_u32(vcvtmq_u32_f32(xf), v_xmax);
  uint32x4_t y0 = vminq_u32(vcvtmq_u32_f32(yf), v_ymax);

  // Get fractional part, or 0 if out of range
  float32x4_t zero = vdupq_n_f32(0.F);
  uint32x4_t x_in_range = vandq_u32(vcgeq_f32(xf, zero), vcltq_u32(x0, v_xmax));
  uint32x4_t y_in_range = vandq_u32(vcgeq_f32(yf, zero), vcltq_u32(y0, v_ymax));
  xfrac = vsubq_f32(xf, vrndmq_f32(xf));
  yfrac = vsubq_f32(yf, vrndmq_f32(yf));

  // x1 = x0 + 1, except if it's already xmax or out of range
  uint32x4_t x1 = vsubq_u32(x0, x_in_range);
  uint32x4_t y1 = vsubq_u32(y0, y_in_range);

  // a: top left, b: top right, c: bottom left, d: bottom right
  a = load_xy<ScalarType, IsLarge>(x0, y0, v_src_stride, src_rows);
  b = load_xy<ScalarType, IsLarge>(x1, y0, v_src_stride, src_rows);
  c = load_xy<ScalarType, IsLarge>(x0, y1, v_src_stride, src_rows);
  d = load_xy<ScalarType, IsLarge>(x1, y1, v_src_stride, src_rows);
}

template <typename ScalarType, bool IsLarge>
void load_quad_pixels_replicate_2ch(FloatVectorPair xy, uint32x4_t v_xmax,
                                    uint32x4_t v_ymax, uint32x4_t v_src_stride,
                                    Rows<const ScalarType> src_rows,
                                    float32x4_t& xfrac, float32x4_t& yfrac,
                                    float32x4x2_t& a, float32x4x2_t& b,
                                    float32x4x2_t& c, float32x4x2_t& d) {
  auto&& [xf, yf] = xy;
  // Truncating convert to int
  uint32x4_t x0 = vminq_u32(vcvtmq_u32_f32(xf), v_xmax);
  uint32x4_t y0 = vminq_u32(vcvtmq_u32_f32(yf), v_ymax);

  // Get fractional part, or 0 if out of range
  float32x4_t zero = vdupq_n_f32(0.F);
  uint32x4_t x_in_range = vandq_u32(vcgeq_f32(xf, zero), vcltq_u32(x0, v_xmax));
  uint32x4_t y_in_range = vandq_u32(vcgeq_f32(yf, zero), vcltq_u32(y0, v_ymax));
  xfrac = vsubq_f32(xf, vrndmq_f32(xf));
  yfrac = vsubq_f32(yf, vrndmq_f32(yf));

  // x1 = x0 + 1, except if it's already xmax or out of range
  uint32x4_t x1 = vsubq_u32(x0, x_in_range);
  uint32x4_t y1 = vsubq_u32(y0, y_in_range);

  // a: top left, b: top right, c: bottom left, d: bottom right
  a = load_xy_2ch<ScalarType, IsLarge>(x0, y0, v_src_stride, src_rows);
  b = load_xy_2ch<ScalarType, IsLarge>(x1, y0, v_src_stride, src_rows);
  c = load_xy_2ch<ScalarType, IsLarge>(x0, y1, v_src_stride, src_rows);
  d = load_xy_2ch<ScalarType, IsLarge>(x1, y1, v_src_stride, src_rows);
}

template <typename ScalarType, bool IsLarge>
void load_quad_pixels_constant(FloatVectorPair xy, uint32x4_t v_xmax,
                               uint32x4_t v_ymax, uint32x4_t v_src_stride,
                               const ScalarType* border_values,
                               Rows<const ScalarType> src_rows,
                               float32x4_t& xfrac, float32x4_t& yfrac,
                               float32x4_t& a, float32x4_t& b, float32x4_t& c,
                               float32x4_t& d) {
  auto&& [xf, yf] = xy;
  // Convert coordinates to integers, truncating towards minus infinity.
  // Negative numbers will become large positive numbers.
  // Since the source width and height is known to be <=2^24 these large
  // positive numbers will always be treated as outside the source image
  // bounds.
  uint32x4_t x0 = vreinterpretq_u32_s32(vcvtmq_s32_f32(xf));
  uint32x4_t y0 = vreinterpretq_u32_s32(vcvtmq_s32_f32(yf));
  uint32x4_t x1 = vaddq(x0, vdupq_n_u32(1));
  uint32x4_t y1 = vaddq(y0, vdupq_n_u32(1));
  xfrac = vsubq_f32(xf, vrndmq_f32(xf));
  yfrac = vsubq_f32(yf, vrndmq_f32(yf));
  uint32x4_t a_in_range, b_in_range, c_in_range, d_in_range;
  {
    uint32x4_t x0_in_range = vcleq_u32(x0, v_xmax);
    uint32x4_t y0_in_range = vcleq_u32(y0, v_ymax);
    uint32x4_t x1_in_range = vcleq_u32(x1, v_xmax);
    uint32x4_t y1_in_range = vcleq_u32(y1, v_ymax);
    a_in_range = vandq(x0_in_range, y0_in_range);
    b_in_range = vandq(x1_in_range, y0_in_range);
    c_in_range = vandq(x0_in_range, y1_in_range);
    d_in_range = vandq(x1_in_range, y1_in_range);
  }
  a = load_xy_or_border<ScalarType, IsLarge>(
      x0, y0, a_in_range, border_values[0], v_src_stride, src_rows);
  b = load_xy_or_border<ScalarType, IsLarge>(
      x1, y0, b_in_range, border_values[0], v_src_stride, src_rows);
  c = load_xy_or_border<ScalarType, IsLarge>(
      x0, y1, c_in_range, border_values[0], v_src_stride, src_rows);
  d = load_xy_or_border<ScalarType, IsLarge>(
      x1, y1, d_in_range, border_values[0], v_src_stride, src_rows);
}

template <typename ScalarType, bool IsLarge>
void load_quad_pixels_constant_2ch(FloatVectorPair xy, uint32x4_t v_xmax,
                                   uint32x4_t v_ymax, uint32x4_t v_src_stride,
                                   const ScalarType* border_values,
                                   Rows<const ScalarType> src_rows,
                                   float32x4_t& xfrac, float32x4_t& yfrac,
                                   float32x4x2_t& a, float32x4x2_t& b,
                                   float32x4x2_t& c, float32x4x2_t& d) {
  auto&& [xf, yf] = xy;
  // Convert coordinates to integers, truncating towards minus infinity.
  // Negative numbers will become large positive numbers.
  // Since the source width and height is known to be <=2^24 these large
  // positive numbers will always be treated as outside the source image
  // bounds.
  uint32x4_t x0 = vreinterpretq_u32_s32(vcvtmq_s32_f32(xf));
  uint32x4_t y0 = vreinterpretq_u32_s32(vcvtmq_s32_f32(yf));
  uint32x4_t x1 = vaddq(x0, vdupq_n_u32(1));
  uint32x4_t y1 = vaddq(y0, vdupq_n_u32(1));
  xfrac = vsubq_f32(xf, vrndmq_f32(xf));
  yfrac = vsubq_f32(yf, vrndmq_f32(yf));
  uint32x4_t a_in_range, b_in_range, c_in_range, d_in_range;
  {
    uint32x4_t x0_in_range = vcleq_u32(x0, v_xmax);
    uint32x4_t y0_in_range = vcleq_u32(y0, v_ymax);
    uint32x4_t x1_in_range = vcleq_u32(x1, v_xmax);
    uint32x4_t y1_in_range = vcleq_u32(y1, v_ymax);
    a_in_range = vandq(x0_in_range, y0_in_range);
    b_in_range = vandq(x1_in_range, y0_in_range);
    c_in_range = vandq(x0_in_range, y1_in_range);
    d_in_range = vandq(x1_in_range, y1_in_range);
  }
  a = load_xy_or_border_2ch<ScalarType, IsLarge>(
      x0, y0, a_in_range, border_values, v_src_stride, src_rows);
  b = load_xy_or_border_2ch<ScalarType, IsLarge>(
      x1, y0, b_in_range, border_values, v_src_stride, src_rows);
  c = load_xy_or_border_2ch<ScalarType, IsLarge>(
      x0, y1, c_in_range, border_values, v_src_stride, src_rows);
  d = load_xy_or_border_2ch<ScalarType, IsLarge>(
      x1, y1, d_in_range, border_values, v_src_stride, src_rows);
}

inline uint32x4_t lerp_2d(float32x4_t xfrac, float32x4_t yfrac, float32x4_t a,
                          float32x4_t b, float32x4_t c, float32x4_t d) {
  float32x4_t line0 = vmlaq_f32(a, vsubq_f32(b, a), xfrac);
  float32x4_t line1 = vmlaq_f32(c, vsubq_f32(d, c), xfrac);
  float32x4_t result = vmlaq_f32(line0, vsubq_f32(line1, line0), yfrac);
  return vcvtaq_u32_f32(result);
}

template <typename ScalarType, bool IsLarge, size_t Channels>
void transform_pixels_replicate(float32x4_t xf, float32x4_t yf,
                                uint32x4_t v_xmax, uint32x4_t v_ymax,
                                uint32x4_t v_src_element_stride,
                                Rows<const ScalarType> src_rows,
                                Columns<ScalarType> dst) {
  // Round to nearest, with Ties To Away (i.e. round 0.5 up)
  // Clamp coordinates to within the dimensions of the source image
  // (vcvtaq already converted negative values to 0)
  uint32x4_t x = vminq_u32(vcvtaq_u32_f32(xf), v_xmax);
  uint32x4_t y = vminq_u32(vcvtaq_u32_f32(yf), v_ymax);
  if constexpr (Channels == 2) {
    // Multiply x with the number of channels
    x = vshlq_n_u32(x, 1);
  }
  // Copy pixels from source
  if constexpr (IsLarge) {
    uint64x2_t indices_low =
        vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                  vget_low_u32(v_src_element_stride));
    uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                        vget_low_u32(v_src_element_stride));
    if constexpr (Channels == 1) {
      dst[0] = src_rows[vgetq_lane_u64(indices_low, 0)];
      dst[1] = src_rows[vgetq_lane_u64(indices_low, 1)];
      dst[2] = src_rows[vgetq_lane_u64(indices_high, 0)];
      dst[3] = src_rows[vgetq_lane_u64(indices_high, 1)];
    } else {
      const size_t kBytes = Channels * sizeof(ScalarType);
      memcpy(dst.ptr_at(0), &src_rows[vgetq_lane_u64(indices_low, 0)], kBytes);
      memcpy(dst.ptr_at(1), &src_rows[vgetq_lane_u64(indices_low, 1)], kBytes);
      memcpy(dst.ptr_at(2), &src_rows[vgetq_lane_u64(indices_high, 0)], kBytes);
      memcpy(dst.ptr_at(3), &src_rows[vgetq_lane_u64(indices_high, 1)], kBytes);
    }
  } else {
    uint32x4_t indices = vmlaq_u32(x, y, v_src_element_stride);
    if constexpr (Channels == 1) {
      dst[0] = src_rows[vgetq_lane_u32(indices, 0)];
      dst[1] = src_rows[vgetq_lane_u32(indices, 1)];
      dst[2] = src_rows[vgetq_lane_u32(indices, 2)];
      dst[3] = src_rows[vgetq_lane_u32(indices, 3)];
    } else {
      const size_t kBytes = Channels * sizeof(ScalarType);
      memcpy(dst.ptr_at(0), &src_rows[vgetq_lane_u32(indices, 0)], kBytes);
      memcpy(dst.ptr_at(1), &src_rows[vgetq_lane_u32(indices, 1)], kBytes);
      memcpy(dst.ptr_at(2), &src_rows[vgetq_lane_u32(indices, 2)], kBytes);
      memcpy(dst.ptr_at(3), &src_rows[vgetq_lane_u32(indices, 3)], kBytes);
    }
  }
}

template <size_t Lane, typename ScalarType>
static const ScalarType* get_src_or_border_small(
    uint32x4_t in_range, Rows<const ScalarType> src_rows, uint32x4_t indices,
    const ScalarType* border_values) {
  return vgetq_lane_u32(in_range, Lane)
             ? &src_rows[vgetq_lane_u32(indices, Lane)]
             : border_values;
}

template <size_t Lane, typename ScalarType>
static const ScalarType* get_src_or_border_large(
    uint32x4_t in_range, Rows<const ScalarType> src_rows, uint64x2_t indices,
    const ScalarType* border_values) {
  return vgetq_lane_u32(in_range, Lane)
             ? &src_rows[vgetq_lane_u64(indices, Lane % 2)]
             : border_values;
}

template <typename ScalarType, bool IsLarge, size_t Channels>
void transform_pixels_constant(float32x4_t xf, float32x4_t yf,
                               uint32x4_t v_xmax, uint32x4_t v_ymax,
                               uint32x4_t v_src_element_stride,
                               Rows<const ScalarType> src_rows,
                               Columns<ScalarType> dst,
                               const ScalarType* border_values) {
  // Convert coordinates to integers.
  // Negative numbers will become large positive numbers.
  // Since the source width and height is known to be <=2^24 these large
  // positive numbers will always be treated as outside the source image
  // bounds.
  uint32x4_t x = vreinterpretq_u32_s32(vcvtaq_s32_f32(xf));
  uint32x4_t y = vreinterpretq_u32_s32(vcvtaq_s32_f32(yf));
  uint32x4_t in_range = vandq_u32(vcleq_u32(x, v_xmax), vcleq_u32(y, v_ymax));

  // Copy pixels from source
  if constexpr (Channels == 1) {
    if constexpr (IsLarge) {
      uint64x2_t indices_low =
          vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                    vget_low_u32(v_src_element_stride));
      uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                          vget_low_u32(v_src_element_stride));
      dst[0] = *get_src_or_border_large<0>(in_range, src_rows, indices_low,
                                           border_values);
      dst[1] = *get_src_or_border_large<1>(in_range, src_rows, indices_low,
                                           border_values);
      dst[2] = *get_src_or_border_large<2>(in_range, src_rows, indices_high,
                                           border_values);
      dst[3] = *get_src_or_border_large<3>(in_range, src_rows, indices_high,
                                           border_values);
    } else {
      uint32x4_t indices = vmlaq_u32(x, y, v_src_element_stride);
      dst[0] = *get_src_or_border_small<0>(in_range, src_rows, indices,
                                           border_values);
      dst[1] = *get_src_or_border_small<1>(in_range, src_rows, indices,
                                           border_values);
      dst[2] = *get_src_or_border_small<2>(in_range, src_rows, indices,
                                           border_values);
      dst[3] = *get_src_or_border_small<3>(in_range, src_rows, indices,
                                           border_values);
    }
  } else {  // Channels > 1
    const size_t kBytes = Channels * sizeof(ScalarType);
    const ScalarType *pixel0{}, *pixel1{}, *pixel2{}, *pixel3{};
    // Multiply x with the number of channels
    if constexpr (Channels == 2) {
      x = vshlq_n_u32(x, 1);
    } else {
      x = vmulq_n_u32(x, Channels);
    }
    if constexpr (IsLarge) {
      uint64x2_t indices_low =
          vmlal_u32(vmovl_u32(vget_low_u32(x)), vget_low_u32(y),
                    vget_low_u32(v_src_element_stride));
      uint64x2_t indices_high = vmlal_u32(vmovl_high_u32(x), vget_high_u32(y),
                                          vget_low_u32(v_src_element_stride));
      pixel0 = get_src_or_border_large<0>(in_range, src_rows, indices_low,
                                          border_values);
      pixel1 = get_src_or_border_large<1>(in_range, src_rows, indices_low,
                                          border_values);
      pixel2 = get_src_or_border_large<2>(in_range, src_rows, indices_high,
                                          border_values);
      pixel3 = get_src_or_border_large<3>(in_range, src_rows, indices_high,
                                          border_values);

    } else {
      uint32x4_t indices = vmlaq_u32(x, y, v_src_element_stride);
      pixel0 = get_src_or_border_small<0>(in_range, src_rows, indices,
                                          border_values);
      pixel1 = get_src_or_border_small<1>(in_range, src_rows, indices,
                                          border_values);
      pixel2 = get_src_or_border_small<2>(in_range, src_rows, indices,
                                          border_values);
      pixel3 = get_src_or_border_small<3>(in_range, src_rows, indices,
                                          border_values);
    }
    memcpy(dst.ptr_at(0), pixel0, kBytes);
    memcpy(dst.ptr_at(1), pixel1, kBytes);
    memcpy(dst.ptr_at(2), pixel2, kBytes);
    memcpy(dst.ptr_at(3), pixel3, kBytes);
  }
}

template <typename ScalarType, bool IsLarge, size_t Channels,
          kleidicv_border_type_t Border>
void transform_pixels(float32x4_t xf, float32x4_t yf, uint32x4_t v_xmax,
                      uint32x4_t v_ymax, uint32x4_t v_src_element_stride,
                      Rows<const ScalarType> src_rows, Columns<ScalarType> dst,
                      [[maybe_unused]] const ScalarType* border_values) {
  if constexpr (Border == KLEIDICV_BORDER_TYPE_REPLICATE) {
    transform_pixels_replicate<ScalarType, IsLarge, Channels>(
        xf, yf, v_xmax, v_ymax, v_src_element_stride, src_rows, dst);
  } else {
    static_assert(Border == KLEIDICV_BORDER_TYPE_CONSTANT);
    transform_pixels_constant<ScalarType, IsLarge, Channels>(
        xf, yf, v_xmax, v_ymax, v_src_element_stride, src_rows, dst,
        border_values);
  }
}

}  // namespace kleidicv::neon
