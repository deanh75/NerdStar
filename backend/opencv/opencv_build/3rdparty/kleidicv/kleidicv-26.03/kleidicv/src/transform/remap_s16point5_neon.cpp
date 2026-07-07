// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "kleidicv/neon.h"
#include "kleidicv/transform/remap.h"

namespace kleidicv::neon {

template <typename ScalarType>
class RemapS16Point5Replicate;

template <>
class RemapS16Point5Replicate<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = neon::VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = neon::VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5Replicate(Rows<const ScalarType> src_rows, size_t src_width,
                          size_t src_height)
      : src_rows_{src_rows},
        v_src_stride_{vdup_n_u16(static_cast<uint16_t>(src_rows_.stride()))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      MapVector2Type xy = vld2q_s16(&mapxy[0]);
      FracVectorType frac = vld1q_u16(&mapfrac[0]);
      uint16x8_t xfrac =
          vbslq_u16(vcltq_s16(xy.val[0], vdupq_n_s16(0)), vdupq_n_u16(0),
                    // extract xfrac = frac[0:4]
                    vandq_u16(frac, vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1)));
      uint16x8_t yfrac =
          vbslq_u16(vcltq_s16(xy.val[1], vdupq_n_s16(0)), vdupq_n_u16(0),
                    // extract yfrac = frac[5:9]
                    vandq_u16(vshrq_n_u16(frac, REMAP16POINT5_FRAC_BITS),
                              vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1)));
      uint16x8_t nxfrac = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
      uint16x8_t nyfrac = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);

      // Clamp coordinates to within the dimensions of the source image
      uint16x8_t x0 = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[0], v_xmax_)));
      uint16x8_t y0 = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[1], v_ymax_)));

      // x1 = x0 + 1, except if it's already xmax
      uint16x8_t x1 = vsubq_u16(x0, vcltq_s16(xy.val[0], v_xmax_));
      uint16x8_t y1 = vsubq_u16(y0, vcltq_s16(xy.val[1], v_ymax_));

      uint16x4_t dst_low = load_and_interpolate(
          vmovl_u16(vget_low_u16(x0)), vget_low_u16(y0),
          vmovl_u16(vget_low_u16(x1)), vget_low_u16(y1), vget_low_u16(xfrac),
          vget_low_u16(yfrac), vget_low_u16(nxfrac), vget_low_u16(nyfrac));

      uint16x4_t dst_high = load_and_interpolate(
          vmovl_high_u16(x0), vget_high_u16(y0), vmovl_high_u16(x1),
          vget_high_u16(y1), vget_high_u16(xfrac), vget_high_u16(yfrac),
          vget_high_u16(nxfrac), vget_high_u16(nyfrac));

      vst1_u8(&dst[0], vuzp1_u8(dst_low, dst_high));
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };
    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  uint16x4_t load_and_interpolate(uint32x4_t x0, uint16x4_t y0, uint32x4_t x1,
                                  uint16x4_t y1, uint16x4_t xfrac,
                                  uint16x4_t yfrac, uint16x4_t nxfrac,
                                  uint16x4_t nyfrac) {
    // Calculate offsets from coordinates (y * stride + x)
    // a: top left, b: top right, c: bottom left, d: bottom right
    uint32x4_t offset = vmlal_u16(x0, y0, v_src_stride_);
    uint64_t acc =
        static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
        (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
        (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
        (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t a = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    offset = vmlal_u16(x1, y0, v_src_stride_);

    acc = static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t b = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    uint16x4_t line0 = vmla_u16(vmul_u16(xfrac, b), nxfrac, a);

    offset = vmlal_u16(x0, y1, v_src_stride_);

    acc = static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t c = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    uint32x4_t line0_lerpd = vmlal_u16(
        vdupq_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2), line0, nyfrac);

    offset = vmlal_u16(x1, y1, v_src_stride_);

    acc = static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t d = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    uint16x4_t line1 = vmla_u16(vmul_u16(xfrac, d), nxfrac, c);
    return vshrn_n_u32(vmlal_u16(line0_lerpd, line1, yfrac),
                       2 * REMAP16POINT5_FRAC_BITS);
  }

  Rows<const ScalarType> src_rows_;
  uint16x4_t v_src_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
};  // end of class RemapS16Point5Replicate<uint8_t>

// Common interpolation function used by all RemapS16Point5 operations except
// 1-channel u8 with replicated borders (RemapS16Point5Replicate<uint8_t>)
// because that processes one half vector in one step
static uint16x8_t interpolate(uint16x8_t a, uint16x8_t b, uint16x8_t c,
                              uint16x8_t d, uint16x8_t xfrac, uint16x8_t yfrac,
                              uint16x8_t nxfrac, uint16x8_t nyfrac) {
  auto interpolate_horizontal = [](uint16x4_t left, uint16x4_t right,
                                   uint16x4_t frac,
                                   uint16x4_t nfrac) -> uint32x4_t {
    return vmlal_u16(vmull_u16(nfrac, left), frac, right);
  };

  auto interpolate_horizontal_low = [interpolate_horizontal](
                                        uint16x8_t left, uint16x8_t right,
                                        uint16x8_t frac,
                                        uint16x8_t nfrac) -> uint32x4_t {
    return interpolate_horizontal(vget_low_u16(left), vget_low_u16(right),
                                  vget_low_u16(frac), vget_low_u16(nfrac));
  };

  auto interpolate_horizontal_high = [interpolate_horizontal](
                                         uint16x8_t left, uint16x8_t right,
                                         uint16x8_t frac,
                                         uint16x8_t nfrac) -> uint32x4_t {
    return interpolate_horizontal(vget_high_u16(left), vget_high_u16(right),
                                  vget_high_u16(frac), vget_high_u16(nfrac));
  };

  // Offset pixel values by 0.5 before rounding down.
  const uint32x4_t bias = vdupq_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);

  auto interpolate_vertical = [&](uint32x4_t a, uint32x4_t b, uint32x4_t frac,
                                  uint32x4_t nfrac) -> uint32x4_t {
    uint32x4_t res32 = vmlaq_u32(vmlaq_u32(bias, a, nfrac), b, frac);
    return vshrq_n_u32(res32, 2 * REMAP16POINT5_FRAC_BITS);
  };

  uint32x4_t line0_low = interpolate_horizontal_low(a, b, xfrac, nxfrac);
  uint32x4_t line1_low = interpolate_horizontal_low(c, d, xfrac, nxfrac);
  uint32x4_t line0_high = interpolate_horizontal_high(a, b, xfrac, nxfrac);
  uint32x4_t line1_high = interpolate_horizontal_high(c, d, xfrac, nxfrac);

  uint32x4_t lo =
      interpolate_vertical(line0_low, line1_low, vmovl_u16(vget_low_u16(yfrac)),
                           vmovl_u16(vget_low_u16(nyfrac)));
  uint32x4_t hi = interpolate_vertical(
      line0_high, line1_high, vmovl_high_u16(yfrac), vmovl_high_u16(nyfrac));

  // Discard upper 16 bits of each element (low the precision back to original
  // 16 bits)
  uint16x8_t result =
      vuzp1q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi));
  return result;
}

template <>
class RemapS16Point5Replicate<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5Replicate(Rows<const ScalarType> src_rows, size_t src_width,
                          size_t src_height)
      : src_rows_{src_rows},
        v_src_element_stride_{vdupq_n_u16(
            static_cast<uint16_t>(src_rows_.stride() / sizeof(ScalarType)))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))},
        xfrac_{vdupq_n_u16(0)},
        yfrac_{vdupq_n_u16(0)},
        nxfrac_{vdupq_n_u16(0)},
        nyfrac_{vdupq_n_u16(0)},
        x0_{vdupq_n_s16(0)},
        x1_{vdupq_n_s16(0)},
        y0_{vdupq_n_s16(0)},
        y1_{vdupq_n_s16(0)} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      prepare_maps(mapxy, mapfrac);
      transform_pixels(dst);

      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };
    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

  void prepare_maps(Columns<const int16_t> mapxy,
                    Columns<const uint16_t> mapfrac) {
    int16x8x2_t xy = vld2q_s16(&mapxy[0]);
    uint16x8_t frac = vld1q_u16(&mapfrac[0]);
    uint16x8_t frac_max = vdupq_n_u16(REMAP16POINT5_FRAC_MAX);
    uint16x8_t frac_mask = vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1);
    xfrac_ = vbslq_u16(vcltq_s16(xy.val[0], vdupq_n_s16(0)), vdupq_n_u16(0),
                       vandq_u16(frac, frac_mask));
    yfrac_ = vbslq_u16(
        vcltq_s16(xy.val[1], vdupq_n_s16(0)), vdupq_n_u16(0),
        vandq_u16(vshrq_n_u16(frac, REMAP16POINT5_FRAC_BITS), frac_mask));
    nxfrac_ = vsubq_u16(frac_max, xfrac_);
    nyfrac_ = vsubq_u16(frac_max, yfrac_);

    // Clamp coordinates to within the dimensions of the source image
    x0_ = vreinterpretq_u16_s16(
        vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[0], v_xmax_)));
    y0_ = vreinterpretq_u16_s16(
        vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[1], v_ymax_)));

    // x1 = x0 + 1, except if it's already xmax
    x1_ = vsubq_u16(x0_, vcltq_s16(xy.val[0], v_xmax_));
    y1_ = vsubq_u16(y0_, vcltq_s16(xy.val[1], v_ymax_));
  }

  void transform_pixels(Columns<uint16_t> dst) {
    uint16x8_t a = load_pixels(x0_, y0_);
    uint16x8_t b = load_pixels(x1_, y0_);
    uint16x8_t c = load_pixels(x0_, y1_);
    uint16x8_t d = load_pixels(x1_, y1_);

    uint16x8_t result =
        interpolate(a, b, c, d, xfrac_, yfrac_, nxfrac_, nyfrac_);

    vst1q_u16(&dst[0], result);
  }

  uint16x8_t load_pixels(int16x8_t x, int16x8_t y) {
    // Clamp coordinates to within the dimensions of the source image
    uint16x8_t x_clamped =
        vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(x, vdupq_n_s16(0))), v_xmax_);
    uint16x8_t y_clamped =
        vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(y, vdupq_n_s16(0))), v_ymax_);

    // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
    uint32x4_t indices_low =
        vmlal_u16(vmovl_u16(vget_low_u16(x_clamped)), vget_low_u16(y_clamped),
                  vget_low_u16(v_src_element_stride_));
    uint32x4_t indices_high = vmlal_high_u16(vmovl_high_u16(x_clamped),
                                             y_clamped, v_src_element_stride_);

    // Read pixels from source
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

    return pixels;
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x8_t v_src_element_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
  uint16x8_t xfrac_;
  uint16x8_t yfrac_;
  uint16x8_t nxfrac_;
  uint16x8_t nyfrac_;
  int16x8_t x0_;
  int16x8_t x1_;
  int16x8_t y0_;
  int16x8_t y1_;
};  // end of class RemapS16Point5Replicate<uint16_t>

template <typename ScalarType>
class RemapS16Point5ConstantBorder;

template <>
class RemapS16Point5ConstantBorder<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5ConstantBorder(Rows<const ScalarType> src_rows,
                               size_t src_width, size_t src_height,
                               const ScalarType *border_value)
      : src_rows_{src_rows},
        v_src_stride_{vdupq_n_u16(static_cast<uint16_t>(src_rows_.stride()))},
        v_width_{vdupq_n_u16(static_cast<uint16_t>(src_width))},
        v_height_{vdupq_n_u16(static_cast<uint16_t>(src_height))},
        v_border_{vdupq_n_u16(static_cast<uint16_t>(*border_value))} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      int16x8x2_t xy = vld2q_s16(&mapxy[0]);
      uint16x8_t frac = vld1q_u16(&mapfrac[0]);
      uint16x8_t frac_max = vdupq_n_u16(REMAP16POINT5_FRAC_MAX);
      uint16x8_t frac_mask = vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1);
      uint16x8_t xfrac = vandq_u16(frac, frac_mask);
      uint16x8_t yfrac =
          vandq_u16(vshrq_n_u16(frac, REMAP16POINT5_FRAC_BITS), frac_mask);
      uint16x8_t nxfrac = vsubq_u16(frac_max, xfrac);
      uint16x8_t nyfrac = vsubq_u16(frac_max, yfrac);

      uint16x8_t one = vdupq_n_u16(1);
      uint16x8_t x0 = vreinterpretq_u16_s16(xy.val[0]);
      uint16x8_t y0 = vreinterpretq_u16_s16(xy.val[1]);
      uint16x8_t x1 = vaddq_u16(x0, one);
      uint16x8_t y1 = vaddq_u16(y0, one);

      uint16x8_t a = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, x0, y0);
      uint16x8_t b = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, x1, y0);
      uint16x8_t c = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, x0, y1);
      uint16x8_t d = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, x1, y1);

      uint16x8_t result = interpolate(a, b, c, d, xfrac, yfrac, nxfrac, nyfrac);

      vst1_u8(&dst[0], vqmovn_u16(result));
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };
    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  uint16x8_t load_pixels_or_constant_border(Rows<const uint8_t> &src_rows_,
                                            uint16x8_t v_src_element_stride_,
                                            uint16x8_t v_width_,
                                            uint16x8_t v_height_,
                                            uint16x8_t v_border_, uint16x8_t x,
                                            uint16x8_t y) {
    // Find whether coordinates are within the image dimensions.
    // Negative coordinates are interpreted as large values due to the s16->u16
    // reinterpretation.
    uint16x8_t in_range =
        vandq_u16(vcltq_u16(vreinterpretq_u16_s16(x), v_width_),
                  vcltq_u16(vreinterpretq_u16_s16(y), v_height_));

    // Zero out-of-range coordinates.
    x = vandq_u16(in_range, x);
    y = vandq_u16(in_range, y);

    // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
    uint32x4_t indices_low =
        vmlal_u16(vmovl_u16(vget_low_u16(x)), vget_low_u16(y),
                  vget_low_u16(v_src_element_stride_));
    uint32x4_t indices_high =
        vmlal_high_u16(vmovl_high_u16(x), y, v_src_element_stride_);

    // Read pixels from source
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
    return vbslq_u16(in_range, vmovl_u8(pixels), v_border_);
  }

  Rows<const ScalarType> src_rows_;
  uint16x8_t v_src_stride_;
  uint16x8_t v_width_;
  uint16x8_t v_height_;
  uint16x8_t v_border_;
};  // end of class RemapS16Point5ConstantBorder<uint8_t>

template <>
class RemapS16Point5ConstantBorder<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5ConstantBorder(Rows<const ScalarType> src_rows,
                               size_t src_width, size_t src_height,
                               const ScalarType *border_value)
      : src_rows_{src_rows},
        v_src_element_stride_{vdupq_n_u16(
            static_cast<uint16_t>(src_rows_.stride() / sizeof(ScalarType)))},
        v_width_{vdupq_n_u16(static_cast<uint16_t>(src_width))},
        v_height_{vdupq_n_u16(static_cast<uint16_t>(src_height))},
        v_border_{vdupq_n_u16(*border_value)},
        xfrac_{vdupq_n_u16(0)},
        yfrac_{vdupq_n_u16(0)},
        nxfrac_{vdupq_n_u16(0)},
        nyfrac_{vdupq_n_u16(0)},
        x0_{vdupq_n_s16(0)},
        x1_{vdupq_n_s16(0)},
        y0_{vdupq_n_s16(0)},
        y1_{vdupq_n_s16(0)} {}

  void prepare_maps(Columns<const int16_t> mapxy,
                    Columns<const uint16_t> mapfrac) {
    int16x8x2_t xy = vld2q_s16(&mapxy[0]);
    uint16x8_t frac = vld1q_u16(&mapfrac[0]);
    uint16x8_t frac_max = vdupq_n_u16(REMAP16POINT5_FRAC_MAX);
    uint16x8_t frac_mask = vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1);
    xfrac_ = vandq_u16(frac, frac_mask);
    yfrac_ = vandq_u16(vshrq_n_u16(frac, REMAP16POINT5_FRAC_BITS), frac_mask);
    nxfrac_ = vsubq_u16(frac_max, xfrac_);
    nyfrac_ = vsubq_u16(frac_max, yfrac_);

    uint16x8_t one = vdupq_n_u16(1);
    x0_ = xy.val[0];
    y0_ = xy.val[1];
    x1_ = vaddq_u16(x0_, one);
    y1_ = vaddq_u16(y0_, one);
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      prepare_maps(mapxy, mapfrac);
      transform_pixels(dst);

      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };
    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

  void transform_pixels(Columns<uint16_t> dst) {
    uint16x8_t a = load_pixels(x0_, y0_);
    uint16x8_t b = load_pixels(x1_, y0_);
    uint16x8_t c = load_pixels(x0_, y1_);
    uint16x8_t d = load_pixels(x1_, y1_);

    uint16x8_t result =
        interpolate(a, b, c, d, xfrac_, yfrac_, nxfrac_, nyfrac_);

    vst1q_u16(&dst[0], result);
  }

  uint16x8_t load_pixels(uint16x8_t x, uint16x8_t y) {
    // Find whether coordinates are within the image dimensions.
    // Negative coordinates are interpreted as large values due to the s16->u16
    // reinterpretation.
    uint16x8_t in_range =
        vandq_u16(vcltq_u16(vreinterpretq_u16_s16(x), v_width_),
                  vcltq_u16(vreinterpretq_u16_s16(y), v_height_));

    // Zero out-of-range coordinates.
    x = vandq_u16(in_range, x);
    y = vandq_u16(in_range, y);

    // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
    uint32x4_t indices_low =
        vmlal_u16(vmovl_u16(vget_low_u16(x)), vget_low_u16(y),
                  vget_low_u16(v_src_element_stride_));
    uint32x4_t indices_high =
        vmlal_high_u16(vmovl_high_u16(x), y, v_src_element_stride_);

    // Read pixels from source
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
    return vbslq_u16(in_range, pixels, v_border_);
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x8_t v_src_element_stride_;
  uint16x8_t v_width_;
  uint16x8_t v_height_;
  uint16x8_t v_border_;
  uint16x8_t xfrac_;
  uint16x8_t yfrac_;
  uint16x8_t nxfrac_;
  uint16x8_t nyfrac_;
  int16x8_t x0_;
  int16x8_t x1_;
  int16x8_t y0_;
  int16x8_t y1_;
};  // end of class RemapS16Point5ConstantBorder<uint16_t>

inline void get_coordinates(Columns<const int16_t> mapxy,
                            Columns<const uint16_t> mapfrac, uint16x8_t &x,
                            uint16x8_t &y, uint16x8_t &xfrac,
                            uint16x8_t &yfrac) {
  int16x8x2_t xy = vld2q_s16(&mapxy[0]);
  x = xy.val[0];
  y = xy.val[1];

  uint16x8_t frac = vld1q_u16(&mapfrac[0]);
  xfrac = vandq_u16(frac, vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1));
  yfrac = vandq_u16(vshrq_n_u16(frac, REMAP16POINT5_FRAC_BITS),
                    vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1));
}

inline void get_offsets_4ch(uint16x4_t x0, uint16x4_t y0, uint16x4_t x1,
                            uint16x4_t y1, uint32x4_t &offsets_a,
                            uint32x4_t &offsets_b, uint32x4_t &offsets_c,
                            uint32x4_t &offsets_d,
                            uint16x4_t v_src_element_stride) {
  // Multiply by 4 because of channels
  uint32x4_t x0_scaled = vshll_n_u16(x0, 2);
  uint32x4_t x1_scaled = vshll_n_u16(x1, 2);

  // Calculate offsets from coordinates (y * element_stride + x)
  // a: top left, b: top right, c: bottom left, d: bottom right
  offsets_a = vmlal_u16(x0_scaled, y0, v_src_element_stride);
  offsets_b = vmlal_u16(x1_scaled, y0, v_src_element_stride);
  offsets_c = vmlal_u16(x0_scaled, y1, v_src_element_stride);
  offsets_d = vmlal_u16(x1_scaled, y1, v_src_element_stride);
}

inline uint16x8_t create_frac_low_high_u8_4ch(uint8_t frac_low,
                                              uint8_t frac_high) {
  uint8x8_t frac_low_high = {frac_low,  frac_low,  frac_low,  frac_low,
                             frac_high, frac_high, frac_high, frac_high};
  return vmovl_u8(frac_low_high);
}

inline uint64_t load_32bit(const uint8_t *src) {
  uint32_t value = 0;
  memcpy(&value, src, sizeof(uint32_t));
  return static_cast<uint64_t>(value);
}

inline uint8x16_t load_4px_4ch(Rows<const uint8_t> src_rows,
                               uint32x4_t offsets) {
  uint64_t pixels01 = load_32bit(&src_rows[vgetq_lane_u32(offsets, 0)]) |
                      (load_32bit(&src_rows[vgetq_lane_u32(offsets, 1)]) << 32);
  uint64_t pixels23 = load_32bit(&src_rows[vgetq_lane_u32(offsets, 2)]) |
                      (load_32bit(&src_rows[vgetq_lane_u32(offsets, 3)]) << 32);
  return vcombine(vcreate_u8(pixels01), vcreate_u8(pixels23));
}

inline void store_pixels_u8_4ch(uint8x16x2_t res, Columns<uint8_t> dst) {
  using ScalarType = uint8_t;
  neon::VecTraits<ScalarType>::store(res, &dst[0]);
}

inline uint16x8_t load_2px_4ch(Rows<const uint16_t> src_rows,
                               uint32x2_t offsets) {
  return vcombine(vld1_u16(&src_rows[vget_lane_u32(offsets, 0)]),
                  vld1_u16(&src_rows[vget_lane_u32(offsets, 1)]));
}

inline void store_pixels_u16_4ch(uint16x8x4_t res, Columns<uint16_t> dst) {
  using ScalarType = uint16_t;
  neon::VecTraits<ScalarType>::store(res, &dst[0]);
}

// Replicate border specific functions
inline void get_coordinates_replicate(Columns<const int16_t> mapxy,
                                      Columns<const uint16_t> mapfrac,
                                      uint16x8_t &x0, uint16x8_t &y0,
                                      uint16x8_t &x1, uint16x8_t &y1,
                                      uint16x8_t &xfrac, uint16x8_t &yfrac,
                                      int16x8_t v_xmax, int16x8_t v_ymax) {
  get_coordinates(mapxy, mapfrac, x0, y0, xfrac, yfrac);

  // Zero the xfrac (or yfrac) if x (or y) are below zero
  xfrac = vbslq_u16(vcltq_s16(x0, vdupq_n_s16(0)), vdupq_n_u16(0), xfrac);
  yfrac = vbslq_u16(vcltq_s16(y0, vdupq_n_s16(0)), vdupq_n_u16(0), yfrac);

  // Clamp coordinates to within the dimensions of the source image
  x0 = vreinterpretq_u16_s16(vmaxq_s16(vdupq_n_s16(0), vminq_s16(x0, v_xmax)));
  y0 = vreinterpretq_u16_s16(vmaxq_s16(vdupq_n_s16(0), vminq_s16(y0, v_ymax)));

  // x1 = x0 + 1, except if it's already xmax
  x1 = vsubq_u16(x0, vcltq_s16(x0, v_xmax));
  y1 = vsubq_u16(y0, vcltq_s16(y0, v_ymax));
}

inline void load_pixels_u8_4ch_replicate(
    Rows<const uint8_t> src_rows, uint32x4_t offsets_a, uint32x4_t offsets_b,
    uint32x4_t offsets_c, uint32x4_t offsets_d, uint8x16_t &a, uint8x16_t &b,
    uint8x16_t &c, uint8x16_t &d) {
  a = load_4px_4ch(src_rows, offsets_a);
  b = load_4px_4ch(src_rows, offsets_b);
  c = load_4px_4ch(src_rows, offsets_c);
  d = load_4px_4ch(src_rows, offsets_d);
}

inline void load_pixels_u16_4ch_replicate(
    Rows<const uint16_t> src_rows, uint32x4_t offsets_a, uint32x4_t offsets_b,
    uint32x4_t offsets_c, uint32x4_t offsets_d, uint16x8_t &a_lo,
    uint16x8_t &a_hi, uint16x8_t &b_lo, uint16x8_t &b_hi, uint16x8_t &c_lo,
    uint16x8_t &c_hi, uint16x8_t &d_lo, uint16x8_t &d_hi) {
  a_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_a));
  b_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_b));
  c_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_c));
  d_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_d));

  a_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_a));
  b_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_b));
  c_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_c));
  d_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_d));
}

template <typename ScalarType>
class RemapS16Point5Replicate4ch;

template <>
class RemapS16Point5Replicate4ch<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5Replicate4ch(Rows<const ScalarType> src_rows, size_t src_width,
                             size_t src_height)
      : src_rows_{src_rows},
        v_src_stride_{vdup_n_u16(static_cast<uint16_t>(src_rows_.stride()))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      uint16x8_t x0, y0, x1, y1;
      uint16x8_t xfrac, yfrac;
      get_coordinates_replicate(mapxy, mapfrac, x0, y0, x1, y1, xfrac, yfrac,
                                v_xmax_, v_ymax_);

      uint32x4_t offsets_a, offsets_b, offsets_c, offsets_d;
      uint8x16_t a, b, c, d;
      uint8x16x2_t res;

      get_offsets_4ch(vget_low_u16(x0), vget_low_u16(y0), vget_low_u16(x1),
                      vget_low_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_stride_);
      load_pixels_u8_4ch_replicate(src_rows_, offsets_a, offsets_b, offsets_c,
                                   offsets_d, a, b, c, d);

      // Doubled fractions 001122..., low part
      uint16x8_t xfrac2 = vzip1q(xfrac, xfrac);
      uint16x8_t yfrac2 = vzip1q(yfrac, yfrac);
      uint16x8_t nxfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      uint16x8_t nyfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      uint16x8_t res0 = interpolate(
          vmovl_u8(vget_low(a)), vmovl_u8(vget_low(b)), vmovl_u8(vget_low(c)),
          vmovl_u8(vget_low(d)), vzip1q(xfrac2, xfrac2), vzip1q(yfrac2, yfrac2),
          vzip1q(nxfrac2, nxfrac2), vzip1q(nyfrac2, nyfrac2));
      uint16x8_t res1 = interpolate(
          vmovl_high_u8(a), vmovl_high_u8(b), vmovl_high_u8(c),
          vmovl_high_u8(d), vzip2q(xfrac2, xfrac2), vzip2q(yfrac2, yfrac2),
          vzip2q(nxfrac2, nxfrac2), vzip2q(nyfrac2, nyfrac2));
      res.val[0] =
          vuzp1q_u8(vreinterpretq_u8_u16(res0), vreinterpretq_u8_u16(res1));

      get_offsets_4ch(vget_high_u16(x0), vget_high_u16(y0), vget_high_u16(x1),
                      vget_high_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_stride_);
      load_pixels_u8_4ch_replicate(src_rows_, offsets_a, offsets_b, offsets_c,
                                   offsets_d, a, b, c, d);
      // Doubled fractions 001122..., high part
      xfrac2 = vzip2q(xfrac, xfrac);
      yfrac2 = vzip2q(yfrac, yfrac);
      nxfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      nyfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      res0 = interpolate(vmovl_u8(vget_low(a)), vmovl_u8(vget_low(b)),
                         vmovl_u8(vget_low(c)), vmovl_u8(vget_low(d)),
                         vzip1q(xfrac2, xfrac2), vzip1q(yfrac2, yfrac2),
                         vzip1q(nxfrac2, nxfrac2), vzip1q(nyfrac2, nyfrac2));
      res1 = interpolate(vmovl_high_u8(a), vmovl_high_u8(b), vmovl_high_u8(c),
                         vmovl_high_u8(d), vzip2q(xfrac2, xfrac2),
                         vzip2q(yfrac2, yfrac2), vzip2q(nxfrac2, nxfrac2),
                         vzip2q(nyfrac2, nyfrac2));
      res.val[1] =
          vuzp1q_u8(vreinterpretq_u8_u16(res0), vreinterpretq_u8_u16(res1));

      store_pixels_u8_4ch(res, dst);
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x4_t v_src_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
};  // end of class RemapS16Point5Replicate4ch<uint8_t>

template <>
class RemapS16Point5Replicate4ch<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5Replicate4ch(Rows<const ScalarType> src_rows, size_t src_width,
                             size_t src_height)
      : src_rows_{src_rows},
        v_src_element_stride_{vdup_n_u16(
            static_cast<uint16_t>(src_rows_.stride() / sizeof(ScalarType)))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      uint16x8_t x0, y0, x1, y1;
      uint16x8_t xfrac, yfrac;
      get_coordinates_replicate(mapxy, mapfrac, x0, y0, x1, y1, xfrac, yfrac,
                                v_xmax_, v_ymax_);

      uint32x4_t offsets_a, offsets_b, offsets_c, offsets_d;
      uint16x8_t a_low, a_high, b_low, b_high, c_low, c_high, d_low, d_high;
      uint16x8x4_t res;
      get_offsets_4ch(vget_low_u16(x0), vget_low_u16(y0), vget_low_u16(x1),
                      vget_low_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_element_stride_);
      load_pixels_u16_4ch_replicate(src_rows_, offsets_a, offsets_b, offsets_c,
                                    offsets_d, a_low, a_high, b_low, b_high,
                                    c_low, c_high, d_low, d_high);

      // Doubled fractions 001122..., low part
      uint16x8_t xfrac2 = vzip1q(xfrac, xfrac);
      uint16x8_t yfrac2 = vzip1q(yfrac, yfrac);
      uint16x8_t nxfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      uint16x8_t nyfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      res.val[0] =
          interpolate(a_low, b_low, c_low, d_low, vzip1q(xfrac2, xfrac2),
                      vzip1q(yfrac2, yfrac2), vzip1q(nxfrac2, nxfrac2),
                      vzip1q(nyfrac2, nyfrac2));
      res.val[1] =
          interpolate(a_high, b_high, c_high, d_high, vzip2q(xfrac2, xfrac2),
                      vzip2q(yfrac2, yfrac2), vzip2q(nxfrac2, nxfrac2),
                      vzip2q(nyfrac2, nyfrac2));

      get_offsets_4ch(vget_high_u16(x0), vget_high_u16(y0), vget_high_u16(x1),
                      vget_high_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_element_stride_);
      load_pixels_u16_4ch_replicate(src_rows_, offsets_a, offsets_b, offsets_c,
                                    offsets_d, a_low, a_high, b_low, b_high,
                                    c_low, c_high, d_low, d_high);
      // Doubled fractions 001122..., high part
      xfrac2 = vzip2q(xfrac, xfrac);
      yfrac2 = vzip2q(yfrac, yfrac);
      nxfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      nyfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      res.val[2] =
          interpolate(a_low, b_low, c_low, d_low, vzip1q(xfrac2, xfrac2),
                      vzip1q(yfrac2, yfrac2), vzip1q(nxfrac2, nxfrac2),
                      vzip1q(nyfrac2, nyfrac2));
      res.val[3] =
          interpolate(a_high, b_high, c_high, d_high, vzip2q(xfrac2, xfrac2),
                      vzip2q(yfrac2, yfrac2), vzip2q(nxfrac2, nxfrac2),
                      vzip2q(nyfrac2, nyfrac2));

      store_pixels_u16_4ch(res, dst);
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x4_t v_src_element_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
};  // end of class RemapS16Point5Replicate4ch<uint16_t>

// Constant border specific functions
inline void get_coordinates_constant(
    Columns<const int16_t> mapxy, Columns<const uint16_t> mapfrac,
    uint16x8_t v_width, uint16x8_t v_height, uint16x8_t &x0, uint16x8_t &y0,
    uint16x8_t &x1, uint16x8_t &y1, uint16x8_t &xfrac, uint16x8_t &yfrac,
    uint16x8_t &in_range_a, uint16x8_t &in_range_b, uint16x8_t &in_range_c,
    uint16x8_t &in_range_d) {
  get_coordinates(mapxy, mapfrac, x0, y0, xfrac, yfrac);

  uint16x8_t one = vdupq_n_u16(1);
  x1 = vaddq_u16(x0, one);
  y1 = vaddq_u16(y0, one);

  uint16x8_t x0_in_range = vcltq_u16(x0, v_width);
  uint16x8_t y0_in_range = vcltq_u16(y0, v_height);
  uint16x8_t x1_in_range = vcltq_u16(x1, v_width);
  uint16x8_t y1_in_range = vcltq_u16(y1, v_height);

  in_range_a = vandq(x0_in_range, y0_in_range);
  in_range_b = vandq(x1_in_range, y0_in_range);
  in_range_c = vandq(x0_in_range, y1_in_range);
  in_range_d = vandq(x1_in_range, y1_in_range);
}

inline uint32x4_t zero_out_of_range_offsets(uint32x4_t in_range,
                                            uint32x4_t offsets) {
  return vbslq_u32(in_range, offsets, vdupq_n_u32(0));
}

inline uint8x16_t replace_pixel_with_border_u8_4ch(uint32x4_t in_range,
                                                   uint8x16_t pixels,
                                                   uint8x16_t v_border) {
  return vreinterpretq_u8_u32(
      vbslq_u32(in_range, vreinterpretq_u32_u8(pixels), v_border));
}

inline uint16x8_t replace_pixel_with_border_u16_4ch(uint64x2_t in_range,
                                                    uint16x8_t pixels,
                                                    uint16x8_t v_border) {
  return vreinterpretq_u16_u64(
      vbslq_u64(in_range, vreinterpretq_u64_u16(pixels), v_border));
}

inline void load_pixels_u8_4ch_constant(
    Rows<const uint8_t> src_rows, uint32x4_t offsets_a, uint32x4_t offsets_b,
    uint32x4_t offsets_c, uint32x4_t offsets_d, uint32x4_t in_range_a,
    uint32x4_t in_range_b, uint32x4_t in_range_c, uint32x4_t in_range_d,
    uint8x16_t v_border, uint8x16_t &a, uint8x16_t &b, uint8x16_t &c,
    uint8x16_t &d) {
  offsets_a = zero_out_of_range_offsets(in_range_a, offsets_a);
  offsets_b = zero_out_of_range_offsets(in_range_b, offsets_b);
  offsets_c = zero_out_of_range_offsets(in_range_c, offsets_c);
  offsets_d = zero_out_of_range_offsets(in_range_d, offsets_d);

  a = load_4px_4ch(src_rows, offsets_a);
  b = load_4px_4ch(src_rows, offsets_b);
  c = load_4px_4ch(src_rows, offsets_c);
  d = load_4px_4ch(src_rows, offsets_d);

  a = replace_pixel_with_border_u8_4ch(in_range_a, a, v_border);
  b = replace_pixel_with_border_u8_4ch(in_range_b, b, v_border);
  c = replace_pixel_with_border_u8_4ch(in_range_c, c, v_border);
  d = replace_pixel_with_border_u8_4ch(in_range_d, d, v_border);
}

inline void load_pixels_u16_4ch_constant(
    Rows<const uint16_t> src_rows, uint32x4_t offsets_a, uint32x4_t offsets_b,
    uint32x4_t offsets_c, uint32x4_t offsets_d, uint32x4_t in_range_a,
    uint32x4_t in_range_b, uint32x4_t in_range_c, uint32x4_t in_range_d,
    uint16x8_t v_border, uint16x8_t &a_lo, uint16x8_t &a_hi, uint16x8_t &b_lo,
    uint16x8_t &b_hi, uint16x8_t &c_lo, uint16x8_t &c_hi, uint16x8_t &d_lo,
    uint16x8_t &d_hi) {
  offsets_a = zero_out_of_range_offsets(in_range_a, offsets_a);
  offsets_b = zero_out_of_range_offsets(in_range_b, offsets_b);
  offsets_c = zero_out_of_range_offsets(in_range_c, offsets_c);
  offsets_d = zero_out_of_range_offsets(in_range_d, offsets_d);

  a_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_a));
  b_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_b));
  c_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_c));
  d_lo = load_2px_4ch(src_rows, vget_low_u32(offsets_d));

  // Convert bitsets such as in_range to 64bits, making all 1s or all 0s
  auto low32_to_u64 = [](uint32x4_t bitset) {
    return vreinterpretq_u64_s64(
        vmovl_s32(vreinterpret_s32_u32(vget_low_u32(bitset))));
  };

  a_lo = replace_pixel_with_border_u16_4ch(low32_to_u64(in_range_a), a_lo,
                                           v_border);
  b_lo = replace_pixel_with_border_u16_4ch(low32_to_u64(in_range_b), b_lo,
                                           v_border);
  c_lo = replace_pixel_with_border_u16_4ch(low32_to_u64(in_range_c), c_lo,
                                           v_border);
  d_lo = replace_pixel_with_border_u16_4ch(low32_to_u64(in_range_d), d_lo,
                                           v_border);

  a_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_a));
  b_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_b));
  c_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_c));
  d_hi = load_2px_4ch(src_rows, vget_high_u32(offsets_d));

  // Convert bitsets such as in_range to 64bits, making all 1s or all 0s
  auto hi32_to_u64 = [](uint32x4_t bitset) {
    return vreinterpretq_u64_s64(vmovl_high_s32(vreinterpretq_s32_u32(bitset)));
  };

  a_hi = replace_pixel_with_border_u16_4ch(hi32_to_u64(in_range_a), a_hi,
                                           v_border);
  b_hi = replace_pixel_with_border_u16_4ch(hi32_to_u64(in_range_b), b_hi,
                                           v_border);
  c_hi = replace_pixel_with_border_u16_4ch(hi32_to_u64(in_range_c), c_hi,
                                           v_border);
  d_hi = replace_pixel_with_border_u16_4ch(hi32_to_u64(in_range_d), d_hi,
                                           v_border);
}

// Convert bitsets such as in_range to 32bits, making all 1s or all 0s
static uint32x4_t low16_to_s32(uint16x8_t bitset) {
  return vreinterpretq_u32_s32(
      vmovl_s16(vreinterpret_s16_u16(vget_low_u16(bitset))));
}

static uint32x4_t hi16_to_s32(uint16x8_t bitset) {
  return vreinterpretq_u32_s32(vmovl_high_s16(vreinterpretq_s16_u16(bitset)));
}

template <typename ScalarType>
class RemapS16Point5Constant4ch;

template <>
class RemapS16Point5Constant4ch<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5Constant4ch(Rows<const ScalarType> src_rows, size_t src_width,
                            size_t src_height, const ScalarType *border_value)
      : src_rows_{src_rows},
        v_src_stride_{vdup_n_u16(static_cast<uint16_t>(src_rows_.stride()))},
        v_width_{vdupq_n_u16(static_cast<uint16_t>(src_width))},
        v_height_{vdupq_n_u16(static_cast<uint16_t>(src_height))},
        v_border_{} {
    uint32_t border_value_32{};
    memcpy(&border_value_32, border_value, sizeof(uint32_t));
    v_border_ = vreinterpretq_u8_u32(vdupq_n_u32(border_value_32));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      uint16x8_t x0, y0, x1, y1;
      uint16x8_t xfrac, yfrac;
      uint16x8_t in_range_a, in_range_b, in_range_c, in_range_d;
      get_coordinates_constant(mapxy, mapfrac, v_width_, v_height_, x0, y0, x1,
                               y1, xfrac, yfrac, in_range_a, in_range_b,
                               in_range_c, in_range_d);

      uint32x4_t offsets_a, offsets_b, offsets_c, offsets_d;
      uint8x16_t a, b, c, d;
      uint8x16x2_t res;

      get_offsets_4ch(vget_low_u16(x0), vget_low_u16(y0), vget_low_u16(x1),
                      vget_low_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_stride_);

      load_pixels_u8_4ch_constant(
          src_rows_, offsets_a, offsets_b, offsets_c, offsets_d,
          low16_to_s32(in_range_a), low16_to_s32(in_range_b),
          low16_to_s32(in_range_c), low16_to_s32(in_range_d), v_border_, a, b,
          c, d);

      // Doubled fractions 001122..., low part
      uint16x8_t xfrac2 = vzip1q(xfrac, xfrac);
      uint16x8_t yfrac2 = vzip1q(yfrac, yfrac);
      uint16x8_t nxfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      uint16x8_t nyfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      uint16x8_t res0 = interpolate(
          vmovl_u8(vget_low(a)), vmovl_u8(vget_low(b)), vmovl_u8(vget_low(c)),
          vmovl_u8(vget_low(d)), vzip1q(xfrac2, xfrac2), vzip1q(yfrac2, yfrac2),
          vzip1q(nxfrac2, nxfrac2), vzip1q(nyfrac2, nyfrac2));
      uint16x8_t res1 = interpolate(
          vmovl_high_u8(a), vmovl_high_u8(b), vmovl_high_u8(c),
          vmovl_high_u8(d), vzip2q(xfrac2, xfrac2), vzip2q(yfrac2, yfrac2),
          vzip2q(nxfrac2, nxfrac2), vzip2q(nyfrac2, nyfrac2));
      res.val[0] =
          vuzp1q_u8(vreinterpretq_u8_u16(res0), vreinterpretq_u8_u16(res1));

      get_offsets_4ch(vget_high_u16(x0), vget_high_u16(y0), vget_high_u16(x1),
                      vget_high_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_stride_);

      load_pixels_u8_4ch_constant(
          src_rows_, offsets_a, offsets_b, offsets_c, offsets_d,
          hi16_to_s32(in_range_a), hi16_to_s32(in_range_b),
          hi16_to_s32(in_range_c), hi16_to_s32(in_range_d), v_border_, a, b, c,
          d);
      // Doubled fractions 001122..., high part
      xfrac2 = vzip2q(xfrac, xfrac);
      yfrac2 = vzip2q(yfrac, yfrac);
      nxfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      nyfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      res0 = interpolate(vmovl_u8(vget_low(a)), vmovl_u8(vget_low(b)),
                         vmovl_u8(vget_low(c)), vmovl_u8(vget_low(d)),
                         vzip1q(xfrac2, xfrac2), vzip1q(yfrac2, yfrac2),
                         vzip1q(nxfrac2, nxfrac2), vzip1q(nyfrac2, nyfrac2));
      res1 = interpolate(vmovl_high_u8(a), vmovl_high_u8(b), vmovl_high_u8(c),
                         vmovl_high_u8(d), vzip2q(xfrac2, xfrac2),
                         vzip2q(yfrac2, yfrac2), vzip2q(nxfrac2, nxfrac2),
                         vzip2q(nyfrac2, nyfrac2));
      res.val[1] =
          vuzp1q_u8(vreinterpretq_u8_u16(res0), vreinterpretq_u8_u16(res1));

      store_pixels_u8_4ch(res, dst);
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x4_t v_src_stride_;
  uint16x8_t v_width_;
  uint16x8_t v_height_;
  uint8x16_t v_border_;
};  // end of class RemapS16Point5Constant4ch<uint8_t>

template <>
class RemapS16Point5Constant4ch<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using MapVecTraits = neon::VecTraits<int16_t>;

  RemapS16Point5Constant4ch(Rows<const ScalarType> src_rows, size_t src_width,
                            size_t src_height, const ScalarType *border_value)
      : src_rows_{src_rows},
        v_src_element_stride_{vdup_n_u16(
            static_cast<uint16_t>(src_rows_.stride() / sizeof(ScalarType)))},
        v_width_{vdupq_n_u16(static_cast<uint16_t>(src_width))},
        v_height_{vdupq_n_u16(static_cast<uint16_t>(src_height))},
        v_border_{} {
    uint64_t border_value_64{};
    memcpy(&border_value_64, border_value, sizeof(uint64_t));
    v_border_ = vreinterpretq_u16_u64(vdupq_n_u64(border_value_64));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      uint16x8_t x0, y0, x1, y1;
      uint16x8_t xfrac, yfrac;
      uint16x8_t in_range_a, in_range_b, in_range_c, in_range_d;
      get_coordinates_constant(mapxy, mapfrac, v_width_, v_height_, x0, y0, x1,
                               y1, xfrac, yfrac, in_range_a, in_range_b,
                               in_range_c, in_range_d);

      uint32x4_t offsets_a, offsets_b, offsets_c, offsets_d;
      uint16x8_t a_low, a_high, b_low, b_high, c_low, c_high, d_low, d_high;
      uint16x8x4_t res;

      get_offsets_4ch(vget_low_u16(x0), vget_low_u16(y0), vget_low_u16(x1),
                      vget_low_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_element_stride_);

      load_pixels_u16_4ch_constant(
          src_rows_, offsets_a, offsets_b, offsets_c, offsets_d,
          low16_to_s32(in_range_a), low16_to_s32(in_range_b),
          low16_to_s32(in_range_c), low16_to_s32(in_range_d), v_border_, a_low,
          a_high, b_low, b_high, c_low, c_high, d_low, d_high);

      // Doubled fractions 001122..., low part
      uint16x8_t xfrac2 = vzip1q(xfrac, xfrac);
      uint16x8_t yfrac2 = vzip1q(yfrac, yfrac);
      uint16x8_t nxfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      uint16x8_t nyfrac2 =
          vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      res.val[0] =
          interpolate(a_low, b_low, c_low, d_low, vzip1q(xfrac2, xfrac2),
                      vzip1q(yfrac2, yfrac2), vzip1q(nxfrac2, nxfrac2),
                      vzip1q(nyfrac2, nyfrac2));
      res.val[1] =
          interpolate(a_high, b_high, c_high, d_high, vzip2q(xfrac2, xfrac2),
                      vzip2q(yfrac2, yfrac2), vzip2q(nxfrac2, nxfrac2),
                      vzip2q(nyfrac2, nyfrac2));

      get_offsets_4ch(vget_high_u16(x0), vget_high_u16(y0), vget_high_u16(x1),
                      vget_high_u16(y1), offsets_a, offsets_b, offsets_c,
                      offsets_d, v_src_element_stride_);

      load_pixels_u16_4ch_constant(
          src_rows_, offsets_a, offsets_b, offsets_c, offsets_d,
          hi16_to_s32(in_range_a), hi16_to_s32(in_range_b),
          hi16_to_s32(in_range_c), hi16_to_s32(in_range_d), v_border_, a_low,
          a_high, b_low, b_high, c_low, c_high, d_low, d_high);

      // Doubled fractions 001122..., high part
      xfrac2 = vzip2q(xfrac, xfrac);
      yfrac2 = vzip2q(yfrac, yfrac);
      nxfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2);
      nyfrac2 = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2);
      // Quadrupled fractions (00001111) are passed to interpolate
      res.val[2] =
          interpolate(a_low, b_low, c_low, d_low, vzip1q(xfrac2, xfrac2),
                      vzip1q(yfrac2, yfrac2), vzip1q(nxfrac2, nxfrac2),
                      vzip1q(nyfrac2, nyfrac2));
      res.val[3] =
          interpolate(a_high, b_high, c_high, d_high, vzip2q(xfrac2, xfrac2),
                      vzip2q(yfrac2, yfrac2), vzip2q(nxfrac2, nxfrac2),
                      vzip2q(nyfrac2, nyfrac2));

      store_pixels_u16_4ch(res, dst);
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  uint16x4_t v_src_element_stride_;
  uint16x8_t v_width_;
  uint16x8_t v_height_;
  uint16x8_t v_border_;
};  // end of class RemapS16Point5Constant4ch<uint16_t>

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t remap_s16point5(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    const uint16_t *mapfrac, size_t mapfrac_stride,
    [[maybe_unused]] kleidicv_border_type_t border_type,
    [[maybe_unused]] const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapxy, mapxy_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapfrac, mapfrac_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (!remap_s16point5_is_implemented<T>(src_stride, src_width, src_height,
                                         dst_width, border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<const int16_t> mapxy_rows{mapxy, mapxy_stride, 2};
  Rows<const uint16_t> mapfrac_rows{mapfrac, mapfrac_stride, 1};
  Rows<T> dst_rows{dst, dst_stride, channels};
  Rectangle rect{dst_width, dst_height};
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT) {
    if (channels == 1) {
      RemapS16Point5ConstantBorder<T> operation{src_rows, src_width, src_height,
                                                border_value};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    } else {
      assert(channels == 4);
      RemapS16Point5Constant4ch<T> operation{src_rows, src_width, src_height,
                                             border_value};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    }
  } else {
    assert(border_type == KLEIDICV_BORDER_TYPE_REPLICATE);
    if (channels == 1) {
      RemapS16Point5Replicate<T> operation{src_rows, src_width, src_height};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    } else {
      assert(channels == 4);
      RemapS16Point5Replicate4ch<T> operation{src_rows, src_width, src_height};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    }
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(type)                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16point5<type>(    \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      const uint16_t *mapfrac, size_t mapfrac_stride,                          \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint16_t);

}  // namespace kleidicv::neon
