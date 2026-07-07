// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kleidicv/sve2.h"
#include "kleidicv/traits.h"
#include "kleidicv/transform/remap.h"
#include "transform_sve2.h"

namespace kleidicv::sve2 {

template <typename ScalarType>
inline svuint16_t interpolate_16point5(svbool_t pg, svuint16_t frac,
                                       svuint16_t src_a, svuint16_t src_b,
                                       svuint16_t src_c, svuint16_t src_d,
                                       svuint32_t bias);

template <>
inline svuint16_t interpolate_16point5<uint8_t>(
    svbool_t pg, svuint16_t frac, svuint16_t src_a, svuint16_t src_b,
    svuint16_t src_c, svuint16_t src_d, svuint32_t bias) {
  svuint16_t xfrac = svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
  svuint16_t yfrac =
      svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
              svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
  svuint16_t nxfrac =
      svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
  svuint16_t nyfrac =
      svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);
  svuint16_t line0 = svmla_x(pg, svmul_x(pg, xfrac, src_b), nxfrac, src_a);
  svuint16_t line1 = svmla_x(pg, svmul_x(pg, xfrac, src_d), nxfrac, src_c);

  svuint32_t acc_b = svmlalb_u32(bias, line0, nyfrac);
  svuint32_t acc_t = svmlalt_u32(bias, line0, nyfrac);
  acc_b = svmlalb_u32(acc_b, line1, yfrac);
  acc_t = svmlalt_u32(acc_t, line1, yfrac);

  return svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS), acc_t,
                 2ULL * REMAP16POINT5_FRAC_BITS);
}

template <>
inline svuint16_t interpolate_16point5<uint16_t>(
    svbool_t pg, svuint16_t frac, svuint16_t src_a, svuint16_t src_b,
    svuint16_t src_c, svuint16_t src_d, svuint32_t bias) {
  svuint16_t xfrac = svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
  svuint16_t yfrac =
      svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
              svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
  svuint16_t nxfrac =
      svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
  svuint16_t nyfrac =
      svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);
  svuint32_t line0_b = svmla_x(pg, svmullb(xfrac, src_b), svmovlb_u32(nxfrac),
                               svmovlb_u32(src_a));
  svuint32_t line0_t = svmla_x(pg, svmullt(xfrac, src_b), svmovlt_u32(nxfrac),
                               svmovlt_u32(src_a));
  svuint32_t line1_b = svmla_x(pg, svmullb(xfrac, src_d), svmovlb_u32(nxfrac),
                               svmovlb_u32(src_c));
  svuint32_t line1_t = svmla_x(pg, svmullt(xfrac, src_d), svmovlt_u32(nxfrac),
                               svmovlt_u32(src_c));

  svuint32_t acc_b =
      svmla_u32_x(pg, svmla_u32_x(pg, bias, line0_b, svmovlb_u32(nyfrac)),
                  line1_b, svmovlb_u32(yfrac));
  svuint32_t acc_t =
      svmla_u32_x(pg, svmla_u32_x(pg, bias, line0_t, svmovlt_u32(nyfrac)),
                  line1_t, svmovlt_u32(yfrac));

  return svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS), acc_t,
                 2ULL * REMAP16POINT5_FRAC_BITS);
}

template <typename ScalarType>
class RemapS16Point5Replicate;

template <>
class RemapS16Point5Replicate<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5Replicate(Rows<const ScalarType> src_rows, size_t src_width,
                          size_t src_height, svuint16_t& v_src_stride,
                          MapVectorType& v_x_max, MapVectorType& v_y_max)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_stride_ = svdup_u16(src_rows.stride());
    v_xmax_ = svdup_s16(static_cast<int16_t>(src_width - 1));
    v_ymax_ = svdup_s16(static_cast<int16_t>(src_height - 1));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    svuint16_t src_a, src_b, src_c, src_d;

    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);
    auto vector_path = [&](svbool_t pg, ptrdiff_t step) {
      load_source(pg, step, mapxy, src_a, src_b, src_c, src_d);
      interpolate_and_store(pg, step, mapfrac, dst, src_a, src_b, src_c, src_d,
                            bias);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      svbool_t pg = MapVecTraits::svptrue();
      vector_path(pg, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      vector_path(pg, static_cast<ptrdiff_t>(length));
    });
  }

 protected:
  svuint16_t gather_load_src(svbool_t pg_b, svuint32_t offsets_b, svbool_t pg_t,
                             svuint32_t offsets_t) {
    svuint32_t src_b =
        svldnt1ub_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
    svuint32_t src_t =
        svldnt1ub_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);
    return svtrn1_u16(svreinterpret_u16_u32(src_b),
                      svreinterpret_u16_u32(src_t));
  }

  void load_source(svbool_t pg, ptrdiff_t step, Columns<const int16_t>& mapxy,
                   svuint16_t& src_a, svuint16_t& src_b, svuint16_t& src_c,
                   svuint16_t& src_d) {
    MapVector2Type xy = svld2_s16(pg, &mapxy[0]);

    // Clamp coordinates to within the dimensions of the source image
    svuint16_t x0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 0), v_xmax_)));
    svuint16_t y0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 1), v_ymax_)));

    // x1 = x0 + 1, and clamp it too
    svuint16_t x1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 0), 1), v_xmax_)));

    svuint16_t y1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 1), 1), v_ymax_)));
    svbool_t pg_b = svwhilelt_b32(int64_t{0}, (step + 1) / 2);
    svbool_t pg_t = svwhilelt_b32(int64_t{0}, step / 2);

    // Calculate offsets from coordinates (y * stride + x)
    svuint32_t offsets_a_b = svmlalb_u32(svmovlb_u32(x0), y0, v_src_stride_);
    svuint32_t offsets_a_t = svmlalt_u32(svmovlt_u32(x0), y0, v_src_stride_);
    svuint32_t offsets_b_b = svmlalb_u32(svmovlb_u32(x1), y0, v_src_stride_);
    svuint32_t offsets_b_t = svmlalt_u32(svmovlt_u32(x1), y0, v_src_stride_);
    svuint32_t offsets_c_b = svmlalb_u32(svmovlb_u32(x0), y1, v_src_stride_);
    svuint32_t offsets_c_t = svmlalt_u32(svmovlt_u32(x0), y1, v_src_stride_);
    svuint32_t offsets_d_b = svmlalb_u32(svmovlb_u32(x1), y1, v_src_stride_);
    svuint32_t offsets_d_t = svmlalt_u32(svmovlt_u32(x1), y1, v_src_stride_);

    // Load pixels from source
    src_a = gather_load_src(pg_b, offsets_a_b, pg_t, offsets_a_t);
    src_b = gather_load_src(pg_b, offsets_b_b, pg_t, offsets_b_t);
    src_c = gather_load_src(pg_b, offsets_c_b, pg_t, offsets_c_t);
    src_d = gather_load_src(pg_b, offsets_d_b, pg_t, offsets_d_t);
    mapxy += step;
  }

  void interpolate_and_store(svbool_t pg, ptrdiff_t step,
                             Columns<const uint16_t>& mapfrac,
                             Columns<ScalarType>& dst, svuint16_t src_a,
                             svuint16_t src_b, svuint16_t src_c,
                             svuint16_t src_d, svuint32_t bias) {
    FracVectorType frac = svld1_u16(pg, &mapfrac[0]);
    svuint16_t result = interpolate_16point5<uint8_t>(pg, frac, src_a, src_b,
                                                      src_c, src_d, bias);
    svst1b_u16(pg, &dst[0], result);
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint16_t& v_src_stride_;
  MapVectorType& v_xmax_;
  MapVectorType& v_ymax_;
};  // end of class RemapS16Point5Replicate<uint8_t>

template <>
class RemapS16Point5Replicate<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5Replicate(Rows<const ScalarType> src_rows, size_t src_width,
                          size_t src_height, svuint16_t& v_src_stride,
                          MapVectorType& v_x_max, MapVectorType& v_y_max)
      : src_rows_{src_rows},
        v_src_element_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_element_stride_ = svdup_u16(src_rows.stride() / sizeof(ScalarType));
    v_xmax_ = svdup_s16(static_cast<int16_t>(src_width - 1));
    v_ymax_ = svdup_s16(static_cast<int16_t>(src_height - 1));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    svuint16_t src_a, src_b, src_c, src_d;

    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);
    auto vector_path = [&](svbool_t pg, ptrdiff_t step) {
      load_source(pg, step, mapxy, src_a, src_b, src_c, src_d);
      interpolate_and_store(pg, step, mapfrac, dst, src_a, src_b, src_c, src_d,
                            bias);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      svbool_t pg = MapVecTraits::svptrue();
      vector_path(pg, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      vector_path(pg, static_cast<ptrdiff_t>(length));
    });
  }

 protected:
  svuint16_t gather_load_src(svbool_t pg_b, svuint32_t offsets_b, svbool_t pg_t,
                             svuint32_t offsets_t) {
    // Account for the size of the source type when calculating offset
    offsets_b = svlsl_n_u32_x(pg_b, offsets_b, 1);
    offsets_t = svlsl_n_u32_x(pg_t, offsets_t, 1);

    svuint32_t src_b =
        svldnt1uh_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
    svuint32_t src_t =
        svldnt1uh_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);
    return svtrn1_u16(svreinterpret_u16_u32(src_b),
                      svreinterpret_u16_u32(src_t));
  }

  void load_source(svbool_t pg, ptrdiff_t step, Columns<const int16_t>& mapxy,
                   svuint16_t& src_a, svuint16_t& src_b, svuint16_t& src_c,
                   svuint16_t& src_d) {
    MapVector2Type xy = svld2_s16(pg, &mapxy[0]);

    // Clamp coordinates to within the dimensions of the source image
    svuint16_t x0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 0), v_xmax_)));
    svuint16_t y0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 1), v_ymax_)));

    // x1 = x0 + 1, and clamp it too
    svuint16_t x1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 0), 1), v_xmax_)));

    svuint16_t y1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 1), 1), v_ymax_)));
    svbool_t pg_b = svwhilelt_b32(int64_t{0}, (step + 1) / 2);
    svbool_t pg_t = svwhilelt_b32(int64_t{0}, step / 2);

    // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
    svuint32_t offsets_a_b =
        svmlalb_u32(svmovlb_u32(x0), y0, v_src_element_stride_);
    svuint32_t offsets_a_t =
        svmlalt_u32(svmovlt_u32(x0), y0, v_src_element_stride_);
    svuint32_t offsets_b_b =
        svmlalb_u32(svmovlb_u32(x1), y0, v_src_element_stride_);
    svuint32_t offsets_b_t =
        svmlalt_u32(svmovlt_u32(x1), y0, v_src_element_stride_);
    svuint32_t offsets_c_b =
        svmlalb_u32(svmovlb_u32(x0), y1, v_src_element_stride_);
    svuint32_t offsets_c_t =
        svmlalt_u32(svmovlt_u32(x0), y1, v_src_element_stride_);
    svuint32_t offsets_d_b =
        svmlalb_u32(svmovlb_u32(x1), y1, v_src_element_stride_);
    svuint32_t offsets_d_t =
        svmlalt_u32(svmovlt_u32(x1), y1, v_src_element_stride_);

    // Load pixels from source
    src_a = gather_load_src(pg_b, offsets_a_b, pg_t, offsets_a_t);
    src_b = gather_load_src(pg_b, offsets_b_b, pg_t, offsets_b_t);
    src_c = gather_load_src(pg_b, offsets_c_b, pg_t, offsets_c_t);
    src_d = gather_load_src(pg_b, offsets_d_b, pg_t, offsets_d_t);
    mapxy += step;
  }

  void interpolate_and_store(svbool_t pg, ptrdiff_t step,
                             Columns<const uint16_t>& mapfrac,
                             Columns<ScalarType>& dst, svuint16_t src_a,
                             svuint16_t src_b, svuint16_t src_c,
                             svuint16_t src_d, svuint32_t bias) {
    FracVectorType frac = svld1_u16(pg, &mapfrac[0]);
    svuint16_t result = interpolate_16point5<uint16_t>(pg, frac, src_a, src_b,
                                                       src_c, src_d, bias);
    svst1_u16(pg, &dst[0], result);
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint16_t& v_src_element_stride_;
  MapVectorType& v_xmax_;
  MapVectorType& v_ymax_;
};  // end of class RemapS16Point5Replicate<uint16_t>

template <typename ScalarType>
class RemapS16Point5ConstantBorder;

template <>
class RemapS16Point5ConstantBorder<uint8_t> {
 public:
  using ScalarType = uint8_t;

  RemapS16Point5ConstantBorder(Rows<const ScalarType> src_rows,
                               size_t src_width, size_t src_height,
                               const ScalarType* border_value,
                               svuint16_t& v_src_stride, svuint16_t& v_width,
                               svuint16_t& v_height, svuint16_t& v_border)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_width_{v_width},
        v_height_{v_height},
        v_border_{v_border} {
    v_src_stride_ = svdup_u16(src_rows.stride());
    v_width_ = svdup_u16(static_cast<uint16_t>(src_width));
    v_height_ = svdup_u16(static_cast<uint16_t>(src_height));
    v_border_ = svdup_u16(*border_value);
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    svuint16_t one = svdup_n_u16(1);
    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);
    for (size_t i = 0; i < width; i += svcnth()) {
      svbool_t pg = svwhilelt_b16_u64(i, width);

      svuint16x2_t xy =
          svld2_u16(pg, reinterpret_cast<const uint16_t*>(
                            &mapxy[static_cast<ptrdiff_t>(i * 2)]));

      svuint16_t x0 = svget2(xy, 0);
      svuint16_t y0 = svget2(xy, 1);
      svuint16_t x1 = svadd_x(pg, x0, one);
      svuint16_t y1 = svadd_x(pg, y0, one);

      svuint16_t v00 = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, pg, x0, y0);
      svuint16_t v01 = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, pg, x0, y1);
      svuint16_t v10 = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, pg, x1, y0);
      svuint16_t v11 = load_pixels_or_constant_border(
          src_rows_, v_src_stride_, v_width_, v_height_, v_border_, pg, x1, y1);

      svuint16_t frac = svld1_u16(pg, &mapfrac[static_cast<ptrdiff_t>(i)]);
      svuint16_t result =
          interpolate_16point5<uint8_t>(pg, frac, v00, v10, v01, v11, bias);

      svst1b_u16(pg, &dst[static_cast<ptrdiff_t>(i)], result);
    }
  }

 private:
  svuint16_t load_pixels_or_constant_border(Rows<const ScalarType> src_rows_,
                                            svuint16_t& v_src_stride_,
                                            svuint16_t& v_width_,
                                            svuint16_t& v_height_,
                                            svuint16_t& v_border_, svbool_t pg,
                                            svuint16_t x, svuint16_t y) {
    // Find whether coordinates are within the image dimensions.
    svbool_t in_range = svand_b_z(pg, svcmplt_u16(pg, x, v_width_),
                                  svcmplt_u16(pg, y, v_height_));

    // Calculate offsets from coordinates (y * stride + x)
    svuint32_t offsets_b = svmlalb_u32(svmovlb_u32(x), y, v_src_stride_);
    svuint32_t offsets_t = svmlalt_u32(svmovlt_u32(x), y, v_src_stride_);

    svbool_t pg_b = in_range;
    svbool_t pg_t = svtrn2_b16(in_range, svpfalse());

    // Copy pixels from source
    svuint32_t result_b =
        svld1ub_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
    svuint32_t result_t =
        svld1ub_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);

    svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                   svreinterpret_u16_u32(result_t));

    return svsel(in_range, result, v_border_);
  }

  Rows<const ScalarType> src_rows_;
  svuint16_t& v_src_stride_;
  svuint16_t& v_width_;
  svuint16_t& v_height_;
  svuint16_t& v_border_;
};  // end of class RemapS16Point5ConstantBorder<uint8_t>

template <>
class RemapS16Point5ConstantBorder<uint16_t> {
 public:
  using ScalarType = uint16_t;

  RemapS16Point5ConstantBorder(Rows<const ScalarType> src_rows,
                               size_t src_width, size_t src_height,
                               const ScalarType* border_value,
                               svuint16_t& v_src_stride, svuint16_t& v_width,
                               svuint16_t& v_height, svuint16_t& v_border)
      : src_rows_{src_rows},
        v_src_element_stride_{v_src_stride},
        v_width_{v_width},
        v_height_{v_height},
        v_border_{v_border} {
    v_src_element_stride_ = svdup_u16(src_rows.stride() / sizeof(ScalarType));
    v_width_ = svdup_u16(static_cast<uint16_t>(src_width));
    v_height_ = svdup_u16(static_cast<uint16_t>(src_height));
    v_border_ = svdup_u16(*border_value);
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    svuint16_t one = svdup_n_u16(1);
    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);
    for (size_t i = 0; i < width; i += svcnth()) {
      svbool_t pg = svwhilelt_b16_u64(i, width);

      svuint16x2_t xy =
          svld2_u16(pg, reinterpret_cast<const uint16_t*>(
                            &mapxy[static_cast<ptrdiff_t>(i * 2)]));

      svuint16_t x0 = svget2(xy, 0);
      svuint16_t y0 = svget2(xy, 1);
      svuint16_t x1 = svadd_x(pg, x0, one);
      svuint16_t y1 = svadd_x(pg, y0, one);

      svuint16_t v00 = load_pixels_or_constant_border(
          src_rows_, v_src_element_stride_, v_width_, v_height_, v_border_, pg,
          x0, y0);
      svuint16_t v01 = load_pixels_or_constant_border(
          src_rows_, v_src_element_stride_, v_width_, v_height_, v_border_, pg,
          x0, y1);
      svuint16_t v10 = load_pixels_or_constant_border(
          src_rows_, v_src_element_stride_, v_width_, v_height_, v_border_, pg,
          x1, y0);
      svuint16_t v11 = load_pixels_or_constant_border(
          src_rows_, v_src_element_stride_, v_width_, v_height_, v_border_, pg,
          x1, y1);

      svuint16_t frac = svld1_u16(pg, &mapfrac[static_cast<ptrdiff_t>(i)]);
      svuint16_t result =
          interpolate_16point5<uint16_t>(pg, frac, v00, v10, v01, v11, bias);

      svst1_u16(pg, &dst[static_cast<ptrdiff_t>(i)], result);
    }
  }

 private:
  svuint16_t load_pixels_or_constant_border(Rows<const ScalarType> src_rows_,
                                            svuint16_t& v_src_element_stride_,
                                            svuint16_t& v_width_,
                                            svuint16_t& v_height_,
                                            svuint16_t& v_border_, svbool_t pg,
                                            svuint16_t x, svuint16_t y) {
    // Find whether coordinates are within the image dimensions.
    svbool_t in_range = svand_b_z(pg, svcmplt_u16(pg, x, v_width_),
                                  svcmplt_u16(pg, y, v_height_));

    // Calculate offsets from coordinates (y * stride/sizeof(ScalarType) + x)
    svuint32_t offsets_b =
        svmlalb_u32(svmovlb_u32(x), y, v_src_element_stride_);
    svuint32_t offsets_t =
        svmlalt_u32(svmovlt_u32(x), y, v_src_element_stride_);

    svbool_t pg_b = in_range;
    svbool_t pg_t = svtrn2_b16(in_range, svpfalse());

    // Account for the size of the source type when calculating offset
    offsets_b = svlsl_n_u32_x(pg_b, offsets_b, 1);
    offsets_t = svlsl_n_u32_x(pg_t, offsets_t, 1);

    // Copy pixels from source
    svuint32_t result_b =
        svld1uh_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
    svuint32_t result_t =
        svld1uh_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);

    svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                   svreinterpret_u16_u32(result_t));

    return svsel(in_range, result, v_border_);
  }

  Rows<const ScalarType> src_rows_;
  svuint16_t& v_src_element_stride_;
  svuint16_t& v_width_;
  svuint16_t& v_height_;
  svuint16_t& v_border_;
};  // end of class RemapS16Point5ConstantBorder<uint16_t>

template <typename ScalarType>
class RemapS16Point5Replicate4ch;

template <>
class RemapS16Point5Replicate4ch<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5Replicate4ch(Rows<const ScalarType> src_rows, size_t src_width,
                             size_t src_height, svuint16_t& v_src_stride,
                             MapVectorType& v_x_max, MapVectorType& v_y_max)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_stride_ = svdup_u16(src_rows.stride());
    v_xmax_ = svdup_s16(static_cast<int16_t>(src_width - 1));
    v_ymax_ = svdup_s16(static_cast<int16_t>(src_height - 1));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      svbool_t pg = MapVecTraits::svptrue();
      vector_path(pg, mapxy, mapfrac, dst, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      vector_path(pg, mapxy, mapfrac, dst, static_cast<ptrdiff_t>(length));
    });
  }

  void vector_path(svbool_t pg, Columns<const int16_t>& mapxy,
                   Columns<const uint16_t>& mapfrac, Columns<ScalarType>& dst,
                   ptrdiff_t step) {
    MapVector2Type xy = svld2_s16(pg, &mapxy[0]);
    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);

    // Clamp coordinates to within the dimensions of the source image
    svuint16_t x0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 0), v_xmax_)));
    svuint16_t y0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 1), v_ymax_)));

    // x1 = x0 + 1, and clamp it too
    svuint16_t x1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 0), 1), v_xmax_)));

    svuint16_t y1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 1), 1), v_ymax_)));
    svbool_t pg_b = svwhilelt_b32(int64_t{0}, (step + 1) / 2);
    svbool_t pg_t = svwhilelt_b32(int64_t{0}, step / 2);

    // Calculate offsets from coordinates (y * stride + x), x multiplied by 4
    // channels
    auto load_4ch_b = [&](svuint16_t x, svuint16_t y) {
      return svreinterpret_u8_u32(svld1_gather_u32offset_u32(
          pg_b, reinterpret_cast<const uint32_t*>(&src_rows_[0]),
          svmlalb_u32(svshllb_n_u32(x, 2), y, v_src_stride_)));
    };
    auto load_4ch_t = [&](svuint16_t x, svuint16_t y) {
      return svreinterpret_u8_u32(svld1_gather_u32offset_u32(
          pg_t, reinterpret_cast<const uint32_t*>(&src_rows_[0]),
          svmlalt_u32(svshllt_n_u32(x, 2), y, v_src_stride_)));
    };

    FracVectorType frac = svld1_u16(pg, &mapfrac[0]);
    svuint16_t xfrac =
        svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
    svuint16_t yfrac =
        svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
                svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));

    auto lerp2d = [&](svuint16_t xfrac, svuint16_t yfrac, svuint16_t nxfrac,
                      svuint16_t nyfrac, svuint16_t src_a, svuint16_t src_b,
                      svuint16_t src_c, svuint16_t src_d, svuint32_t bias) {
      svuint16_t line0 = svmla_x(
          svptrue_b16(), svmul_x(svptrue_b16(), xfrac, src_b), nxfrac, src_a);
      svuint16_t line1 = svmla_x(
          svptrue_b16(), svmul_x(svptrue_b16(), xfrac, src_d), nxfrac, src_c);

      svuint32_t acc_b = svmlalb_u32(bias, line0, nyfrac);
      svuint32_t acc_t = svmlalt_u32(bias, line0, nyfrac);
      acc_b = svmlalb_u32(acc_b, line1, yfrac);
      acc_t = svmlalt_u32(acc_t, line1, yfrac);

      return svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS), acc_t,
                     2ULL * REMAP16POINT5_FRAC_BITS);
    };

    // bottom part
    svuint8_t a = load_4ch_b(x0, y0);
    svuint8_t b = load_4ch_b(x1, y0);
    svuint8_t c = load_4ch_b(x0, y1);
    svuint8_t d = load_4ch_b(x1, y1);
    // from xfrac, we need the bottom part twice
    svuint16_t xfrac2b = svtrn1_u16(xfrac, xfrac);
    svuint16_t nxfrac2b = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2b);
    svuint16_t yfrac2b = svtrn1_u16(yfrac, yfrac);
    svuint16_t nyfrac2b = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2b);

    // a,b,c,d looks like 12341234...(four channels)
    // bottom is 1313...
    svuint16_t res_bb =
        lerp2d(xfrac2b, yfrac2b, nxfrac2b, nyfrac2b, svmovlb_u16(a),
               svmovlb_u16(b), svmovlb_u16(c), svmovlb_u16(d), bias);
    // top is 2424...
    svuint16_t res_bt =
        lerp2d(xfrac2b, yfrac2b, nxfrac2b, nyfrac2b, svmovlt_u16(a),
               svmovlt_u16(b), svmovlt_u16(c), svmovlt_u16(d), bias);
    svuint8_t res_b =
        svtrn1_u8(svreinterpret_u8_u16(res_bb), svreinterpret_u8_u16(res_bt));

    // top part
    a = load_4ch_t(x0, y0);
    b = load_4ch_t(x1, y0);
    c = load_4ch_t(x0, y1);
    d = load_4ch_t(x1, y1);
    // from xfrac, we need the top part twice
    svuint16_t xfrac2t = svtrn2_u16(xfrac, xfrac);
    svuint16_t nxfrac2t = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2t);
    svuint16_t yfrac2t = svtrn2_u16(yfrac, yfrac);
    svuint16_t nyfrac2t = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2t);

    // a,b,c,d looks like 12341234...(four channels)
    // bottom is 1313...
    svuint16_t res_tb =
        lerp2d(xfrac2t, yfrac2t, nxfrac2t, nyfrac2t, svmovlb_u16(a),
               svmovlb_u16(b), svmovlb_u16(c), svmovlb_u16(d), bias);
    // top is 2424...
    svuint16_t res_tt =
        lerp2d(xfrac2t, yfrac2t, nxfrac2t, nyfrac2t, svmovlt_u16(a),
               svmovlt_u16(b), svmovlt_u16(c), svmovlt_u16(d), bias);
    svuint8_t res_t =
        svtrn1_u8(svreinterpret_u8_u16(res_tb), svreinterpret_u8_u16(res_tt));

    svbool_t pg_low = svwhilelt_b32_u64(0L, static_cast<size_t>(step));
    svbool_t pg_high = svwhilelt_b32_u64(svcntw(), static_cast<size_t>(step));
    svuint32_t res_low =
        svzip1_u32(svreinterpret_u32_u8(res_b), svreinterpret_u32_u8(res_t));
    svuint32_t res_high =
        svzip2_u32(svreinterpret_u32_u8(res_b), svreinterpret_u32_u8(res_t));
    mapxy += step;
    svst1_u32(pg_low, reinterpret_cast<uint32_t*>(&dst[0]), res_low);
    svst1_u32(pg_high, reinterpret_cast<uint32_t*>(&dst[0]) + svcntw(),
              res_high);
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint16_t& v_src_stride_;
  MapVectorType& v_xmax_;
  MapVectorType& v_ymax_;
};  // end of class RemapS16Point5Replicate4ch<uint8_t>

template <>
class RemapS16Point5Replicate4ch<uint16_t> {
 public:
  using ScalarType = uint16_t;

  RemapS16Point5Replicate4ch(Rows<const ScalarType> src_rows, size_t src_width,
                             size_t src_height, svuint32_t& v_src_stride,
                             svint32_t& v_x_max, svint32_t& v_y_max)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_stride_ = svdup_u32(src_rows.stride());
    v_xmax_ = svdup_s32(static_cast<int32_t>(src_width - 1));
    v_ymax_ = svdup_s32(static_cast<int32_t>(src_height - 1));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    LoopUnroll loop{width, svcntw()};
    loop.unroll_once([&](size_t step) {
      vector_path(svptrue_b32(), svptrue_b64(), svptrue_b64(), svptrue_b64(),
                  svptrue_b64(), mapxy, mapfrac, dst,
                  static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = svwhilelt_b32_u64(step, step + length);
      svbool_t pg64_b = svtrn1_b32(pg, svpfalse());
      svbool_t pg64_t = svtrn2_b32(pg, svpfalse());
      svbool_t pg_low = svzip1_b32(pg, svpfalse());
      svbool_t pg_high = svzip2_b32(pg, svpfalse());
      vector_path(pg, pg64_b, pg64_t, pg_low, pg_high, mapxy, mapfrac, dst,
                  static_cast<ptrdiff_t>(length));
    });
  }

  void vector_path(svbool_t pg, svbool_t pg64_b, svbool_t pg64_t,
                   svbool_t pg_low, svbool_t pg_high,
                   Columns<const int16_t>& mapxy,
                   Columns<const uint16_t>& mapfrac, Columns<ScalarType>& dst,
                   ptrdiff_t step) {
    // Load one vector of xy: even coordinates are x, odd are y
    svint16_t xy = svreinterpret_s16_u32(
        svld1_u32(pg, reinterpret_cast<const uint32_t*>(&mapxy[0])));
    svint32_t x = svmovlb(xy);
    svint32_t y = svmovlt(xy);
    // Clamp coordinates to within the dimensions of the source image
    svuint32_t x0 = svreinterpret_u32_s32(
        svmax_x(pg, svdup_n_s32(0), svmin_x(pg, x, v_xmax_)));
    svuint32_t y0 = svreinterpret_u32_s32(
        svmax_x(pg, svdup_n_s32(0), svmin_x(pg, y, v_ymax_)));

    // x1 = x0 + 1, and clamp it too
    svuint32_t x1 = svreinterpret_u32_s32(svmax_x(
        pg, svdup_n_s32(0), svmin_x(pg, svqadd_n_s32_x(pg, x, 1), v_xmax_)));
    svuint32_t y1 = svreinterpret_u32_s32(svmax_x(
        pg, svdup_n_s32(0), svmin_x(pg, svqadd_n_s32_x(pg, y, 1), v_ymax_)));

    auto load_4ch = [&](svbool_t pg, svuint64_t offsets) {
      return svreinterpret_u16_u64(svld1_gather_u64offset_u64(
          pg, reinterpret_cast<const uint64_t*>(&src_rows_[0]), offsets));
    };

    svuint16_t xfrac, yfrac, nxfrac, nyfrac;
    {
      // Fractions are loaded into even lanes
      svuint16_t rawfrac = svreinterpret_u16_u32(svld1uh_u32(pg, &mapfrac[0]));

      // Fractions are doubled, 00112233... (will be doubled again later)
      svuint16_t frac = svtrn1(rawfrac, rawfrac);

      xfrac = svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
      yfrac = svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
                      svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
      nxfrac = svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
      nyfrac = svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);
    }

    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);

    auto lerp2d = [&](svuint16_t xfrac, svuint16_t yfrac, svuint16_t nxfrac,
                      svuint16_t nyfrac, svuint16_t src_a, svuint16_t src_b,
                      svuint16_t src_c, svuint16_t src_d, svuint32_t bias) {
      svuint32_t line0_b = svmlalb(svmullb(xfrac, src_b), nxfrac, src_a);
      svuint32_t line0_t = svmlalt(svmullt(xfrac, src_b), nxfrac, src_a);
      svuint32_t line1_b = svmlalb(svmullb(xfrac, src_d), nxfrac, src_c);
      svuint32_t line1_t = svmlalt(svmullt(xfrac, src_d), nxfrac, src_c);

      svuint32_t acc_b =
          svmla_u32_x(svptrue_b32(), bias, line0_b, svmovlb_u32(nyfrac));
      svuint32_t acc_t =
          svmla_u32_x(svptrue_b32(), bias, line0_t, svmovlt_u32(nyfrac));
      acc_b = svmla_u32_x(svptrue_b32(), acc_b, line1_b, svmovlb_u32(yfrac));
      acc_t = svmla_u32_x(svptrue_b32(), acc_t, line1_t, svmovlt_u32(yfrac));

      return svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS), acc_t,
                     2ULL * REMAP16POINT5_FRAC_BITS);
    };

    // Data is 4x16 = 64 bits, twice as wide as the widened coords (32-bit)
    // Calculation is done in 2 parts, top and bottom
    svuint16_t res_b, res_t;

    {  // bottom
      svuint64_t x0w = svshllb_n_u64(x0, 3);
      svuint64_t x1w = svshllb_n_u64(x1, 3);
      svuint64_t ys0w = svmullb_u64(y0, v_src_stride_);
      svuint64_t ys1w = svmullb_u64(y1, v_src_stride_);
      svuint64_t offsets_a = svadd_x(pg64_b, x0w, ys0w);
      svuint64_t offsets_b = svadd_x(pg64_b, x1w, ys0w);
      svuint64_t offsets_c = svadd_x(pg64_b, x0w, ys1w);
      svuint64_t offsets_d = svadd_x(pg64_b, x1w, ys1w);

      svuint16_t a = load_4ch(pg64_b, offsets_a);
      svuint16_t b = load_4ch(pg64_b, offsets_b);
      svuint16_t c = load_4ch(pg64_b, offsets_c);
      svuint16_t d = load_4ch(pg64_b, offsets_d);

      // Copy even lanes twice -> 000022224444... these are the "bottom"
      // fractions
      svuint16_t xfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(xfrac), svreinterpret_u32_u16(xfrac)));
      svuint16_t nxfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(nxfrac), svreinterpret_u32_u16(nxfrac)));
      svuint16_t yfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(yfrac), svreinterpret_u32_u16(yfrac)));
      svuint16_t nyfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(nyfrac), svreinterpret_u32_u16(nyfrac)));

      res_b = lerp2d(xfr, yfr, nxfr, nyfr, a, b, c, d, bias);
    }

    {  // top
      svuint64_t x0w = svshllt_n_u64(x0, 3);
      svuint64_t x1w = svshllt_n_u64(x1, 3);
      svuint64_t ys0w = svmullt_u64(y0, v_src_stride_);
      svuint64_t ys1w = svmullt_u64(y1, v_src_stride_);
      svuint64_t offsets_a = svadd_x(pg64_b, x0w, ys0w);
      svuint64_t offsets_b = svadd_x(pg64_b, x1w, ys0w);
      svuint64_t offsets_c = svadd_x(pg64_b, x0w, ys1w);
      svuint64_t offsets_d = svadd_x(pg64_b, x1w, ys1w);

      svuint16_t a = load_4ch(pg64_t, offsets_a);
      svuint16_t b = load_4ch(pg64_t, offsets_b);
      svuint16_t c = load_4ch(pg64_t, offsets_c);
      svuint16_t d = load_4ch(pg64_t, offsets_d);

      // Copy odd lanes twice -> 111133335555... these are the "top"
      // fractions
      svuint16_t xfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(xfrac), svreinterpret_u32_u16(xfrac)));
      svuint16_t nxfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(nxfrac), svreinterpret_u32_u16(nxfrac)));
      svuint16_t yfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(yfrac), svreinterpret_u32_u16(yfrac)));
      svuint16_t nyfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(nyfrac), svreinterpret_u32_u16(nyfrac)));

      res_t = lerp2d(xfr, yfr, nxfr, nyfr, a, b, c, d, bias);
    }

    svuint64_t res_low =
        svzip1_u64(svreinterpret_u64_u16(res_b), svreinterpret_u64_u16(res_t));
    svuint64_t res_high =
        svzip2_u64(svreinterpret_u64_u16(res_b), svreinterpret_u64_u16(res_t));
    svst1_u64(pg_low, reinterpret_cast<uint64_t*>(&dst[0]), res_low);
    svst1_u64(pg_high, reinterpret_cast<uint64_t*>(&dst[0]) + svcntd(),
              res_high);
    mapxy += step;
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint32_t& v_src_stride_;
  svint32_t& v_xmax_;
  svint32_t& v_ymax_;
};  // end of class RemapS16Point5Replicate4ch<uint16_t>

template <typename ScalarType>
class RemapS16Point5Constant4ch;

template <>
class RemapS16Point5Constant4ch<uint8_t> {
 public:
  using ScalarType = uint8_t;

  RemapS16Point5Constant4ch(Rows<const ScalarType> src_rows, size_t src_width,
                            size_t src_height, const ScalarType* border_value,
                            svuint16_t& v_src_stride, svuint16_t& v_x_max,
                            svuint16_t& v_y_max, svuint32_t& v_border)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max},
        v_border_{v_border} {
    v_src_stride_ = svdup_u16(src_rows.stride());
    v_xmax_ = svdup_u16(static_cast<uint16_t>(src_width - 1));
    v_ymax_ = svdup_u16(static_cast<uint16_t>(src_height - 1));
    uint32_t border_value_u32{};
    memcpy(&border_value_u32, border_value, sizeof(uint32_t));
    v_border_ = svdup_u32(border_value_u32);
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    LoopUnroll loop{width, svcnth()};
    loop.unroll_once([&](size_t step) {
      svbool_t pg = svptrue_b16();
      vector_path(pg, mapxy, mapfrac, dst, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = svwhilelt_b16_u64(step - length, step);
      vector_path(pg, mapxy, mapfrac, dst, static_cast<ptrdiff_t>(length));
    });
  }

  void vector_path(svbool_t pg, Columns<const int16_t>& mapxy,
                   Columns<const uint16_t>& mapfrac, Columns<ScalarType>& dst,
                   ptrdiff_t step) {
    svuint16x2_t xy =
        svld2_u16(pg, reinterpret_cast<const uint16_t*>(&mapxy[0]));
    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);

    // Negative values become big positive ones
    svuint16_t x0 = svget2(xy, 0);
    svuint16_t y0 = svget2(xy, 1);
    svuint16_t x1 = svadd_n_u16_x(pg, x0, 1);
    svuint16_t y1 = svadd_n_u16_x(pg, y0, 1);

    // Calculate offsets from coordinates (y * stride + x), x multiplied by 4
    // channels
    auto load_4ch_or_border_b = [&](svuint16_t x, svuint16_t y) {
      svbool_t in_range_b16 =
          svand_b_z(pg, svcmple(pg, x, v_xmax_), svcmple(pg, y, v_ymax_));
      svbool_t in_range = svtrn1_b16(in_range_b16, svpfalse());
      svuint32_t image = svld1_gather_u32offset_u32(
          in_range, reinterpret_cast<const uint32_t*>(&src_rows_[0]),
          svmlalb_u32(svshllb_n_u32(x, 2), y, v_src_stride_));
      return svreinterpret_u8_u32(svsel(in_range, image, v_border_));
    };
    auto load_4ch_or_border_t = [&](svuint16_t x, svuint16_t y) {
      svbool_t in_range_b16 =
          svand_b_z(pg, svcmple(pg, x, v_xmax_), svcmple(pg, y, v_ymax_));
      svbool_t in_range = svtrn2_b16(in_range_b16, svpfalse());
      svuint32_t image = svld1_gather_u32offset_u32(
          in_range, reinterpret_cast<const uint32_t*>(&src_rows_[0]),
          svmlalt_u32(svshllt_n_u32(x, 2), y, v_src_stride_));
      return svreinterpret_u8_u32(svsel(in_range, image, v_border_));
    };

    svuint16_t frac = svld1_u16(pg, &mapfrac[0]);
    svuint16_t xfrac =
        svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
    svuint16_t yfrac =
        svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
                svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));

    auto lerp2d = [&](svuint16_t xfrac, svuint16_t yfrac, svuint16_t nxfrac,
                      svuint16_t nyfrac, svuint16_t src_a, svuint16_t src_b,
                      svuint16_t src_c, svuint16_t src_d, svuint32_t bias) {
      svuint16_t line0 = svmla_x(
          svptrue_b16(), svmul_x(svptrue_b16(), xfrac, src_b), nxfrac, src_a);
      svuint16_t line1 = svmla_x(
          svptrue_b16(), svmul_x(svptrue_b16(), xfrac, src_d), nxfrac, src_c);

      svuint32_t acc_b = svmlalb_u32(bias, line0, nyfrac);
      svuint32_t acc_t = svmlalt_u32(bias, line0, nyfrac);
      acc_b = svmlalb_u32(acc_b, line1, yfrac);
      acc_t = svmlalt_u32(acc_t, line1, yfrac);

      return svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS), acc_t,
                     2ULL * REMAP16POINT5_FRAC_BITS);
    };

    // bottom part
    svuint8_t a = load_4ch_or_border_b(x0, y0);
    svuint8_t b = load_4ch_or_border_b(x1, y0);
    svuint8_t c = load_4ch_or_border_b(x0, y1);
    svuint8_t d = load_4ch_or_border_b(x1, y1);
    // from xfrac, we need the bottom part twice
    svuint16_t xfrac2b = svtrn1_u16(xfrac, xfrac);
    svuint16_t nxfrac2b = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2b);
    svuint16_t yfrac2b = svtrn1_u16(yfrac, yfrac);
    svuint16_t nyfrac2b = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2b);

    // a,b,c,d looks like 12341234...(four channels)
    // bottom is 1313...
    svuint16_t res_bb =
        lerp2d(xfrac2b, yfrac2b, nxfrac2b, nyfrac2b, svmovlb_u16(a),
               svmovlb_u16(b), svmovlb_u16(c), svmovlb_u16(d), bias);
    // top is 2424...
    svuint16_t res_bt =
        lerp2d(xfrac2b, yfrac2b, nxfrac2b, nyfrac2b, svmovlt_u16(a),
               svmovlt_u16(b), svmovlt_u16(c), svmovlt_u16(d), bias);
    svuint8_t res_b =
        svtrn1_u8(svreinterpret_u8_u16(res_bb), svreinterpret_u8_u16(res_bt));

    // top part
    a = load_4ch_or_border_t(x0, y0);
    b = load_4ch_or_border_t(x1, y0);
    c = load_4ch_or_border_t(x0, y1);
    d = load_4ch_or_border_t(x1, y1);
    // from xfrac, we need the top part twice
    svuint16_t xfrac2t = svtrn2_u16(xfrac, xfrac);
    svuint16_t nxfrac2t = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac2t);
    svuint16_t yfrac2t = svtrn2_u16(yfrac, yfrac);
    svuint16_t nyfrac2t = svsub_u16_x(
        svptrue_b16(), svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac2t);

    // a,b,c,d looks like 12341234...(four channels)
    // bottom is 1313...
    svuint16_t res_tb =
        lerp2d(xfrac2t, yfrac2t, nxfrac2t, nyfrac2t, svmovlb_u16(a),
               svmovlb_u16(b), svmovlb_u16(c), svmovlb_u16(d), bias);
    // top is 2424...
    svuint16_t res_tt =
        lerp2d(xfrac2t, yfrac2t, nxfrac2t, nyfrac2t, svmovlt_u16(a),
               svmovlt_u16(b), svmovlt_u16(c), svmovlt_u16(d), bias);
    svuint8_t res_t =
        svtrn1_u8(svreinterpret_u8_u16(res_tb), svreinterpret_u8_u16(res_tt));

    svbool_t pg_low = svwhilelt_b32_u64(0L, static_cast<size_t>(step));
    svbool_t pg_high = svwhilelt_b32_u64(svcntw(), static_cast<size_t>(step));
    svuint32_t res_low =
        svzip1_u32(svreinterpret_u32_u8(res_b), svreinterpret_u32_u8(res_t));
    svuint32_t res_high =
        svzip2_u32(svreinterpret_u32_u8(res_b), svreinterpret_u32_u8(res_t));
    mapxy += step;
    svst1_u32(pg_low, reinterpret_cast<uint32_t*>(&dst[0]), res_low);
    svst1_u32(pg_high, reinterpret_cast<uint32_t*>(&dst[0]) + svcntw(),
              res_high);
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint16_t& v_src_stride_;
  svuint16_t& v_xmax_;
  svuint16_t& v_ymax_;
  svuint32_t& v_border_;
};  // end of class RemapS16Point5Constant4ch<uint8_t>

template <>
class RemapS16Point5Constant4ch<uint16_t> {
 public:
  using ScalarType = uint16_t;

  RemapS16Point5Constant4ch(Rows<const ScalarType> src_rows, size_t src_width,
                            size_t src_height, const ScalarType* border_value,
                            svuint32_t& v_src_stride, svuint32_t& v_x_max,
                            svuint32_t& v_y_max, svuint64_t& v_border)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max},
        v_border_{v_border} {
    v_src_stride_ = svdup_u32(src_rows.stride());
    v_xmax_ = svdup_u32(static_cast<uint32_t>(src_width - 1));
    v_ymax_ = svdup_u32(static_cast<uint32_t>(src_height - 1));
    uint64_t border_value_u64{};
    memcpy(&border_value_u64, border_value, sizeof(uint64_t));
    v_border_ = svdup_u64(border_value_u64);
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    LoopUnroll loop{width, svcntw()};
    loop.unroll_once([&](size_t step) {
      vector_path(svptrue_b32(), svptrue_b64(), svptrue_b64(), mapxy, mapfrac,
                  dst, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = svwhilelt_b32_u64(step, step + length);
      svbool_t pg_low = svzip1_b32(pg, svpfalse());
      svbool_t pg_high = svzip2_b32(pg, svpfalse());
      vector_path(pg, pg_low, pg_high, mapxy, mapfrac, dst,
                  static_cast<ptrdiff_t>(length));
    });
  }

  void vector_path(svbool_t pg, svbool_t pg_low, svbool_t pg_high,
                   Columns<const int16_t>& mapxy,
                   Columns<const uint16_t>& mapfrac, Columns<ScalarType>& dst,
                   ptrdiff_t step) {
    // Load one vector of xy: even coordinates are x, odd are y
    svint16_t xy = svreinterpret_s16_u32(
        svld1_u32(pg, reinterpret_cast<const uint32_t*>(&mapxy[0])));

    // Negative values become big positive ones
    // Widening is signed, so 16-bit -1 becomes 32-bit -1
    svuint32_t x0 = svreinterpret_u32_s32(svmovlb(xy));
    svuint32_t y0 = svreinterpret_u32_s32(svmovlt(xy));
    svuint32_t x1 = svadd_n_u32_x(pg, x0, 1);
    svuint32_t y1 = svadd_n_u32_x(pg, y0, 1);

    auto load_4ch_or_border_b = [&](svuint32_t x, svuint32_t y) {
      svbool_t in_range_b32 =
          svand_b_z(pg, svcmple(pg, x, v_xmax_), svcmple(pg, y, v_ymax_));
      svbool_t in_range = svtrn1_b32(in_range_b32, svpfalse());
      svuint64_t image = svld1_gather_u64offset_u64(
          in_range, reinterpret_cast<const uint64_t*>(&src_rows_[0]),
          svmlalb_u64(svshllb_n_u64(x, 3), y, v_src_stride_));
      return svreinterpret_u16_u64(svsel(in_range, image, v_border_));
    };

    auto load_4ch_or_border_t = [&](svuint32_t x, svuint32_t y) {
      svbool_t in_range_b32 =
          svand_b_z(pg, svcmple(pg, x, v_xmax_), svcmple(pg, y, v_ymax_));
      svbool_t in_range = svtrn2_b32(in_range_b32, svpfalse());
      svuint64_t image = svld1_gather_u64offset_u64(
          in_range, reinterpret_cast<const uint64_t*>(&src_rows_[0]),
          svmlalt_u64(svshllt_n_u64(x, 3), y, v_src_stride_));
      return svreinterpret_u16_u64(svsel(in_range, image, v_border_));
    };

    svuint16_t xfrac, yfrac, nxfrac, nyfrac;
    {
      // Fractions are loaded into even lanes
      svuint16_t rawfrac = svreinterpret_u16_u32(svld1uh_u32(pg, &mapfrac[0]));

      // Fractions are doubled, 00112233... (will be doubled again later)
      svuint16_t frac = svtrn1(rawfrac, rawfrac);

      xfrac = svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
      yfrac = svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
                      svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
      nxfrac = svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
      nyfrac = svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);
    }

    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);

    auto lerp2d = [&](svuint16_t xfrac, svuint16_t yfrac, svuint16_t nxfrac,
                      svuint16_t nyfrac, svuint16_t src_a, svuint16_t src_b,
                      svuint16_t src_c, svuint16_t src_d, svuint32_t bias) {
      svuint32_t line0_b = svmlalb(svmullb(xfrac, src_b), nxfrac, src_a);
      svuint32_t line0_t = svmlalt(svmullt(xfrac, src_b), nxfrac, src_a);
      svuint32_t line1_b = svmlalb(svmullb(xfrac, src_d), nxfrac, src_c);
      svuint32_t line1_t = svmlalt(svmullt(xfrac, src_d), nxfrac, src_c);

      svuint32_t acc_b =
          svmla_u32_x(svptrue_b32(), bias, line0_b, svmovlb_u32(nyfrac));
      svuint32_t acc_t =
          svmla_u32_x(svptrue_b32(), bias, line0_t, svmovlt_u32(nyfrac));
      acc_b = svmla_u32_x(svptrue_b32(), acc_b, line1_b, svmovlb_u32(yfrac));
      acc_t = svmla_u32_x(svptrue_b32(), acc_t, line1_t, svmovlt_u32(yfrac));

      return svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS), acc_t,
                     2ULL * REMAP16POINT5_FRAC_BITS);
    };

    // Data is 4x16 = 64 bits, twice as wide as the widened coords (32-bit)
    // Calculation is done in 2 parts, top and bottom
    svuint16_t res_b, res_t;

    {  // bottom
      svuint16_t a = load_4ch_or_border_b(x0, y0);
      svuint16_t b = load_4ch_or_border_b(x1, y0);
      svuint16_t c = load_4ch_or_border_b(x0, y1);
      svuint16_t d = load_4ch_or_border_b(x1, y1);

      // Copy even lanes twice -> 000022224444... these are the "bottom"
      // fractions
      svuint16_t xfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(xfrac), svreinterpret_u32_u16(xfrac)));
      svuint16_t nxfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(nxfrac), svreinterpret_u32_u16(nxfrac)));
      svuint16_t yfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(yfrac), svreinterpret_u32_u16(yfrac)));
      svuint16_t nyfr = svreinterpret_u16_u32(svtrn1_u32(
          svreinterpret_u32_u16(nyfrac), svreinterpret_u32_u16(nyfrac)));

      res_b = lerp2d(xfr, yfr, nxfr, nyfr, a, b, c, d, bias);
    }

    {  // top
      svuint16_t a = load_4ch_or_border_t(x0, y0);
      svuint16_t b = load_4ch_or_border_t(x1, y0);
      svuint16_t c = load_4ch_or_border_t(x0, y1);
      svuint16_t d = load_4ch_or_border_t(x1, y1);

      // Copy odd lanes twice -> 111133335555... these are the "top"
      // fractions
      svuint16_t xfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(xfrac), svreinterpret_u32_u16(xfrac)));
      svuint16_t nxfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(nxfrac), svreinterpret_u32_u16(nxfrac)));
      svuint16_t yfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(yfrac), svreinterpret_u32_u16(yfrac)));
      svuint16_t nyfr = svreinterpret_u16_u32(svtrn2_u32(
          svreinterpret_u32_u16(nyfrac), svreinterpret_u32_u16(nyfrac)));

      res_t = lerp2d(xfr, yfr, nxfr, nyfr, a, b, c, d, bias);
    }

    svuint64_t res_low =
        svzip1_u64(svreinterpret_u64_u16(res_b), svreinterpret_u64_u16(res_t));
    svuint64_t res_high =
        svzip2_u64(svreinterpret_u64_u16(res_b), svreinterpret_u64_u16(res_t));
    svst1_u64(pg_low, reinterpret_cast<uint64_t*>(&dst[0]), res_low);
    svst1_u64(pg_high, reinterpret_cast<uint64_t*>(&dst[0]) + svcntd(),
              res_high);
    mapxy += step;
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint32_t& v_src_stride_;
  svuint32_t& v_xmax_;
  svuint32_t& v_ymax_;
  svuint64_t& v_border_;
};  // end of class RemapS16Point5Constant4ch<uint16_t>

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t remap_s16point5(const T* src, size_t src_stride,
                                 size_t src_width, size_t src_height, T* dst,
                                 size_t dst_stride, size_t dst_width,
                                 size_t dst_height, size_t channels,
                                 const int16_t* mapxy, size_t mapxy_stride,
                                 const uint16_t* mapfrac, size_t mapfrac_stride,
                                 kleidicv_border_type_t border_type,
                                 const T* border_value) {
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
  svuint16_t sv_src_stride;
  Rectangle rect{dst_width, dst_height};

  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT) {
    if (channels == 1) {
      svuint16_t sv_width, sv_height, sv_border;
      RemapS16Point5ConstantBorder<T> operation{
          src_rows,      src_width, src_height, border_value,
          sv_src_stride, sv_width,  sv_height,  sv_border};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    } else {
      assert(channels == 4);
      typedef typename double_element_width<T>::type DoubleType;
      typedef typename double_element_width<DoubleType>::type QuadType;
      typename VecTraits<DoubleType>::VectorType sv_width, sv_height,
          sv_src_stride;
      typename VecTraits<QuadType>::VectorType sv_border;
      RemapS16Point5Constant4ch<T> operation{
          src_rows,      src_width, src_height, border_value,
          sv_src_stride, sv_width,  sv_height,  sv_border};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    }
  } else {
    assert(border_type == KLEIDICV_BORDER_TYPE_REPLICATE);
    svint16_t sv_xmax, sv_ymax;
    if (channels == 1) {
      RemapS16Point5Replicate<T> operation{src_rows,      src_width, src_height,
                                           sv_src_stride, sv_xmax,   sv_ymax};
      zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
    } else {
      assert(channels == 4);
      if constexpr (std::is_same<T, uint8_t>::value) {
        RemapS16Point5Replicate4ch<T> operation{
            src_rows, src_width, src_height, sv_src_stride, sv_xmax, sv_ymax};
        zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
      }
      if constexpr (std::is_same<T, uint16_t>::value) {
        svuint32_t stride;
        svint32_t xmax, ymax;
        RemapS16Point5Replicate4ch<T> operation{src_rows, src_width, src_height,
                                                stride,   xmax,      ymax};
        zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
      }
    }
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(type)                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16point5<type>(    \
      const type* src, size_t src_stride, size_t src_width, size_t src_height, \
      type* dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t* mapxy, size_t mapxy_stride,              \
      const uint16_t* mapfrac, size_t mapfrac_stride,                          \
      kleidicv_border_type_t border_type, const type* border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint16_t);

}  // namespace kleidicv::sve2
