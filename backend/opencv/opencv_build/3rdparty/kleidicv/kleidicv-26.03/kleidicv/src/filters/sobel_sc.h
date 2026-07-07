// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SOBEL_SC_H
#define KLEIDICV_SOBEL_SC_H

#include "kleidicv/filters/separable_filter_3x3_sc.h"
#include "kleidicv/filters/sobel.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/workspace/separable.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Template for 3x3 Sobel filters which calculate horizontal derivative
// approximations, often denoted as Gx.
//
//      [ -1, 0, 1 ]   [ 1 ]
//  F = [ -2, 0, 2 ] = [ 2 ] * [ -1,  0, 1 ]
//      [ -1, 0, 1 ]   [ 1 ]
template <typename T>
class HorizontalSobel3x3;

// 3x3 Sobel filter for uint8_t types which calculates horizontal derivative
// approximations, often denoted as Gx.
template <>
class HorizontalSobel3x3<uint8_t> {
 public:
  using SourceType = uint8_t;
  using BufferType = int16_t;
  using DestinationType = int16_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<svuint8_t> src[3],
                            BufferType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_u16_b = svaddlb(src[0], src[2]);
    svuint16_t acc_u16_t = svaddlt(src[0], src[2]);
    acc_u16_b = svmlalb(acc_u16_b, src[1], svdup_n_u8(2));
    acc_u16_t = svmlalt(acc_u16_t, src[1], svdup_n_u8(2));

    svint16x2_t interleaved =
        svcreate2(svreinterpret_s16(acc_u16_b), svreinterpret_s16(acc_u16_t));
    svst2(pg, &dst[0], interleaved);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<svint16_t> src[3],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    svst1(pg, &dst[0], svsub_x(pg, src[2], src[0]));
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    // Explicitly narrow. Overflow is permitted.
    dst[0] = static_cast<DestinationType>(src[2] - src[0]);
  }
};  // end of class HorizontalSobel3x3<uint8_t>

// Template for 3x3 Sobel filters which calculate vertical derivative
// approximations, often denoted as Gy.
//
//      [ -1, -2, 1 ]   [ -1 ]
//  F = [  0,  0, 0 ] = [  0 ] * [ 1,  2, 1 ]
//      [  1,  2, 1 ]   [  1 ]
template <typename T>
class VerticalSobel3x3;

// 3x3 Sobel filter for uint8_t types which calculates vertical derivative
// approximations, often denoted as Gy.
template <>
class VerticalSobel3x3<uint8_t> {
 public:
  using SourceType = uint8_t;
  using BufferType = int16_t;
  using DestinationType = int16_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<svuint8_t> src[3],
                            BufferType *dst) const KLEIDICV_STREAMING {
    svuint16_t acc_u16_b = svsublb(src[2], src[0]);
    svuint16_t acc_u16_t = svsublt(src[2], src[0]);

    svint16x2_t interleaved =
        svcreate2(svreinterpret_s16(acc_u16_b), svreinterpret_s16(acc_u16_t));
    svst2(pg, &dst[0], interleaved);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<svint16_t> src[3],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    svint16_t acc = svadd_x(pg, src[0], src[2]);
    acc = svmad_s16_x(pg, src[1], svdup_n_s16(2), acc);
    svst1(pg, &dst[0], acc);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    // Explicitly narrow. Overflow is permitted.
    dst[0] = static_cast<DestinationType>(src[0] + 2 * src[1] + src[2]);
  }
};  // end of class VerticalSobel3x3<uint8_t>

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8_sc(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<int16_t> dst_rows{dst, dst_stride, channels};
  using HorizontalSobel3x3_t = HorizontalSobel3x3<uint8_t>;
  constexpr size_t intermediate_size{
      sizeof(typename HorizontalSobel3x3_t::BufferType)};

  auto workspace_variant =
      SeparableFilterWorkspace::create(rect, channels, intermediate_size);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<SeparableFilterWorkspace>(&workspace_variant);

  HorizontalSobel3x3_t vertical_sobel;
  SeparableFilter3x3<HorizontalSobel3x3_t> filter{vertical_sobel};
  workspace.process(y_begin, y_end, src_rows, dst_rows,
                    FixedBorderType::REPLICATE, filter);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8_sc(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<int16_t> dst_rows{dst, dst_stride, channels};
  using VerticalSobel3x3_t = VerticalSobel3x3<uint8_t>;
  constexpr size_t intermediate_size{
      sizeof(typename VerticalSobel3x3_t::BufferType)};

  auto workspace_variant =
      SeparableFilterWorkspace::create(rect, channels, intermediate_size);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<SeparableFilterWorkspace>(&workspace_variant);

  VerticalSobel3x3_t vertical_sobel;
  SeparableFilter3x3<VerticalSobel3x3_t> filter{vertical_sobel};
  workspace.process(y_begin, y_end, src_rows, dst_rows,
                    FixedBorderType::REPLICATE, filter);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SOBEL_SC_H
