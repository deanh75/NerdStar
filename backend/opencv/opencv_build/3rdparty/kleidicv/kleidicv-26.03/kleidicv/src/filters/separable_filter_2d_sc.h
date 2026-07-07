// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SEPARABLE_FILTER_2D_SC_H
#define KLEIDICV_SEPARABLE_FILTER_2D_SC_H

#include <limits>

#include "kleidicv/filters/separable_filter_5x5_sc.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/workspace/separable.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType, size_t KernelSize>
class SeparableFilter2D;

template <>
class SeparableFilter2D<uint8_t, 5> {
 public:
  using SourceType = uint8_t;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using BufferType = uint16_t;
  using BufferVectorType = typename VecTraits<BufferType>::VectorType;
  using BufferDoubleVectorType = typename VecTraits<BufferType>::Vector2Type;
  using DestinationType = uint8_t;

  SeparableFilter2D(
      const SourceType *kernel_x, BufferVectorType &kernel_x_0_u16,
      BufferVectorType &kernel_x_1_u16, BufferVectorType &kernel_x_2_u16,
      BufferVectorType &kernel_x_3_u16, BufferVectorType &kernel_x_4_u16,
      SourceVectorType &kernel_y_0_u8, SourceVectorType &kernel_y_1_u8,
      SourceVectorType &kernel_y_2_u8, SourceVectorType &kernel_y_3_u8,
      SourceVectorType &kernel_y_4_u8)
      : kernel_x_(kernel_x),
        kernel_x_0_u16_(kernel_x_0_u16),
        kernel_x_1_u16_(kernel_x_1_u16),
        kernel_x_2_u16_(kernel_x_2_u16),
        kernel_x_3_u16_(kernel_x_3_u16),
        kernel_x_4_u16_(kernel_x_4_u16),

        kernel_y_0_u8_(kernel_y_0_u8),
        kernel_y_1_u8_(kernel_y_1_u8),
        kernel_y_2_u8_(kernel_y_2_u8),
        kernel_y_3_u8_(kernel_y_3_u8),
        kernel_y_4_u8_(kernel_y_4_u8) {}

  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<SourceVectorType> src[5],
                            BufferType *dst) const KLEIDICV_STREAMING {
    // 0
    BufferVectorType acc_b = svmullb_u16(src[0], kernel_y_0_u8_);
    BufferVectorType acc_t = svmullt_u16(src[0], kernel_y_0_u8_);

    // 1
    BufferVectorType vec_b = svmullb_u16(src[1], kernel_y_1_u8_);
    BufferVectorType vec_t = svmullt_u16(src[1], kernel_y_1_u8_);
    acc_b = svqadd_u16_x(pg, acc_b, vec_b);
    acc_t = svqadd_u16_x(pg, acc_t, vec_t);

    // 2
    vec_b = svmullb_u16(src[2], kernel_y_2_u8_);
    vec_t = svmullt_u16(src[2], kernel_y_2_u8_);
    acc_b = svqadd_u16_x(pg, acc_b, vec_b);
    acc_t = svqadd_u16_x(pg, acc_t, vec_t);

    // 3
    vec_b = svmullb_u16(src[3], kernel_y_3_u8_);
    vec_t = svmullt_u16(src[3], kernel_y_3_u8_);
    acc_b = svqadd_u16_x(pg, acc_b, vec_b);
    acc_t = svqadd_u16_x(pg, acc_t, vec_t);

    // 4
    vec_b = svmullb_u16(src[4], kernel_y_4_u8_);
    vec_t = svmullt_u16(src[4], kernel_y_4_u8_);
    acc_b = svqadd_u16_x(pg, acc_b, vec_b);
    acc_t = svqadd_u16_x(pg, acc_t, vec_t);

    BufferDoubleVectorType interleaved = svcreate2_u16(acc_b, acc_t);
    svst2(pg, &dst[0], interleaved);
  }

  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<BufferVectorType> src[5],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    // 0
    svuint32_t acc_b = svmullb_u32(src[0], kernel_x_0_u16_);
    svuint32_t acc_t = svmullt_u32(src[0], kernel_x_0_u16_);

    // 1
    acc_b = svmlalb_u32(acc_b, src[1], kernel_x_1_u16_);
    acc_t = svmlalt_u32(acc_t, src[1], kernel_x_1_u16_);

    // 2
    acc_b = svmlalb_u32(acc_b, src[2], kernel_x_2_u16_);
    acc_t = svmlalt_u32(acc_t, src[2], kernel_x_2_u16_);

    // 3
    acc_b = svmlalb_u32(acc_b, src[3], kernel_x_3_u16_);
    acc_t = svmlalt_u32(acc_t, src[3], kernel_x_3_u16_);

    // 4
    acc_b = svmlalb_u32(acc_b, src[4], kernel_x_4_u16_);
    acc_t = svmlalt_u32(acc_t, src[4], kernel_x_4_u16_);

    svuint16_t acc_u16_b = svqxtnb_u32(acc_b);
    svuint16_t acc_u16 = svqxtnt_u32(acc_u16_b, acc_t);

    svbool_t greater =
        svcmpgt_n_u16(pg, acc_u16, std::numeric_limits<SourceType>::max());
    acc_u16 =
        svdup_n_u16_m(acc_u16, greater, std::numeric_limits<SourceType>::max());

    svst1b_u16(pg, &dst[0], acc_u16);
  }

  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    SourceType acc;  // NOLINT
    if (__builtin_mul_overflow(src[0], kernel_x_[0], &acc)) {
      dst[0] = std::numeric_limits<SourceType>::max();
      return;
    }

    for (size_t i = 1; i < 5; i++) {
      SourceType temp;  // NOLINT
      if (__builtin_mul_overflow(src[i], kernel_x_[i], &temp)) {
        dst[0] = std::numeric_limits<SourceType>::max();
        return;
      }
      if (__builtin_add_overflow(acc, temp, &acc)) {
        dst[0] = std::numeric_limits<SourceType>::max();
        return;
      }
    }

    dst[0] = acc;
  }

 private:
  const SourceType *kernel_x_;

  BufferVectorType &kernel_x_0_u16_;
  BufferVectorType &kernel_x_1_u16_;
  BufferVectorType &kernel_x_2_u16_;
  BufferVectorType &kernel_x_3_u16_;
  BufferVectorType &kernel_x_4_u16_;

  SourceVectorType &kernel_y_0_u8_;
  SourceVectorType &kernel_y_1_u8_;
  SourceVectorType &kernel_y_2_u8_;
  SourceVectorType &kernel_y_3_u8_;
  SourceVectorType &kernel_y_4_u8_;
};  // end of class SeparableFilter2D<uint8_t, 5>

template <>
class SeparableFilter2D<uint16_t, 5> {
 public:
  using SourceType = uint16_t;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using BufferType = uint32_t;
  using BufferVectorType = typename VecTraits<BufferType>::VectorType;
  using BufferDoubleVectorType = typename VecTraits<BufferType>::Vector2Type;
  using DestinationType = uint16_t;

  SeparableFilter2D(
      const SourceType *kernel_x, BufferVectorType &kernel_x_0_u32,
      BufferVectorType &kernel_x_1_u32, BufferVectorType &kernel_x_2_u32,
      BufferVectorType &kernel_x_3_u32, BufferVectorType &kernel_x_4_u32,
      SourceVectorType &kernel_y_0_u16, SourceVectorType &kernel_y_1_u16,
      SourceVectorType &kernel_y_2_u16, SourceVectorType &kernel_y_3_u16,
      SourceVectorType &kernel_y_4_u16)
      : kernel_x_(kernel_x),
        kernel_x_0_u32_(kernel_x_0_u32),
        kernel_x_1_u32_(kernel_x_1_u32),
        kernel_x_2_u32_(kernel_x_2_u32),
        kernel_x_3_u32_(kernel_x_3_u32),
        kernel_x_4_u32_(kernel_x_4_u32),

        kernel_y_0_u16_(kernel_y_0_u16),
        kernel_y_1_u16_(kernel_y_1_u16),
        kernel_y_2_u16_(kernel_y_2_u16),
        kernel_y_3_u16_(kernel_y_3_u16),
        kernel_y_4_u16_(kernel_y_4_u16) {}

  void vertical_vector_path(svbool_t pg,
                            std::reference_wrapper<SourceVectorType> src[5],
                            BufferType *dst) const KLEIDICV_STREAMING {
    // 0
    BufferVectorType acc_b = svmullb_u32(src[0], kernel_y_0_u16_);
    BufferVectorType acc_t = svmullt_u32(src[0], kernel_y_0_u16_);

    // 1
    BufferVectorType vec_b = svmullb_u32(src[1], kernel_y_1_u16_);
    BufferVectorType vec_t = svmullt_u32(src[1], kernel_y_1_u16_);
    acc_b = svqadd_u32_x(pg, acc_b, vec_b);
    acc_t = svqadd_u32_x(pg, acc_t, vec_t);

    // 2
    vec_b = svmullb_u32(src[2], kernel_y_2_u16_);
    vec_t = svmullt_u32(src[2], kernel_y_2_u16_);
    acc_b = svqadd_u32_x(pg, acc_b, vec_b);
    acc_t = svqadd_u32_x(pg, acc_t, vec_t);

    // 3
    vec_b = svmullb_u32(src[3], kernel_y_3_u16_);
    vec_t = svmullt_u32(src[3], kernel_y_3_u16_);
    acc_b = svqadd_u32_x(pg, acc_b, vec_b);
    acc_t = svqadd_u32_x(pg, acc_t, vec_t);

    // 4
    vec_b = svmullb_u32(src[4], kernel_y_4_u16_);
    vec_t = svmullt_u32(src[4], kernel_y_4_u16_);
    acc_b = svqadd_u32_x(pg, acc_b, vec_b);
    acc_t = svqadd_u32_x(pg, acc_t, vec_t);

    BufferDoubleVectorType interleaved = svcreate2_u32(acc_b, acc_t);
    svst2(pg, &dst[0], interleaved);
  }

  void horizontal_vector_path(svbool_t pg,
                              std::reference_wrapper<BufferVectorType> src[5],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    // 0
    svuint64_t acc_b = svmullb_u64(src[0], kernel_x_0_u32_);
    svuint64_t acc_t = svmullt_u64(src[0], kernel_x_0_u32_);

    // 1
    acc_b = svmlalb_u64(acc_b, src[1], kernel_x_1_u32_);
    acc_t = svmlalt_u64(acc_t, src[1], kernel_x_1_u32_);

    // 2
    acc_b = svmlalb_u64(acc_b, src[2], kernel_x_2_u32_);
    acc_t = svmlalt_u64(acc_t, src[2], kernel_x_2_u32_);

    // 3
    acc_b = svmlalb_u64(acc_b, src[3], kernel_x_3_u32_);
    acc_t = svmlalt_u64(acc_t, src[3], kernel_x_3_u32_);

    // 4
    acc_b = svmlalb_u64(acc_b, src[4], kernel_x_4_u32_);
    acc_t = svmlalt_u64(acc_t, src[4], kernel_x_4_u32_);

    svuint32_t acc_u32_b = svqxtnb_u64(acc_b);
    svuint32_t acc_u32 = svqxtnt_u64(acc_u32_b, acc_t);

    svbool_t greater =
        svcmpgt_n_u32(pg, acc_u32, std::numeric_limits<SourceType>::max());
    acc_u32 =
        svdup_n_u32_m(acc_u32, greater, std::numeric_limits<SourceType>::max());

    svst1h_u32(pg, &dst[0], acc_u32);
  }

  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const KLEIDICV_STREAMING {
    SourceType acc;  // Avoid cppcoreguidelines-init-variables. NOLINT
    if (__builtin_mul_overflow(src[0], kernel_x_[0], &acc)) {
      dst[0] = std::numeric_limits<SourceType>::max();
      return;
    }

    for (size_t i = 1; i < 5; i++) {
      SourceType temp;  // Avoid cppcoreguidelines-init-variables. NOLINT
      if (__builtin_mul_overflow(src[i], kernel_x_[i], &temp)) {
        dst[0] = std::numeric_limits<SourceType>::max();
        return;
      }
      if (__builtin_add_overflow(acc, temp, &acc)) {
        dst[0] = std::numeric_limits<SourceType>::max();
        return;
      }
    }

    dst[0] = acc;
  }

 private:
  const SourceType *kernel_x_;

  BufferVectorType &kernel_x_0_u32_;
  BufferVectorType &kernel_x_1_u32_;
  BufferVectorType &kernel_x_2_u32_;
  BufferVectorType &kernel_x_3_u32_;
  BufferVectorType &kernel_x_4_u32_;

  SourceVectorType &kernel_y_0_u16_;
  SourceVectorType &kernel_y_1_u16_;
  SourceVectorType &kernel_y_2_u16_;
  SourceVectorType &kernel_y_3_u16_;
  SourceVectorType &kernel_y_4_u16_;
};  // end of class SeparableFilter2D<uint16_t, 5>

template <typename T>
kleidicv_error_t separable_filter_2d_stripe_sc(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t /*kernel_width*/, const T *kernel_y,
    size_t /*kernel_height*/,
    FixedBorderType fixed_border_type) KLEIDICV_STREAMING {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  CHECK_POINTERS(kernel_x, kernel_y);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};

  using SeparableFilterClass = SeparableFilter2D<T, 5>;
  constexpr size_t intermediate_size{
      sizeof(typename SeparableFilterClass::BufferType)};

  auto workspace_variant =
      SeparableFilterWorkspace::create(rect, channels, intermediate_size);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<SeparableFilterWorkspace>(&workspace_variant);

  using WiderT = typename double_element_width<T>::type;
  using KernelXVectorTraits = VecTraits<WiderT>;
  using KernelXVectorT = typename KernelXVectorTraits::VectorType;
  using KernelYVectorTraits = VecTraits<T>;
  using KernelYVectorT = typename KernelYVectorTraits::VectorType;

  KernelXVectorT kernel_x_0 = KernelXVectorTraits::svdup(kernel_x[0]);
  KernelXVectorT kernel_x_1 = KernelXVectorTraits::svdup(kernel_x[1]);
  KernelXVectorT kernel_x_2 = KernelXVectorTraits::svdup(kernel_x[2]);
  KernelXVectorT kernel_x_3 = KernelXVectorTraits::svdup(kernel_x[3]);
  KernelXVectorT kernel_x_4 = KernelXVectorTraits::svdup(kernel_x[4]);

  KernelYVectorT kernel_y_0 = KernelYVectorTraits::svdup(kernel_y[0]);
  KernelYVectorT kernel_y_1 = KernelYVectorTraits::svdup(kernel_y[1]);
  KernelYVectorT kernel_y_2 = KernelYVectorTraits::svdup(kernel_y[2]);
  KernelYVectorT kernel_y_3 = KernelYVectorTraits::svdup(kernel_y[3]);
  KernelYVectorT kernel_y_4 = KernelYVectorTraits::svdup(kernel_y[4]);

  SeparableFilterClass filterClass{
      kernel_x,   kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_x_4,
      kernel_y_0, kernel_y_1, kernel_y_2, kernel_y_3, kernel_y_4};
  SeparableFilter<SeparableFilterClass, 5> filter{filterClass};

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};
  workspace.process(y_begin, y_end, src_rows, dst_rows, fixed_border_type,
                    filter);

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SEPARABLE_FILTER_2D_SC_H
