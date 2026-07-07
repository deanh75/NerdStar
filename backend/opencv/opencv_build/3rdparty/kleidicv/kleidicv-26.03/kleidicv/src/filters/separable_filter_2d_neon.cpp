// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/separable_filter_2d.h"
#include "kleidicv/filters/separable_filter_5x5_neon.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/workspace/separable.h"

namespace kleidicv::neon {

template <typename ScalarType, size_t KernelSize>
class SeparableFilter2D;

template <>
class SeparableFilter2D<uint8_t, 5> {
 public:
  using SourceType = uint8_t;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using BufferType = uint16_t;
  using BufferVectorType = typename VecTraits<BufferType>::VectorType;
  using DestinationType = uint8_t;

  // Ignored because vectors are initialized in the constructor body.
  // NOLINTNEXTLINE - hicpp-member-init
  SeparableFilter2D(const SourceType *kernel_x, const SourceType *kernel_y)
      : kernel_x_(kernel_x), kernel_y_(kernel_y) {
    for (size_t i = 0; i < 5; i++) {
      kernel_x_u16_[i] = vdupq_n_u16(kernel_x[i]);
      kernel_y_u8_[i] = vdupq_n_u8(kernel_y[i]);
    }
  }

  void vertical_vector_path(SourceVectorType src[5], BufferType *dst) const {
    BufferVectorType acc_l =
        vmull_u8(vget_low_u8(src[0]), vget_low_u8(kernel_y_u8_[0]));
    BufferVectorType acc_h = vmull_high_u8(src[0], kernel_y_u8_[0]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 1; i < 5; i++) {
      BufferVectorType vec_l =
          vmull_u8(vget_low_u8(src[i]), vget_low_u8(kernel_y_u8_[i]));
      BufferVectorType vec_h = vmull_high_u8(src[i], kernel_y_u8_[i]);

      acc_l = vqaddq_u16(acc_l, vec_l);
      acc_h = vqaddq_u16(acc_h, vec_h);
    }

    vst1q_u16(&dst[0], acc_l);
    vst1q_u16(&dst[8], acc_h);
  }

  void vertical_scalar_path(const SourceType src[5], BufferType *dst) const {
    BufferType acc = static_cast<BufferType>(src[0]) * kernel_y_[0];
    for (size_t i = 1; i < 5; i++) {
      BufferType temp = static_cast<BufferType>(src[i]) * kernel_y_[i];
      if (__builtin_add_overflow(acc, temp, &acc)) {
        dst[0] = std::numeric_limits<SourceType>::max();
        return;
      }
    }

    dst[0] = acc;
  }

  void horizontal_vector_path(BufferVectorType src[5],
                              DestinationType *dst) const {
    uint32x4_t acc_l =
        vmull_u16(vget_low_u16(src[0]), vget_low_u16(kernel_x_u16_[0]));
    uint32x4_t acc_h = vmull_high_u16(src[0], kernel_x_u16_[0]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 1; i < 5; i++) {
      acc_l = vmlal_u16(acc_l, vget_low_u16(src[i]),
                        vget_low_u16(kernel_x_u16_[i]));
      acc_h = vmlal_high_u16(acc_h, src[i], kernel_x_u16_[i]);
    }

    uint16x8_t acc_u16 = vcombine_u16(vqmovn_u32(acc_l), vqmovn_u32(acc_h));
    uint8x8_t result = vqmovn_u16(acc_u16);
    vst1_u8(&dst[0], result);
  }

  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const {
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
  const SourceType *kernel_y_;

  BufferVectorType kernel_x_u16_[5];
  SourceVectorType kernel_y_u8_[5];
};  // end of class SeparableFilter2D<uint8_t, 5>

template <>
class SeparableFilter2D<uint16_t, 5> {
 public:
  using SourceType = uint16_t;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using BufferType = uint32_t;
  using BufferVectorType = typename VecTraits<BufferType>::VectorType;
  using DestinationType = uint16_t;

  // Ignored because vectors are initialized in the constructor body.
  // NOLINTNEXTLINE - hicpp-member-init
  SeparableFilter2D(const SourceType *kernel_x, const SourceType *kernel_y)
      : kernel_x_(kernel_x), kernel_y_(kernel_y) {
    for (size_t i = 0; i < 5; i++) {
      kernel_x_u32_[i] = vdupq_n_u32(kernel_x[i]);
      kernel_y_u16_[i] = vdupq_n_u16(kernel_y[i]);
    }
  }

  void vertical_vector_path(SourceVectorType src[5], BufferType *dst) const {
    BufferVectorType acc_l =
        vmull_u16(vget_low_u16(src[0]), vget_low_u16(kernel_y_u16_[0]));
    BufferVectorType acc_h = vmull_high_u16(src[0], kernel_y_u16_[0]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 1; i < 5; i++) {
      BufferVectorType vec_l =
          vmull_u16(vget_low_u16(src[i]), vget_low_u16(kernel_y_u16_[i]));
      BufferVectorType vec_h = vmull_high_u16(src[i], kernel_y_u16_[i]);

      acc_l = vqaddq_u32(acc_l, vec_l);
      acc_h = vqaddq_u32(acc_h, vec_h);
    }

    vst1q_u32(&dst[0], acc_l);
    vst1q_u32(&dst[4], acc_h);
  }

  void vertical_scalar_path(const SourceType src[5], BufferType *dst) const {
    BufferType acc = static_cast<BufferType>(src[0]) * kernel_y_[0];
    for (size_t i = 1; i < 5; i++) {
      BufferType temp = static_cast<BufferType>(src[i]) * kernel_y_[i];
      if (__builtin_add_overflow(acc, temp, &acc)) {
        dst[0] = std::numeric_limits<SourceType>::max();
        return;
      }
    }

    dst[0] = acc;
  }

  void horizontal_vector_path(BufferVectorType src[5],
                              DestinationType *dst) const {
    uint64x2_t acc_l =
        vmull_u32(vget_low_u32(src[0]), vget_low_u32(kernel_x_u32_[0]));
    uint64x2_t acc_h = vmull_high_u32(src[0], kernel_x_u32_[0]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 1; i < 5; i++) {
      acc_l = vmlal_u32(acc_l, vget_low_u32(src[i]),
                        vget_low_u32(kernel_x_u32_[i]));
      acc_h = vmlal_high_u32(acc_h, src[i], kernel_x_u32_[i]);
    }

    uint32x4_t acc_u32 = vcombine_u32(vqmovn_u64(acc_l), vqmovn_u64(acc_h));
    uint16x4_t result = vqmovn_u32(acc_u32);
    vst1_u16(&dst[0], result);
  }

  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const {
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
  const SourceType *kernel_y_;

  BufferVectorType kernel_x_u32_[5];
  SourceVectorType kernel_y_u16_[5];
};  // end of class SeparableFilter2D<uint16_t, 5>

template <typename T>
kleidicv_error_t separable_filter_2d_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t /*kernel_width*/, const T *kernel_y,
    size_t /*kernel_height*/, FixedBorderType fixed_border_type) {
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

  SeparableFilterClass filterClass{kernel_x, kernel_y};
  SeparableFilter<SeparableFilterClass, 5> filter{filterClass};

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};
  workspace.process(y_begin, y_end, src_rows, dst_rows, fixed_border_type,
                    filter);

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                    \
  separable_filter_2d_stripe<type>(                                     \
      const type *src, size_t src_stride, type *dst, size_t dst_stride, \
      size_t width, size_t height, size_t y_begin, size_t y_end,        \
      size_t channels, const type *kernel_x, size_t kernel_width,       \
      const type *kernel_y, size_t kernel_height, FixedBorderType border_type)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);

}  // namespace kleidicv::neon
