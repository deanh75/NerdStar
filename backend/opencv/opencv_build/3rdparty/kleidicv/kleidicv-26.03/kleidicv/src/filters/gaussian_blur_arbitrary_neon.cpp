// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstddef>

#include "border_generic_neon.h"
#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/neon.h"
#include "kleidicv/workspace/border_types.h"
#include "kleidicv/workspace/separable.h"

namespace kleidicv::neon {

// Template for arbitrary kernel size Gaussian Blur filters.
template <typename ScalarType, FixedBorderType>
class GaussianBlurArbitrary;

template <FixedBorderType BorderT>
class GaussianBlurArbitrary<uint8_t, BorderT> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint8_t;
  using DestinationType = uint8_t;
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits = typename neon::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderType = FixedBorderType;

  GaussianBlurArbitrary(const uint8_t *half_kernel, ptrdiff_t half_kernel_size,
                        Rectangle &rect, size_t channels)
      : half_kernel_size_(half_kernel_size),
        half_kernel_(half_kernel),
        width_(static_cast<ptrdiff_t>(rect.width())),
        vertical_border_(rect.height()),
        horizontal_border_(rect.width(), channels) {}

  // Not border-affected parts
  void process_arbitrary_vertical(size_t width, Rows<const SourceType> src_rows,
                                  Rows<BufferType> buffer_rows) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) {
      vertical_vector_path(src_rows, buffer_rows, index);
    });

    loop.tail([&](size_t index) {
      vertical_scalar_path(src_rows, buffer_rows, index);
    });
  }

  // Border-affected parts
  void process_arbitrary_border_vertical(size_t width,
                                         Rows<const SourceType> src_rows,
                                         ptrdiff_t row_index,
                                         Rows<BufferType> buffer_rows) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t column_index) {
      vertical_border_vector_path(src_rows, buffer_rows, row_index,
                                  column_index);
    });

    loop.tail([&](size_t column_index) {
      vertical_border_scalar_path(src_rows, buffer_rows, row_index,
                                  column_index);
    });
  }

  void process_arbitrary_horizontal(
      size_t width, size_t kernel_size, Rows<BufferType> buffer_rows,
      Rows<DestinationType> dst_rows) KLEIDICV_STREAMING {
    size_t x = 0;
    // Assume that there is always a widening when calculating, so the
    // horizontal vector path processes double-width vectors
    const size_t num_lanes = BufferVecTraits::num_lanes() / 2;
    const size_t block_len = num_lanes;
    const size_t margin = kernel_size / 2;
    const size_t border_len = buffer_rows.channels() * margin;
    const size_t border_process_len =
        ((border_len + block_len - 1) / block_len) * block_len;

    for (; x < border_process_len; x += num_lanes) {
      horizontal_left_border_vector_path(buffer_rows, dst_rows, x);
    }

    // Process data which is not affected by any borders in bulk.
    if (width * buffer_rows.channels() > 2 * border_process_len) {
      size_t total_width_without_borders =
          width * buffer_rows.channels() - 2 * border_process_len;

      LoopUnroll2<TryToAvoidTailLoop> loop{total_width_without_borders,
                                           BufferVecTraits::num_lanes()};

      loop.unroll_twice([&](size_t index) {
        horizontal_vector_path(buffer_rows, dst_rows, x + index);
        horizontal_vector_path(buffer_rows, dst_rows,
                               x + index + BufferVecTraits::num_lanes());
      });

      loop.unroll_once([&](size_t index) {
        horizontal_vector_path(buffer_rows, dst_rows, x + index);
      });

      loop.tail([&](size_t index) {
        horizontal_scalar_path(buffer_rows, dst_rows, x + index);
      });

      x += total_width_without_borders;
    } else {
      // rewind if needed, so we'll have exact vector paths at the right side
      x = width * buffer_rows.channels() - border_process_len;
    }

    for (; x < width * buffer_rows.channels(); x += num_lanes) {
      horizontal_right_border_vector_path(buffer_rows, dst_rows, x);
    }
  }

 private:
  void vertical_vector_path(Rows<const SourceType> src_rows,
                            Rows<BufferType> dst_rows, ptrdiff_t x) const {
    uint8x16_t src_mid = vld1q_u8(&src_rows[x]);
    uint8x8_t half_kernel_mid = vdup_n_u8(half_kernel_[half_kernel_size_ - 1]);
    uint16x8_t acc_l = vmull_u8(vget_low_u8(src_mid), half_kernel_mid);
    uint16x8_t acc_h = vmull_u8(vget_high_u8(src_mid), half_kernel_mid);

    ptrdiff_t i = 0;
    // Unroll 4 times
    for (; i < half_kernel_size_ - 4; i += 4) {
      uint8x16_t src_i = vld1q_u8(&src_rows.at(i - half_kernel_size_ + 1)[x]);
      uint8x16_t src_j = vld1q_u8(&src_rows.at(half_kernel_size_ - i - 1)[x]);
      uint16x8_t vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      uint16x8_t vec_h = vaddl_high_u8(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      uint16x8_t prod0_l = vmulq_u16(vec_l, coeff);
      uint16x8_t prod0_h = vmulq_u16(vec_h, coeff);

      src_i = vld1q_u8(&src_rows.at(i + 2 - half_kernel_size_)[x]);
      src_j = vld1q_u8(&src_rows.at(half_kernel_size_ - i - 2)[x]);
      vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      vec_h = vaddl_high_u8(src_i, src_j);
      coeff = vdupq_n_u16(half_kernel_[i + 1]);
      uint16x8_t prod1_l = vmulq_u16(vec_l, coeff);
      uint16x8_t prod1_h = vmulq_u16(vec_h, coeff);

      src_i = vld1q_u8(&src_rows.at(i + 3 - half_kernel_size_)[x]);
      src_j = vld1q_u8(&src_rows.at(half_kernel_size_ - i - 3)[x]);
      vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      vec_h = vaddl_high_u8(src_i, src_j);
      coeff = vdupq_n_u16(half_kernel_[i + 2]);
      uint16x8_t prod2_l = vmulq_u16(vec_l, coeff);
      uint16x8_t prod2_h = vmulq_u16(vec_h, coeff);

      src_i = vld1q_u8(&src_rows.at(i + 4 - half_kernel_size_)[x]);
      src_j = vld1q_u8(&src_rows.at(half_kernel_size_ - i - 4)[x]);
      vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      vec_h = vaddl_high_u8(src_i, src_j);
      coeff = vdupq_n_u16(half_kernel_[i + 3]);
      uint16x8_t prod3_l = vmulq_u16(vec_l, coeff);
      uint16x8_t prod3_h = vmulq_u16(vec_h, coeff);

      uint16x8_t acc0_l = vaddq_u16(prod0_l, prod1_l);
      uint16x8_t acc0_h = vaddq_u16(prod0_h, prod1_h);
      uint16x8_t acc1_l = vaddq_u16(prod2_l, prod3_l);
      uint16x8_t acc1_h = vaddq_u16(prod2_h, prod3_h);

      uint16x8_t acc_new_l = vaddq_u16(acc0_l, acc1_l);
      uint16x8_t acc_new_h = vaddq_u16(acc0_h, acc1_h);

      acc_l = vaddq_u16(acc_l, acc_new_l);
      acc_h = vaddq_u16(acc_h, acc_new_h);
    }

    for (; i < half_kernel_size_ - 1; ++i) {
      uint8x16_t src_i = vld1q_u8(&src_rows.at(i - half_kernel_size_ + 1)[x]);
      uint8x16_t src_j = vld1q_u8(&src_rows.at(half_kernel_size_ - i - 1)[x]);
      uint16x8_t vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      uint16x8_t vec_h = vaddl_high_u8(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc_l = vmlaq_u16(acc_l, vec_l, coeff);
      acc_h = vmlaq_u16(acc_h, vec_h, coeff);
    }

    // Rounding
    acc_l = vqaddq_u16(acc_l, vdupq_n_u16(128));
    acc_h = vqaddq_u16(acc_h, vdupq_n_u16(128));
    // Keep only the highest 8 bits
    uint8x16_t result =
        vuzp2q_u8(vreinterpretq_u8_u16(acc_l), vreinterpretq_u8_u16(acc_h));
    neon::VecTraits<uint8_t>::store(result, &dst_rows[x]);
  }

  // Where y is affected by border
  void vertical_border_vector_path(Rows<const SourceType> src_rows,
                                   Rows<BufferType> dst_rows, ptrdiff_t y,
                                   ptrdiff_t x) const {
    uint8x16_t src_mid = vld1q_u8(&src_rows.at(y)[x]);
    uint8x8_t half_kernel_mid = vdup_n_u8(half_kernel_[half_kernel_size_ - 1]);
    uint16x8_t acc_l = vmull_u8(vget_low_u8(src_mid), half_kernel_mid);
    uint16x8_t acc_h = vmull_u8(vget_high_u8(src_mid), half_kernel_mid);

    ptrdiff_t i = 0;
    for (; i < half_kernel_size_ - 1; ++i) {
      uint8x16_t src_i = vld1q_u8(&src_rows.at(
          vertical_border_.get_row(y - half_kernel_size_ + 1 + i))[x]);
      uint8x16_t src_j = vld1q_u8(&src_rows.at(
          vertical_border_.get_row(y + half_kernel_size_ - 1 - i))[x]);
      uint16x8_t vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      uint16x8_t vec_h = vaddl_high_u8(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc_l = vmlaq_u16(acc_l, vec_l, coeff);
      acc_h = vmlaq_u16(acc_h, vec_h, coeff);
    }

    // Rounding
    acc_l = vqaddq_u16(acc_l, vdupq_n_u16(128));
    acc_h = vqaddq_u16(acc_h, vdupq_n_u16(128));
    // Keep only the highest 8 bits
    uint8x16_t result =
        vuzp2q_u8(vreinterpretq_u8_u16(acc_l), vreinterpretq_u8_u16(acc_h));
    neon::VecTraits<uint8_t>::store(result, &dst_rows[x]);
  }

  void vertical_scalar_path(Rows<const SourceType> src_rows,
                            Rows<BufferType> dst_rows, ptrdiff_t x) const {
    uint32_t acc = static_cast<uint32_t>(src_rows[x]) *
                   half_kernel_[half_kernel_size_ - 1];

    for (ptrdiff_t i = 0; i < half_kernel_size_ - 1; i++) {
      acc += (src_rows.at(i + 1 - half_kernel_size_)[x] +
              src_rows.at(half_kernel_size_ - i - 1)[x]) *
             half_kernel_[i];
    }

    dst_rows[x] = static_cast<BufferType>(rounding_shift_right(acc, 8));
  }

  void vertical_border_scalar_path(Rows<const SourceType> src_rows,
                                   Rows<BufferType> dst_rows, ptrdiff_t y,
                                   ptrdiff_t x) const {
    uint32_t acc = src_rows.at(y)[x] * half_kernel_[half_kernel_size_ - 1];

    for (ptrdiff_t i = 0; i < half_kernel_size_ - 1; i++) {
      acc += (src_rows.at(
                  vertical_border_.get_row(y + i + 1 - half_kernel_size_))[x] +
              src_rows.at(
                  vertical_border_.get_row(y + half_kernel_size_ - i - 1))[x]) *
             half_kernel_[i];
    }

    dst_rows[x] = static_cast<BufferType>(rounding_shift_right(acc, 8));
  }

  void horizontal_vector_path(Rows<BufferType> src_rows,
                              Rows<DestinationType> dst_rows,
                              ptrdiff_t x) const {
    // very similar to the vertical path, the difference is only the loading
    // pattern
    uint8x16_t src_mid = vld1q_u8(&src_rows[x]);
    uint8x8_t half_kernel_mid = vdup_n_u8(half_kernel_[half_kernel_size_ - 1]);
    uint16x8_t acc_l = vmull_u8(vget_low_u8(src_mid), half_kernel_mid);
    uint16x8_t acc_h = vmull_u8(vget_high_u8(src_mid), half_kernel_mid);

    ptrdiff_t ch = static_cast<ptrdiff_t>(src_rows.channels()),
              left = x - ch * (half_kernel_size_ - 1),
              right = x + ch * (half_kernel_size_ - 1);
    for (ptrdiff_t i = 0; i < half_kernel_size_ - 1; ++i) {
      uint8x16_t src_i = vld1q_u8(&src_rows[left + i * ch]);
      uint8x16_t src_j = vld1q_u8(&src_rows[right - i * ch]);
      uint16x8_t vec_l = vaddl_u8(vget_low_u8(src_i), vget_low_u8(src_j));
      uint16x8_t vec_h = vaddl_high_u8(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc_l = vmlaq_u16(acc_l, vec_l, coeff);
      acc_h = vmlaq_u16(acc_h, vec_h, coeff);
    }

    // Rounding
    acc_l = vqaddq_u16(acc_l, vdupq_n_u16(128));
    acc_h = vqaddq_u16(acc_h, vdupq_n_u16(128));
    // Keep only the highest 8 bits
    uint8x16_t result =
        vuzp2q_u8(vreinterpretq_u8_u16(acc_l), vreinterpretq_u8_u16(acc_h));
    neon::VecTraits<uint8_t>::store(result, &dst_rows[x]);
  }

  void horizontal_left_border_vector_path(Rows<BufferType> src_rows,
                                          Rows<DestinationType> dst_rows,
                                          ptrdiff_t x) const {
    // similar to the simple horizontal path, except the loading pattern:
    // - this is loading indirect columns, and half of that data
    uint16x8_t src_mid = vmovl_u8(vld1_u8(&src_rows[x]));
    uint16x8_t acc = vmulq_n_u16(src_mid, half_kernel_[half_kernel_size_ - 1]);

    ptrdiff_t ch = static_cast<ptrdiff_t>(src_rows.channels());
    ptrdiff_t i = 0, left = x - ch * (half_kernel_size_ - 1),
              right = x + ch * (half_kernel_size_ - 1);
    for (; i * ch + left < 0; ++i) {
      uint16x8_t src_i = horizontal_border_.load_left(src_rows, left + i * ch);
      uint16x8_t src_j = vmovl_u8(vld1_u8(&src_rows[right - i * ch]));
      uint16x8_t vec = vaddq_u16(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc = vmlaq_u16(acc, vec, coeff);
    }

    for (; i < half_kernel_size_ - 1; ++i) {
      uint16x8_t src_i = vmovl_u8(vld1_u8(&src_rows[left + i * ch]));
      uint16x8_t src_j = vmovl_u8(vld1_u8(&src_rows[right - i * ch]));
      uint16x8_t vec = vaddq_u16(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc = vmlaq_u16(acc, vec, coeff);
    }

    // Store only the highest 8 bits
    uint8x8_t result = vrshrn_n_u16(acc, 8);
    vst1_u8(&dst_rows[x], result);
  }

  void horizontal_right_border_vector_path(Rows<BufferType> src_rows,
                                           Rows<DestinationType> dst_rows,
                                           ptrdiff_t x) const {
    // similar to the simple horizontal path, except the loading pattern:
    // - this is loading indirect columns, and half of that data
    uint16x8_t src_mid = vmovl_u8(vld1_u8(&src_rows[x]));
    uint16x8_t acc = vmulq_n_u16(src_mid, half_kernel_[half_kernel_size_ - 1]);

    ptrdiff_t ch = static_cast<ptrdiff_t>(src_rows.channels());
    ptrdiff_t i = 0, left = x - ch * (half_kernel_size_ - 1),
              right = x + ch * (half_kernel_size_ - 1);
    for (; right - i * ch > width_ * ch - 8; ++i) {
      uint16x8_t src_i = vmovl_u8(vld1_u8(&src_rows[left + i * ch]));
      uint16x8_t src_j =
          horizontal_border_.load_right(src_rows, right - i * ch);
      uint16x8_t vec = vaddq_u16(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc = vmlaq_u16(acc, vec, coeff);
    }

    for (; i < half_kernel_size_ - 1; ++i) {
      uint16x8_t src_i = vmovl_u8(vld1_u8(&src_rows[left + i * ch]));
      uint16x8_t src_j = vmovl_u8(vld1_u8(&src_rows[right - i * ch]));
      uint16x8_t vec = vaddq_u16(src_i, src_j);
      uint16x8_t coeff = vdupq_n_u16(half_kernel_[i]);
      acc = vmlaq_u16(acc, vec, coeff);
    }

    // Store only the highest 8 bits
    uint8x8_t result = vrshrn_n_u16(acc, 8);
    vst1_u8(&dst_rows[x], result);
  }

  void horizontal_scalar_path(Rows<BufferType> src_rows,
                              Rows<DestinationType> dst_rows,
                              ptrdiff_t x) const {
    uint32_t acc = static_cast<uint32_t>(src_rows[x]) *
                   half_kernel_[half_kernel_size_ - 1];
    ptrdiff_t ch = static_cast<ptrdiff_t>(src_rows.channels());
    ptrdiff_t channel_offset = x % ch;
    ptrdiff_t left_col = x / ch - (half_kernel_size_ - 1),
              right_col = x / ch + (half_kernel_size_ - 1);

    for (ptrdiff_t i = 0; i < half_kernel_size_ - 1; i++) {
      acc += (static_cast<uint32_t>(
                  src_rows[horizontal_border_.get_column(left_col + i) * ch +
                           channel_offset]) +
              static_cast<uint32_t>(
                  src_rows[horizontal_border_.get_column(right_col - i) * ch +
                           channel_offset])) *
             half_kernel_[i];
    }

    dst_rows[x] = static_cast<DestinationType>(rounding_shift_right(acc, 8));
  }

  const ptrdiff_t half_kernel_size_;
  const uint8_t *half_kernel_;
  const ptrdiff_t width_;
  KLEIDICV_TARGET_NAMESPACE::GenericBorderVertical<BorderT> vertical_border_;
  KLEIDICV_TARGET_NAMESPACE::GenericBorderHorizontal<BorderT>
      horizontal_border_;
};  // end of class GaussianBlurArbitrary<uint8_t>

template <typename ScalarType>
static kleidicv_error_t gaussian_blur_arbitrary_kernel_size(
    const ScalarType *src, size_t src_stride, ScalarType *dst,
    size_t dst_stride, Rectangle &rect, size_t kernel_size, size_t y_begin,
    size_t y_end, size_t channels, float sigma, FixedBorderType border_type) {
  Rows<const ScalarType> src_rows{src, src_stride, channels};
  Rows<ScalarType> dst_rows{dst, dst_stride, channels};
  // Only replicated border is implemented so far.
  using GaussianBlurFilter =
      GaussianBlurArbitrary<ScalarType, FixedBorderType::REPLICATE>;
  constexpr size_t intermediate_size{
      sizeof(typename GaussianBlurFilter::BufferType)};

  auto workspace_variant =
      SeparableFilterWorkspace::create(rect, channels, intermediate_size);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<SeparableFilterWorkspace>(&workspace_variant);

  const ptrdiff_t kHalfKernelSize =
      static_cast<ptrdiff_t>(get_half_kernel_size(kernel_size));
  uint8_t half_kernel[128];
  bool success =
      generate_gaussian_half_kernel(half_kernel, kHalfKernelSize, sigma);
  if (success) {
    GaussianBlurFilter filter{half_kernel, kHalfKernelSize, rect,
                              src_rows.channels()};
    workspace.process_arbitrary(kernel_size, y_begin, y_end, src_rows, dst_rows,
                                border_type, filter);
  } else {
    // Sigma is too small that the middle point would get all the weight
    // => it's just a copy.
    for (size_t row = y_begin; row < y_end; ++row) {
      std::memcpy(static_cast<void *>(&dst_rows.at(row)[0]),
                  static_cast<const void *>(&src_rows.at(row)[0]),
                  rect.width() * sizeof(ScalarType) * dst_rows.channels());
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gaussian_blur_arbitrary_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t /*kernel_height*/, float sigma_x,
    float /*sigma_y*/, FixedBorderType fixed_border_type) {
  if (auto result =
          gaussian_blur_checks(src, src_stride, dst, dst_stride, width, height);
      result != KLEIDICV_OK) {
    return result;
  }

  Rectangle rect{width, height};

  return gaussian_blur_arbitrary_kernel_size(
      src, src_stride, dst, dst_stride, rect, kernel_width, y_begin, y_end,
      channels, sigma_x, fixed_border_type);
}

}  // namespace kleidicv::neon
