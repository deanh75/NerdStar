// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/filters/scharr.h"
#include "kleidicv/neon.h"
#include "kleidicv/types.h"
#include "kleidicv/utils.h"

namespace kleidicv::neon {

// Scharr filtering in both horizontal and vertical directions, horizontal and
// vertical derivative approximations are stored interleaved.
//
// The applied weights for the horizontal approximation, as the kernel is
// mirrored both vertically and horizontally during the convolution:
//      [  -3   0   3 ]   [  3 ]
//  F = [ -10   0  10 ] = [ 10 ] * [ -1, 0, 1 ]
//      [  -3   0   3 ]   [  3 ]
//
// The applied weights for the vertical approximation, as the kernel is mirrored
// both vertically and horizontally during the convolution:
//      [ -3 -10  -3 ]   [ -1 ]
//  F = [  0,  0,  0 ] = [  0 ] * [ 3, 10, 3 ]
//      [  3  10   3 ]   [  1 ]
//
class ScharrInterleaved {
  using SourceType = uint8_t;
  using SourceVecTraits = VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using SourceVector2Type = typename SourceVecTraits::Vector2Type;
  using BufferType = int16_t;
  using BufferVecTraits = VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BufferVector2Type = typename BufferVecTraits::Vector2Type;
  using BufferVector4Type = typename BufferVecTraits::Vector4Type;
  using DestinationType = int16_t;
  using DestinationVecTraits = VecTraits<DestinationType>;
  using DestinationVectorType = typename DestinationVecTraits::VectorType;
  using DestinationVector2Type = typename DestinationVecTraits::Vector2Type;

 public:
  ScharrInterleaved(Rows<BufferType> hori_deriv_buffer,
                    Rows<BufferType> vert_deriv_buffer, size_t width)
      : hori_deriv_buffer_(hori_deriv_buffer),
        vert_deriv_buffer_(vert_deriv_buffer),
        width_(width),
        const_3_s16_(vdupq_n_s16(3)),
        const_10_u8_(vdupq_n_u8(10)),
        const_10_s16_(vdupq_n_s16(10)) {}

  KLEIDICV_FORCE_INLINE void process(Rows<const SourceType> src_rows,
                                     Rows<DestinationType> dst_rows,
                                     size_t y_begin, size_t y_end) {
    for (size_t i = y_begin; i < y_end; ++i) {
      process_vertical(src_rows.at(static_cast<ptrdiff_t>(i)));
      process_horizontal(dst_rows.at(static_cast<ptrdiff_t>(i)));
    }
  }

 private:
  KLEIDICV_FORCE_INLINE BufferVector4Type
  vertical_vector_path(SourceVectorType src[3]) {
    // Horizontal derivative approximation
    uint16x8_t hori_acc_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[2]));
    uint16x8_t hori_acc_h = vaddl_high_u8(src[0], src[2]);

    hori_acc_l = vmulq_u16(hori_acc_l, const_3_s16_);
    hori_acc_h = vmulq_u16(hori_acc_h, const_3_s16_);

    hori_acc_l =
        vmlal_u8(hori_acc_l, vget_low_u8(src[1]), vget_low_u8(const_10_u8_));
    hori_acc_h = vmlal_high_u8(hori_acc_h, src[1], const_10_u8_);

    // Vertical derivative approximation
    uint16x8_t vert_acc_l = vsubl_u8(vget_low_u8(src[2]), vget_low_u8(src[0]));
    uint16x8_t vert_acc_h = vsubl_high_u8(src[2], src[0]);

    return {
        vreinterpretq_s16_u16(hori_acc_l), vreinterpretq_s16_u16(hori_acc_h),
        vreinterpretq_s16_u16(vert_acc_l), vreinterpretq_s16_u16(vert_acc_h)};
  }

  KLEIDICV_FORCE_INLINE void process_vertical(Rows<const SourceType> src_rows) {
    LoopUnroll2 loop{width_ * src_rows.channels(), kSourceVecNumLanes};

    loop.unroll_twice([&](ptrdiff_t index) {
      SourceVector2Type src[3];
      SourceVecTraits::load(&src_rows.at(0)[index], src[0]);
      SourceVecTraits::load(&src_rows.at(1)[index], src[1]);
      SourceVecTraits::load(&src_rows.at(2)[index], src[2]);

      SourceVectorType src_a[3] = {src[0].val[0], src[1].val[0], src[2].val[0]};
      BufferVector4Type res_a = vertical_vector_path(src_a);

      SourceVectorType src_b[3] = {src[0].val[1], src[1].val[1], src[2].val[1]};
      BufferVector4Type res_b = vertical_vector_path(src_b);

      BufferVector4Type hori_derivs = {res_a.val[0], res_a.val[1], res_b.val[0],
                                       res_b.val[1]};
      BufferVector4Type vert_derivs = {res_a.val[2], res_a.val[3], res_b.val[2],
                                       res_b.val[3]};

      BufferVecTraits::store(hori_derivs, &hori_deriv_buffer_[index]);
      BufferVecTraits::store(vert_derivs, &vert_deriv_buffer_[index]);
    });

    loop.unroll_once([&](ptrdiff_t index) {
      SourceVectorType src[3];
      SourceVecTraits::load(&src_rows.at(0)[index], src[0]);
      SourceVecTraits::load(&src_rows.at(1)[index], src[1]);
      SourceVecTraits::load(&src_rows.at(2)[index], src[2]);

      BufferVector4Type res = vertical_vector_path(src);

      BufferVector2Type hori_pair = {res.val[0], res.val[1]};
      BufferVector2Type vert_pair = {res.val[2], res.val[3]};

      BufferVecTraits::store(hori_pair, &hori_deriv_buffer_[index]);
      BufferVecTraits::store(vert_pair, &vert_deriv_buffer_[index]);
    });

    loop.tail([&](ptrdiff_t index) {
      hori_deriv_buffer_[index] = static_cast<BufferType>(
          (src_rows.at(0)[index] + src_rows.at(2)[index]) * 3 +
          src_rows.at(1)[index] * 10);

      vert_deriv_buffer_[index] = static_cast<BufferType>(
          src_rows.at(2)[index] - src_rows.at(0)[index]);
    });
  }

  KLEIDICV_FORCE_INLINE DestinationVectorType
  horizontal_vector_path_hori_approx(BufferVectorType buff[2]) {
    return vsubq_s16(buff[1], buff[0]);
  }

  KLEIDICV_FORCE_INLINE DestinationVectorType
  horizontal_vector_path_vert_approx(BufferVectorType buff[3]) {
    BufferVectorType a = vaddq_u16(buff[0], buff[2]);
    a = vaddq_u16(a, vaddq_u16(a, a));
    return vmlaq_u16(a, buff[1], const_10_s16_);
  }

  KLEIDICV_FORCE_INLINE void process_horizontal(
      Rows<DestinationType> dst_rows) {
    // width is decremented by 2 as the result has less columns.
    LoopUnroll2 loop{(width_ - 2) * hori_deriv_buffer_.channels(),
                     kBufferVecNumLanes};
    const ptrdiff_t channel =
        static_cast<ptrdiff_t>(hori_deriv_buffer_.channels());

    loop.unroll_twice([&](ptrdiff_t index) {
      BufferVector2Type hori_buff[2];
      BufferVecTraits::load(&hori_deriv_buffer_[index], hori_buff[0]);
      BufferVecTraits::load(&hori_deriv_buffer_[index + channel * 2],
                            hori_buff[1]);

      BufferVectorType hori_buff_a[2] = {hori_buff[0].val[0],
                                         hori_buff[1].val[0]};
      DestinationVectorType hori_approx_res_a =
          horizontal_vector_path_hori_approx(hori_buff_a);

      BufferVectorType hori_buff_b[2] = {hori_buff[0].val[1],
                                         hori_buff[1].val[1]};
      DestinationVectorType hori_approx_res_b =
          horizontal_vector_path_hori_approx(hori_buff_b);

      BufferVector2Type vert_buff[3];
      BufferVecTraits::load(&vert_deriv_buffer_[index], vert_buff[0]);
      BufferVecTraits::load(&vert_deriv_buffer_[index + channel], vert_buff[1]);
      BufferVecTraits::load(&vert_deriv_buffer_[index + channel * 2],
                            vert_buff[2]);

      BufferVectorType vert_buff_a[3] = {
          vert_buff[0].val[0], vert_buff[1].val[0], vert_buff[2].val[0]};
      DestinationVectorType vert_approx_res_a =
          horizontal_vector_path_vert_approx(vert_buff_a);

      BufferVectorType vert_buff_b[3] = {
          vert_buff[0].val[1], vert_buff[1].val[1], vert_buff[2].val[1]};
      DestinationVectorType vert_approx_res_b =
          horizontal_vector_path_vert_approx(vert_buff_b);

      vst2q(&dst_rows[index * 2], {hori_approx_res_a, vert_approx_res_a});
      vst2q(&dst_rows[(index + kBufferVecNumLanes) * 2],
            {hori_approx_res_b, vert_approx_res_b});
    });

    loop.unroll_once([&](ptrdiff_t index) {
      BufferVectorType hori_buff[2];
      BufferVecTraits::load(&hori_deriv_buffer_[index], hori_buff[0]);
      BufferVecTraits::load(&hori_deriv_buffer_[index + channel * 2],
                            hori_buff[1]);
      DestinationVectorType hori_approx_res =
          horizontal_vector_path_hori_approx(hori_buff);

      BufferVectorType vert_buff[3];
      BufferVecTraits::load(&vert_deriv_buffer_[index], vert_buff[0]);
      BufferVecTraits::load(&vert_deriv_buffer_[index + channel], vert_buff[1]);
      BufferVecTraits::load(&vert_deriv_buffer_[index + channel * 2],
                            vert_buff[2]);
      DestinationVectorType vert_approx_res =
          horizontal_vector_path_vert_approx(vert_buff);

      vst2q(&dst_rows[index * 2], {hori_approx_res, vert_approx_res});
    });

    loop.tail([&](ptrdiff_t index) {
      dst_rows[index * 2] = static_cast<DestinationType>(
          // For some reason clang-tidy thinks these accesses are invalid
          // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign,
          // clang-analyzer-core.UndefinedBinaryOperatorResult)
          hori_deriv_buffer_[index + channel * 2] - hori_deriv_buffer_[index]);
      // NOLINTEND(clang-analyzer-core.uninitialized.Assign,
      // clang-analyzer-core.UndefinedBinaryOperatorResult)

      dst_rows[index * 2 + 1] = static_cast<DestinationType>(
          (vert_deriv_buffer_[index] +
           vert_deriv_buffer_[index + channel * 2]) *
              3 +
          vert_deriv_buffer_[index + channel] * 10);
    });
  }

  Rows<BufferType> hori_deriv_buffer_;
  Rows<BufferType> vert_deriv_buffer_;
  size_t width_;
  int16x8_t const_3_s16_;
  uint8x16_t const_10_u8_;
  int16x8_t const_10_s16_;

  static constexpr ptrdiff_t kSourceVecNumLanes =
      static_cast<ptrdiff_t>(SourceVecTraits::num_lanes());
  static constexpr ptrdiff_t kBufferVecNumLanes =
      static_cast<ptrdiff_t>(BufferVecTraits::num_lanes());
};  // end of class ScharrInterleaved

class ScharrBufferDeleter {
 public:
  void operator()(void *ptr) const { std::free(ptr); }
};

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end) {
  // Does not include checks for whether the operation is implemented.
  // This must be done earlier, by scharr_interleaved_is_implemented.
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, src_height);
  CHECK_IMAGE_SIZE(src_width, src_height);

  size_t buffer_stride = src_width * src_channels * sizeof(int16_t);
  // Buffer has two rows, one for the horizontal derivative approximation, one
  // for the vertical one.
  size_t buffer_height = 2;
  // Memory is allocated with malloc to avoid its initialization.
  void *allocation = std::malloc(buffer_stride * buffer_height);

  if (!allocation) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  std::unique_ptr<int16_t, ScharrBufferDeleter> buffer(
      reinterpret_cast<int16_t *>(allocation));

  Rows<const uint8_t> src_rows{src, src_stride, src_channels};

  // Result is treated as it has double the channel number compared to the
  // input.
  Rows<int16_t> dst_rows{dst, dst_stride, src_channels * 2};

  Rows<int16_t> hori_deriv_buffer{buffer.get(), buffer_stride, src_channels};

  int16_t *vert_deriv_ptr = reinterpret_cast<int16_t *>(
      reinterpret_cast<uint8_t *>(buffer.get()) + buffer_stride);
  Rows<int16_t> vert_deriv_buffer{vert_deriv_ptr, buffer_stride, src_channels};

  ScharrInterleaved(hori_deriv_buffer, vert_deriv_buffer, src_width)
      .process(src_rows, dst_rows, y_begin, y_end);

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
