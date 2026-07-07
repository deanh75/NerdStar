// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/sve2.h"
#include "kleidicv/types.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

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
  using SourceVector3Type = typename SourceVecTraits::Vector3Type;
  using BufferType = int16_t;
  using BufferVecTraits = VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BufferVector2Type = typename BufferVecTraits::Vector2Type;
  using BufferVector3Type = typename BufferVecTraits::Vector3Type;
  using DestinationType = int16_t;
  using DestinationVecTraits = VecTraits<DestinationType>;
  using DestinationVectorType = typename DestinationVecTraits::VectorType;

 public:
  ScharrInterleaved(Rows<BufferType> hori_deriv_buffer,
                    Rows<BufferType> vert_deriv_buffer,
                    size_t width) KLEIDICV_STREAMING
      : hori_deriv_buffer_(hori_deriv_buffer),
        vert_deriv_buffer_(vert_deriv_buffer),
        width_(width) {}

  void process(Rows<const SourceType> src_rows, Rows<DestinationType> dst_rows,
               size_t y_begin, size_t y_end) KLEIDICV_STREAMING {
    for (size_t i = y_begin; i < y_end; ++i) {
      process_vertical(src_rows.at(static_cast<ptrdiff_t>(i)));
      process_horizontal(dst_rows.at(static_cast<ptrdiff_t>(i)));
    }
  }

 private:
  void vertical_vector_path_x2(svbool_t pg, Rows<const SourceType> src_rows,
                               ptrdiff_t index) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    svcount_t pc8 = SourceVecTraits::svptrue_c();
    SourceVector2Type src0 = svld1_x2(pc8, &src_rows.at(0)[index]);
    SourceVector2Type src1 = svld1_x2(pc8, &src_rows.at(1)[index]);
    SourceVector2Type src2 = svld1_x2(pc8, &src_rows.at(2)[index]);
    SourceVector3Type src_a =
        svcreate3(svget2(src0, 0), svget2(src1, 0), svget2(src2, 0));
    SourceVector3Type src_b =
        svcreate3(svget2(src0, 1), svget2(src1, 1), svget2(src2, 1));
#else
    SourceVector3Type src_a = svcreate3(svld1(pg, &src_rows.at(0)[index]),
                                        svld1(pg, &src_rows.at(1)[index]),
                                        svld1(pg, &src_rows.at(2)[index]));
    SourceVector3Type src_b =
        svcreate3(svld1_vnum(pg, &src_rows.at(0)[index], 1),
                  svld1_vnum(pg, &src_rows.at(1)[index], 1),
                  svld1_vnum(pg, &src_rows.at(2)[index], 1));
#endif

    svint16x2_t hori_interleaved_a = vertical_compute_hori_approx(pg, src_a);
    svint16x2_t vert_interleaved_a = vertical_compute_vert_approx(src_a);

    svint16x2_t hori_interleaved_b = vertical_compute_hori_approx(pg, src_b);
    svint16x2_t vert_interleaved_b = vertical_compute_vert_approx(src_b);

    svst2(pg, &hori_deriv_buffer_[index], hori_interleaved_a);
    svst2_vnum(pg, &hori_deriv_buffer_[index], 2, hori_interleaved_b);

    svst2(pg, &vert_deriv_buffer_[index], vert_interleaved_a);
    svst2_vnum(pg, &vert_deriv_buffer_[index], 2, vert_interleaved_b);
  }

  void vertical_vector_path(svbool_t pg, Rows<const SourceType> src_rows,
                            ptrdiff_t index) KLEIDICV_STREAMING {
    SourceVectorType src_0 = svld1(pg, &src_rows.at(0)[index]);
    SourceVectorType src_1 = svld1(pg, &src_rows.at(1)[index]);
    SourceVectorType src_2 = svld1(pg, &src_rows.at(2)[index]);

    SourceVector3Type src = svcreate3(src_0, src_1, src_2);
    svint16x2_t hori_interleaved = vertical_compute_hori_approx(pg, src);
    svint16x2_t vert_interleaved = vertical_compute_vert_approx(src);

    svst2(pg, &hori_deriv_buffer_[index], hori_interleaved);
    svst2(pg, &vert_deriv_buffer_[index], vert_interleaved);
  }

  svint16x2_t vertical_compute_hori_approx(svbool_t pg, SourceVector3Type src)
      KLEIDICV_STREAMING {
    SourceVectorType src_0 = svget3(src, 0);
    SourceVectorType src_1 = svget3(src, 1);
    SourceVectorType src_2 = svget3(src, 2);
    svuint16_t hori_acc_b = svaddlb(src_0, src_2);
    svuint16_t hori_acc_t = svaddlt(src_0, src_2);

    hori_acc_b = svmul_n_u16_x(pg, hori_acc_b, 3);
    hori_acc_t = svmul_n_u16_x(pg, hori_acc_t, 3);

    hori_acc_b = svmlalb_n_u16(hori_acc_b, src_1, 10);
    hori_acc_t = svmlalt_n_u16(hori_acc_t, src_1, 10);

    return svcreate2(svreinterpret_s16(hori_acc_b),
                     svreinterpret_s16(hori_acc_t));
  }

  svint16x2_t vertical_compute_vert_approx(SourceVector3Type src)
      KLEIDICV_STREAMING {
    SourceVectorType src_0 = svget3(src, 0);
    SourceVectorType src_2 = svget3(src, 2);
    svuint16_t vert_acc_b = svsublb(src_2, src_0);
    svuint16_t vert_acc_t = svsublt(src_2, src_0);

    return svcreate2(svreinterpret_s16(vert_acc_b),
                     svreinterpret_s16(vert_acc_t));
  }

  void process_vertical(Rows<const SourceType> src_rows) KLEIDICV_STREAMING {
    LoopUnroll2 loop{width_ * src_rows.channels(),
                     SourceVecTraits::num_lanes()};
    svbool_t pg_all = SourceVecTraits::svptrue();

    loop.unroll_twice([&](ptrdiff_t index) KLEIDICV_STREAMING {
      vertical_vector_path_x2(pg_all, src_rows, index);
    });

    loop.unroll_once([&](ptrdiff_t index) KLEIDICV_STREAMING {
      vertical_vector_path(pg_all, src_rows, index);
    });

    loop.remaining([&](ptrdiff_t index, ptrdiff_t length) KLEIDICV_STREAMING {
      svbool_t pg = SourceVecTraits::svwhilelt(index, length);
      vertical_vector_path(pg, src_rows, index);
    });
  }

  void horizontal_vector_path_x2(svbool_t pg, Rows<DestinationType> dst_rows,
                                 ptrdiff_t index,
                                 ptrdiff_t channel) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    svcount_t pc16 = BufferVecTraits::svptrue_c();
    BufferVector2Type hori_0 = svld1_x2(pc16, &hori_deriv_buffer_[index]);
    BufferVector2Type hori_1 =
        svld1_x2(pc16, &hori_deriv_buffer_[index + channel * 2]);
    BufferVector2Type vert_0 = svld1_x2(pc16, &vert_deriv_buffer_[index]);
    BufferVector2Type vert_1 =
        svld1_x2(pc16, &vert_deriv_buffer_[index + channel]);
    BufferVector2Type vert_2 =
        svld1_x2(pc16, &vert_deriv_buffer_[index + channel * 2]);
    BufferVector2Type hori_a = svcreate2(svget2(hori_0, 0), svget2(hori_1, 0));
    BufferVector2Type hori_b = svcreate2(svget2(hori_0, 1), svget2(hori_1, 1));

    BufferVector3Type vert_a =
        svcreate3(svget2(vert_0, 0), svget2(vert_1, 0), svget2(vert_2, 0));
    BufferVector3Type vert_b =
        svcreate3(svget2(vert_0, 1), svget2(vert_1, 1), svget2(vert_2, 1));
#else
    BufferVectorType hori_buff_0 = svld1(pg, &hori_deriv_buffer_[index]);
    BufferVectorType hori_buff_3 =
        svld1_vnum(pg, &hori_deriv_buffer_[index], 1);
    BufferVectorType hori_buff_2 =
        svld1(pg, &hori_deriv_buffer_[index + channel * 2]);
    BufferVectorType hori_buff_4 =
        svld1_vnum(pg, &hori_deriv_buffer_[index + channel * 2], 1);

    BufferVectorType vert_buff_0 = svld1(pg, &vert_deriv_buffer_[index]);
    BufferVectorType vert_buff_3 =
        svld1_vnum(pg, &vert_deriv_buffer_[index], 1);
    BufferVectorType vert_buff_1 =
        svld1(pg, &vert_deriv_buffer_[index + channel]);
    BufferVectorType vert_buff_2 =
        svld1(pg, &vert_deriv_buffer_[index + channel * 2]);
    BufferVectorType vert_buff_4 =
        svld1_vnum(pg, &vert_deriv_buffer_[index + channel], 1);
    BufferVectorType vert_buff_5 =
        svld1_vnum(pg, &vert_deriv_buffer_[index + channel * 2], 1);

    BufferVector2Type hori_a = svcreate2(hori_buff_0, hori_buff_2);
    BufferVector2Type hori_b = svcreate2(hori_buff_3, hori_buff_4);

    BufferVector3Type vert_a = svcreate3(vert_buff_0, vert_buff_1, vert_buff_2);
    BufferVector3Type vert_b = svcreate3(vert_buff_3, vert_buff_4, vert_buff_5);
#endif

    svint16x2_t interleaved_result_a =
        horizontal_compute_interleaved(pg, hori_a, vert_a);

    svint16x2_t interleaved_result_b =
        horizontal_compute_interleaved(pg, hori_b, vert_b);

    svst2(pg, &dst_rows[index * 2], interleaved_result_a);
    svst2_vnum(pg, &dst_rows[index * 2], 2, interleaved_result_b);
  }

  void horizontal_vector_path(svbool_t pg, Rows<DestinationType> dst_rows,
                              ptrdiff_t index,
                              ptrdiff_t channel) KLEIDICV_STREAMING {
    // Horizontal derivative approximation
    BufferVector2Type hori =
        svcreate2(svld1(pg, &hori_deriv_buffer_[index]),
                  svld1(pg, &hori_deriv_buffer_[index + channel * 2]));

    // Vertical derivative approximation
    BufferVector3Type vert =
        svcreate3(svld1(pg, &vert_deriv_buffer_[index]),
                  svld1(pg, &vert_deriv_buffer_[index + channel]),
                  svld1(pg, &vert_deriv_buffer_[index + channel * 2]));

    svint16x2_t interleaved_result =
        horizontal_compute_interleaved(pg, hori, vert);
    svst2(pg, &dst_rows[index * 2], interleaved_result);
  }

  svint16x2_t horizontal_compute_interleaved(
      svbool_t pg, BufferVector2Type hori,
      BufferVector3Type vert) KLEIDICV_STREAMING {
    BufferVectorType hori_buff_0 = svget2(hori, 0);
    BufferVectorType hori_buff_2 = svget2(hori, 1);
    BufferVectorType vert_buff_0 = svget3(vert, 0);
    BufferVectorType vert_buff_1 = svget3(vert, 1);
    BufferVectorType vert_buff_2 = svget3(vert, 2);
    DestinationVectorType hori_result = svsub_x(pg, hori_buff_2, hori_buff_0);

    DestinationVectorType vert_result = svadd_x(pg, vert_buff_0, vert_buff_2);
    vert_result = svmul_n_s16_x(pg, vert_result, 3);
    vert_result = svmad_s16_x(pg, vert_buff_1, svdup_n_s16(10), vert_result);

    return svcreate2(hori_result, vert_result);
  }

  void process_horizontal(Rows<DestinationType> dst_rows) KLEIDICV_STREAMING {
    // width is decremented by 2 as the result has less columns.
    LoopUnroll2 loop{(width_ - 2) * hori_deriv_buffer_.channels(),
                     BufferVecTraits::num_lanes()};
    svbool_t pg_all = BufferVecTraits::svptrue();
    const ptrdiff_t channel =
        static_cast<ptrdiff_t>(hori_deriv_buffer_.channels());

    loop.unroll_twice([&](ptrdiff_t index) KLEIDICV_STREAMING {
      horizontal_vector_path_x2(pg_all, dst_rows, index, channel);
    });

    loop.unroll_once([&](ptrdiff_t index) KLEIDICV_STREAMING {
      horizontal_vector_path(pg_all, dst_rows, index, channel);
    });

    loop.remaining([&](ptrdiff_t index, ptrdiff_t length) KLEIDICV_STREAMING {
      svbool_t pg = BufferVecTraits::svwhilelt(index, length);
      horizontal_vector_path(pg, dst_rows, index, channel);
    });
  }

  Rows<BufferType> hori_deriv_buffer_;
  Rows<BufferType> vert_deriv_buffer_;
  size_t width_;
};  // end of class ScharrInterleaved

class ScharrBufferDeleter {
 public:
  void operator()(void *ptr) const { std::free(ptr); }
};

static kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8_sc(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end) KLEIDICV_STREAMING {
  // Does not include checks for whether the operation is implemented.
  // This must be done earlier, by scharr_interleaved_is_implemented.
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, src_height);
  CHECK_IMAGE_SIZE(src_width, src_height);

  // Allocating more elements because in case of SVE interleaving stores are
  // governed by one predicate. For example, if a predicate requires 7 uint8_t
  // elements and an algorithm performs widening to 16 bits, the resulting
  // interleaving store will still be governed by the same predicate, thus
  // storing 8 elements. Choosing '3' to account for svst4().
  size_t buffer_stride = ((src_width * src_channels) + 3) * sizeof(int16_t);
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
}  // namespace KLEIDICV_TARGET_NAMESPACE
