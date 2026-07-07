// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/resize/resize_linear.h"

namespace kleidicv::neon {

/// Resizes source data by averaging 4 elements to one.
/// In-place operation is not supported.
///
/// Only even source dimensions are supported.
/// Only single-channel images are supported.
/// The source is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Example of 2x2 to 1x1 conversion:
/// ```
/// | a | b | --> | (a+b+c+d)/4 |
/// | c | d |
/// ```

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride) {
  using VecTraits = neon::VecTraits<uint8_t>;
  constexpr size_t kVectorLengthX2 = kVectorLength * 2;
  constexpr size_t kVectorLengthX4 = kVectorLength * 4;

  for (; src_height >= 2;
       src_height -= 2, src += (src_stride * 2), dst += dst_stride) {
    const uint8_t *src_l = src;
    uint8_t *dst_l = dst;
    size_t src_width_l = src_width;

    for (; src_width_l >= kVectorLengthX4; src_width_l -= kVectorLengthX4,
                                           dst_l += kVectorLengthX2,
                                           src_l += kVectorLengthX4) {
      KLEIDICV_PREFETCH(src_l + 1024);
      KLEIDICV_PREFETCH(src_l + src_stride + 1024);

      uint8x16x4_t top_line, bottom_line;
      uint16x8_t top_line_pairs_summed[4];
      uint16x8_t bottom_line_pairs_summed[4];
      uint16x8_t result_before_averaging[4];
      uint8x16x2_t result;

      VecTraits::load(src_l, top_line);
      VecTraits::load(&src_l[src_stride], bottom_line);

      top_line_pairs_summed[0] = vpaddlq_u8(top_line.val[0]);
      top_line_pairs_summed[1] = vpaddlq_u8(top_line.val[1]);
      top_line_pairs_summed[2] = vpaddlq_u8(top_line.val[2]);
      top_line_pairs_summed[3] = vpaddlq_u8(top_line.val[3]);

      bottom_line_pairs_summed[0] = vpaddlq_u8(bottom_line.val[0]);
      bottom_line_pairs_summed[1] = vpaddlq_u8(bottom_line.val[1]);
      bottom_line_pairs_summed[2] = vpaddlq_u8(bottom_line.val[2]);
      bottom_line_pairs_summed[3] = vpaddlq_u8(bottom_line.val[3]);

      result_before_averaging[0] =
          vaddq_u16(top_line_pairs_summed[0], bottom_line_pairs_summed[0]);
      result_before_averaging[1] =
          vaddq_u16(top_line_pairs_summed[1], bottom_line_pairs_summed[1]);
      result_before_averaging[2] =
          vaddq_u16(top_line_pairs_summed[2], bottom_line_pairs_summed[2]);
      result_before_averaging[3] =
          vaddq_u16(top_line_pairs_summed[3], bottom_line_pairs_summed[3]);

      result.val[0] =
          vrshrn_high_n_u16(vrshrn_n_u16(result_before_averaging[0], 2),
                            result_before_averaging[1], 2);
      result.val[1] =
          vrshrn_high_n_u16(vrshrn_n_u16(result_before_averaging[2], 2),
                            result_before_averaging[3], 2);

      VecTraits::store(result, dst_l);
    }

    for (; src_width_l > 1; src_width_l -= 2, src_l += 2, ++dst_l) {
      disable_loop_vectorization();
      *dst_l = rounding_shift_right<uint16_t>(
          static_cast<uint16_t>(*src_l) + *(src_l + 1) + *(src_l + src_stride) +
              *(src_l + src_stride + 1),
          2);
    }
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
