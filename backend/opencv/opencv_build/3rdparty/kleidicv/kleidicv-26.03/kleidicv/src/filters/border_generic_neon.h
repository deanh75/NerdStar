// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_BORDER_GENERIC_NEON_H
#define KLEIDICV_WORKSPACE_BORDER_GENERIC_NEON_H

#include <algorithm>
#include <cstddef>

#include "kleidicv/neon.h"
#include "kleidicv/types.h"
#include "kleidicv/workspace/border_types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Border offsets for generic filters.
template <kleidicv::FixedBorderType BorderType>
class GenericBorderHorizontal final {
 public:
  GenericBorderHorizontal(size_t width, size_t channels)
      : width_(static_cast<ptrdiff_t>(width)),
        channels_{static_cast<ptrdiff_t>(channels)},
        data_indices_{0ULL | (1ULL << 8) | (2ULL << 16) | (3ULL << 24) |
                      (4ULL << 32) | (5ULL << 40) | (6ULL << 48) |
                      (7ULL << 56)},
        border_indices_left_{0},
        border_indices_right_{0} {
    // The result will take some elements from the image (data), and the
    // remaining parts from the border.
    // An index vector is prepared here to help the process, e.g. for replicated
    // borders and 3 channels, the constructed index vector will look like this:
    //               [1, 2, 0, 1, 2, 3, 4, 5]
    // (0,1,2 is repeated until index 0 is reached, when the image data begins)
    // Right side is similar, but it is the [5,6,7] that repeats after.
    for (ptrdiff_t i = 0; i < 8; ++i) {
      // channels_*8 - 1 - i:   23, 22, 21, 20, 19, 18, 17, 16
      // % channels:             2,  1,  0,  2,  1,  0,  2,  1
      border_indices_left_ =
          (border_indices_left_ << 8) | ((channels_ * 8 - 1 - i) % channels_);
      // (7 - i):                7, 6, 5, 4, 3, 2, 1, 0
      // % channels:             1, 0, 2, 1  0, 2, 1, 0
      border_indices_right_ =
          (border_indices_right_ << 8) | (((7 - i) % channels) + 8 - channels_);
    }
  }

  // Raw column can be bigger than width-1 or less than 0
  ptrdiff_t get_column(ptrdiff_t raw_column) const {
    // TODO more border types, this is only the Replicated
    return std::max<ptrdiff_t>(std::min<ptrdiff_t>(raw_column, width_ - 1),
                               ptrdiff_t{0});
  }

  // Assuming that start_offset is <= 0
  uint16x8_t load_left(Rows<const uint8_t> src_rows,
                       ptrdiff_t start_offset) const {
    if constexpr (BorderType == FixedBorderType::REPLICATE) {
      uint8x8_t data = vld1_u8(&src_rows[0]);
      uint64_t indices{};
      if (start_offset > -8) {
        ptrdiff_t shift = -8 * start_offset;
        indices =
            ((border_indices_left_ >> (64 - shift)) | (data_indices_ << shift));
      } else {
        ptrdiff_t shift = ((-start_offset - 8) % channels_) * 8;
        indices = (((border_indices_left_ >> (8 * channels_ - shift)) &
                    ((1 << shift) - 1)) |
                   (border_indices_left_ << shift));
      }
      return vmovl_u8(vtbl1_u8(data, vreinterpret_u8_u64(uint64x1_t{indices})));
    }
  }

  // Assuming that start_offset is >= width - 8
  uint16x8_t load_right(Rows<const uint8_t> src_rows,
                        ptrdiff_t start_offset) const {
    if constexpr (BorderType == FixedBorderType::REPLICATE) {
      uint8x8_t data = vld1_u8(&src_rows[width_ * channels_ - 8]);
      uint64_t indices{};
      ptrdiff_t shift = 8 * (start_offset - (width_ * channels_ - 8));
      if (shift < 64) {
        indices =
            (data_indices_ >> shift) | (border_indices_right_ << (64 - shift));
      } else {
        shift = ((start_offset - width_ * channels_) % channels_) * 8;
        indices = shift == 0
                      ? border_indices_right_
                      : (((border_indices_right_ >> (8 * channels_ - shift))
                          << (64 - shift)) |
                         (border_indices_right_ >> shift));
      }
      return vmovl_u8(vtbl1_u8(data, vreinterpret_u8_u64(uint64x1_t{indices})));
    }
  }

 private:
  ptrdiff_t width_;
  ptrdiff_t channels_;
  uint64_t data_indices_, border_indices_left_, border_indices_right_;
};  // end of class GenericBorderHorizontal<BorderType>

// Border offsets for generic filters.
template <kleidicv::FixedBorderType BorderType>
class GenericBorderVertical final {
 public:
  explicit GenericBorderVertical(size_t height)
      : height_(static_cast<ptrdiff_t>(height)) {}

  // Raw column can be bigger than width-1 or less than 0
  ptrdiff_t get_row(ptrdiff_t raw_row) const {
    // TODO more border types, this is only the Replicated
    return std::max<ptrdiff_t>(std::min<ptrdiff_t>(raw_row, height_ - 1),
                               ptrdiff_t{0});
  }

 private:
  ptrdiff_t height_;
};  // end of class GenericBorderVertical<BorderType>

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_BORDER_GENERIC_NEON_H
