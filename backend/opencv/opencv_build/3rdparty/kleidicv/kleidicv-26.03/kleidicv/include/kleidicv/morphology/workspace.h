// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MORPHOLOGY_WORKSPACE_H
#define KLEIDICV_MORPHOLOGY_WORKSPACE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <variant>

#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

#if KLEIDICV_TARGET_SME
#include <arm_sme.h>
#endif

namespace KLEIDICV_TARGET_NAMESPACE {

// Workspace for morphological operations.
class MorphologyWorkspace final {
 public:
  enum class BorderType {
    CONSTANT,
    REPLICATE,
  };

  static std::optional<BorderType> get_border_type(
      kleidicv_border_type_t border_type) KLEIDICV_STREAMING {
    switch (border_type) {
      case KLEIDICV_BORDER_TYPE_REPLICATE:
        return BorderType::REPLICATE;
      case KLEIDICV_BORDER_TYPE_CONSTANT:
        return BorderType::CONSTANT;
      default:
        return std::optional<BorderType>();
    }
  }

  template <typename T>
  class CopyDataMemcpy {
   public:
    constexpr void operator()(Rows<const T> src_rows, Rows<T> dst_rows,
                              size_t length) const KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME
      __arm_sc_memcpy(static_cast<void *>(&dst_rows[0]),
                      static_cast<const void *>(&src_rows[0]),
                      length * sizeof(T) * dst_rows.channels());
#else
      std::memcpy(static_cast<void *>(&dst_rows[0]),
                  static_cast<const void *>(&src_rows[0]),
                  length * sizeof(T) * dst_rows.channels());
#endif
    }
  };

  // MorphologyWorkspace is only constructible with create().
  MorphologyWorkspace() = delete;

  static std::variant<MorphologyWorkspace, kleidicv_error_t> create(
      Rectangle kernel, Point anchor, BorderType border_type,
      const uint8_t *border_value, size_t channels, size_t type_size,
      Rectangle image_size) KLEIDICV_STREAMING {
    if (anchor.x() >= kernel.width() || anchor.y() >= kernel.height()) {
      return KLEIDICV_ERROR_RANGE;
    }

    if (border_type == BorderType::CONSTANT && border_value == nullptr) {
      return KLEIDICV_ERROR_NULL_POINTER;
    }

    // These values are arbitrarily choosen.
    const size_t rows_per_iteration =
        std::max(2 * kernel.height(), static_cast<size_t>(32ULL));
    // To avoid load/store penalties.
    const size_t kAlignment = 16;
    Margin margin{kernel, anchor};

    // A single wide row which can hold one row worth of data in addition
    // to left and right margins.
    size_t wide_rows_width =
        margin.left() + image_size.width() + margin.right();
    size_t wide_rows_stride = wide_rows_width * channels;
    wide_rows_stride = align_up(wide_rows_stride, kAlignment);
    size_t wide_rows_height = 1UL;  // There is only one wide row.
    size_t wide_rows_size = wide_rows_stride * wide_rows_height;
    wide_rows_size += kAlignment - 1;

    // Multiple buffer rows to hold rows without any borders.
    size_t buffer_rows_width = type_size * image_size.width();
    size_t buffer_rows_stride = buffer_rows_width * channels;
    buffer_rows_stride = align_up(buffer_rows_stride, kAlignment);
    size_t buffer_rows_height = 2 * rows_per_iteration;
    size_t buffer_rows_size = 0UL;
    if (__builtin_mul_overflow(buffer_rows_stride, buffer_rows_height,
                               &buffer_rows_size)) {
      return KLEIDICV_ERROR_RANGE;
    }
    buffer_rows_size += kAlignment - 1;

    // Storage for indirect row access.
    size_t indirect_row_storage_size = 3 * rows_per_iteration * sizeof(void *);

    // Try to allocate the buffers at once.
    size_t allocation_size =
        indirect_row_storage_size + buffer_rows_size + wide_rows_size;
    uint8_t *allocation =
        reinterpret_cast<uint8_t *>(std::malloc(allocation_size));
    if (!allocation) {
      return KLEIDICV_ERROR_ALLOCATION;
    }

    size_t wide_rows_src_width = image_size.width();

    auto *buffer_rows_address = &allocation[indirect_row_storage_size];
    buffer_rows_address = align_up(buffer_rows_address, kAlignment);
    ptrdiff_t buffer_rows_offset = buffer_rows_address - allocation;

    auto *wide_rows_address =
        &allocation[indirect_row_storage_size + buffer_rows_size];
    wide_rows_address += margin.left() * channels;
    wide_rows_address = align_up(wide_rows_address, kAlignment);
    wide_rows_address -= margin.left() * channels;
    ptrdiff_t wide_rows_offset = wide_rows_address - allocation;

    std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT> border_values{};
    if (border_type == BorderType::CONSTANT) {
      for (size_t i = 0; i < channels; ++i) {
        border_values[i] = border_value[i];
      }
    }

    return MorphologyWorkspace{image_size,
                               margin,
                               border_type,
                               border_values,
                               rows_per_iteration,
                               wide_rows_src_width,
                               channels,
                               0,
                               0,
                               buffer_rows_offset,
                               buffer_rows_stride,
                               wide_rows_offset,
                               wide_rows_stride,
                               allocation};
  }

 private:
  MorphologyWorkspace(
      Rectangle image_size, Margin margin, BorderType border_type,
      std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT> border_values,
      size_t rows_per_iteration, size_t wide_rows_src_width, size_t channels,
      size_t horizontal_height, size_t vertical_height,
      ptrdiff_t buffer_rows_offset, size_t buffer_rows_stride,
      ptrdiff_t wide_rows_offset, size_t wide_rows_stride,
      uint8_t *allocation) KLEIDICV_STREAMING
      : image_size_{image_size},
        margin_{margin},
        border_type_{border_type},
        border_value_{border_values},
        rows_per_iteration_{rows_per_iteration},
        wide_rows_src_width_{wide_rows_src_width},
        channels_{channels},
        horizontal_height_{horizontal_height},
        vertical_height_{vertical_height},
        buffer_rows_offset_{buffer_rows_offset},
        buffer_rows_stride_{buffer_rows_stride},
        wide_rows_offset_{wide_rows_offset},
        wide_rows_stride_{wide_rows_stride},
        data_{allocation, &std::free} {}

 public:
  // Function complexity is just above the threshold, but this is readable this
  // way, ignore the warning.
  // NOLINTBEGIN(readability-function-cognitive-complexity)
  template <typename O>
  void process(Rows<const typename O::SourceType> src_rows,
               Rows<typename O::DestinationType> dst_rows,
               O operation) KLEIDICV_STREAMING {
    using S = typename O::SourceType;
    using B = typename O::BufferType;
    typename O::CopyData copy_data{};

    if (KLEIDICV_UNLIKELY(image_size_.width() == 0 ||
                          image_size_.height() == 0)) {
      return;
    }

    // Wide rows which can hold data with left and right margins.
    auto wide_rows =
        Rows{reinterpret_cast<S *>(&data_.get()[wide_rows_offset_]),
             wide_rows_stride_, channels_};

    // Double buffered indirect rows to access the buffer rows.
    auto db_indirect_rows = DoubleBufferedIndirectRows{
        reinterpret_cast<B **>(&data_.get()[0]), rows_per_iteration_,
        Rows{reinterpret_cast<B *>(&data_.get()[buffer_rows_offset_]),
             buffer_rows_stride_, channels_}};

    // [Step 1] Initialize workspace.
    horizontal_height_ =
        margin_.top() + image_size_.height() + margin_.bottom();
    vertical_height_ = image_size_.height();
    size_t row_index = 0;

    // Used by replicate border type.
    auto first_src_rows = src_rows;
    auto last_src_rows = src_rows.at(image_size_.height() - 1);

    size_t horizontal_height = get_next_horizontal_height();
    for (size_t index = 0; index < horizontal_height; ++index) {
      switch (border_type_) {
        case BorderType::CONSTANT: {
          make_constant_border(wide_rows, 0, margin_.left());

          if (row_index < margin_.top() ||
              row_index >= margin_.top() + image_size_.height()) {
            make_constant_border(wide_rows, margin_.left(),
                                 wide_rows_src_width_);
          } else {
            copy_data(src_rows, wide_rows.at(0, margin_.left()),
                      wide_rows_src_width_);
            // Advance source rows.
            ++src_rows;
          }

          make_constant_border(wide_rows, margin_.left() + wide_rows_src_width_,
                               margin_.right());

          // Advance counters.
          ++row_index;
        } break;

        case BorderType::REPLICATE: {
          Rows<const S> current_src_row;

          if (row_index < margin_.top()) {
            current_src_row = first_src_rows;
          } else if (row_index < (margin_.top() + image_size_.height())) {
            current_src_row = src_rows;
            // Advance source rows.
            ++src_rows;
          } else {
            current_src_row = last_src_rows;
          }

          replicate_border(current_src_row, wide_rows, 0, 0, margin_.left());
          copy_data(current_src_row, wide_rows.at(0, margin_.left()),
                    wide_rows_src_width_);
          replicate_border(current_src_row, wide_rows, wide_rows_src_width_ - 1,
                           margin_.left() + wide_rows_src_width_,
                           margin_.right());

          // Advance counters.
          ++row_index;
        } break;
      }  // switch (border_type_)

      // [Step 2] Process the preloaded data.
      operation.process_horizontal(Rectangle{image_size_.width(), 1UL},
                                   wide_rows,
                                   db_indirect_rows.write_at().at(index));
    }  // for (...; index < horizontal_height; ...)

    db_indirect_rows.swap();

    // [Step 3] Process any remaining data.
    while (vertical_height_) {
      size_t horizontal_height = get_next_horizontal_height();
      for (size_t index = 0; index < horizontal_height; ++index) {
        switch (border_type_) {
          case BorderType::CONSTANT: {
            if (row_index < (margin_.top() + image_size_.height())) {
              // Constant left and right borders with source data.
              copy_data(src_rows, wide_rows.at(0, margin_.left()),
                        wide_rows_src_width_);
              // Advance source rows.
              ++src_rows;
            } else {
              make_constant_border(wide_rows, margin_.left(),
                                   wide_rows_src_width_);
            }

            // Advance row counter.
            ++row_index;
          } break;

          case BorderType::REPLICATE: {
            Rows<const S> current_src_row;

            if (row_index < (margin_.top() + image_size_.height())) {
              current_src_row = src_rows;
              // Advance source rows.
              ++src_rows;
            } else {
              current_src_row = last_src_rows;
            }

            replicate_border(current_src_row, wide_rows, 0, 0, margin_.left());
            copy_data(current_src_row, wide_rows.at(0, margin_.left()),
                      wide_rows_src_width_);
            replicate_border(
                current_src_row, wide_rows, wide_rows_src_width_ - 1,
                margin_.left() + wide_rows_src_width_, margin_.right());

            // Advance counters.
            ++row_index;
          } break;
        }  // switch (border_type_)

        operation.process_horizontal(Rectangle{image_size_.width(), 1UL},
                                     wide_rows,
                                     db_indirect_rows.write_at().at(index));
      }  // for (...; index < horizontal_height; ...)

      size_t next_vertical_height = get_next_vertical_height();
      operation.process_vertical(
          Rectangle{image_size_.width(), next_vertical_height},
          db_indirect_rows.read_at(), dst_rows);
      dst_rows += next_vertical_height;

      db_indirect_rows.swap();
    }
  }
  // NOLINTEND(readability-function-cognitive-complexity)

 private:
  // The number of wide rows to process in the next iteration.
  [[nodiscard]] size_t get_next_horizontal_height() KLEIDICV_STREAMING {
    size_t height = std::min(horizontal_height_, rows_per_iteration_);
    horizontal_height_ -= height;
    return height;
  }

  // The number of indirect rows to process in the next iteration.
  [[nodiscard]] size_t get_next_vertical_height() KLEIDICV_STREAMING {
    size_t height = std::min(vertical_height_, rows_per_iteration_);
    vertical_height_ -= height;
    return height;
  }

  template <typename T>
  void make_constant_border(Rows<T> dst_rows, size_t dst_index,
                            size_t count) KLEIDICV_STREAMING {
    auto dst = &dst_rows.at(0, dst_index)[0];
    for (size_t index = 0; index < count; ++index) {
      for (size_t channel = 0; channel < dst_rows.channels(); ++channel) {
        dst[index * dst_rows.channels() + channel] = border_value_[channel];
      }
    }
  }

  template <typename T>
  void replicate_border(Rows<const T> src_rows, Rows<T> dst_rows,
                        size_t src_index, size_t dst_index,
                        size_t count) KLEIDICV_STREAMING {
    if (!count) {
      return;
    }

    for (size_t channel = 0; channel < src_rows.channels(); ++channel) {
      for (size_t index = dst_index; index < dst_index + count; ++index) {
        dst_rows.at(0, index)[channel] = src_rows.at(0, src_index)[channel];
      }
    }
  }

  Rectangle image_size_;
  Margin margin_;
  BorderType border_type_;
  std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT> border_value_;

  // Number of wide rows in this workspace.
  size_t rows_per_iteration_;
  // Size of the data in bytes within a row.
  size_t wide_rows_src_width_;
  // The number of channels.
  size_t channels_;
  // Remaining height to process in horizontal direction.
  size_t horizontal_height_;
  // Remaining height to process in vertical direction.
  size_t vertical_height_;
  // Offset in bytes to the buffer rows from &data_[0].
  ptrdiff_t buffer_rows_offset_;
  // Stride of the buffer rows.
  size_t buffer_rows_stride_;
  // Offset in bytes to the wide rows from &data_[0].
  ptrdiff_t wide_rows_offset_;
  // Stride of the wide rows.
  size_t wide_rows_stride_;
  // Workspace buffer
  std::unique_ptr<uint8_t, decltype(&std::free)> data_;
};  // end of class MorphologyWorkspace

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MORPHOLOGY_WORKSPACE_H
