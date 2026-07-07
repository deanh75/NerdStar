// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_BUILD_OPTICAL_FLOW_LK_PYRAMID_H
#define KLEIDICV_BUILD_OPTICAL_FLOW_LK_PYRAMID_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/blur_and_downsample.h"
#include "kleidicv/filters/scharr.h"

namespace kleidicv {

// Shared parameter validation and actual-level computation used by the
// build entry points.
inline kleidicv_error_t validate_and_compute_build_optical_flow_pyr_lk_levels(
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height,
    size_t* actual_level_count) {
  if (actual_level_count == nullptr) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (width == 0 || height == 0 || channels == 0 ||
      channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT || level_count == 0 ||
      window_width <= 2 || window_height <= 2 ||
      window_width > KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE ||
      window_height > KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE) {
    return KLEIDICV_ERROR_RANGE;
  }

  const size_t row_bytes = width * channels;
  if (height > 1 && src_stride < row_bytes) {
    return KLEIDICV_ERROR_RANGE;
  }

  size_t computed_level_count = 0;
  size_t level_width = width;
  size_t level_height = height;
  for (size_t requested = 0; requested < level_count; ++requested) {
    ++computed_level_count;
    const size_t next_width = (level_width + 1) / 2;
    const size_t next_height = (level_height + 1) / 2;
    if (next_width <= window_width || next_height <= window_height) {
      break;
    }
    level_width = next_width;
    level_height = next_height;
  }

  *actual_level_count = computed_level_count;
  return KLEIDICV_OK;
}

// Internal storage/build class used by:
// - kleidicv_build_optical_flow_pyr_lk_pyramid
// - kleidicv_thread_build_optical_flow_pyr_lk_pyramid
//
// It owns image + Scharr pyramids and provides allocation/build helpers.
// Per-level metadata for both pyramids.
// - image_data points to the level in the uint8 image pyramid.
// - scharr_data points to the level in the int16 interleaved dx/dy pyramid.
struct OpticalFlowLevel {
  size_t width;
  size_t height;
  size_t image_stride;
  size_t scharr_stride;
  uint8_t* image_data;
  int16_t* scharr_data;
};

class OpticalFlowLKPyramid;

class OpticalFlowLKPyramidDeleter {
 public:
  void operator()(OpticalFlowLKPyramid* ptr) const;
};

class OpticalFlowLKPyramid final {
 public:
  using Pointer =
      std::unique_ptr<OpticalFlowLKPyramid, OpticalFlowLKPyramidDeleter>;

  OpticalFlowLKPyramid(const OpticalFlowLKPyramid&) = delete;
  OpticalFlowLKPyramid& operator=(const OpticalFlowLKPyramid&) = delete;
  OpticalFlowLKPyramid(OpticalFlowLKPyramid&&) = delete;
  OpticalFlowLKPyramid& operator=(OpticalFlowLKPyramid&&) = delete;

  // This function has many guarded allocation/overflow checks by design.
  // Keep explicit checks for safety and suppress cognitive-complexity warning.
  // NOLINTBEGIN(readability-function-cognitive-complexity)
  // Allocate one contiguous block that stores metadata and all level buffers.
  // TODO(future): Revisit object-lifetime handling for this allocation path.
  //
  // This code allocates one raw storage block with std::malloc() and then
  // treats the beginning of that block as an OpticalFlowLKPyramid object via
  // reinterpret_cast. In C++17, malloc() only provides raw storage; it does not
  // begin the lifetime of a class-type object. Writing members through the
  // casted OpticalFlowLKPyramid* therefore relies on object-lifetime
  // assumptions that should be cleaned up.
  //
  // The packed single-allocation layout is intentional and should be preserved:
  //   [OpticalFlowLKPyramid header][levels table][image storage][scharr
  //   storage]
  //
  // A future fix should explicitly make this layout lifetime-safe, for example
  // by either:
  // - starting the OpticalFlowLKPyramid lifetime in the allocated storage and
  //   pairing destruction correctly in the deleter, or
  // - refactoring the allocation header into a raw/trivial storage struct whose
  //   lifetime rules are valid for malloc-backed storage in C++17.
  static kleidicv_error_t allocate(Pointer& pyramid, size_t level_count,
                                   size_t width, size_t height, size_t channels,
                                   size_t border_width, size_t border_height) {
    // First sizing pass: compute total bytes required for all image levels and
    // all Scharr levels.
    size_t total_image_bytes = 0;
    size_t total_scharr_bytes = 0;
    size_t level_width = width;
    size_t level_height = height;
    for (size_t level = 0; level < level_count; ++level) {
      LevelStorageLayout layout{};
      if (kleidicv_error_t err = compute_level_storage_layout(
              level_width, level_height, channels, border_width, border_height,
              &layout)) {
        return err;
      }

      // Accumulate total storage required across all levels.
      total_image_bytes += layout.image_bytes;
      total_scharr_bytes += layout.scharr_bytes;

      // LK pyramid geometry uses ceil division by 2 between levels.
      level_width = (level_width + 1) / 2;
      level_height = (level_height + 1) / 2;
    }

    size_t levels_bytes = 0;
    // Metadata table footprint: one OpticalFlowLevel entry per pyramid level.
    levels_bytes = level_count * sizeof(OpticalFlowLevel);

    // Single-allocation packed layout:
    // [object header][levels table][image storage][scharr storage]
    //
    // Each section start is aligned for its element type, and each size/offset
    // update is overflow-checked to keep allocation math safe.
    size_t levels_offset = 0;
    size_t image_offset = 0;
    size_t scharr_offset = 0;
    size_t allocation_size = 0;
    // GCOVR_EXCL_START
    if (!align_up_size(sizeof(OpticalFlowLKPyramid), alignof(OpticalFlowLevel),
                       &levels_offset) ||
        add_overflow_size(levels_offset, levels_bytes, &image_offset) ||
        !align_up_size(image_offset, alignof(int16_t), &image_offset) ||
        add_overflow_size(image_offset, total_image_bytes, &scharr_offset) ||
        !align_up_size(scharr_offset, alignof(int16_t), &scharr_offset) ||
        add_overflow_size(scharr_offset, total_scharr_bytes,
                          &allocation_size)) {
      // Unreachable for practical API paths after previous bounded-size checks.
      // Keep as final allocation-layout overflow guard.
      return KLEIDICV_ERROR_RANGE;
    }
    // GCOVR_EXCL_STOP

    void* allocation = std::malloc(allocation_size);
    if (allocation == nullptr) {
      pyramid = Pointer{};
      return KLEIDICV_ERROR_ALLOCATION;
    }

    auto* instance = reinterpret_cast<OpticalFlowLKPyramid*>(allocation);
    pyramid = Pointer{instance};

    instance->level_count_ = level_count;
    instance->channels_ = channels;
    instance->border_width_ = border_width;
    instance->border_height_ = border_height;

    auto* allocation_base = reinterpret_cast<uint8_t*>(instance);
    instance->levels_ =
        reinterpret_cast<OpticalFlowLevel*>(allocation_base + levels_offset);

    auto* images_cursor = allocation_base + image_offset;
    auto* scharr_cursor = allocation_base + scharr_offset;

    // Second pass: fill metadata and point level entries into the contiguous
    // block.
    level_width = width;
    level_height = height;
    for (size_t level = 0; level < level_count; ++level) {
      // Bind the metadata entry for this pyramid level and store the logical
      // (interior) image size that callers see through the getter APIs.
      auto& entry = instance->levels_[level];
      entry.width = level_width;
      entry.height = level_height;

      LevelStorageLayout layout{};
      // Assumption: this second pass reuses dimensions already validated in the
      // first pass, so layout computation succeeds.
      [[maybe_unused]] const kleidicv_error_t layout_err =
          compute_level_storage_layout(level_width, level_height, channels,
                                       border_width, border_height, &layout);
      entry.image_stride = layout.image_stride;
      entry.scharr_stride = layout.scharr_stride;

      // images_cursor points at the beginning of this level's bordered image
      // allocation.
      auto* image_base = images_cursor;

      // Expose a pointer to the interior top-left pixel by skipping the full
      // configured border in both dimensions.
      //
      // This allows border writes/reads via pointer arithmetic around
      // entry.image_data without extra allocations/copies.
      entry.image_data = image_base +
                         instance->border_height_ * entry.image_stride +
                         instance->border_width_ * channels;

      // scharr_cursor points at the beginning of this level's bordered Scharr
      // allocation (int16_t elements viewed through byte cursor).
      auto* scharr_base = reinterpret_cast<int16_t*>(scharr_cursor);

      // Convert Scharr stride bytes to element count for pointer arithmetic.
      const size_t scharr_stride_elems = entry.scharr_stride / sizeof(int16_t);

      // Interior starts after the full configured Scharr border.
      const size_t scharr_interior_offset =
          instance->border_height_ * scharr_stride_elems +
          instance->border_width_ * (channels * 2);
      entry.scharr_data = scharr_base + scharr_interior_offset;

      // Move cursors to the next level's storage region within the same
      // allocation block.
      images_cursor += layout.image_bytes;
      scharr_cursor += layout.scharr_bytes;

      // Next level dimensions follow the LK pyramid halving rule
      // (ceil division by 2).
      level_width = (level_width + 1) / 2;
      level_height = (level_height + 1) / 2;
    }

    return KLEIDICV_OK;
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  // Default creation entry point.
  //
  // This overload wires the standard single-threaded kernels
  // (blur_and_downsample + scharr_interleaved) and forwards to the generic
  // overload below.
  //
  // Note on naming:
  // There are intentionally two functions named create():
  // 1) this one: convenience wrapper with default kernel selection
  // 2) the templated overload: shared implementation with injectable
  //    kernels (used by both single-thread and threaded builders).
  template <bool kUseSME>
  kleidicv_error_t create(const uint8_t* src, size_t src_stride) {
    auto blur_and_downsample_fn =
        [](const uint8_t* blur_src, size_t blur_src_stride, size_t blur_width,
           size_t blur_height, uint8_t* blur_dst, size_t blur_dst_stride,
           size_t blur_channels, kleidicv_border_type_t border_type) {
          if constexpr (kUseSME) {
            return kleidicv_blur_and_downsample_u8_sme(
                blur_src, blur_src_stride, blur_width, blur_height, blur_dst,
                blur_dst_stride, blur_channels, border_type);
          }
          return kleidicv_blur_and_downsample_u8(
              blur_src, blur_src_stride, blur_width, blur_height, blur_dst,
              blur_dst_stride, blur_channels, border_type);
        };

    auto scharr_fn = [](const uint8_t* scharr_src, size_t scharr_src_stride,
                        size_t scharr_width, size_t scharr_height,
                        size_t scharr_channels, int16_t* scharr_dst,
                        size_t scharr_dst_stride) {
      if constexpr (kUseSME) {
        return kleidicv_scharr_interleaved_s16_u8_sme(
            scharr_src, scharr_src_stride, scharr_width, scharr_height,
            scharr_channels, scharr_dst, scharr_dst_stride);
      }
      return kleidicv_scharr_interleaved_s16_u8(
          scharr_src, scharr_src_stride, scharr_width, scharr_height,
          scharr_channels, scharr_dst, scharr_dst_stride);
    };

    return create(src, src_stride, blur_and_downsample_fn, scharr_fn);
  }

  // Generic creation implementation.
  //
  // Callers provide the concrete blur/downsample and Scharr functions, which
  // allows reusing the same pyramid-construction logic for different execution
  // paths (for example single-thread vs threaded variants) without duplicating
  // the algorithm.
  template <typename BlurAndDownsampleFn, typename ScharrFn>
  kleidicv_error_t create(const uint8_t* src, size_t src_stride,
                          BlurAndDownsampleFn blur_and_downsample_fn,
                          ScharrFn scharr_fn) {
    auto build_scharr_for_level = [&](size_t level_index) -> kleidicv_error_t {
      const auto& entry = levels_[level_index];
      const size_t bordered_width = entry.width + 2;
      const size_t bordered_height = entry.height + 2;

      // Zero bordered Scharr storage so derivative border is constant.
      const size_t scharr_height_with_border =
          entry.height + 2 * border_height_;
      // Assumption: allocation layout and validated parameter limits guarantee
      // this product fits in size_t for supported API flows.
      const size_t scharr_bytes =
          entry.scharr_stride * scharr_height_with_border;

      auto* scharr_base = reinterpret_cast<uint8_t*>(entry.scharr_data) -
                          border_height_ * entry.scharr_stride -
                          border_width_ * channels_ * 2 * sizeof(int16_t);
      std::memset(scharr_base, 0, scharr_bytes);

      const auto* bordered_src =
          entry.image_data - entry.image_stride - channels_;
      // GCOVR_EXCL_START
      if (kleidicv_error_t err = scharr_fn(
              bordered_src, entry.image_stride, bordered_width, bordered_height,
              channels_, entry.scharr_data, entry.scharr_stride)) {
        return err;
      }
      // GCOVR_EXCL_STOP
      return KLEIDICV_OK;
    };

    // Build level 0 image and Scharr.
    {
      auto& level0 = levels_[0];
      const size_t interior_bytes_per_row = level0.width * channels_;
      for (size_t y = 0; y < level0.height; ++y) {
        std::memcpy(level0.image_data + y * level0.image_stride,
                    src + y * src_stride, interior_bytes_per_row);
      }
      fill_reflect_101_border_in_place(level0.image_data, level0.image_stride,
                                       level0.width, level0.height, channels_,
                                       border_width_, border_height_);
      // GCOVR_EXCL_START
      if (kleidicv_error_t err = build_scharr_for_level(0)) {
        return err;
      }
      // GCOVR_EXCL_STOP
    }

    // Build remaining levels. Each level is completed immediately:
    // image -> border fill -> Scharr.
    for (size_t level = 1; level < level_count_; ++level) {
      const auto& previous = levels_[level - 1];
      auto& current = levels_[level];

      // GCOVR_EXCL_START
      if (kleidicv_error_t err = blur_and_downsample_fn(
              previous.image_data, previous.image_stride, previous.width,
              previous.height, current.image_data, current.image_stride,
              channels_, KLEIDICV_BORDER_TYPE_REVERSE)) {
        return err;
      }
      // GCOVR_EXCL_STOP

      fill_reflect_101_border_in_place(current.image_data, current.image_stride,
                                       current.width, current.height, channels_,
                                       border_width_, border_height_);

      // GCOVR_EXCL_START
      if (kleidicv_error_t err = build_scharr_for_level(level)) {
        return err;
      }
      // GCOVR_EXCL_STOP
    }

    return KLEIDICV_OK;
  }

  // Returns how many pyramid levels were allocated and populated.
  // Valid level indices are in the range [0, level_count()).
  size_t level_count() const { return level_count_; }
  size_t channels() const { return channels_; }
  size_t border_width() const { return border_width_; }
  size_t border_height() const { return border_height_; }

  // Returns metadata and data pointers for one level.
  // The returned reference is non-owning and remains valid while this
  // OpticalFlowLKPyramid instance is alive.
  const OpticalFlowLevel& level(size_t index) const { return levels_[index]; }

 private:
  struct LevelStorageLayout {
    size_t image_stride;
    size_t scharr_stride;
    size_t image_bytes;
    size_t scharr_bytes;
  };

  static kleidicv_error_t compute_level_storage_layout(
      size_t level_width, size_t level_height, size_t channels,
      size_t border_width, size_t border_height, LevelStorageLayout* out) {
    size_t image_width_with_border = 0;
    size_t image_height_with_border = 0;
    image_width_with_border = level_width + 2 * border_width;
    image_height_with_border = level_height + 2 * border_height;
    size_t image_stride = 0;
    if (__builtin_mul_overflow(image_width_with_border, channels,
                               &image_stride)) {
      return KLEIDICV_ERROR_RANGE;
    }

    size_t scharr_elements_stride = 0;
    size_t scharr_stride = 0;
    if (__builtin_mul_overflow(image_width_with_border, channels,
                               &scharr_elements_stride) ||
        __builtin_mul_overflow(scharr_elements_stride, static_cast<size_t>(2),
                               &scharr_elements_stride) ||
        __builtin_mul_overflow(scharr_elements_stride, sizeof(int16_t),
                               &scharr_stride)) {
      return KLEIDICV_ERROR_RANGE;
    }

    size_t image_bytes = 0;
    if (__builtin_mul_overflow(image_stride, image_height_with_border,
                               &image_bytes)) {
      return KLEIDICV_ERROR_RANGE;
    }

    size_t scharr_bytes = 0;
    if (__builtin_mul_overflow(scharr_stride, image_height_with_border,
                               &scharr_bytes)) {
      return KLEIDICV_ERROR_RANGE;
    }

    out->image_stride = image_stride;
    out->scharr_stride = scharr_stride;
    out->image_bytes = image_bytes;
    out->scharr_bytes = scharr_bytes;
    return KLEIDICV_OK;
  }

  // Rounds value up to the next multiple of alignment.
  //
  // Returns false only if the intermediate value + (alignment - 1) overflows,
  // otherwise stores the aligned result in `*aligned_value` and returns true.
  static bool align_up_size(size_t value, size_t alignment,
                            size_t* aligned_value) {
    const size_t addend = alignment - 1;
    size_t with_addend = 0;
    if (__builtin_add_overflow(value, addend, &with_addend)) {
      // Unreachable for this file's call sites after bounded size checks.
      // Kept to preserve generic overflow-safe utility behavior.
      return false;  // GCOVR_EXCL_LINE
    }
    *aligned_value = with_addend & ~addend;
    return true;
  }

  // Adds two size_t values and reports overflow.
  //
  // Mirrors __builtin_add_overflow semantics:
  // - returns true  -> overflow happened
  // - returns false -> result is valid and written to `*out`
  static bool add_overflow_size(size_t a, size_t b, size_t* out) {
    return __builtin_add_overflow(a, b, out);
  }

  // Maps an index (including out-of-range indices) into [0, length)
  // using reflect-101 border behavior.
  static size_t border_index_reflect_101(ptrdiff_t index, size_t length) {
    if (length <= 1) {
      return 0;
    }

    ptrdiff_t reflected = index;
    while (reflected < 0 || reflected >= static_cast<ptrdiff_t>(length)) {
      if (reflected < 0) {
        reflected = -reflected;
      } else {
        reflected = 2 * static_cast<ptrdiff_t>(length) - reflected - 2;
      }
    }
    return static_cast<size_t>(reflected);
  }

  static KLEIDICV_FORCE_INLINE void copy_pixel_channels(uint8_t* dst,
                                                        const uint8_t* src,
                                                        size_t channels) {
    switch (channels) {
      case 1:
        dst[0] = src[0];
        break;
      case 2:
        dst[0] = src[0];
        dst[1] = src[1];
        break;
      case 3:
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        break;
      case 4:
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        break;
      default:
        std::memcpy(dst, src, channels);
        break;
    }
  }

  static void copy_reflected_row_segment(uint8_t* dst, const uint8_t* src_row,
                                         const size_t* reflected_x,
                                         size_t reflected_count,
                                         size_t channels) {
    for (size_t i = 0; i < reflected_count; ++i) {
      copy_pixel_channels(dst + i * channels,
                          src_row + reflected_x[i] * channels, channels);
    }
  }

  static void fill_reflect_101_border_in_place(uint8_t* interior, size_t stride,
                                               size_t width, size_t height,
                                               size_t channels,
                                               size_t border_width,
                                               size_t border_height) {
    if (border_width == 0 && border_height == 0) {
      return;
    }

    const ptrdiff_t height_end = static_cast<ptrdiff_t>(height);
    const ptrdiff_t border_height_i = static_cast<ptrdiff_t>(border_height);
    const size_t row_bytes = width * channels;
    const size_t halo_bytes = border_width * channels;

    size_t* reflected_left_x = nullptr;
    size_t* reflected_right_x = nullptr;
    if (border_width > 0) {
      reflected_left_x =
          static_cast<size_t*>(std::malloc(border_width * sizeof(size_t)));
      reflected_right_x =
          static_cast<size_t*>(std::malloc(border_width * sizeof(size_t)));
      if (reflected_left_x == nullptr || reflected_right_x == nullptr) {
        std::free(reflected_left_x);
        std::free(reflected_right_x);
        return;
      }

      for (size_t x = 0; x < border_width; ++x) {
        reflected_left_x[x] = border_index_reflect_101(
            static_cast<ptrdiff_t>(x) - static_cast<ptrdiff_t>(border_width),
            width);
        reflected_right_x[x] =
            border_index_reflect_101(static_cast<ptrdiff_t>(width + x), width);
      }
    }

    auto fill_full_border_row = [&](ptrdiff_t dst_y, size_t src_y) {
      uint8_t* dst_row = interior + dst_y * static_cast<ptrdiff_t>(stride) -
                         static_cast<ptrdiff_t>(halo_bytes);
      const uint8_t* src_row = interior + src_y * stride;

      if (border_width > 0) {
        copy_reflected_row_segment(dst_row, src_row, reflected_left_x,
                                   border_width, channels);
      }

      std::memcpy(dst_row + halo_bytes, src_row, row_bytes);

      if (border_width > 0) {
        copy_reflected_row_segment(dst_row + halo_bytes + row_bytes, src_row,
                                   reflected_right_x, border_width, channels);
      }
    };

    for (ptrdiff_t y = -border_height_i; y < 0; ++y) {
      fill_full_border_row(y, border_index_reflect_101(y, height));
    }

    for (ptrdiff_t y = 0; y < height_end; ++y) {
      uint8_t* dst_row = interior + y * static_cast<ptrdiff_t>(stride) -
                         static_cast<ptrdiff_t>(halo_bytes);
      const uint8_t* src_row = interior + y * stride;

      if (border_width > 0) {
        copy_reflected_row_segment(dst_row, src_row, reflected_left_x,
                                   border_width, channels);
        copy_reflected_row_segment(dst_row + halo_bytes + row_bytes, src_row,
                                   reflected_right_x, border_width, channels);
      }
    }

    for (ptrdiff_t y = height_end; y < height_end + border_height_i; ++y) {
      fill_full_border_row(y, border_index_reflect_101(y, height));
    }

    std::free(reflected_left_x);
    std::free(reflected_right_x);
  }

  size_t level_count_;
  size_t channels_;
  size_t border_width_;
  size_t border_height_;
  OpticalFlowLevel* levels_;
};

static_assert(std::is_trivially_destructible<OpticalFlowLKPyramid>::value,
              "malloc-only pyramid storage requires trivial destruction");

inline void OpticalFlowLKPyramidDeleter::operator()(
    OpticalFlowLKPyramid* ptr) const {
  if (ptr != nullptr) {
    std::free(ptr);
  }
}

}  // namespace kleidicv

#endif  // KLEIDICV_BUILD_OPTICAL_FLOW_LK_PYRAMID_H
