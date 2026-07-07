// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/median_blur.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "median_blur_border_handling.h"

namespace kleidicv::neon {

// B. Weiss, "Fast Median and Bilateral Filtering," in *ACM SIGGRAPH 2006
// Papers*, ACM, New York, NY, USA, pp. 519-526, 2006.
// The paper is currently available at:
// http://mesh.brown.edu/engn1610/refs/Weiss-siggraph2006.pdf
class MedianBlurSmallHist {
 public:
  MedianBlurSmallHist() : fine{}, coarse{} {}

  void process_pixels_with_horizontal_borders(
      Rectangle image_dimensions, Point starting_coordinates,
      Point ending_coordinates, Rows<const uint8_t> src_rows,
      Rows<uint8_t> dst_rows, size_t ksize, FixedBorderType border_type) {
    const size_t kMargin = ksize / 2;

    for (size_t w = starting_coordinates.x(); w < ending_coordinates.x(); w++) {
      for (ptrdiff_t ch = 0; ch < static_cast<ptrdiff_t>(src_rows.channels());
           ch++) {
        scalar_clear_histogram();

        // We initialize with ksize rows to allow merging of
        // histogram increment and decrement operations in the main loop.
        // This extra initial load enables a single update phase and avoids
        // splitting the logic into separate steps.
        for (size_t r = 0; r < ksize; r++) {
          for (size_t c = 0; c < ksize; c++) {
            const ptrdiff_t valid_h =
                get_physical_index(starting_coordinates.y() + r - kMargin,
                                   image_dimensions.height(), border_type);
            const ptrdiff_t valid_w = get_physical_index(
                w + c - kMargin, image_dimensions.width(), border_type);

            uint8_t pixel = src_rows.at(valid_h, valid_w)[ch];

            scalar_initialize_histogram(pixel);
          }
        }

        const uint8_t median_value = scalar_find_median(ksize);

        dst_rows.at(static_cast<ptrdiff_t>(starting_coordinates.y()),
                    static_cast<ptrdiff_t>(w))[ch] = median_value;

        for (size_t h = starting_coordinates.y() + 1;
             h < ending_coordinates.y(); h++) {
          const ptrdiff_t valid_new_h = get_physical_index(
              h + kMargin, image_dimensions.height(), border_type);

          const ptrdiff_t valid_old_h = get_physical_index(
              h - kMargin - 1, image_dimensions.height(), border_type);

          for (size_t c = 0; c < ksize; c++) {
            const ptrdiff_t valid_w = get_physical_index(
                w + c - kMargin, image_dimensions.width(), border_type);

            uint8_t incoming_pixel = src_rows.at(valid_new_h, valid_w)[ch];

            uint8_t outgoing_pixel = src_rows.at(valid_old_h, valid_w)[ch];

            scalar_update_histogram(incoming_pixel, outgoing_pixel);
          }

          const uint8_t median_value = scalar_find_median(ksize);

          dst_rows.at(static_cast<ptrdiff_t>(h),
                      static_cast<ptrdiff_t>(w))[ch] = median_value;
        }
      }
    }
  }

  void process_pixels_without_horizontal_borders(
      Rectangle image_dimensions, Point starting_coordinates,
      Point ending_coordinates, Rows<const uint8_t> src_rows,
      Rows<uint8_t> dst_rows, size_t ksize, FixedBorderType border_type) {
    const size_t step = sizeof(uint8x16_t);
    const size_t KMargin_w = (ksize / 2) * src_rows.channels();
    const size_t KMargin_h = (ksize / 2);

    for (size_t w = starting_coordinates.x(); w < ending_coordinates.x();
         w += step) {
      vector_clear_histogram();

      // We initialize with ksize rows to allow merging of
      // histogram increment and decrement operations in the main loop.
      // This extra initial load enables a single update phase and avoids
      // splitting the logic into separate steps.
      for (size_t r = 0; r < ksize; r++) {
        const ptrdiff_t vertical_index =
            get_physical_index(starting_coordinates.y() + r - KMargin_h,
                               image_dimensions.height(), border_type);

        for (size_t c = 0; c < ksize; c++) {
          const size_t horizontal_index =
              w + c * src_rows.channels() - KMargin_w;

          uint8x16_t pixel = vld1q_u8(
              &src_rows[vertical_index * src_rows.stride() + horizontal_index]);

          vector_initialize_histogram(pixel);
        }
      }

      const uint8x16_t median_value = vector_find_median(ksize);

      vst1q_u8(&dst_rows[starting_coordinates.y() * dst_rows.stride() + w],
               median_value);

      for (size_t h = starting_coordinates.y() + 1; h < ending_coordinates.y();
           ++h) {
        const ptrdiff_t vertical_index_new = get_physical_index(
            h + KMargin_h, image_dimensions.height(), border_type);

        const ptrdiff_t vertical_index_old = get_physical_index(
            h - KMargin_h - 1, image_dimensions.height(), border_type);

        for (size_t c = 0; c < ksize; c++) {
          size_t horizontal_index = w + c * src_rows.channels() - KMargin_w;

          uint8x16_t incoming_pixels =
              vld1q_u8(&src_rows[vertical_index_new * src_rows.stride() +
                                 horizontal_index]);

          uint8x16_t outgoing_pixels =
              vld1q_u8(&src_rows[vertical_index_old * src_rows.stride() +
                                 horizontal_index]);

          vector_update_histogram(incoming_pixels, outgoing_pixels);
        }

        const uint8x16_t median_value = vector_find_median(ksize);

        vst1q_u8(&dst_rows[h * dst_rows.stride() + w], median_value);
      }
    }
  }

 private:
  // The 'fine' and 'coarse' histograms are shared between both scalar and
  // vector operations. Their buffer sizes are allocated based on the
  // vectorized case to ensure compatibility and avoid reallocation.
  // In case of vectorized execution, 'fine' and 'coarse' are actually
  // implemented as 16 interleaved histograms, one per vector lane.
  uint8_t fine[4096];
  uint8_t coarse[256];

  // In scalar_clear_histogram, we only clear the relevant portions of the
  // 'fine' and 'coarse' buffers that are actually used during computation. This
  // avoids unnecessary memory operations.
  void scalar_clear_histogram() {
    memset(fine, 0, sizeof(fine[0]) * 256);
    memset(coarse, 0, sizeof(coarse[0]) * 16);
  }

  void vector_clear_histogram() {
    memset(fine, 0, sizeof(uint8_t) * 4096);
    memset(coarse, 0, sizeof(uint8_t) * 256);
  }

  // Before the main vertical loop over 'height', the histogram must be
  // initialized for each new 'width'. This is done using either
  // scalar_initialize_histogram or vector_initialize_histogram depending on the
  // processing mode. These functions preload the histogram using rows from the
  // source image to enable efficient sliding window updates during vertical
  // traversal.
  void scalar_initialize_histogram(uint8_t incoming_pixel) {
    fine[incoming_pixel]++;
    coarse[incoming_pixel >> 4]++;
  }

  void vector_initialize_histogram(uint8x16_t& incoming_pixels) {
    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 16; i++) {
      fine[incoming_pixels[i] * 16 + i]++;
    }

    incoming_pixels = vshrq_n_u8(incoming_pixels, 4);

    uint8x16_t* vec_coarse = reinterpret_cast<uint8x16_t*>(coarse);
    vec_coarse[0] = vsubq(vec_coarse[0], vceqzq_u8(incoming_pixels));
    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 1; i < 16; i++) {
      uint8x16_t index = vdupq_n_u8(i);
      vec_coarse[i] = vsubq(vec_coarse[i], vceqq(incoming_pixels, index));
    }
  }

  // During vertical traversal (the main 'height' loop), each sliding window
  // iteration introduces a new incoming row and removes an outgoing one. The
  // histogram must be updated accordingly by subtracting the contributions of
  // the outgoing row and adding those of the incoming row.
  // In many cases, incoming and outgoing pixels may be equal, so we perform a
  // conditional check to avoid unnecessary updates.
  // Both increment and decrement operations are handled inside the same
  // function (scalar_update_histogram / vector_update_histogram) for
  // efficiency.
  void scalar_update_histogram(uint8_t& incoming_pixel,
                               uint8_t& outgoing_pixel) {
    if (incoming_pixel != outgoing_pixel) {
      fine[incoming_pixel]++;
      coarse[incoming_pixel >> 4]++;
      fine[outgoing_pixel]--;
      coarse[outgoing_pixel >> 4]--;
    }
  }

  void vector_update_histogram(uint8x16_t& incoming_pixels,
                               uint8x16_t& outgoing_pixels) {
    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 16; i++) {
      fine[incoming_pixels[i] * 16 + i]++;
      fine[outgoing_pixels[i] * 16 + i]--;
    }

    uint8x16_t* vec_coarse = reinterpret_cast<uint8x16_t*>(coarse);
    incoming_pixels = vshrq_n<4>(incoming_pixels);
    outgoing_pixels = vshrq_n<4>(outgoing_pixels);

    uint8x16_t delta =
        vsubq(vceqzq_u8(outgoing_pixels), vceqzq_u8(incoming_pixels));
    vec_coarse[0] = vaddq(vec_coarse[0], delta);

    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 1; i < 16; i++) {
      uint8x16_t index = vdupq_n_u8(i);
      delta =
          vsubq(vceqq(outgoing_pixels, index), vceqq(incoming_pixels, index));
      vec_coarse[i] = vaddq(vec_coarse[i], delta);
    }
  }

  // To find the median efficiently, we first scan the coarse histogram to
  // identify the segment (coarse bin) where the median value lies. This helps
  // narrow down the search range in the fine histogram. Once the correct coarse
  // bin is located, we scan the corresponding segment in the fine histogram
  // until the cumulative distribution function (CDF) reaches the target CDF
  uint8_t scalar_find_median(size_t ksize) {
    // The target median index in a sorted window
    const uint8_t target_cdf = (ksize * ksize) / 2;

    // Variables for histogram scanning
    uint8_t cumulative_sum = 0;
    int fine_index = 0;
    int coarse_index = 0;

    // Phase 1: Coarse histogram scan to find the correct bin range
    while (true) {
      if ((cumulative_sum + coarse[coarse_index]) > target_cdf) {
        fine_index = coarse_index * 16;
        break;
      }
      cumulative_sum += coarse[coarse_index];
      coarse_index++;
    }

    // Phase 2: Fine histogram scan to locate the exact median value
    while (true) {
      cumulative_sum += fine[fine_index];
      if (cumulative_sum > target_cdf) {
        break;
      }
      fine_index++;
    }

    return fine_index;
  }

  uint8x16_t vector_find_median(size_t ksize) {
    // Calculate the target median index based on kernel size
    const uint8x16_t target_cdf = vdupq_n_u8((ksize * ksize) / 2);

    // Cumulative sum vector used for tracking the running histogram total
    uint8x16_t cumulative_sum = vdupq_n_u8(0);

    // Coarse histogram pointer (used to narrow the search)
    uint8x16_t* coarse_histogram = reinterpret_cast<uint8x16_t*>(coarse);

    // Coarse pass: Locate the coarse histogram bin range likely containing the
    // median value This step identifies the starting fine histogram index for
    // each lane, based on cumulative counts. It does not find the actual median
    // yet.
    int coarse_index = 0;
    int fine_index = 0;

    while (true) {
      uint8x16_t cumulative_sum_next =
          vaddq(cumulative_sum, coarse_histogram[coarse_index]);
      uint8x16_t coarse_threshold_exceeded =
          vcgtq_u8(cumulative_sum_next, target_cdf);

      if (any_lane_set(coarse_threshold_exceeded)) {
        fine_index = coarse_index * 16;
        break;
      }

      cumulative_sum = cumulative_sum_next;
      coarse_index++;
    }

    // Fine pass: Scan the fine histogram to find the exact median per lane
    uint8x16_t median_result = vdupq_n_u8(0);
    uint8x16_t lane_found_mask = vdupq_n_u8(0);
    uint8x16_t* fine_histogram = reinterpret_cast<uint8x16_t*>(fine);

    while (true) {
      cumulative_sum = vaddq(cumulative_sum, fine_histogram[fine_index]);

      uint8x16_t still_searching_mask = vceqzq_u8(lane_found_mask);
      median_result =
          vbslq_u8(still_searching_mask, vdupq_n_u8(fine_index), median_result);
      lane_found_mask =
          vorrq_u8(lane_found_mask, vcgtq_u8(cumulative_sum, target_cdf));

      if (all_lane_set(lane_found_mask)) {
        break;
      }
      fine_index++;
    }

    return median_result;
  }

  bool all_lane_set(uint8x16_t& v_u8) {
    uint32x4_t v_u32 = vreinterpretq_u32_u8(v_u8);
    return vminvq_u32(v_u32) == 0xffffffff;
  }

  bool any_lane_set(uint8x16_t v_u8) {
    uint32x4_t v_u32 = vreinterpretq_u32_u8(v_u8);
    return vmaxvq_u32(v_u32) != 0;
  }
};

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t median_blur_small_hist_stripe_u8(
    const uint8_t* src, size_t src_stride, uint8_t* dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type) {
  Rectangle image_dimensions{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<uint8_t> dst_rows{dst, dst_stride, channels};
  MedianBlurSmallHist median_filter;
  const size_t kMargin = kernel_width / 2;

  // Process left border
  size_t starting_width = 0;
  const size_t processing_left_width = kMargin;
  Point starting_left_coordinates{starting_width, y_begin};
  Point ending_left_coordinates{starting_width + processing_left_width, y_end};

  median_filter.process_pixels_with_horizontal_borders(
      image_dimensions, starting_left_coordinates, ending_left_coordinates,
      src_rows, dst_rows, kernel_height, border_type);

  // Process center region
  starting_width = processing_left_width;
  // Compute the width of the center region that can be processed with NEON
  // instructions. Subtract 2 * kMargin to exclude left and right borders, which
  // are handled separately using scalar code due to varying border modes (e.g.,
  // REPLICATE, REFLECT, WRAP, REVERSE). Align the remaining width down to the
  // nearest multiple of 16 to match NEON's 128-bit register width (16 bytes for
  // uint8x16_t).
  const size_t processing_center_width = ((width - 2 * kMargin) / 16) * 16;
  Point starting_center_coordinates{starting_width * channels, y_begin};
  Point ending_center_coordinates{
      (processing_center_width + starting_width) * channels, y_end};

  median_filter.process_pixels_without_horizontal_borders(
      image_dimensions, starting_center_coordinates, ending_center_coordinates,
      src_rows, dst_rows, kernel_height, border_type);

  // Process right border
  starting_width = processing_left_width + processing_center_width;
  const size_t processing_right_width =
      width - processing_left_width - processing_center_width;
  Point starting_right_coordinates{starting_width, y_begin};
  Point ending_right_coordinates{starting_width + processing_right_width,
                                 y_end};

  median_filter.process_pixels_with_horizontal_borders(
      image_dimensions, starting_right_coordinates, ending_right_coordinates,
      src_rows, dst_rows, kernel_height, border_type);

  return KLEIDICV_OK;
}
}  // namespace kleidicv::neon
