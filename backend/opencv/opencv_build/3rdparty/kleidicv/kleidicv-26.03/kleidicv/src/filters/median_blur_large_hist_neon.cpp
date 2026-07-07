// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/median_blur.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "median_blur_border_handling.h"

namespace kleidicv::neon {

template <bool is_single_channel>
class MedianBlurLargeHist {
 public:
  MedianBlurLargeHist(size_t channels, size_t kMargin)
      : patched_coarse{static_cast<uint16_t*>(
            malloc((16 + 256) * (patch_length + 2 * kMargin) * channels *
                   sizeof(uint16_t)))},
        patched_fine{
            &patched_coarse[16 * (patch_length + 2 * kMargin) * channels]},
        H{},
        luc{} {}

  ~MedianBlurLargeHist() { free(patched_coarse); }

  void process_pixels_without_horizontal_borders(
      Rectangle image_dimensions, Point starting_coordinates,
      Point ending_coordinates, Rows<const uint8_t> src_rows,
      Rows<uint8_t> dst_rows, size_t ksize, FixedBorderType border_type) {
    const size_t kMargin = ksize / 2;

    for (size_t w = starting_coordinates.x(); w < ending_coordinates.x();
         w += patch_length) {
      const size_t total_patch_span =
          std::min(ending_coordinates.x() - w, patch_length) + kMargin * 2;

      Rows<uint8_t> shifted_dst{
          &dst_rows[0] + static_cast<ptrdiff_t>(w) * dst_rows.channels(),
          dst_rows.stride(), dst_rows.channels()};

      clear_patched_histogram(total_patch_span * src_rows.channels());

      for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(src_rows.channels());
           ++c) {
        clear_lookup_table();

        // We initialize with ksize rows to allow merging of
        // histogram increment and decrement operations in the main loop.
        // This extra initial load enables a single update phase and avoids
        // splitting the logic into separate steps.
        for (size_t r = 0; r < ksize; ++r) {
          const ptrdiff_t valid_h =
              get_physical_index(starting_coordinates.y() + r - kMargin,
                                 image_dimensions.height(), border_type);
          initialize_patched_histogram_without_horizontal_borders(
              src_rows, c, valid_h, total_patch_span, w, kMargin);
        }
        compute_patch_median_from_histogram(starting_coordinates.y(), c,
                                            total_patch_span, kMargin, ksize,
                                            shifted_dst);
      }

      for (size_t h = starting_coordinates.y() + 1; h < ending_coordinates.y();
           ++h) {
        const ptrdiff_t valid_old_h = get_physical_index(
            h - kMargin - 1, image_dimensions.height(), border_type);

        const ptrdiff_t valid_new_h = get_physical_index(
            h + kMargin, image_dimensions.height(), border_type);

        for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(src_rows.channels());
             ++c) {
          clear_lookup_table();

          update_patch_histogram_without_horizontal_borders(
              src_rows, c, valid_old_h, valid_new_h, total_patch_span, w,
              kMargin);

          compute_patch_median_from_histogram(h, c, total_patch_span, kMargin,
                                              ksize, shifted_dst);
        }
      }
    }
  }

  void process_pixels_with_horizontal_borders(
      Rectangle image_dimensions, Point starting_coordinates,
      Point ending_coordinates, Rows<const uint8_t> src_rows,
      Rows<uint8_t> dst_rows, size_t ksize, FixedBorderType border_type) {
    const size_t kMargin = ksize / 2;

    for (size_t w = starting_coordinates.x(); w < ending_coordinates.x();
         w += patch_length) {
      const size_t total_patch_span =
          std::min(ending_coordinates.x() - w, patch_length) + kMargin * 2;
      Rows<uint8_t> shifted_dst{
          &dst_rows[0] + static_cast<ptrdiff_t>(w) * dst_rows.channels(),
          dst_rows.stride(), dst_rows.channels()};

      clear_patched_histogram(total_patch_span * src_rows.channels());

      for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(src_rows.channels());
           ++c) {
        clear_lookup_table();
        // We initialize with ksize rows to allow merging of
        // histogram increment and decrement operations in the main loop.
        // This extra initial load enables a single update phase and avoids
        // splitting the logic into separate steps.
        for (size_t r = 0; r < ksize; ++r) {
          const ptrdiff_t valid_h =
              get_physical_index(starting_coordinates.y() + r - kMargin,
                                 image_dimensions.height(), border_type);
          initialize_patched_histogram_with_horizontal_borders(
              src_rows, c, valid_h, total_patch_span, w, kMargin,
              image_dimensions.width(), border_type);
        }
        compute_patch_median_from_histogram(starting_coordinates.y(), c,
                                            total_patch_span, kMargin, ksize,
                                            shifted_dst);
      }

      for (size_t h = starting_coordinates.y() + 1; h < ending_coordinates.y();
           ++h) {
        const ptrdiff_t valid_old_h = get_physical_index(
            h - kMargin - 1, image_dimensions.height(), border_type);
        const ptrdiff_t valid_new_h = get_physical_index(
            h + kMargin, image_dimensions.height(), border_type);
        for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(src_rows.channels());
             ++c) {
          clear_lookup_table();

          update_patch_histogram_with_horizontal_borders(
              src_rows, c, valid_old_h, valid_new_h, total_patch_span, w,
              kMargin, image_dimensions.width(), border_type);

          compute_patch_median_from_histogram(h, c, total_patch_span, kMargin,
                                              ksize, shifted_dst);
        }
      }
    }
  }

 private:
  // Histogram Buffer Layout Explanation
  // -----------------------------------
  // patched_coarse:
  //   - Conceptually a 3D array:
  //       coarse[channel_idx][patch_offset][coarse_bin]
  //   - coarse_bin = incoming_pixel >> 4 (in the range of [0, 15])
  //   - Flattened as:
  //       coarse_offset = 16 * (patch_length * channel_idx + patch_offset)
  //                       + (incoming_pixel >> 4);
  //
  // patched_fine:
  //   - Conceptually a 4D array:
  //       fine[channel_idx][coarse_bin][patch_offset][fine_bin]
  //   - coarse_bin = incoming_pixel >> 4 (in the range of [0, 15])
  //   - fine_bin   = incoming_pixel & 0xF (in the range of [0, 15])
  //   - Flattened as:
  //       fine_offset = 16 * (patch_length * (16 * channel_idx + coarse_bin)
  //                           + patch_offset) + fine_bin;
  //
  // This layout enables fast linear access while preserving the hierarchical
  // structure of histograms per channel and patch position.
  constexpr static size_t patch_length = 512;
  uint16_t* patched_coarse;
  uint16_t* patched_fine;

  // Clear only the portion of the patched histogram buffers that will be used
  // for the current patch.
  // Since these buffers are large, there's no need to zero out the entire
  // allocation - only the section relevant to the current total patch size is
  // cleared for efficiency.
  void clear_patched_histogram(size_t total_patch_size) {
    std::memset(patched_coarse, 0,
                16 * total_patch_size * sizeof(patched_coarse[0]));
    std::memset(patched_fine, 0,
                256 * total_patch_size * sizeof(patched_fine[0]));
  }

  void scalar_initialize_patched_histogram(int incoming_pixel,
                                           size_t channel_idx,
                                           size_t patch_length,
                                           size_t patch_offset) {
    const size_t coarse_offset =
        16 * (patch_length * channel_idx + patch_offset) +
        (incoming_pixel >> 4);
    const size_t fine_offset =
        16 * (patch_length * (16 * channel_idx + (incoming_pixel >> 4)) +
              patch_offset) +
        (incoming_pixel & 0xF);
    patched_coarse[coarse_offset]++;
    patched_fine[fine_offset]++;
  }

  void vector_initialize_patched_histogram(uint8x16_t& incoming_pixels,
                                           size_t channel_idx,
                                           size_t patch_length,
                                           size_t patch_offset) {
    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 16; i++) {
      const size_t coarse_offset_incoming =
          16 * (patch_length * channel_idx + patch_offset + i) +
          (incoming_pixels[i] >> 4);

      const size_t fine_offset_incoming =
          16 * (patch_length * (16 * channel_idx + (incoming_pixels[i] >> 4)) +
                patch_offset + i) +
          (incoming_pixels[i] & 0xF);

      patched_coarse[coarse_offset_incoming]++;
      patched_fine[fine_offset_incoming]++;
    }
  }

  void initialize_patched_histogram_without_horizontal_borders(
      Rows<const uint8_t> src_rows, ptrdiff_t c, ptrdiff_t valid_h,
      size_t total_patch_span, size_t starting_width, size_t kMargin) {
    size_t vector_part = 0;
    if constexpr (is_single_channel) {
      vector_part = (total_patch_span >> 4) << 4;
      for (size_t patch_offset = 0; patch_offset < vector_part;
           patch_offset += 16) {
        const ptrdiff_t valid_w =
            static_cast<ptrdiff_t>(starting_width + patch_offset - kMargin);
        auto incoming_pixels = vld1q_u8(&src_rows.at(valid_h, valid_w)[c]);
        vector_initialize_patched_histogram(incoming_pixels, c,
                                            total_patch_span, patch_offset);
      }
    }

    for (size_t patch_offset = vector_part; patch_offset < total_patch_span;
         ++patch_offset) {
      const ptrdiff_t valid_w =
          static_cast<ptrdiff_t>(starting_width + patch_offset - kMargin);
      auto incoming_pixel = src_rows.at(valid_h, valid_w)[c];

      scalar_initialize_patched_histogram(incoming_pixel, c, total_patch_span,
                                          patch_offset);
    }
  }

  void initialize_patched_histogram_with_horizontal_borders(
      Rows<const uint8_t> src_rows, ptrdiff_t c, ptrdiff_t valid_h,
      size_t total_patch_span, size_t starting_width, size_t kMargin,
      size_t width, FixedBorderType border_type) {
    for (size_t patch_offset = 0; patch_offset < total_patch_span;
         ++patch_offset) {
      ptrdiff_t valid_w = get_physical_index(
          starting_width + patch_offset - kMargin, width, border_type);

      auto incoming_pixel = src_rows.at(valid_h, valid_w)[c];

      scalar_initialize_patched_histogram(incoming_pixel, c, total_patch_span,
                                          patch_offset);
    }
  }

  // During vertical traversal (the main 'height' loop), each sliding window
  // iteration introduces a new incoming row and removes an outgoing one. The
  // histogram must be updated accordingly by subtracting the contributions of
  // the outgoing row and adding those of the incoming row.
  // Both increment and decrement operations are handled inside the same
  // function.
  void scalar_update_patch_histogram(int outgoing_pixel, int incoming_pixel,
                                     size_t channel_idx, size_t patch_length,
                                     size_t patch_offset) {
    const size_t coarse_offset_base =
        16 * (patch_length * channel_idx + patch_offset);
    const size_t fine_offset_base =
        16 * patch_length * 16 * channel_idx + 16 * patch_offset;

    const size_t pixel_new_shift_right_4 = (incoming_pixel >> 4);
    const size_t mask_new_pixel = (incoming_pixel & 0xF);
    const size_t pixel_old_shift_right_4 = (outgoing_pixel >> 4);
    const size_t mask_old_pixel = (outgoing_pixel & 0xF);

    const size_t fine_new_offset = fine_offset_base + mask_new_pixel +
                                   16 * patch_length * pixel_new_shift_right_4;
    const size_t coarse_new_offset =
        coarse_offset_base + pixel_new_shift_right_4;

    const size_t fine_old_offset = fine_offset_base + mask_old_pixel +
                                   16 * patch_length * pixel_old_shift_right_4;
    const size_t coarse_old_offset =
        coarse_offset_base + pixel_old_shift_right_4;

    patched_coarse[coarse_new_offset]++;
    patched_coarse[coarse_old_offset]--;
    patched_fine[fine_new_offset]++;
    patched_fine[fine_old_offset]--;
  }

  void vector_update_patch_histogram(uint8x16_t& outgoing_pixels,
                                     uint8x16_t& incoming_pixels,
                                     size_t channel_idx, size_t patch_length,
                                     size_t patch_offset) {
    KLEIDICV_FORCE_LOOP_UNROLL
    for (int i = 0; i < 16; i++) {
      const size_t coarse_offset_incoming =
          16 * (patch_length * channel_idx + patch_offset + i) +
          (incoming_pixels[i] >> 4);

      const size_t coarse_offset_outgoing =
          16 * (patch_length * channel_idx + patch_offset + i) +
          (outgoing_pixels[i] >> 4);

      const size_t fine_offset_incoming =
          16 * (patch_length * (16 * channel_idx + (incoming_pixels[i] >> 4)) +
                patch_offset + i) +
          (incoming_pixels[i] & 0xF);

      const size_t fine_offset_outgoing =
          16 * (patch_length * (16 * channel_idx + (outgoing_pixels[i] >> 4)) +
                patch_offset + i) +
          (outgoing_pixels[i] & 0xF);

      patched_coarse[coarse_offset_incoming]++;
      patched_coarse[coarse_offset_outgoing]--;
      patched_fine[fine_offset_incoming]++;
      patched_fine[fine_offset_outgoing]--;
    }
  }

  void update_patch_histogram_without_horizontal_borders(
      Rows<const uint8_t> src_rows, ptrdiff_t c, ptrdiff_t valid_old_h,
      ptrdiff_t valid_new_h, size_t total_patch_span, size_t starting_width,
      size_t kMargin) {
    size_t vector_part = 0;
    if constexpr (is_single_channel) {
      vector_part = (total_patch_span >> 4) << 4;
      for (size_t patch_offset = 0; patch_offset < vector_part;
           patch_offset += 16) {
        const ptrdiff_t valid_w =
            static_cast<ptrdiff_t>(starting_width + patch_offset - kMargin);
        auto outgoing_pixels = vld1q_u8(&src_rows.at(valid_old_h, valid_w)[c]);
        auto incoming_pixels = vld1q_u8(&src_rows.at(valid_new_h, valid_w)[c]);
        vector_update_patch_histogram(outgoing_pixels, incoming_pixels, c,
                                      total_patch_span, patch_offset);
      }
    }

    for (size_t patch_offset = vector_part; patch_offset < total_patch_span;
         ++patch_offset) {
      const ptrdiff_t valid_w =
          static_cast<ptrdiff_t>(starting_width + patch_offset - kMargin);
      auto outgoing_pixel = src_rows.at(valid_old_h, valid_w)[c];
      auto incoming_pixel = src_rows.at(valid_new_h, valid_w)[c];
      scalar_update_patch_histogram(outgoing_pixel, incoming_pixel, c,
                                    total_patch_span, patch_offset);
    }
  }

  void update_patch_histogram_with_horizontal_borders(
      Rows<const uint8_t> src_rows, ptrdiff_t c, ptrdiff_t valid_old_h,
      ptrdiff_t valid_new_h, size_t total_patch_span, size_t starting_width,
      size_t kMargin, size_t width, FixedBorderType border_type) {
    for (size_t patch_offset = 0; patch_offset < total_patch_span;
         ++patch_offset) {
      const ptrdiff_t valid_w = get_physical_index(
          starting_width + patch_offset - kMargin, width, border_type);
      auto outgoing_pixel = src_rows.at(valid_old_h, valid_w)[c];
      auto incoming_pixel = src_rows.at(valid_new_h, valid_w)[c];
      scalar_update_patch_histogram(outgoing_pixel, incoming_pixel, c,
                                    total_patch_span, patch_offset);
    }
  }

  // `H` and `luc` are used for computing the median value for a single
  // output pixel:
  // - `H` is a histogram structure holding both the coarse and fine
  // bins needed
  //   for the current element's median calculation.
  // - `luc` is a lookup table that stores the last processed offset
  // (index)
  //   for each coarse bin. This allows incremental fine histogram
  //   updates instead of full recalculation when histogram overlap is
  //   high. Since neighboring patches in natural images often have
  //   similar pixel values, reusing previous histogram state can
  //   significantly reduce processing time.
  typedef struct {
    uint16_t coarse[16];
    uint16_t fine[16][16];
  } Histogram;

  Histogram H;
  uint16_t luc[16];

  void clear_lookup_table(void) {
    std::memset(&H, 0, sizeof(Histogram));
    std::memset(luc, 0, 16 * sizeof(uint16_t));
  }

  void initialize_coarse(uint16_t* px, uint16x8_t& v_coarsel,
                         uint16x8_t& v_coarseh, size_t kMargin) {
    for (size_t i = 0; i < 2 * kMargin; ++i, px += 16) {
      v_coarsel = vaddq_u16(v_coarsel, vld1q_u16(px));
      v_coarseh = vaddq_u16(v_coarseh, vld1q_u16(px + 8));
    }
  }

  void increment_coarse(uint16_t* px, uint16x8_t& v_coarsel,
                        uint16x8_t& v_coarseh) {
    v_coarsel = vaddq_u16(v_coarsel, vld1q_u16(px));
    v_coarseh = vaddq_u16(v_coarseh, vld1q_u16(px + 8));
    vst1q_u16(H.coarse, v_coarsel);
    vst1q_u16(H.coarse + 8, v_coarseh);
  }

  void decrement_coarse(uint16_t* px, uint16x8_t& v_coarsel,
                        uint16x8_t& v_coarseh) {
    v_coarsel = vsubq_u16(v_coarsel, vld1q_u16(px));
    v_coarseh = vsubq_u16(v_coarseh, vld1q_u16(px + 8));
  }

  void initialize_fine(uint16_t* px, uint16x8_t& v_finel, uint16x8_t& v_fineh,
                       uint16_t& luc, size_t patch_offset, size_t kMargin,
                       size_t total_patch_span) {
    for (luc = static_cast<uint16_t>(patch_offset - kMargin);
         luc < static_cast<uint16_t>(
                   std::min(patch_offset + kMargin + 1, total_patch_span));
         ++luc, px += 16) {
      v_finel = vaddq_u16(v_finel, vld1q_u16(px));
      v_fineh = vaddq_u16(v_fineh, vld1q_u16(px + 8));
    }
  }

  void update_fine(uint16_t* px, uint16x8_t& v_finel, uint16x8_t& v_fineh,
                   uint16_t& luc, size_t patch_offset, size_t kMargin,
                   size_t total_patch_span) {
    for (; luc < static_cast<uint16_t>(patch_offset + kMargin + 1); ++luc) {
      constexpr ptrdiff_t stride = 16;
      const ptrdiff_t patch_span_limit =
          static_cast<ptrdiff_t>(total_patch_span - 1);
      const ptrdiff_t safe_luc = static_cast<ptrdiff_t>(luc);

      const ptrdiff_t base_offset =
          stride * std::min(safe_luc, patch_span_limit);
      const ptrdiff_t old_offset =
          stride * std::max(safe_luc - static_cast<ptrdiff_t>(2 * kMargin + 1),
                            ptrdiff_t{0});
      const uint16x8_t new_vecl = vld1q_u16(px + base_offset);
      const uint16x8_t new_vech = vld1q_u16(px + base_offset + 8);
      const uint16x8_t old_vecl = vld1q_u16(px + old_offset);
      const uint16x8_t old_vech = vld1q_u16(px + old_offset + 8);
      v_finel = vsubq_u16(vaddq_u16(v_finel, new_vecl), old_vecl);
      v_fineh = vsubq_u16(vaddq_u16(v_fineh, new_vech), old_vech);
    }
  }

  size_t find_coarse_index(size_t& cdf, size_t median_index) {
    size_t coarse_index = 0;
    while (true) {
      cdf += H.coarse[coarse_index];
      if (cdf > median_index) {
        cdf -= H.coarse[coarse_index];
        break;
      }
      coarse_index++;
    }
    return coarse_index;
  }

  uint8_t find_median(size_t& cdf, size_t median_index, size_t coarse_index) {
    uint16_t* segment = H.fine[coarse_index];
    size_t fine_index = 0;
    while (true) {
      cdf += segment[fine_index];
      if (cdf > median_index) {
        fine_index = (16 * coarse_index + fine_index);
        break;
      }
      fine_index++;
    }
    return uint8_t(fine_index);
  }

  void compute_patch_median_from_histogram(size_t h, ptrdiff_t c,
                                           size_t total_patch_span,
                                           size_t kMargin, size_t ksize,
                                           Rows<uint8_t> dst) {
    uint16x8_t v_coarsel = vld1q_u16(H.coarse);
    uint16x8_t v_coarseh = vld1q_u16(H.coarse + 8);

    // Before starting the main patch loop to compute medians for each element,
    // we initialize the coarse histogram buffer with the first (ksize - 1).
    // This allows each subsequent iteration in the patch loop to perform only
    // one addition and one subtraction.
    uint16_t* px = patched_coarse + 16 * total_patch_span * c;
    initialize_coarse(px, v_coarsel, v_coarseh, kMargin);

    for (size_t patch_offset = kMargin;
         patch_offset < total_patch_span - kMargin; patch_offset++) {
      size_t median_index = (ksize * ksize) / 2, cdf = 0;

      px = patched_coarse +
           16 * (total_patch_span * c +
                 std::min(patch_offset + kMargin, total_patch_span - 1));
      increment_coarse(px, v_coarsel, v_coarseh);

      // Find median at coarse level
      size_t coarse_index = find_coarse_index(cdf, median_index);

      uint16x8_t v_finel;
      uint16x8_t v_fineh;
      // Check whether the fine histogram (H.fine[coarse_index]) for the
      // current patch position needs to be freshly initialized or can be
      // incrementally updated. This decision hinges on the `luc` (Last Used
      // Coordinate) table, which records the last horizontal patch offset
      // processed for each coarse bin.
      //
      // The condition is true in two scenarios:
      //
      // 1. **First-Time Initialization**:
      //    - This is the first time we are accessing this coarse bin
      //    (`coarse_index`) at the current patch position.
      //    - We compute the full fine histogram from scratch by summing
      //    over `ksize` rows centered at the patch position.
      //    - We accumulate the results into the `v_finel` and `v_fineh`
      //    vector registers.
      //    - These vectors are then stored into `H.fine[coarse_index]`.
      //    - The `luc` table is updated to reflect the last index
      //    processed.
      //
      // 2. **Window Movement Causes Loss of Overlap**:
      //    - The sliding window has moved enough that it no longer
      //    sufficiently overlaps the region
      //      used to compute the previously cached fine histogram (i.e.,
      //      `luc[coarse_index]` is too far behind).
      //    - We must reinitialize the fine histogram to ensure accuracy.
      //
      // Otherwise:
      //
      // - We reuse the previously computed fine histogram stored in
      // `H.fine[coarse_index]`.
      // - We only update it incrementally using the `update_fine()`
      // function, which:
      //     - Adds the new values entering the window.
      //     - Subtracts the values leaving the window.
      // - This avoids the need for a full re-scan, leveraging temporal
      // locality between neighboring pixels.
      //
      // This lookup-based optimization significantly improves performance,
      // as neighboring filter windows often overlap heavily - especially for
      // small strides and moderate kernel sizes.
      //
      // After this step, the fine histogram is ready. The next phase scans
      // `H.fine[coarse_index]` to identify the fine bin where the
      // cumulative sum crosses the median threshold. This gives us the
      // final median value for the output pixel.
      if (luc[coarse_index] <= patch_offset - kMargin) {
        v_finel = vdupq_n_u16(0);
        v_fineh = vdupq_n_u16(0);
        px = patched_fine + static_cast<ptrdiff_t>(16) *
                                (static_cast<ptrdiff_t>(total_patch_span) *
                                     (16 * c + coarse_index) +
                                 patch_offset - kMargin);

        initialize_fine(px, v_finel, v_fineh, luc[coarse_index], patch_offset,
                        kMargin, total_patch_span);

      } else {
        v_finel = vld1q_u16(H.fine[coarse_index]);
        v_fineh = vld1q_u16(H.fine[coarse_index] + 8);
        px = patched_fine + 16 * total_patch_span * (16 * c + coarse_index);
        update_fine(px, v_finel, v_fineh, luc[coarse_index], patch_offset,
                    kMargin, total_patch_span);
      }

      px = patched_coarse +
           static_cast<ptrdiff_t>(16) *
               (total_patch_span * c +
                std::max(patch_offset - kMargin, static_cast<size_t>(0)));

      vst1q_u16(H.fine[coarse_index], v_finel);
      vst1q_u16(H.fine[coarse_index] + 8, v_fineh);

      decrement_coarse(px, v_coarsel, v_coarseh);

      // Find median at fine level
      dst.at(static_cast<ptrdiff_t>(h),
             static_cast<ptrdiff_t>(patch_offset - kMargin))[c] =
          find_median(cdf, median_index, coarse_index);
    }
  }
};

template <bool is_single_channel>
void median_process(Rectangle image_dimensions, Rows<const uint8_t> src_rows,
                    Rows<uint8_t> dst_rows, size_t y_begin, size_t y_end,
                    size_t kernel_width, size_t kernel_height,
                    FixedBorderType border_type) {
  size_t kMargin = (kernel_width - 1) / 2;
  MedianBlurLargeHist<is_single_channel> median_filter{src_rows.channels(),
                                                       kMargin};

  // Process left border
  size_t starting_width = 0;
  size_t processing_left_width = kMargin;
  Point starting_left_coordinates{starting_width, y_begin};
  Point ending_left_coordinates{starting_width + processing_left_width, y_end};
  median_filter.process_pixels_with_horizontal_borders(
      image_dimensions, starting_left_coordinates, ending_left_coordinates,
      src_rows, dst_rows, kernel_height, border_type);

  // Process center region
  starting_width = processing_left_width;
  size_t processing_center_width = image_dimensions.width() - 2 * kMargin;
  Point starting_center_coordinates{starting_width, y_begin};
  Point ending_center_coordinates{starting_width + processing_center_width,
                                  y_end};
  median_filter.process_pixels_without_horizontal_borders(
      image_dimensions, starting_center_coordinates, ending_center_coordinates,
      src_rows, dst_rows, kernel_height, border_type);

  // Process right border
  starting_width = processing_left_width + processing_center_width;
  size_t processing_right_width = image_dimensions.width() -
                                  processing_left_width -
                                  processing_center_width;
  Point starting_right_coordinates{starting_width, y_begin};
  Point ending_right_coordinates{starting_width + processing_right_width,
                                 y_end};
  median_filter.process_pixels_with_horizontal_borders(
      image_dimensions, starting_right_coordinates, ending_right_coordinates,
      src_rows, dst_rows, kernel_height, border_type);
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t median_blur_large_hist_stripe_u8(
    const uint8_t* src, size_t src_stride, uint8_t* dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type) {
  Rectangle image_dimensions{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<uint8_t> dst_rows{dst, dst_stride, channels};

  if (channels == 1) {
    median_process<true>(image_dimensions, src_rows, dst_rows, y_begin, y_end,
                         kernel_width, kernel_height, border_type);
  } else {
    median_process<false>(image_dimensions, src_rows, dst_rows, y_begin, y_end,
                          kernel_width, kernel_height, border_type);
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
