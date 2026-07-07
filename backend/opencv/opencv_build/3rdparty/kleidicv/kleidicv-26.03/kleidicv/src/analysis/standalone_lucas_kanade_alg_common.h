// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_COMMON_H
#define KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_COMMON_H

#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <variant>

#include "kleidicv/analysis/standalone_lucas_kanade_alg.h"
#include "kleidicv/containers/small_buffer.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {
// Fixed-point configuration for subpixel interpolation in LK sampling.
// Matches OpenCV's Lucas-Kanade fixed-point settings for conformity.
static constexpr int kLucasKanadeAlgFractionBits = 14;
static constexpr float kLucasKanadeAlgFixedPointDescale = (1.0F / (1 << 20));

template <int fraction_bits>
static KLEIDICV_FORCE_INLINE int round_fixed_point(int x) KLEIDICV_STREAMING {
  int half = 1 << (fraction_bits - 1);
  return (x + half) >> fraction_bits;
}

// Function complexity is above the threshold, but keeping the validation
// in one place is clearer here.
// NOLINTBEGIN(readability-function-cognitive-complexity)
static kleidicv_error_t validate_standalone_lucas_kanade_alg_u8_args(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *scharr_data, size_t scharr_stride_bytes,
    const uint8_t *next_data, size_t next_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, int window_width, int window_height) {
  if (width <= 0 || height <= 0 || channels <= 0 || window_width <= 0 ||
      window_height <= 0) {
    return KLEIDICV_ERROR_RANGE;
  }
  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  CHECK_IMAGE_SIZE(static_cast<size_t>(width), static_cast<size_t>(height));
  CHECK_POINTERS(prev_data, scharr_data, next_data);
  if (point_count > 0) {
    CHECK_POINTERS(prev_points, next_points);
  }
  CHECK_POINTER_AND_STRIDE(prev_data, prev_data_stride,
                           static_cast<size_t>(height));
  CHECK_POINTER_AND_STRIDE(next_data, next_stride, static_cast<size_t>(height));
  CHECK_POINTER_AND_STRIDE(scharr_data, scharr_stride_bytes,
                           static_cast<size_t>(height));
  if ((scharr_stride_bytes % sizeof(int16_t)) != 0) {
    return KLEIDICV_ERROR_ALIGNMENT;
  }

  const size_t row_elems =
      static_cast<size_t>(width) * static_cast<size_t>(channels);
  const size_t image_row_bytes = row_elems * sizeof(uint8_t);
  if (prev_data_stride < image_row_bytes || next_stride < image_row_bytes) {
    return KLEIDICV_ERROR_RANGE;
  }

  const size_t scharr_row_bytes = row_elems * 2U * sizeof(int16_t);
  if (scharr_stride_bytes < scharr_row_bytes) {
    return KLEIDICV_ERROR_RANGE;
  }

  if (window_width > KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE ||
      window_height > KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE) {
    return KLEIDICV_ERROR_RANGE;
  }

  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

template <typename Impl>
class LucasKanadeLevelTracker {
 public:
  // Function complexity is just above the threshold, but this is readable this
  // way, ignore the warning.
  // NOLINTBEGIN(readability-function-cognitive-complexity)
  // Single-level Lucas-Kanade tracker for a batch of points on a
  // multi-channel image. Uses precomputed Scharr derivatives of the previous
  // frame and
  // iteratively refines `next_points`.
  static KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t compute(
      int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
      size_t prev_data_stride, const int16_t *scharr_data,
      size_t scharr_stride_bytes, const uint8_t *next_data, size_t next_stride,
      int width, int height, int channels, const float *prev_points,
      float *next_points, size_t point_count, uint8_t *status, float *err,
      int window_width, int window_height, int termination_count,
      double termination_epsilon, bool get_min_eigen_vals,
      float min_eigen_vals_threshold) KLEIDICV_STREAMING {
    // Core Lucas-Kanade iteration at a single pyramid level:
    // 1) Sample the previous frame + gradients at subpixel accuracy.
    // 2) Build the G matrix (IxIx, IxIy, IyIy) and check eigen thresholds.
    // 3) Iterate: warp the next frame toward the previous, accumulate b, solve
    //    the 2x2 system for delta, and update the displacement until converged.

    const ptrdiff_t scharr_stride_elements =
        static_cast<ptrdiff_t>(scharr_stride_bytes / sizeof(int16_t));

    const float half_window_width = static_cast<float>(window_width - 1) * 0.5F;
    const float half_window_height =
        static_cast<float>(window_height - 1) * 0.5F;

    // Process each point independently.
    for (size_t point_index = 0; point_index < point_count; point_index++) {
      const float prev_x = prev_points[point_index * 2] - half_window_width;
      const float prev_y =
          prev_points[point_index * 2 + 1] - half_window_height;
      const int prev_xi = static_cast<int>(floorf(prev_x));
      const int prev_yi = static_cast<int>(floorf(prev_y));

      if (point_out_of_bounds(prev_xi, prev_yi, width, height, window_width,
                              window_height)) {
        if (status) {
          status[point_index] = false;
        }
        continue;
      }

      const auto [coeff_tl, coeff_tr, coeff_bl, coeff_br] =
          get_lerp_params<kLucasKanadeAlgFractionBits>(
              prev_x - static_cast<float>(prev_xi),
              prev_y - static_cast<float>(prev_yi));

      // Build the local structure tensor for LK and reject weak gradients.
      StructureTensor tensor = build_structure_tensor(
          window, scharr_window, prev_data, prev_data_stride, scharr_data,
          scharr_stride_elements, channels, prev_xi, prev_yi, window_width,
          window_height, coeff_tl, coeff_tr, coeff_bl, coeff_br);

      if (err && get_min_eigen_vals) {
        err[point_index] = tensor.min_eigen_val;
      }

      if (texture_too_weak(tensor, min_eigen_vals_threshold)) {
        if (status) {
          status[point_index] = false;
        }
        continue;
      }

      const float inverse_determinant = 1.0F / tensor.determinant;

      // Newton iterations refine the displacement in `next_points`.
      const bool point_ok = refine_point_position(
          next_points, point_index, half_window_width, half_window_height,
          window_width, window_height, width, height, channels,
          termination_count, termination_epsilon, tensor.sum_scharr_xx,
          tensor.sum_scharr_xy, tensor.sum_scharr_yy, inverse_determinant,
          next_data, next_stride, window, scharr_window);

      if (!point_ok) {
        if (status) {
          status[point_index] = false;
        }
        continue;
      }

      if (!status || status[point_index]) {
        float next_x = next_points[point_index * 2] - half_window_width;
        float next_y = next_points[point_index * 2 + 1] - half_window_height;
        const int next_xi = static_cast<int>(floorf(next_x));
        const int next_yi = static_cast<int>(floorf(next_y));

        if (point_out_of_bounds(next_xi, next_yi, width, height, window_width,
                                window_height)) {
          if (status) {
            status[point_index] = false;
          }
        } else if (err && !get_min_eigen_vals) {
          const auto [coeff_tl, coeff_tr, coeff_bl, coeff_br] =
              get_lerp_params<kLucasKanadeAlgFractionBits>(
                  next_x - static_cast<float>(next_xi),
                  next_y - static_cast<float>(next_yi));
          // Photometric error only computed for successfully tracked points.
          err[point_index] = compute_photometric_error(
              next_data, next_stride, window, window_width, window_height,
              channels, next_xi, next_yi, coeff_tl, coeff_tr, coeff_bl,
              coeff_br);
        }
      }
    }
    return KLEIDICV_OK;
  }
  // NOLINTEND(readability-function-cognitive-complexity)
 private:
  struct StructureTensor {
    float sum_scharr_xx = 0;
    float sum_scharr_xy = 0;
    float sum_scharr_yy = 0;
    float determinant = 0;
    float min_eigen_val = 0;
  };

  template <int shift>
  static KLEIDICV_FORCE_INLINE std::array<int16_t, 4> get_lerp_params(
      float x, float y) KLEIDICV_STREAMING {
    // Bilinear weights (top-left, top-right, bottom-left, bottom-right) scaled
    // to fixed-point so the kernel can sample subpixel locations without
    // floats.
    int16_t a =
        static_cast<int16_t>(rintf((1.0F - x) * (1.0F - y) * (1 << shift)));
    int16_t b = static_cast<int16_t>(rintf(x * (1.0F - y) * (1 << shift)));
    int16_t c = static_cast<int16_t>(rintf((1.0F - x) * y * (1 << shift)));
    int16_t d = (1 << shift) - a - b - c;
    return {a, b, c, d};
  }

  static KLEIDICV_FORCE_INLINE bool point_out_of_bounds(
      int x, int y, int width, int height, int window_width,
      int window_height) KLEIDICV_STREAMING {
    return x < -window_width || x >= width || y < -window_height || y >= height;
  }

  static KLEIDICV_FORCE_INLINE StructureTensor compute_structure_tensor(
      float sum_scharr_xx, float sum_scharr_xy, float sum_scharr_yy,
      int window_width, int window_height) KLEIDICV_STREAMING {
    StructureTensor tensor{};
    tensor.sum_scharr_xx = sum_scharr_xx;
    tensor.sum_scharr_xy = sum_scharr_xy;
    tensor.sum_scharr_yy = sum_scharr_yy;
    tensor.determinant = tensor.sum_scharr_xx * tensor.sum_scharr_yy -
                         powf(tensor.sum_scharr_xy, 2);
    tensor.min_eigen_val =
        (tensor.sum_scharr_yy + tensor.sum_scharr_xx -
         sqrtf(powf(tensor.sum_scharr_xx - tensor.sum_scharr_yy, 2) +
               4 * powf(tensor.sum_scharr_xy, 2))) /
        static_cast<float>(2 * window_width * window_height);
    return tensor;
  }

  static KLEIDICV_FORCE_INLINE StructureTensor build_structure_tensor(
      int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
      size_t prev_data_stride, const int16_t *scharr_data,
      ptrdiff_t scharr_stride_elements, int channels, int prev_xi, int prev_yi,
      int window_width, int window_height, int16_t coeff_tl, int16_t coeff_tr,
      int16_t coeff_bl, int16_t coeff_br) KLEIDICV_STREAMING {
    float sum_scharr_xx = 0, sum_scharr_xy = 0, sum_scharr_yy = 0;
    Impl::sample_patch_and_gradients(
        window, scharr_window, prev_data, prev_data_stride, scharr_data,
        scharr_stride_elements, channels, prev_xi, prev_yi, window_width,
        window_height, coeff_tl, coeff_tr, coeff_bl, coeff_br, sum_scharr_xx,
        sum_scharr_xy, sum_scharr_yy);
    return compute_structure_tensor(sum_scharr_xx, sum_scharr_xy, sum_scharr_yy,
                                    window_width, window_height);
  }

  static KLEIDICV_FORCE_INLINE bool texture_too_weak(
      const StructureTensor &tensor,
      float min_eigen_vals_threshold) KLEIDICV_STREAMING {
    return tensor.min_eigen_val < min_eigen_vals_threshold ||
           tensor.determinant < FLT_EPSILON;
  }

  static KLEIDICV_FORCE_INLINE bool refine_point_position(
      float *next_points, size_t point_index, float half_window_width,
      float half_window_height, int window_width, int window_height, int width,
      int height, int channels, int termination_count,
      double termination_epsilon, float sum_scharr_xx, float sum_scharr_xy,
      float sum_scharr_yy, float inverse_determinant, const uint8_t *next_data,
      size_t next_stride, int16_t *window,
      int16_t *scharr_window) KLEIDICV_STREAMING {
    float next_x = next_points[point_index * 2] - half_window_width;
    float next_y = next_points[point_index * 2 + 1] - half_window_height;
    float prev_velocity_x = 0, prev_velocity_y = 0;

    for (int j = 0; j < termination_count; j++) {
      const int next_xi = static_cast<int>(floorf(next_x));
      const int next_yi = static_cast<int>(floorf(next_y));

      if (point_out_of_bounds(next_xi, next_yi, width, height, window_width,
                              window_height)) {
        return false;
      }

      const auto [coeff_tl, coeff_tr, coeff_bl, coeff_br] =
          get_lerp_params<kLucasKanadeAlgFractionBits>(
              next_x - static_cast<float>(next_xi),
              next_y - static_cast<float>(next_yi));

      float sum_diff_scharr_x = 0, sum_diff_scharr_y = 0;
      Impl::accumulate_mismatch_vector(
          next_data, next_stride, window, scharr_window, channels, next_xi,
          next_yi, window_width, window_height, coeff_tl, coeff_tr, coeff_bl,
          coeff_br, sum_diff_scharr_x, sum_diff_scharr_y);

      const float velocity_x = (sum_scharr_xy * sum_diff_scharr_y -
                                sum_scharr_yy * sum_diff_scharr_x) *
                               inverse_determinant;
      const float velocity_y = (sum_scharr_xy * sum_diff_scharr_x -
                                sum_scharr_xx * sum_diff_scharr_y) *
                               inverse_determinant;

      next_x += velocity_x;
      next_y += velocity_y;
      next_points[point_index * 2] = next_x + half_window_width;
      next_points[point_index * 2 + 1] = next_y + half_window_height;

      if (velocity_x * velocity_x + velocity_y * velocity_y <=
          termination_epsilon) {
        break;
      }

      if (j != 0 && fabsf(velocity_x + prev_velocity_x) < 0.01F &&
          fabsf(velocity_y + prev_velocity_y) < 0.01F) {
        next_points[point_index * 2] -= velocity_x * 0.5F;
        next_points[point_index * 2 + 1] -= velocity_y * 0.5F;
        break;
      }
      prev_velocity_x = velocity_x;
      prev_velocity_y = velocity_y;
    }

    return true;
  }

  static KLEIDICV_FORCE_INLINE float compute_photometric_error(
      const uint8_t *next_data, size_t next_stride, int16_t *window,
      int window_width, int window_height, int channels, int next_xi,
      int next_yi, int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl,
      int16_t coeff_br) KLEIDICV_STREAMING {
    float errval = 0;
    for (int y = 0; y < window_height; y++) {
      int16_t *window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;
      const uint8_t *next_row0 =
          next_data + (y + next_yi) * static_cast<ptrdiff_t>(next_stride) +
          static_cast<ptrdiff_t>(next_xi) * channels;
      const uint8_t *next_row1 = next_row0 + next_stride;

      for (int x = 0; x < window_width * channels; x++) {
        int diff =
            round_fixed_point<kLucasKanadeAlgFractionBits - 5>(
                next_row0[x] * coeff_tl + next_row0[x + channels] * coeff_tr +
                next_row1[x] * coeff_bl + next_row1[x + channels] * coeff_br) -
            window_row[x];
        errval += fabsf(static_cast<float>(diff));
      }
    }
    return errval * 1.0F /
           static_cast<float>(32L * window_width * channels * window_height);
  }
};

// Allocates standalone Lucas-Kanade algorithm window buffers.
// If the required buffer size is no greater than StackBufferSize then the
// buffers will be allocated on the stack. Typical window width is 21.
template <int StackBufferSize = 21UL * 21UL * 3UL>
class LucasKanadePatchBuffer {
 public:
  LucasKanadePatchBuffer() = delete;

  static std::variant<LucasKanadePatchBuffer, kleidicv_error_t> create(
      int window_width, int window_height, int channels) KLEIDICV_STREAMING {
    size_t patch_size = 0;
    size_t total_size = 0;
    compute_buffer_sizes(window_width, window_height, channels, patch_size,
                         total_size);

    LucasKanadePatchBuffer buffer{total_size, patch_size};
    if (!buffer.window()) {
      return KLEIDICV_ERROR_ALLOCATION;
    }

    return std::move(buffer);
  }

  int16_t *window() { return window_buffer_.get(); }
  int16_t *deriv_window() { return deriv_window_; }

  LucasKanadePatchBuffer(LucasKanadePatchBuffer &&other) KLEIDICV_STREAMING
      : window_buffer_(std::move(other.window_buffer_)),
        deriv_window_(window_buffer_.get()
                          ? window_buffer_.get() + other.patch_size_
                          : nullptr),
        patch_size_(other.patch_size_) {}

  LucasKanadePatchBuffer(const LucasKanadePatchBuffer &) = delete;
  LucasKanadePatchBuffer &operator=(const LucasKanadePatchBuffer &) = delete;
  LucasKanadePatchBuffer &operator=(LucasKanadePatchBuffer &&) = delete;

 private:
  static void compute_buffer_sizes(int window_width, int window_height,
                                   int channels, size_t &patch_size,
                                   size_t &total_size) KLEIDICV_STREAMING {
    patch_size = static_cast<size_t>(window_width) *
                 static_cast<size_t>(window_height) *
                 static_cast<size_t>(channels);
    total_size = patch_size * 3U;
  }

  LucasKanadePatchBuffer(size_t total_size, size_t patch_size)
      : window_buffer_(total_size),
        deriv_window_(window_buffer_.get() ? window_buffer_.get() + patch_size
                                           : nullptr),
        patch_size_(patch_size) {}

  kleidicv::SmallBuffer<int16_t, StackBufferSize> window_buffer_;
  int16_t *deriv_window_;
  size_t patch_size_;
};

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_COMMON_H
