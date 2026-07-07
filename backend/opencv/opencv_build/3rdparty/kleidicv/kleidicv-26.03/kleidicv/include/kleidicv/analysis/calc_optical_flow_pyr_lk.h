// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CALC_OPTICAL_FLOW_PYR_LK_H
#define KLEIDICV_CALC_OPTICAL_FLOW_PYR_LK_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include "kleidicv/analysis/build_optical_flow_pyr_lk_pyramid.h"
#include "kleidicv/analysis/standalone_lucas_kanade_alg.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

class OpticalFlowPyrLKCalc final {
 public:
  template <bool kUseSME>
  static kleidicv_error_t calc_from_pyramid(
      const kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
      const kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
      const float* prev_points, float* next_points, size_t point_count,
      uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
    if (point_count == 0) {
      return KLEIDICV_OK;
    }

    CHECK_POINTERS(prev_pyramid, next_pyramid, prev_points, next_points,
                   status);
    if (kleidicv_error_t error_code = validate_context(context)) {
      return error_code;
    }

    PyramidPair pyramid_pair{};
    if (kleidicv_error_t error_code = validate_and_prepare_pyramid_pair(
            prev_pyramid, next_pyramid, context, &pyramid_pair)) {
      return error_code;
    }

    // Scratch buffers keep coordinates in the current pyramid level space.
    size_t points_buffer_bytes = 0;
    if (kleidicv_error_t error_code =
            calculate_points_buffer_bytes(point_count, &points_buffer_bytes)) {
      return error_code;
    }

    initialize_calc_status(status, point_count);

    PointBuffers point_buffers(points_buffer_bytes);
    if (!point_buffers.is_valid()) {
      return KLEIDICV_ERROR_ALLOCATION;
    }

    // Lucas-Kanade is applied coarse-to-fine: start at the top level and
    // refine the estimate while descending to level 0.
    const int effective_max_level_i =
        static_cast<int>(pyramid_pair.effective_max_level);
    for (int level_i = effective_max_level_i; level_i >= 0; --level_i) {
      prepare_level_points(prev_points, next_points, point_count, context,
                           pyramid_pair.effective_max_level, level_i,
                           point_buffers.prev.get(), point_buffers.next.get());

      if (kleidicv_error_t error_code = run_level_calc<kUseSME>(
              pyramid_pair, level_i, point_count, status, err, context,
              point_buffers.prev.get(), point_buffers.next.get())) {
        return error_code;
      }
    }

    copy_point_coordinates(next_points, point_buffers.next.get(), point_count);
    return KLEIDICV_OK;
  }

  template <bool kUseSME>
  static kleidicv_error_t calc_from_images(
      const uint8_t* prev_image, size_t prev_image_stride,
      const uint8_t* next_image, size_t next_image_stride, size_t width,
      size_t height, size_t channels, const float* prev_points,
      float* next_points, size_t point_count, uint8_t* status, float* err,
      kleidicv_optflow_lk_context_t context) {
    if (point_count == 0) {
      return KLEIDICV_OK;
    }

    if (context.max_level == std::numeric_limits<size_t>::max()) {
      return KLEIDICV_ERROR_RANGE;
    }

    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid{};
    kleidicv_error_t kle_err{};
    if constexpr (kUseSME) {
      kle_err = kleidicv_build_optical_flow_pyr_lk_pyramid_sme(
          &prev_pyramid, prev_image, prev_image_stride, width, height, channels,
          context.max_level + 1, context.window_width, context.window_height);

    } else {
      kle_err = kleidicv_build_optical_flow_pyr_lk_pyramid(
          &prev_pyramid, prev_image, prev_image_stride, width, height, channels,
          context.max_level + 1, context.window_width, context.window_height);
    }
    if (kle_err != KLEIDICV_OK) {
      return kle_err;
    }

    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid{};
    if constexpr (kUseSME) {
      kle_err = kleidicv_build_optical_flow_pyr_lk_pyramid_sme(
          &next_pyramid, next_image, next_image_stride, width, height, channels,
          context.max_level + 1, context.window_width, context.window_height);

    } else {
      kle_err = kleidicv_build_optical_flow_pyr_lk_pyramid(
          &next_pyramid, next_image, next_image_stride, width, height, channels,
          context.max_level + 1, context.window_width, context.window_height);
    }
    if (kle_err != KLEIDICV_OK) {
      (void)kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid);
      return kle_err;
    }

    kle_err = calc_from_pyramid<kUseSME>(prev_pyramid, next_pyramid,
                                         prev_points, next_points, point_count,
                                         status, err, context);

    (void)kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid);
    (void)kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid);

    return kle_err;
  }

 private:
  struct PyramidPair {
    const OpticalFlowLKPyramid* prev_pyramid;
    const OpticalFlowLKPyramid* next_pyramid;
    size_t effective_max_level;
    size_t channels;
  };

  struct PyramidBorders {
    size_t prev_width;
    size_t prev_height;
    size_t next_width;
    size_t next_height;
    size_t prev_double_width;
    size_t next_double_width;
  };

  struct PointBuffers {
    std::unique_ptr<float[], decltype(&std::free)> prev;
    std::unique_ptr<float[], decltype(&std::free)> next;

    explicit PointBuffers(size_t buffer_bytes)
        : prev(static_cast<float*>(std::malloc(buffer_bytes)), &std::free),
          next(static_cast<float*>(std::malloc(buffer_bytes)), &std::free) {}

    bool is_valid() const { return prev != nullptr && next != nullptr; }
  };

  static bool fits_in_int(size_t value) {
    return value <= static_cast<size_t>(std::numeric_limits<int>::max());
  }

  static kleidicv_error_t validate_context(
      kleidicv_optflow_lk_context_t context) {
    if (context.window_width <= 2 || context.window_height <= 2 ||
        context.termination_count == 0 || context.termination_epsilon < 0.0F ||
        (context.flags & ~(KLEIDICV_USE_INITIAL_FLOW | KLEIDICV_GET_MIN_EIG))) {
      return KLEIDICV_ERROR_RANGE;
    }

    if (!fits_in_int(context.window_width) ||
        !fits_in_int(context.window_height) ||
        !fits_in_int(context.termination_count)) {
      return KLEIDICV_ERROR_RANGE;
    }

    return KLEIDICV_OK;
  }

  static kleidicv_error_t calculate_points_buffer_bytes(
      size_t point_count, size_t* points_buffer_bytes) {
    size_t point_values_count = 0;
    if (__builtin_mul_overflow(point_count, static_cast<size_t>(2),
                               &point_values_count) ||
        __builtin_mul_overflow(point_values_count, sizeof(float),
                               points_buffer_bytes)) {
      return KLEIDICV_ERROR_RANGE;
    }
    return KLEIDICV_OK;
  }

  static void prepare_level_points(const float* prev_points,
                                   const float* next_points, size_t point_count,
                                   kleidicv_optflow_lk_context_t context,
                                   size_t top_level, int level_i,
                                   float* prev_points_level,
                                   float* next_points_level) {
    const float scale = std::ldexp(1.0F, -level_i);
    const bool is_top_level = static_cast<size_t>(level_i) == top_level;
    const bool use_initial_flow =
        (context.flags & KLEIDICV_USE_INITIAL_FLOW) != 0;

    for (size_t point = 0; point < point_count; ++point) {
      const size_t x_index = point * 2;
      const size_t y_index = x_index + 1;
      prev_points_level[x_index] = prev_points[x_index] * scale;
      prev_points_level[y_index] = prev_points[y_index] * scale;

      if (is_top_level) {
        initialize_top_level_next_point(next_points, use_initial_flow, x_index,
                                        y_index, scale, prev_points_level,
                                        next_points_level);
        continue;
      }

      next_points_level[x_index] *= 2.0F;
      next_points_level[y_index] *= 2.0F;
    }
  }

  static void initialize_top_level_next_point(const float* next_points,
                                              bool use_initial_flow,
                                              size_t x_index, size_t y_index,
                                              float scale,
                                              const float* prev_points_level,
                                              float* next_points_level) {
    if (use_initial_flow) {
      next_points_level[x_index] = next_points[x_index] * scale;
      next_points_level[y_index] = next_points[y_index] * scale;
      return;
    }

    next_points_level[x_index] = prev_points_level[x_index];
    next_points_level[y_index] = prev_points_level[y_index];
  }

  template <bool kUseSME>
  static kleidicv_error_t run_level_calc(const PyramidPair& pyramid_pair,
                                         int level_i, size_t point_count,
                                         uint8_t* status, float* err,
                                         kleidicv_optflow_lk_context_t context,
                                         float* prev_points_level,
                                         float* next_points_level) {
    const size_t level = static_cast<size_t>(level_i);
    const auto& prev_level = pyramid_pair.prev_pyramid->level(level);
    const auto& next_level = pyramid_pair.next_pyramid->level(level);
    if (!fits_in_int(prev_level.width) || !fits_in_int(prev_level.height)) {
      return KLEIDICV_ERROR_RANGE;
    }

    uint8_t* status_at_level = (level_i == 0) ? status : nullptr;
    float* error_at_level = err;
    const double squared_termination_epsilon =
        static_cast<double>(context.termination_epsilon) *
        static_cast<double>(context.termination_epsilon);

    if constexpr (kUseSME) {
      return kleidicv_standalone_lucas_kanade_alg_u8_sme(
          prev_level.image_data, prev_level.image_stride,
          prev_level.scharr_data, prev_level.scharr_stride,
          next_level.image_data, next_level.image_stride,
          static_cast<int>(prev_level.width),
          static_cast<int>(prev_level.height),
          static_cast<int>(pyramid_pair.channels), prev_points_level,
          next_points_level, point_count, status_at_level, error_at_level,
          static_cast<int>(context.window_width),
          static_cast<int>(context.window_height),
          static_cast<int>(context.termination_count),
          squared_termination_epsilon,
          (context.flags & KLEIDICV_GET_MIN_EIG) != 0,
          context.min_eig_threshold);
    }
    return kleidicv_standalone_lucas_kanade_alg_u8(
        prev_level.image_data, prev_level.image_stride, prev_level.scharr_data,
        prev_level.scharr_stride, next_level.image_data,
        next_level.image_stride, static_cast<int>(prev_level.width),
        static_cast<int>(prev_level.height),
        static_cast<int>(pyramid_pair.channels), prev_points_level,
        next_points_level, point_count, status_at_level, error_at_level,
        static_cast<int>(context.window_width),
        static_cast<int>(context.window_height),
        static_cast<int>(context.termination_count),
        squared_termination_epsilon,
        (context.flags & KLEIDICV_GET_MIN_EIG) != 0, context.min_eig_threshold);
  }

  static kleidicv_error_t validate_and_prepare_pyramid_pair(
      const kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
      const kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
      kleidicv_optflow_lk_context_t context, PyramidPair* pyramid_pair) {
    const auto* prev_pyramid_view =
        reinterpret_cast<const OpticalFlowLKPyramid*>(prev_pyramid);
    const auto* next_pyramid_view =
        reinterpret_cast<const OpticalFlowLKPyramid*>(next_pyramid);

    const size_t prev_level_count = prev_pyramid_view->level_count();
    const size_t next_level_count = next_pyramid_view->level_count();
    size_t effective_max_level = 0;
    size_t channels = 0;
    PyramidBorders borders{};
    if (kleidicv_error_t error_code = prepare_pyramid_pair_metadata(
            prev_pyramid_view, next_pyramid_view, prev_level_count,
            next_level_count, context, &effective_max_level, &channels,
            &borders)) {
      return error_code;
    }

    if (kleidicv_error_t error_code =
            validate_pyramid_levels(*prev_pyramid_view, *next_pyramid_view,
                                    effective_max_level, channels, borders)) {
      return error_code;
    }

    pyramid_pair->prev_pyramid = prev_pyramid_view;
    pyramid_pair->next_pyramid = next_pyramid_view;
    pyramid_pair->effective_max_level = effective_max_level;
    pyramid_pair->channels = channels;
    return KLEIDICV_OK;
  }

  static kleidicv_error_t prepare_pyramid_pair_metadata(
      const OpticalFlowLKPyramid* prev_pyramid_view,
      const OpticalFlowLKPyramid* next_pyramid_view, size_t prev_level_count,
      size_t next_level_count, kleidicv_optflow_lk_context_t context,
      size_t* effective_max_level, size_t* channels, PyramidBorders* borders) {
    if (prev_level_count == 0 || next_level_count == 0) {
      return KLEIDICV_ERROR_RANGE;
    }

    *effective_max_level =
        std::min(context.max_level,
                 std::min(prev_level_count - 1, next_level_count - 1));
    if (!fits_in_int(*effective_max_level)) {
      return KLEIDICV_ERROR_RANGE;
    }

    if (kleidicv_error_t error_code =
            validate_base_levels(*prev_pyramid_view, *next_pyramid_view)) {
      return error_code;
    }

    *channels = prev_pyramid_view->channels();
    if (!fits_in_int(*channels) || *channels != next_pyramid_view->channels()) {
      return KLEIDICV_ERROR_RANGE;
    }

    return prepare_pyramid_borders(*prev_pyramid_view, *next_pyramid_view,
                                   context, borders);
  }

  static kleidicv_error_t validate_base_levels(
      const OpticalFlowLKPyramid& prev_pyramid,
      const OpticalFlowLKPyramid& next_pyramid) {
    const auto& prev_level0 = prev_pyramid.level(0);
    const auto& next_level0 = next_pyramid.level(0);
    if (prev_level0.width != next_level0.width ||
        prev_level0.height != next_level0.height) {
      return KLEIDICV_ERROR_RANGE;
    }
    return KLEIDICV_OK;
  }

  static kleidicv_error_t prepare_pyramid_borders(
      const OpticalFlowLKPyramid& prev_pyramid,
      const OpticalFlowLKPyramid& next_pyramid,
      kleidicv_optflow_lk_context_t context, PyramidBorders* borders) {
    borders->prev_width = prev_pyramid.border_width();
    borders->prev_height = prev_pyramid.border_height();
    borders->next_width = next_pyramid.border_width();
    borders->next_height = next_pyramid.border_height();

    if (context.window_width > borders->prev_width ||
        context.window_height > borders->prev_height ||
        context.window_width > borders->next_width ||
        context.window_height > borders->next_height) {
      return KLEIDICV_ERROR_RANGE;
    }

    if (__builtin_mul_overflow(borders->prev_width, static_cast<size_t>(2),
                               &borders->prev_double_width) ||
        __builtin_mul_overflow(borders->next_width, static_cast<size_t>(2),
                               &borders->next_double_width)) {
      return KLEIDICV_ERROR_RANGE;
    }
    return KLEIDICV_OK;
  }

  static kleidicv_error_t validate_pyramid_levels(
      const OpticalFlowLKPyramid& prev_pyramid,
      const OpticalFlowLKPyramid& next_pyramid, size_t effective_max_level,
      size_t channels, const PyramidBorders& borders) {
    const size_t scharr_bytes_per_pixel = channels * 2 * sizeof(int16_t);
    for (size_t level = 0; level <= effective_max_level; ++level) {
      if (kleidicv_error_t error_code = validate_pyramid_level(
              prev_pyramid.level(level), next_pyramid.level(level), channels,
              scharr_bytes_per_pixel, borders)) {
        return error_code;
      }
    }
    return KLEIDICV_OK;
  }

  static kleidicv_error_t validate_pyramid_level(
      const OpticalFlowLevel& prev_level, const OpticalFlowLevel& next_level,
      size_t channels, size_t scharr_bytes_per_pixel,
      const PyramidBorders& borders) {
    if (prev_level.width != next_level.width ||
        prev_level.height != next_level.height) {
      return KLEIDICV_ERROR_RANGE;
    }

    size_t prev_image_width_with_border = 0;
    size_t next_image_width_with_border = 0;
    if (__builtin_add_overflow(prev_level.width, borders.prev_double_width,
                               &prev_image_width_with_border) ||
        __builtin_add_overflow(next_level.width, borders.next_double_width,
                               &next_image_width_with_border)) {
      return KLEIDICV_ERROR_RANGE;
    }

    return validate_pyramid_level_strides(
        prev_level, next_level, prev_image_width_with_border,
        next_image_width_with_border, channels, scharr_bytes_per_pixel);
  }

  static kleidicv_error_t validate_pyramid_level_strides(
      const OpticalFlowLevel& prev_level, const OpticalFlowLevel& next_level,
      size_t prev_image_width_with_border, size_t next_image_width_with_border,
      size_t channels, size_t scharr_bytes_per_pixel) {
    size_t min_prev_image_stride = 0;
    size_t min_next_image_stride = 0;
    if (__builtin_mul_overflow(prev_image_width_with_border, channels,
                               &min_prev_image_stride) ||
        __builtin_mul_overflow(next_image_width_with_border, channels,
                               &min_next_image_stride)) {
      return KLEIDICV_ERROR_RANGE;
    }
    if (prev_level.image_stride < min_prev_image_stride ||
        next_level.image_stride < min_next_image_stride) {
      return KLEIDICV_ERROR_RANGE;
    }

    size_t min_prev_scharr_stride = 0;
    size_t min_next_scharr_stride = 0;
    if (__builtin_mul_overflow(prev_image_width_with_border,
                               scharr_bytes_per_pixel,
                               &min_prev_scharr_stride) ||
        __builtin_mul_overflow(next_image_width_with_border,
                               scharr_bytes_per_pixel,
                               &min_next_scharr_stride)) {
      return KLEIDICV_ERROR_RANGE;
    }
    if (prev_level.scharr_stride < min_prev_scharr_stride ||
        next_level.scharr_stride < min_next_scharr_stride) {
      return KLEIDICV_ERROR_RANGE;
    }

    return KLEIDICV_OK;
  }

  static void initialize_calc_status(uint8_t* status, size_t point_count) {
    std::memset(status, 1, point_count);
  }

  static void copy_point_coordinates(float* dst_points, const float* src_points,
                                     size_t point_count) {
    std::memcpy(dst_points, src_points, point_count * 2 * sizeof(float));
  }
};

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_CALC_OPTICAL_FLOW_PYR_LK_H
