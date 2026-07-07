// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/analysis/standalone_lucas_kanade_alg.h"
#include "standalone_lucas_kanade_alg_sc.h"

namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold) {
  if (kleidicv_error_t validation_error =
          validate_standalone_lucas_kanade_alg_u8_args(
              prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
              next_data, next_data_stride, width, height, channels, prev_points,
              next_points, point_count, window_width, window_height)) {
    return validation_error;
  }

  auto window_buffer_or_error =
      LucasKanadePatchBuffer<>::create(window_width, window_height, channels);
  if (!std::holds_alternative<LucasKanadePatchBuffer<>>(
          window_buffer_or_error)) {
    return std::get<kleidicv_error_t>(window_buffer_or_error);
  }
  auto &window_buffer =
      std::get<LucasKanadePatchBuffer<>>(window_buffer_or_error);
  return LucasKanadeLevelTracker<StandaloneLucasKanadeAlg>::compute(
      window_buffer.window(), window_buffer.deriv_window(), prev_data,
      prev_data_stride, prev_deriv_data, prev_deriv_stride, next_data,
      next_data_stride, width, height, channels, prev_points, next_points,
      point_count, status, err, window_width, window_height, termination_count,
      termination_epsilon, get_min_eigen_vals, min_eigen_vals_threshold);
}

}  // namespace kleidicv::sve2
