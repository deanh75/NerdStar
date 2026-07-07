// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_H
#define KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_H

#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {

kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold);

}  // namespace neon

namespace sve2 {

kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold);

}  // namespace sve2

namespace sme {

kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold);

}  // namespace sme

namespace sme2 {

kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold);

}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_STANDALONE_LUCAS_KANADE_ALG_H
