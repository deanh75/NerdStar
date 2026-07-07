// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/analysis/calc_optical_flow_pyr_lk.h"

extern "C" {

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return KLEIDICV_TARGET_NAMESPACE::OpticalFlowPyrLKCalc::calc_from_pyramid<
      false>(prev_pyramid, next_pyramid, prev_points, next_points, point_count,
             status, err, context);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_from_pyramid_sme(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return KLEIDICV_TARGET_NAMESPACE::OpticalFlowPyrLKCalc::calc_from_pyramid<
      true>(prev_pyramid, next_pyramid, prev_points, next_points, point_count,
            status, err, context);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8(
    const uint8_t* prev_img, size_t prev_img_stride, const uint8_t* next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return KLEIDICV_TARGET_NAMESPACE::OpticalFlowPyrLKCalc::calc_from_images<
      false>(prev_img, prev_img_stride, next_img, next_img_stride, width,
             height, channels, prev_points, next_points, point_count, status,
             err, context);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_sme(
    const uint8_t* prev_img, size_t prev_img_stride, const uint8_t* next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return KLEIDICV_TARGET_NAMESPACE::OpticalFlowPyrLKCalc::calc_from_images<
      true>(prev_img, prev_img_stride, next_img, next_img_stride, width, height,
            channels, prev_points, next_points, point_count, status, err,
            context);
}

}  // extern "C"
