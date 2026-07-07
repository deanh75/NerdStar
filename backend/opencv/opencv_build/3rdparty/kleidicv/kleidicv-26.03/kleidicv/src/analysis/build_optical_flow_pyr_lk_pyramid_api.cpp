// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include "kleidicv/analysis/build_optical_flow_pyr_lk_pyramid.h"
#include "kleidicv/utils.h"

namespace kleidicv {

template <bool kUseSME>
kleidicv_error_t build_optical_flow_pyr_lk_pyramid(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height) {
  CHECK_POINTERS(pyramid);

  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  size_t actual_level_count = 0;
  if (kleidicv_error_t err =
          validate_and_compute_build_optical_flow_pyr_lk_levels(
              src_stride, width, height, channels, level_count, window_width,
              window_height, &actual_level_count)) {
    return err;
  }

  OpticalFlowLKPyramid::Pointer pyramid_storage;
  if (kleidicv_error_t err = OpticalFlowLKPyramid::allocate(
          pyramid_storage, actual_level_count, width, height, channels,
          window_width, window_height)) {
    return err;
  }

  if (kleidicv_error_t err =
          pyramid_storage->create<kUseSME>(src, src_stride)) {
    return err;
  }

  *pyramid = reinterpret_cast<kleidicv_optical_flow_pyr_lk_pyramid_t*>(
      pyramid_storage.release());
  return KLEIDICV_OK;
}

}  // namespace kleidicv

extern "C" {

using kleidicv::OpticalFlowLKPyramid;

kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height) {
  return kleidicv::build_optical_flow_pyr_lk_pyramid<false>(
      pyramid, src, src_stride, width, height, channels, level_count,
      window_width, window_height);
}

kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid_sme(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height) {
  return kleidicv::build_optical_flow_pyr_lk_pyramid<true>(
      pyramid, src, src_stride, width, height, channels, level_count,
      window_width, window_height);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_release(
    kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid) {
  CHECK_POINTERS(pyramid);
  // Deliberately create and immediately destroy a unique_ptr to delete the
  // opaque pyramid storage.
  // NOLINTBEGIN(bugprone-unused-raii)
  OpticalFlowLKPyramid::Pointer{
      reinterpret_cast<OpticalFlowLKPyramid*>(pyramid)};
  // NOLINTEND(bugprone-unused-raii)
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid,
    size_t* level_count) {
  CHECK_POINTERS(pyramid, level_count);
  const auto* typed = reinterpret_cast<const OpticalFlowLKPyramid*>(pyramid);
  *level_count = typed->level_count();
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t level,
    const uint8_t** data, size_t* stride, size_t* width, size_t* height) {
  CHECK_POINTERS(pyramid, data, stride, width, height);
  const auto* typed = reinterpret_cast<const OpticalFlowLKPyramid*>(pyramid);
  if (level >= typed->level_count()) {
    return KLEIDICV_ERROR_RANGE;
  }

  const auto& entry = typed->level(level);
  *data = entry.image_data;
  *stride = entry.image_stride;
  *width = entry.width;
  *height = entry.height;
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t level,
    const int16_t** data, size_t* stride, size_t* width, size_t* height) {
  CHECK_POINTERS(pyramid, data, stride, width, height);
  const auto* typed = reinterpret_cast<const OpticalFlowLKPyramid*>(pyramid);
  if (level >= typed->level_count()) {
    return KLEIDICV_ERROR_RANGE;
  }

  const auto& entry = typed->level(level);
  *data = entry.scharr_data;
  *stride = entry.scharr_stride;
  *width = entry.width;
  *height = entry.height;
  return KLEIDICV_OK;
}

}  // extern "C"
