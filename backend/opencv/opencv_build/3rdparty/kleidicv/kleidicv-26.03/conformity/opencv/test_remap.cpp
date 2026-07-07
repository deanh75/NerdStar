// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc/hal/interface.h"
#include "tests.h"

const int kMaxHeight = 36, kMaxWidth = 32;

template <class ScalarType, int Channels>
cv::Mat get_source_mat(int format) {
  auto generate_source = [&]() {
    cv::Mat m(kMaxHeight, kMaxWidth, format);
    const int64_t kMaxValue = std::numeric_limits<ScalarType>::max();
    for (size_t row = 0; row < kMaxHeight; ++row) {
      for (size_t column = 0; column < kMaxWidth; ++column) {
        // Create as many different differences between neighbouring pixels as
        // possible
        cv::Vec<ScalarType, Channels> pixel_value;
        for (size_t ch = 0; ch < Channels; ++ch) {
          size_t counter = row + column + ch;
          pixel_value[ch] =
              (counter % 2) ? kMaxValue : (counter % (kMaxValue + 1));
        }
        m.at<cv::Vec<ScalarType, Channels>>(row, column) = pixel_value;
      }
    }
    return m;
  };
  static cv::Mat source = generate_source();
  return source;
}

// BorderValue is interpreted as 1/1000, i.e. 500 for 0.5
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
cv::Mat exec_remap_s16(cv::Mat& mapxy_mat) {
  cv::Mat source_mat = get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
  cv::Mat result(mapxy_mat.rows, mapxy_mat.cols, Format);
  cv::Mat empty;
  remap(source_mat, result, mapxy_mat, empty, Interpolation, BorderMode,
        BorderValue / 1000.0);
  return result;
}

#if MANAGER
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
bool test_remap_s16(int index, RecreatedMessageQueue& request_queue,
                    RecreatedMessageQueue& reply_queue) {
  cv::Mat source_mat = get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
  cv::RNG rng(0);

  for (size_t w = 5; w <= kMaxWidth; w += 3) {
    for (size_t h = 5; h <= kMaxHeight; h += 2) {
      cv::Mat source_mat =
          get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
      cv::Mat mapxy_mat(w, h, CV_16SC2);
      rng.fill(mapxy_mat, cv::RNG::UNIFORM, -3, kMaxWidth + 3);

      cv::Mat actual_mat = exec_remap_s16<ScalarType, Format, Interpolation,
                                          BorderMode, BorderValue>(mapxy_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, mapxy_mat);

      bool success =
          (CV_MAT_DEPTH(Format) == CV_8U &&
           !are_matrices_different<uint8_t>(0, actual_mat, expected_mat)) ||
          (CV_MAT_DEPTH(Format) == CV_16U &&
           !are_matrices_different<uint16_t>(0, actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(w, h, source_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }
  return false;
}
#endif

// BorderValue is interpreted as 1/1000, i.e. 500 for 0.5
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
cv::Mat exec_remap_s16point5(cv::Mat& map_mat) {
  cv::Mat empty;
  // integer part is 16SC2, twice as much data as the fractional part, 16UC1
  int height = map_mat.rows * 2 / 3;
  cv::Mat mapxy_mat = map_mat.rowRange(0, height);
  ushort* p_frac = map_mat.rowRange(height, map_mat.rows).ptr<ushort>();
  cv::Mat mapfrac_mat{height, map_mat.cols, CV_16UC1, p_frac};
  cv::Mat result(mapxy_mat.rows, mapxy_mat.cols, Format);
  cv::Mat source_mat = get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
  remap(source_mat, result, mapxy_mat, mapfrac_mat, Interpolation, BorderMode,
        BorderValue / 1000.0);
  return result;
}

#if MANAGER
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
bool test_remap_s16point5(int index, RecreatedMessageQueue& request_queue,
                          RecreatedMessageQueue& reply_queue) {
  cv::Mat source_mat = get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
  cv::RNG rng(0);

  for (int w = 5; w <= kMaxWidth; ++w) {
    for (int h = 6; h <= kMaxHeight; h += 2) {  // h must be even
      cv::Mat map_mat(h * 3 / 2, w, CV_16SC2);
      cv::Mat mapxy_mat = map_mat.rowRange(0, h);
      ushort* p_frac = map_mat.rowRange(h, map_mat.rows).ptr<ushort>();
      cv::Mat mapfrac_mat{h, map_mat.cols, CV_16UC1, p_frac};
      rng.fill(mapxy_mat, cv::RNG::UNIFORM, -3, kMaxWidth + 3);
      // Test out of range fractional part too
      rng.fill(mapfrac_mat, cv::RNG::UNIFORM, 0, cv::INTER_TAB_SIZE2 * 3 / 2);

      cv::Mat actual_mat =
          exec_remap_s16point5<ScalarType, Format, Interpolation, BorderMode,
                               BorderValue>(map_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, map_mat);

      bool success =
          (CV_MAT_DEPTH(Format) == CV_8U &&
           !are_matrices_different<uint8_t>(1, actual_mat, expected_mat)) ||
          (CV_MAT_DEPTH(Format) == CV_16U &&
           !are_matrices_different<uint16_t>(1, actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(w, h, source_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }
  return false;
}
#endif

// BorderValue is interpreted as 1/1000, i.e. 500 for 0.5
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
cv::Mat exec_remap_f32(cv::Mat& mapxy_mat) {
  cv::Mat source_mat = get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
  cv::Mat result(mapxy_mat.rows, mapxy_mat.cols, Format);

  cv::Mat mapx_mat = mapxy_mat.rowRange(0, mapxy_mat.rows / 2);
  cv::Mat mapy_mat = mapxy_mat.rowRange(mapxy_mat.rows / 2, mapxy_mat.rows);

  remap(source_mat, result, mapx_mat, mapy_mat, Interpolation, BorderMode,
        BorderValue / 1000.0);
  return result;
}

#if MANAGER
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
bool test_remap_f32(int index, RecreatedMessageQueue& request_queue,
                    RecreatedMessageQueue& reply_queue) {
  cv::Mat source_mat = get_source_mat<ScalarType, CV_MAT_CN(Format)>(Format);
  cv::RNG rng(0);

  for (size_t w = 5; w <= kMaxWidth * 2; w += 3) {
    for (size_t h = 5; h <= kMaxHeight * 2; h += 2) {
      cv::Mat map_mat(h * 2, w, CV_32FC1);
      cv::Mat mapx_mat = map_mat.rowRange(0, h);
      cv::Mat mapy_mat = map_mat.rowRange(h, map_mat.rows);
      for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
          // Values from -0.49 to 0.49, so exactly 0.5 is excluded

          // Reason: When rounding floating point values to integer, OpenCV does
          // scalar rounding that works differently based on the rounding
          // environment. E.g. it can use "Rounding to nearest, ties to even",
          // while KleidiCV always uses "Rounding to nearest, towards plus
          // infinity". To prevent these differences, values with exactly 0.5
          // fractional part are excluded.
          float divisor = (1.01 * 0x1p32);
          float epsilon = 0x1p-16;
          float fractionX = rng.next() / divisor - 0.5F + epsilon;
          float fractionY = rng.next() / divisor - 0.5F + epsilon;
          mapx_mat.at<float>(y, x) =
              (static_cast<int32_t>(rng.next() % (kMaxWidth + 6)) - 3) +
              fractionX;
          mapy_mat.at<float>(y, x) =
              (static_cast<int32_t>(rng.next() % (kMaxHeight + 6)) - 3) +
              fractionY;
        }
      }

      cv::Mat actual_mat = exec_remap_f32<ScalarType, Format, Interpolation,
                                          BorderMode, BorderValue>(map_mat);

      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, map_mat);

      bool success = false;

      if constexpr (Interpolation == cv::INTER_NEAREST) {
        success =
            (CV_MAT_DEPTH(Format) == CV_8U &&
             !are_matrices_different<uint8_t>(0, actual_mat, expected_mat)) ||
            (CV_MAT_DEPTH(Format) == CV_16U &&
             !are_matrices_different<uint16_t>(0, actual_mat, expected_mat));
      } else {
        // Reference algorithm in OpenCV uses integer interpolation with 5 bits
        // fractional part while KleidiCV uses float arithmetic. That means
        // KleidiCV has better accuracy, so a difference is expected between
        // OpenCV and KleidiCV results. Please check the remap accuracy test in
        // this file for more details.
        success =
            (CV_MAT_DEPTH(Format) == CV_8U &&
             !are_matrices_different<uint8_t>(8, actual_mat, expected_mat)) ||
            (CV_MAT_DEPTH(Format) == CV_16U &&
             !are_matrices_different<uint16_t>(2032, actual_mat, expected_mat));
      }

      if (!success) {
        fail_print_matrices(w, h, source_mat, actual_mat, expected_mat);
        std::cout << "=== mapx_mat:" << std::endl;
        std::cout << mapx_mat << std::endl << std::endl;
        std::cout << "=== mapy_mat:" << std::endl;
        std::cout << mapy_mat << std::endl << std::endl;
        return true;
      }
    }
  }
  return false;
}

#endif

template <typename T>
cv::Mat exec_remap_f32_linear_accuracy(cv::Mat& map) {
  // These vaules can trigger biggest difference in the result.
  cv::Mat source =
      (cv::Mat_<T>(2, 2) << 0, 0, 0, std::numeric_limits<T>::max());
  cv::Mat result;
  // Same map used for both x and y directions.
  remap(source, result, map, map, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

  return result;
}

#if MANAGER
// Reference algorithm in OpenCV uses integer interpolation with 5 bits
// fractional part while KleidiCV uses float arithmetic for the linear
// extrapolation.
//
// The maximum difference between the implementations can be obtained by:
// granularity = 1 / 2^5
// map_value (to trigger maximal difference ) = 1 - granularity / 2
//
// Results in one dimension:
// KleidiCV's result = RoundToNearestInt(<max element value> * map_value)
// OpenCV's result = <max element value>
//
// Results in two dimensions:
// KleidiCV's result =
//   RoundToNearestInt(<max element value> * map_value * map_value)
// OpenCV's result = <max element value>
//
// That means the maximum error (difference between the results) is:
//  * 8 in case u8.
//  * 2032 in case of u16.
template <typename T>
bool test_remap_f32_linear_accuracy(int index,
                                    RecreatedMessageQueue& request_queue,
                                    RecreatedMessageQueue& reply_queue) {
  // Map value to trigger biggest possible difference as described above.
  float granularity = 1 / pow(2.0f, 5.0f);
  float map_value = (1 - granularity / 2);

  // If the width of the result image is less than 4 then KleidiCV rejects the
  // operation and fallback happens to OpenCV's implementation. But for this
  // accuracy test only the first pixel counts. This map is used for both x and
  // y directions.
  cv::Mat map = (cv::Mat_<float>(1, 4) << map_value, 0, 0, 0);

  // KleidiCV results after float calculations
  T expected_kleidicv_value = std::is_same_v<T, uint8_t> ? 247 : 63503;
  // OpenCV still returns with max element value
  T expected_opencv_value = std::numeric_limits<T>::max();

  // Check results
  cv::Mat opencv_result =
      get_expected_from_subordinate(index, request_queue, reply_queue, map);
  if (opencv_result.at<T>(0, 0) != expected_opencv_value) {
    std::cout << "Error! Unexpected OpenCV value at " << __FILE__ << ':'
              << __LINE__ << "\n\n";
    return true;
  }
  cv::Mat kleidicv_result = exec_remap_f32_linear_accuracy<T>(map);
  if (kleidicv_result.at<T>(0, 0) != expected_kleidicv_value) {
    std::cout << "Error! Unexpected KleidiCV value at " << __FILE__ << ':'
              << __LINE__ << "\n\n";
    return true;
  }

  // Map value is decreased by the smallest possible value to trigger the change
  // of OpenCV results.
  map_value = std::nextafterf(map_value, 0.0);
  map = (cv::Mat_<float>(1, 4) << map_value, 0, 0, 0);

  // New expected OpenCV results
  expected_opencv_value = std::is_same_v<T, uint8_t> ? 239 : 61503;
  // No need to update expected KleidiCV results

  // Check results
  opencv_result =
      get_expected_from_subordinate(index, request_queue, reply_queue, map);
  if (opencv_result.at<T>(0, 0) != expected_opencv_value) {
    std::cout << "Error! Unexpected OpenCV value at " << __FILE__ << ':'
              << __LINE__ << "\n\n";
    return true;
  }
  kleidicv_result = exec_remap_f32_linear_accuracy<T>(map);
  if (kleidicv_result.at<T>(0, 0) != expected_kleidicv_value) {
    std::cout << "Error! Unexpected KleidiCV value at " << __FILE__ << ':'
              << __LINE__ << "\n\n";
    return true;
  }

  return false;
}
#endif

std::vector<test>& remap_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("RemapS16 uint8 Replicate", (test_remap_s16<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_s16<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16 uint16 Replicate", (test_remap_s16<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_s16<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16 uint8 Constant", (test_remap_s16<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>), (exec_remap_s16<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapS16 uint16 Constant", (test_remap_s16<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>), (exec_remap_s16<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>)),

    TEST("RemapS16Point5 uint8 Replicate", (test_remap_s16point5<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_s16point5<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16Point5 uint8 Replicate 4ch", (test_remap_s16point5<uint8_t, CV_8UC4, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_s16point5<uint8_t, CV_8UC4, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16Point5 uint16 Replicate", (test_remap_s16point5<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_s16point5<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16Point5 uint16 Replicate 4ch", (test_remap_s16point5<uint16_t, CV_16UC4, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_s16point5<uint16_t, CV_16UC4, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16Point5 uint8 Constant", (test_remap_s16point5<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>), (exec_remap_s16point5<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapS16Point5 uint8 Constant 4ch", (test_remap_s16point5<uint8_t, CV_8UC4, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>), (exec_remap_s16point5<uint8_t, CV_8UC4, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapS16Point5 uint16 Constant", (test_remap_s16point5<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>), (exec_remap_s16point5<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapS16Point5 uint16 Constant 4ch", (test_remap_s16point5<uint16_t, CV_16UC4, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>), (exec_remap_s16point5<uint16_t, CV_16UC4, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>)),

    TEST("RemapF32 uint8 Replicate Linear 1ch", (test_remap_f32<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint16 Replicate Linear 1ch", (test_remap_f32<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint8 Constant Linear 1ch", (test_remap_f32<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>), (exec_remap_f32<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapF32 uint16 Constant Linear 1ch", (test_remap_f32<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 123210>), (exec_remap_f32<uint16_t, CV_16UC1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 123210>)),
    TEST("RemapF32 uint8 Replicate Nearest 1ch", (test_remap_f32<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint16 Replicate Nearest 1ch", (test_remap_f32<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint8 Constant Nearest 1ch", (test_remap_f32<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>), (exec_remap_f32<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapF32 uint16 Constant Nearest 1ch", (test_remap_f32<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 123210>), (exec_remap_f32<uint16_t, CV_16UC1, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 123210>)),

    TEST("RemapF32 uint8 Replicate Linear 2ch", (test_remap_f32<uint8_t, CV_8UC2, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint8_t, CV_8UC2, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint16 Replicate Linear 2ch", (test_remap_f32<uint16_t, CV_16UC2, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint16_t, CV_16UC2, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint8 Constant Linear 2ch", (test_remap_f32<uint8_t, CV_8UC2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>), (exec_remap_f32<uint8_t, CV_8UC2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapF32 uint16 Constant Linear 2ch", (test_remap_f32<uint16_t, CV_16UC2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 123210>), (exec_remap_f32<uint16_t, CV_16UC2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 123210>)),
    TEST("RemapF32 uint8 Replicate Nearest 2ch", (test_remap_f32<uint8_t, CV_8UC2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint8_t, CV_8UC2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint16 Replicate Nearest 2ch", (test_remap_f32<uint16_t, CV_16UC2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_f32<uint16_t, CV_16UC2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapF32 uint8 Constant Nearest 2ch", (test_remap_f32<uint8_t, CV_8UC2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>), (exec_remap_f32<uint8_t, CV_8UC2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 12321>)),
    TEST("RemapF32 uint16 Constant Nearest 2ch", (test_remap_f32<uint16_t, CV_16UC2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 123210>), (exec_remap_f32<uint16_t, CV_16UC2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 123210>)),

    TEST("RemapF32 Linear uint8 accuracy", test_remap_f32_linear_accuracy<uint8_t>, exec_remap_f32_linear_accuracy<uint8_t>),
    TEST("RemapF32 Linear uint16 accuracy", test_remap_f32_linear_accuracy<uint16_t>, exec_remap_f32_linear_accuracy<uint16_t>)
  };
  // clang-format on
  return tests;
}
