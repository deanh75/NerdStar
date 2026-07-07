// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "tests.h"

template <int RotateCode>
static cv::Mat exec_rotate(cv::Mat& input_mat) {
  cv::Mat result;
  cv::rotate(input_mat, result, RotateCode);
  return result;
}

#if MANAGER
template <int RotateCode, typename TypeParam, size_t Channels>
bool test_rotate(int index, RecreatedMessageQueue& request_queue,
                 RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t height = 5; height <= 16; ++height) {
    for (size_t width = 5; width <= 16; ++width) {
      cv::Mat input(height, width,
                    get_opencv_matrix_type<TypeParam, Channels>());
      rng.fill(input, cv::RNG::UNIFORM, std::numeric_limits<TypeParam>::min(),
               std::numeric_limits<TypeParam>::max());

      cv::Mat actual = exec_rotate<RotateCode>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      if (are_matrices_different<TypeParam>(0, actual, expected)) {
        fail_print_matrices(input.rows, input.cols, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

#define ROTATE_TEST(code, type, channels, label)              \
  TEST("Rotate " #code ", " #channels " channel (" label ")", \
       (test_rotate<cv::code, type, channels>), (exec_rotate<cv::code>))

std::vector<test>& rotate_tests_get() {
  static std::vector<test> tests = {
      ROTATE_TEST(ROTATE_90_CLOCKWISE, uint8_t, 1, "U8"),
      ROTATE_TEST(ROTATE_90_CLOCKWISE, uint8_t, 2, "U8"),
      ROTATE_TEST(ROTATE_90_CLOCKWISE, uint8_t, 4, "U8"),
      ROTATE_TEST(ROTATE_90_CLOCKWISE, uint16_t, 1, "U16"),
      ROTATE_TEST(ROTATE_90_CLOCKWISE, uint16_t, 2, "U16"),
      ROTATE_TEST(ROTATE_90_CLOCKWISE, uint16_t, 4, "U16"),
      ROTATE_TEST(ROTATE_90_COUNTERCLOCKWISE, uint8_t, 1, "U8"),
      ROTATE_TEST(ROTATE_90_COUNTERCLOCKWISE, uint8_t, 2, "U8"),
      ROTATE_TEST(ROTATE_90_COUNTERCLOCKWISE, uint8_t, 4, "U8"),
      ROTATE_TEST(ROTATE_90_COUNTERCLOCKWISE, uint16_t, 1, "U16"),
      ROTATE_TEST(ROTATE_90_COUNTERCLOCKWISE, uint16_t, 2, "U16"),
      ROTATE_TEST(ROTATE_90_COUNTERCLOCKWISE, uint16_t, 4, "U16"),
  };
  return tests;
}
