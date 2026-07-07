// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <vector>

#include "tests.h"

template <typename T, size_t Format>
cv::Mat exec_sum(cv::Mat& input) {
  // If sum is implemented for multi channels, these dimensions must be modified
  cv::Mat result(1, 1, Format, cv::sum(input));
  return result;
}

#if MANAGER
template <typename T, size_t Format>
bool test_sum(int index, RecreatedMessageQueue& request_queue,
              RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t height = 2; height <= 128; height *= 2) {
    for (size_t width = 2; width <= 128; width *= 2) {
      cv::Mat input(height, width, Format);
      rng.fill(input, cv::RNG::UNIFORM, -10, 100);
      cv::Mat actual = exec_sum<T, Format>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      if (are_float_matrices_different<T>(0.001, actual, expected)) {
        fail_print_matrices(height, width, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& sum_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("sum_f32", (test_sum<float, CV_32FC1>), (exec_sum<float, CV_32FC1>)),
  };
  // clang-format on
  return tests;
}
