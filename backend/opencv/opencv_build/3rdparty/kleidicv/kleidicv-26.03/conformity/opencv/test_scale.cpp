// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "opencv2/core/hal/interface.h"
#include "utils.h"

template <int Scale, int Shift, size_t dstFormat, bool InPlace>
cv::Mat exec_scale(cv::Mat& input_mat) {
  if constexpr (InPlace) {
    input_mat.convertTo(input_mat, -1, Scale / 1000.0, Shift / 1000.0);
    return input_mat;
  }
  cv::Mat result;
  input_mat.convertTo(result, dstFormat, Scale / 1000.0, Shift / 1000.0);
  return result;
}

#if MANAGER
template <int Scale, int Shift, size_t srcFormat, size_t dstFormat,
          bool InPlace>
bool test_scale(int index, RecreatedMessageQueue& request_queue,
                RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t rows = 3; rows <= 8; ++rows) {
    for (size_t cols = 3; cols <= 33; ++cols) {
      cv::Mat input_mat(rows, cols, srcFormat);
      if (CV_MAT_DEPTH(srcFormat) == CV_8U) {
        rng.fill(input_mat, cv::RNG::UNIFORM, 0.0, 255.0);
      }
      if (CV_MAT_DEPTH(srcFormat) == CV_32F) {
        rng.fill(input_mat, cv::RNG::NORMAL, 0.0, 1e4);
      }
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, input_mat);
      cv::Mat actual_mat =
          exec_scale<Scale, Shift, dstFormat, InPlace>(input_mat);
      bool success =
          (CV_MAT_DEPTH(dstFormat) == CV_32F &&
           !are_float_matrices_different<float>(1e-5, actual_mat,
                                                expected_mat)) ||
          (CV_MAT_DEPTH(dstFormat) == CV_16F &&
           !are_float16_matrices_different(0, actual_mat, expected_mat)) ||
          (CV_MAT_DEPTH(dstFormat) == CV_8U &&
           !are_matrices_different<uint8_t>(0, actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(rows, cols, input_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& scale_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Scale float32, scale=1.0, shift=2.0", (test_scale<1000, 2000, CV_32FC4, CV_32FC4, false>), (exec_scale<1000, 2000, CV_32FC4, false>)),
    TEST("Scale float32, scale=-10.0, shift=0.0", (test_scale<(-10000), 0, CV_32FC1, CV_32FC1, false>), (exec_scale<(-10000), 0, CV_32FC1, false>)),
    TEST("Scale float32, scale=3.14, shift=2.72", (test_scale<3140, 2720, CV_32FC2, CV_32FC2, false>), (exec_scale<3140, 2720, CV_32FC2, false>)),
    TEST("Scale float32, scale=7e5, shift=8e-3", (test_scale<700000000, 8, CV_32FC3, CV_32FC3, false>), (exec_scale<700000000, 8, CV_32FC3, false>)),
    TEST("Scale float32, scale=1e-3, shift=8e-3", (test_scale<1, 8, CV_32FC4, CV_32FC4, false>), (exec_scale<1, 8, CV_32FC4, false>)),
    TEST("Scale float32, scale=1.0, shift=2.0, in-place", (test_scale<1000, 2000, CV_32FC4, CV_32FC4, true>), (exec_scale<1000, 2000, CV_32FC4, true>)),
    TEST("Scale float32, scale=-10.0, shift=0.0, in-place", (test_scale<(-10000), 0, CV_32FC1, CV_32FC1, true>), (exec_scale<(-10000), 0, CV_32FC1, true>)),
    TEST("Scale float32, scale=3.14, shift=2.72, in-place", (test_scale<3140, 2720, CV_32FC2, CV_32FC2, true>), (exec_scale<3140, 2720, CV_32FC2, true>)),
    TEST("Scale float32, scale=7e5, shift=8e-3, in-place", (test_scale<700000000, 8, CV_32FC3, CV_32FC3, true>), (exec_scale<700000000, 8, CV_32FC3, true>)),
    TEST("Scale float32, scale=1e-3, shift=8e-3, in-place", (test_scale<1, 8, CV_32FC4, CV_32FC4, true>), (exec_scale<1, 8, CV_32FC4, true>)),

    TEST("Scale uint8, scale=1.0, shift=2.0", (test_scale<1000, 2000, CV_8UC4, CV_8UC4, false>), (exec_scale<1000, 2000, CV_8UC4, false>)),
    TEST("Scale uint8, scale=-10.0, shift=0.0", (test_scale<(-10000), 0, CV_8UC1, CV_8UC1, false>), (exec_scale<(-10000), 0, CV_8UC1, false>)),
    TEST("Scale uint8, scale=3.14, shift=-2.72", (test_scale<3140, -2720, CV_8UC2, CV_8UC2, false>), (exec_scale<3140, -2720, CV_8UC2, false>)),
    TEST("Scale uint8, scale=17.17, shift=3.9", (test_scale<17170, 3900, CV_8UC3, CV_8UC3, false>), (exec_scale<17170, 3900, CV_8UC3, false>)),
    TEST("Scale uint8, scale=0.13, shift=230", (test_scale<130, 230000, CV_8UC4, CV_8UC4, false>), (exec_scale<130, 230000, CV_8UC4, false>)),
    TEST("Scale uint8, scale=1.0, shift=2.0, in-place", (test_scale<1000, 2000, CV_8UC4, CV_8UC4, true>), (exec_scale<1000, 2000, CV_8UC4, true>)),
    TEST("Scale uint8, scale=-10.0, shift=0.0, in-place", (test_scale<(-10000), 0, CV_8UC1, CV_8UC1, true>), (exec_scale<(-10000), 0, CV_8UC1, true>)),
    TEST("Scale uint8, scale=3.14, shift=-2.72, in-place", (test_scale<3140, -2720, CV_8UC2, CV_8UC2, true>), (exec_scale<3140, -2720, CV_8UC2, true>)),
    TEST("Scale uint8, scale=17.17, shift=3.9, in-place", (test_scale<17170, 3900, CV_8UC3, CV_8UC3, true>), (exec_scale<17170, 3900, CV_8UC3, true>)),
    TEST("Scale uint8, scale=0.13, shift=230, in-place", (test_scale<130, 230000, CV_8UC4, CV_8UC4, true>), (exec_scale<130, 230000, CV_8UC4, true>)),

    TEST("Scale uint8 to float16, scale=1.0, shift=2.0", (test_scale<1000, 2000, CV_8UC4, CV_16FC4, false>), (exec_scale<1000, 2000, CV_16FC4, false>)),
    TEST("Scale uint8 to float16, scale=-10.0, shift=0.0", (test_scale<(-10000), 0, CV_8UC1, CV_16FC1, false>), (exec_scale<(-10000), 0, CV_16FC1, false>)),
    TEST("Scale uint8 to float16, scale=3.14, shift=-2.72", (test_scale<3140, -2720, CV_8UC2, CV_16FC2, false>), (exec_scale<3140, -2720, CV_16FC2, false>)),
    TEST("Scale uint8 to float16, scale=0.75, shift=141.42", (test_scale<750, 141420, CV_8UC3, CV_16FC3, false>), (exec_scale<750, 141420, CV_16FC3, false>)),
    TEST("Scale uint8 to float16, scale=0.75, shift=-15", (test_scale<750, -10000, CV_8UC4, CV_16FC4, false>), (exec_scale<750, -10000, CV_16FC4, false>)),
  };
  // clang-format on
  return tests;
}
