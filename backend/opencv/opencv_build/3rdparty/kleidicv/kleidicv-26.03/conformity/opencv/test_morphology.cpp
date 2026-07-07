// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "tests.h"

template <bool Erode, bool DefaultBorder>
static cv::Mat exec_morphology(cv::Mat& input_mat) {
  // Running the operation in-place
  cv::Mat in_place = input_mat.clone();
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::Scalar border_value = DefaultBorder
                                ? cv::morphologyDefaultBorderValue()
                                : cv::Scalar(123, -DBL_MAX, FLT_MAX, 0);
  auto func = Erode ? &cv::erode : &cv::dilate;
  func(in_place, in_place, kernel, cv::Point(-1, -1), 3, cv::BORDER_CONSTANT,
       border_value);
  return in_place;
}

#if MANAGER
template <bool Erode, bool DefaultBorder, size_t Channels>
bool test_morphology(int index, RecreatedMessageQueue& request_queue,
                     RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, CV_8UC(Channels));
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);

      cv::Mat actual = exec_morphology<Erode, DefaultBorder>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      if (are_matrices_different<uint8_t>(0, actual, expected)) {
        fail_print_matrices(x, y, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& morphology_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Dilate, default border, 1 channel", (test_morphology<false, true, 1>), (exec_morphology<false, true>)),
    TEST("Dilate, default border, 2 channel", (test_morphology<false, true, 2>), (exec_morphology<false, true>)),
    TEST("Dilate, default border, 3 channel", (test_morphology<false, true, 3>), (exec_morphology<false, true>)),
    TEST("Dilate, default border, 4 channel", (test_morphology<false, true, 4>), (exec_morphology<false, true>)),
    TEST("Erode, default border, 1 channel", (test_morphology<true, true, 1>), (exec_morphology<true, true>)),
    TEST("Erode, default border, 2 channel", (test_morphology<true, true, 2>), (exec_morphology<true, true>)),
    TEST("Erode, default border, 3 channel", (test_morphology<true, true, 3>), (exec_morphology<true, true>)),
    TEST("Erode, default border, 4 channel", (test_morphology<true, true, 4>), (exec_morphology<true, true>)),
    TEST("Dilate, custom border, 1 channel", (test_morphology<false, false, 1>), (exec_morphology<false, false>)),
    TEST("Dilate, custom border, 2 channel", (test_morphology<false, false, 2>), (exec_morphology<false, false>)),
    TEST("Dilate, custom border, 3 channel", (test_morphology<false, false, 3>), (exec_morphology<false, false>)),
    TEST("Dilate, custom border, 4 channel", (test_morphology<false, false, 4>), (exec_morphology<false, false>)),
    TEST("Erode, custom border, 1 channel", (test_morphology<true, false, 1>), (exec_morphology<true, false>)),
    TEST("Erode, custom border, 2 channel", (test_morphology<true, false, 2>), (exec_morphology<true, false>)),
    TEST("Erode, custom border, 3 channel", (test_morphology<true, false, 3>), (exec_morphology<true, false>)),
    TEST("Erode, custom border, 4 channel", (test_morphology<true, false, 4>), (exec_morphology<true, false>)),
  };
  // clang-format on
  return tests;
}
