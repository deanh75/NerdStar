// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/video.hpp"
#include "tests.h"

static cv::Mat exec_standalone_lucas_kanade_alg(cv::Mat& input_mat) {
  int mid = input_mat.rows / 2;
  cv::Mat prevImg = input_mat.rowRange(0, mid).clone();
  cv::Mat nextImg = input_mat.rowRange(mid, input_mat.rows).clone();

  std::vector<cv::Point2f> prevPts, nextPts;
  std::vector<uint8_t> status;
  std::vector<float> err;

  const size_t point_count_x = prevImg.cols / 30;
  const size_t point_count_y = prevImg.rows / 30;

  for (size_t j = 0; j != point_count_y; ++j) {
    for (size_t i = 0; i != point_count_x; ++i) {
      float x = i * prevImg.cols / point_count_x;
      float y = j * prevImg.rows / point_count_y;
      prevPts.push_back(cv::Point2f(x, y));
    }
  }

  prevPts.push_back(cv::Point2f(0, 0));
  prevPts.push_back(cv::Point2f(prevImg.cols - 0.5F, 0));
  prevPts.push_back(cv::Point2f(0, prevImg.rows - 0.5F));
  prevPts.push_back(cv::Point2f(prevImg.cols - 0.5F, prevImg.rows - 0.5F));

  cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err);

  cv::Mat result(0, 1, CV_32FC1);
  for (const auto& p : nextPts) {
    result.push_back(p.x);
    result.push_back(p.y);
  }
  for (auto s : status) {
    result.push_back(static_cast<float>(s));
  }
  for (auto e : err) {
    result.push_back(e);
  }

  return result;
}

#if MANAGER
template <size_t channels>
bool test_standalone_lucas_kanade_alg(int index,
                                      RecreatedMessageQueue& request_queue,
                                      RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  const size_t width = 80;
  const size_t height = 60;

  cv::Mat input_mat(height * 2, width, CV_8UC(channels));
  rng.fill(input_mat, cv::RNG::UNIFORM, 0, 255);

  // Rotate top half of image, rotate it and scale it a little, and put that in
  // the bottom half of the image.
  cv::warpAffine(input_mat(cv::Rect(0, 0, width, height)),
                 input_mat(cv::Rect(0, height, width, height)),
                 cv::getRotationMatrix2D(cv::Point2f(), 5, 0.95F),
                 cv::Size(width, height), cv::INTER_LINEAR,
                 cv::BORDER_REFLECT_101);

  cv::Mat actual_mat = exec_standalone_lucas_kanade_alg(input_mat);
  cv::Mat expected_mat = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input_mat);

  if (are_float_matrices_different<float>(1e-3F, actual_mat, expected_mat)) {
    fail_print_matrices(width, height, input_mat, actual_mat, expected_mat);
    return true;
  }

  return false;
}
#endif

std::vector<test>& standalone_lucas_kanade_alg_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Standalone Lucas-Kanade Alg 1 channel", (test_standalone_lucas_kanade_alg<1>), exec_standalone_lucas_kanade_alg),
    TEST("Standalone Lucas-Kanade Alg 2 channel", (test_standalone_lucas_kanade_alg<2>), exec_standalone_lucas_kanade_alg),
    TEST("Standalone Lucas-Kanade Alg 3 channel", (test_standalone_lucas_kanade_alg<3>), exec_standalone_lucas_kanade_alg),
  };
  // clang-format on
  return tests;
}
