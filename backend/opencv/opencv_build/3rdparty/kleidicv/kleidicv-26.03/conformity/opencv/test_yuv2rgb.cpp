// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tests.h"

template <bool SwitchBlue, size_t Channels>
cv::Mat exec_yuv2rgb(cv::Mat& input) {
  cv::Mat result;
  if constexpr (SwitchBlue) {
    cv::cvtColor(input, result, cv::COLOR_YUV2RGB, Channels);
  } else {
    cv::cvtColor(input, result, cv::COLOR_YUV2BGR, Channels);
  }
  return result;
}

#if MANAGER
template <bool SwitchBlue, size_t Channels>
bool test_yuv2rgb(int index, RecreatedMessageQueue& request_queue,
                  RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, CV_8UC3);
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);
      cv::Mat actual = exec_yuv2rgb<SwitchBlue, Channels>(input);
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

std::vector<test>& yuv2rgb_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
      TEST("YUV2RGB",  (test_yuv2rgb<false, 3>), (exec_yuv2rgb<false, 3>)),
      TEST("YUV2BGR",  (test_yuv2rgb<true,  3>), (exec_yuv2rgb<true,  3>)),
      TEST("YUV2RGBA", (test_yuv2rgb<false, 4>), (exec_yuv2rgb<false, 4>)),
      TEST("YUV2BGRA", (test_yuv2rgb<true,  4>), (exec_yuv2rgb<true,  4>))
    };
  // clang-format on
  return tests;
}
