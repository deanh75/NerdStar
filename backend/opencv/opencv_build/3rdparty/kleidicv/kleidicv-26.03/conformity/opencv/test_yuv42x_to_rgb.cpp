// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "tests.h"

template <int Code>
static cv::Mat exec_cvtcolor(cv::Mat& input) {
  cv::Mat result;
  cv::cvtColor(input, result, Code);
  return result;
}

#if MANAGER
template <int Code, int input_channel>
bool test_yuv42x_to_rgb(int index, RecreatedMessageQueue& request_queue,
                        RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  auto check = [&](size_t x, size_t y) -> bool {
    // Multiply height by 1.5 to fit Y & UV planes.
    cv::Mat input(y * 3 / 2, x, CV_8UC(input_channel));
    rng.fill(input, cv::RNG::UNIFORM, 0, 255);

    cv::Mat actual = exec_cvtcolor<Code>(input);
    cv::Mat expected =
        get_expected_from_subordinate(index, request_queue, reply_queue, input);

    if (are_matrices_different<uint8_t>(0, actual, expected)) {
      fail_print_matrices(x, y, input, actual, expected);
      return true;
    }
    return false;
  };

  // OpenCV only accepts images with an even number of columns & rows to some
  // YUV formats like YUV420.
  for (size_t x = 48; x <= 64; x += 2) {
    for (size_t y = 2; y <= 16; y += 2) {
      if (check(x, y)) {
        return true;
      }
    }
  }

  // Check taller images - this number of rows was necessary to trigger a bug on
  // a machine with 64 cores.
  if (check(2, 1000)) {
    return true;
  }

  return false;
}
#endif

#define CVTCOLOR_TEST(code, channel)                           \
  TEST(#code, (test_yuv42x_to_rgb<cv::COLOR_##code, channel>), \
       exec_cvtcolor<cv::COLOR_##code>)

std::vector<test>& yuv42x_to_rgb_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
      CVTCOLOR_TEST(YUV2RGB_UYVY, 2),
      CVTCOLOR_TEST(YUV2BGR_UYVY, 2),
      CVTCOLOR_TEST(YUV2RGBA_UYVY, 2),
      CVTCOLOR_TEST(YUV2BGRA_UYVY, 2),
      CVTCOLOR_TEST(YUV2RGB_YUY2, 2),
      CVTCOLOR_TEST(YUV2BGR_YUY2, 2),
      CVTCOLOR_TEST(YUV2RGB_YVYU, 2),
      CVTCOLOR_TEST(YUV2BGR_YVYU, 2),
      CVTCOLOR_TEST(YUV2RGBA_YUY2, 2),
      CVTCOLOR_TEST(YUV2BGRA_YUY2, 2),
      CVTCOLOR_TEST(YUV2RGBA_YVYU, 2),
      CVTCOLOR_TEST(YUV2BGRA_YVYU, 2),
      CVTCOLOR_TEST(YUV2BGRA_YV12, 1),
      CVTCOLOR_TEST(YUV2RGB_YV12, 1),
      CVTCOLOR_TEST(YUV2RGBA_YV12, 1),
      CVTCOLOR_TEST(YUV2BGR_IYUV, 1),
      CVTCOLOR_TEST(YUV2BGRA_IYUV, 1),
      CVTCOLOR_TEST(YUV2RGB_IYUV, 1),
      CVTCOLOR_TEST(YUV2RGBA_IYUV, 1),
      CVTCOLOR_TEST(YUV2BGR_NV12, 1),
      CVTCOLOR_TEST(YUV2RGB_NV12, 1),
      CVTCOLOR_TEST(YUV2BGRA_NV12, 1),
      CVTCOLOR_TEST(YUV2RGBA_NV12, 1),
      CVTCOLOR_TEST(YUV2BGR_NV21, 1),
      CVTCOLOR_TEST(YUV2RGB_NV21, 1),
      CVTCOLOR_TEST(YUV2BGRA_NV21, 1),
      CVTCOLOR_TEST(YUV2RGBA_NV21, 1),
  };
  // clang-format on
  return tests;
}
