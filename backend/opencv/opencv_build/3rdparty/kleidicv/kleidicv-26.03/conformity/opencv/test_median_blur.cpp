// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tests.h"

template <size_t KernelSize>
cv::Mat exec_median_blur(cv::Mat& input) {
  cv::Mat result;
  cv::medianBlur(input, result, KernelSize);
  return result;
}

#if MANAGER
template <size_t KernelSize, typename TypeParam, size_t Channels>
bool test_median_blur(int index, RecreatedMessageQueue& request_queue,
                      RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  size_t size_min{0};
  size_t size_max{0};
  size_t step{0};

  if constexpr (KernelSize <= 15) {
    size_min = KernelSize - 1;
    size_max = 2 * KernelSize + 16;
    step = 1;
  } else {
    size_min = KernelSize - 1;
    size_max = KernelSize + 16;
    step = 16;
  }

  for (size_t w = size_min; w <= size_max; w += step) {
    for (size_t h = size_min; h <= size_max; h += step) {
      cv::Mat input(w, h, get_opencv_matrix_type<TypeParam, Channels>());
      rng.fill(input, cv::RNG::UNIFORM, std::numeric_limits<TypeParam>::min(),
               std::numeric_limits<TypeParam>::max());

      cv::Mat actual = exec_median_blur<KernelSize>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      if (are_matrices_different<TypeParam>(0, actual, expected)) {
        fail_print_matrices(w, h, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& median_blur_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Median 3x3, 1 channel (U8)",    (test_median_blur<3, uint8_t, 1>),   exec_median_blur<3>),
    TEST("Median 3x3, 3 channel (U8)",    (test_median_blur<3, uint8_t, 3>),   exec_median_blur<3>),
    TEST("Median 3x3, 4 channel (U8)",    (test_median_blur<3, uint8_t, 4>),   exec_median_blur<3>),
    TEST("Median 3x3, 1 channel (U16)",   (test_median_blur<3, uint16_t, 1>),  exec_median_blur<3>),
    TEST("Median 3x3, 3 channel (U16)",   (test_median_blur<3, uint16_t, 3>),  exec_median_blur<3>),
    TEST("Median 3x3, 4 channel (U16)",   (test_median_blur<3, uint16_t, 4>),  exec_median_blur<3>),
    TEST("Median 3x3, 1 channel (S16)",   (test_median_blur<3, int16_t, 1>),   exec_median_blur<3>),
    TEST("Median 3x3, 3 channel (S16)",   (test_median_blur<3, int16_t, 3>),   exec_median_blur<3>),
    TEST("Median 3x3, 4 channel (S16)",   (test_median_blur<3, int16_t, 4>),   exec_median_blur<3>),
    TEST("Median 3x3, 1 channel (F32)",   (test_median_blur<3, float, 1>),     exec_median_blur<3>),
    TEST("Median 3x3, 3 channel (F32)",   (test_median_blur<3, float, 3>),     exec_median_blur<3>),
    TEST("Median 3x3, 4 channel (F32)",   (test_median_blur<3, float, 4>),     exec_median_blur<3>),
    TEST("Median 5x5, 1 channel (U8)",    (test_median_blur<5, uint8_t, 1>),   exec_median_blur<5>),
    TEST("Median 5x5, 3 channel (U8)",    (test_median_blur<5, uint8_t, 3>),   exec_median_blur<5>),
    TEST("Median 5x5, 4 channel (U8)",    (test_median_blur<5, uint8_t, 4>),   exec_median_blur<5>),
    TEST("Median 5x5, 1 channel (U16)",   (test_median_blur<5, uint16_t, 1>),  exec_median_blur<5>),
    TEST("Median 5x5, 3 channel (U16)",   (test_median_blur<5, uint16_t, 3>),  exec_median_blur<5>),
    TEST("Median 5x5, 4 channel (U16)",   (test_median_blur<5, uint16_t, 4>),  exec_median_blur<5>),
    TEST("Median 5x5, 1 channel (S16)",   (test_median_blur<5, int16_t, 1>),   exec_median_blur<5>),
    TEST("Median 5x5, 3 channel (S16)",   (test_median_blur<5, int16_t, 3>),   exec_median_blur<5>),
    TEST("Median 5x5, 4 channel (S16)",   (test_median_blur<5, int16_t, 4>),   exec_median_blur<5>),
    TEST("Median 5x5, 1 channel (F32)",   (test_median_blur<5, float, 1>),     exec_median_blur<5>),
    TEST("Median 5x5, 3 channel (F32)",   (test_median_blur<5, float, 3>),     exec_median_blur<5>),
    TEST("Median 5x5, 4 channel (F32)",   (test_median_blur<5, float, 4>),     exec_median_blur<5>),
    TEST("Median 7x7, 1 channel (U8)",    (test_median_blur<7, uint8_t, 1>),   exec_median_blur<7>),
    TEST("Median 7x7, 3 channel (U8)",    (test_median_blur<7, uint8_t, 3>),   exec_median_blur<7>),
    TEST("Median 7x7, 4 channel (U8)",    (test_median_blur<7, uint8_t, 4>),   exec_median_blur<7>),
    TEST("Median 9x9, 1 channel (U8)",    (test_median_blur<9, uint8_t, 1>),   exec_median_blur<9>),
    TEST("Median 9x9, 3 channel (U8)",    (test_median_blur<9, uint8_t, 3>),   exec_median_blur<9>),
    TEST("Median 9x9, 4 channel (U8)",    (test_median_blur<9, uint8_t, 4>),   exec_median_blur<9>),
    TEST("Median 11x11, 1 channel (U8)",  (test_median_blur<11, uint8_t, 1>),  exec_median_blur<11>),
    TEST("Median 11x11, 3 channel (U8)",  (test_median_blur<11, uint8_t, 3>),  exec_median_blur<11>),
    TEST("Median 11x11, 4 channel (U8)",  (test_median_blur<11, uint8_t, 4>),  exec_median_blur<11>),
    TEST("Median 13x13, 1 channel (U8)",  (test_median_blur<13, uint8_t, 1>),  exec_median_blur<13>),
    TEST("Median 13x13, 3 channel (U8)",  (test_median_blur<13, uint8_t, 3>),  exec_median_blur<13>),
    TEST("Median 13x13, 4 channel (U8)",  (test_median_blur<13, uint8_t, 4>),  exec_median_blur<13>),
    TEST("Median 15x15, 1 channel (U8)",  (test_median_blur<15, uint8_t, 1>),  exec_median_blur<15>),
    TEST("Median 15x15, 3 channel (U8)",  (test_median_blur<15, uint8_t, 3>),  exec_median_blur<15>),
    TEST("Median 15x15, 4 channel (U8)",  (test_median_blur<15, uint8_t, 4>),  exec_median_blur<15>),
    TEST("Median 17x17, 1 channel (U8)",  (test_median_blur<17, uint8_t, 1>),  exec_median_blur<17>),
    TEST("Median 17x17, 3 channel (U8)",  (test_median_blur<17, uint8_t, 3>),  exec_median_blur<17>),
    TEST("Median 17x17, 4 channel (U8)",  (test_median_blur<17, uint8_t, 4>),  exec_median_blur<17>),
    TEST("Median 27x27, 1 channel (U8)",  (test_median_blur<27, uint8_t, 1>),  exec_median_blur<27>),
    TEST("Median 27x27, 3 channel (U8)",  (test_median_blur<27, uint8_t, 3>),  exec_median_blur<27>),
    TEST("Median 27x27, 4 channel (U8)",  (test_median_blur<27, uint8_t, 4>),  exec_median_blur<27>),
    TEST("Median 35x35, 1 channel (U8)",  (test_median_blur<35, uint8_t, 1>),  exec_median_blur<35>),
    TEST("Median 35x35, 3 channel (U8)",  (test_median_blur<35, uint8_t, 3>),  exec_median_blur<35>),
    TEST("Median 35x35, 4 channel (U8)",  (test_median_blur<35, uint8_t, 4>),  exec_median_blur<35>),
    TEST("Median 255x255, 1 channel (U8)",  (test_median_blur<255, uint8_t, 1>),  exec_median_blur<255>),
    TEST("Median 255x255, 3 channel (U8)",  (test_median_blur<255, uint8_t, 3>),  exec_median_blur<255>),
    TEST("Median 255x255, 4 channel (U8)",  (test_median_blur<255, uint8_t, 4>),  exec_median_blur<255>),
  };
  // clang-format on
  return tests;
}
