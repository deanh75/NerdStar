// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "tests.h"

template <size_t KernelSize, size_t BorderType>
cv::Mat exec_gaussian_blur(cv::Mat& input) {
  double sigma =
      *reinterpret_cast<double*>(&input.at<uint8_t>(input.rows - 2, 0));
  cv::Mat input_mat = input.rowRange(0, input.rows - 2);
  cv::Size kernel(KernelSize, KernelSize);
  cv::Mat result;
  // cv::BORDER_ISOLATED` is needed to be OR-ed into `borderType`, as otherwise
  // submatrix inputs are not supported
  cv::GaussianBlur(input_mat, result, kernel, sigma, sigma,
                   BorderType | cv::BORDER_ISOLATED, cv::ALGO_HINT_APPROX);
  return result;
}

#if MANAGER
template <size_t KernelSize, size_t BorderType, size_t Channels,
          bool Binomial = true>
bool test_gaussian_blur(int index, RecreatedMessageQueue& request_queue,
                        RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  size_t size_min = std::max<size_t>(4, KernelSize - 1);
  size_t size_max = std::max<size_t>(16, 2 * KernelSize + 2);

  for (size_t y = size_min; y <= size_max; ++y) {
    for (size_t x = size_min; x <= size_max; ++x) {
      // Two extra lines allocated to be sure sigma can be placed next to the
      // real input
      cv::Mat input(y + 2, x, CV_8UC(Channels));
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);

      double sigma = 0.0;

      if constexpr (!Binomial) {
        // cv::rng returns [0,1) range in case of float or double, so it is
        // multiplied by 10
        sigma = static_cast<double>(rng) * 10;
      }

      // sigma is embedded into the input matrix
      *reinterpret_cast<double*>(&input.at<uint8_t>(input.rows - 2, 0)) = sigma;

      cv::Mat actual = exec_gaussian_blur<KernelSize, BorderType>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      uint8_t threshold = 0;
      // Bit-exact operation is only guaranteed for kernels up to 9x9 when the
      // binomial variant is used. For bigger kernels, and for all the
      // CustomSigma variants, a small difference is allowed.
      if constexpr (KernelSize > 9 || !Binomial) {
        threshold = 1;
      }

      if (are_matrices_different<uint8_t>(threshold, actual, expected)) {
        fail_print_matrices(y, x, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& gaussian_blur_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 1 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 1>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 2 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 2>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 3 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 3>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 4 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 4>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 1 channel, random sigma", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 1, false>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 2 channel, random sigma", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 2, false>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 3 channel, random sigma", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 3, false>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT_101, 4 channel, random sigma", (test_gaussian_blur<3, cv::BORDER_REFLECT_101, 4, false>), (exec_gaussian_blur<3, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 3x3, BORDER_REFLECT, 1 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT, 1>), (exec_gaussian_blur<3, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT, 2 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT, 2>), (exec_gaussian_blur<3, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT, 3 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT, 3>), (exec_gaussian_blur<3, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 3x3, BORDER_REFLECT, 4 channel", (test_gaussian_blur<3, cv::BORDER_REFLECT, 4>), (exec_gaussian_blur<3, cv::BORDER_REFLECT>)),

    TEST("Gaussian blur 3x3, BORDER_WRAP, 1 channel", (test_gaussian_blur<3, cv::BORDER_WRAP, 1>), (exec_gaussian_blur<3, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 3x3, BORDER_WRAP, 2 channel", (test_gaussian_blur<3, cv::BORDER_WRAP, 2>), (exec_gaussian_blur<3, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 3x3, BORDER_WRAP, 3 channel", (test_gaussian_blur<3, cv::BORDER_WRAP, 3>), (exec_gaussian_blur<3, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 3x3, BORDER_WRAP, 4 channel", (test_gaussian_blur<3, cv::BORDER_WRAP, 4>), (exec_gaussian_blur<3, cv::BORDER_WRAP>)),

    TEST("Gaussian blur 3x3, BORDER_REPLICATE, 1 channel", (test_gaussian_blur<3, cv::BORDER_REPLICATE, 1>), (exec_gaussian_blur<3, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 3x3, BORDER_REPLICATE, 2 channel", (test_gaussian_blur<3, cv::BORDER_REPLICATE, 2>), (exec_gaussian_blur<3, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 3x3, BORDER_REPLICATE, 3 channel", (test_gaussian_blur<3, cv::BORDER_REPLICATE, 3>), (exec_gaussian_blur<3, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 3x3, BORDER_REPLICATE, 4 channel", (test_gaussian_blur<3, cv::BORDER_REPLICATE, 4>), (exec_gaussian_blur<3, cv::BORDER_REPLICATE>)),

    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 1 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 1>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 2 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 2>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 3 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 3>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 4 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 4>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 1 channel, random sigma", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 1, false>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 2 channel, random sigma", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 2, false>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 3 channel, random sigma", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 3, false>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT_101, 4 channel, random sigma", (test_gaussian_blur<5, cv::BORDER_REFLECT_101, 4, false>), (exec_gaussian_blur<5, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 5x5, BORDER_REFLECT, 1 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT, 1>), (exec_gaussian_blur<5, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT, 2 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT, 2>), (exec_gaussian_blur<5, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT, 3 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT, 3>), (exec_gaussian_blur<5, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 5x5, BORDER_REFLECT, 4 channel", (test_gaussian_blur<5, cv::BORDER_REFLECT, 4>), (exec_gaussian_blur<5, cv::BORDER_REFLECT>)),

    TEST("Gaussian blur 5x5, BORDER_WRAP, 1 channel", (test_gaussian_blur<5, cv::BORDER_WRAP, 1>), (exec_gaussian_blur<5, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 5x5, BORDER_WRAP, 2 channel", (test_gaussian_blur<5, cv::BORDER_WRAP, 2>), (exec_gaussian_blur<5, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 5x5, BORDER_WRAP, 3 channel", (test_gaussian_blur<5, cv::BORDER_WRAP, 3>), (exec_gaussian_blur<5, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 5x5, BORDER_WRAP, 4 channel", (test_gaussian_blur<5, cv::BORDER_WRAP, 4>), (exec_gaussian_blur<5, cv::BORDER_WRAP>)),

    TEST("Gaussian blur 5x5, BORDER_REPLICATE, 1 channel", (test_gaussian_blur<5, cv::BORDER_REPLICATE, 1>), (exec_gaussian_blur<5, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 5x5, BORDER_REPLICATE, 2 channel", (test_gaussian_blur<5, cv::BORDER_REPLICATE, 2>), (exec_gaussian_blur<5, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 5x5, BORDER_REPLICATE, 3 channel", (test_gaussian_blur<5, cv::BORDER_REPLICATE, 3>), (exec_gaussian_blur<5, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 5x5, BORDER_REPLICATE, 4 channel", (test_gaussian_blur<5, cv::BORDER_REPLICATE, 4>), (exec_gaussian_blur<5, cv::BORDER_REPLICATE>)),

    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 1 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 1>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 2 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 2>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 3 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 3>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 4 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 4>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 1 channel, random sigma", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 1, false>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 2 channel, random sigma", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 2, false>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 3 channel, random sigma", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 3, false>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT_101, 4 channel, random sigma", (test_gaussian_blur<7, cv::BORDER_REFLECT_101, 4, false>), (exec_gaussian_blur<7, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 7x7, BORDER_REFLECT, 1 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT, 1>), (exec_gaussian_blur<7, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT, 2 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT, 2>), (exec_gaussian_blur<7, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT, 3 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT, 3>), (exec_gaussian_blur<7, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 7x7, BORDER_REFLECT, 4 channel", (test_gaussian_blur<7, cv::BORDER_REFLECT, 4>), (exec_gaussian_blur<7, cv::BORDER_REFLECT>)),

    TEST("Gaussian blur 7x7, BORDER_WRAP, 1 channel", (test_gaussian_blur<7, cv::BORDER_WRAP, 1>), (exec_gaussian_blur<7, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 7x7, BORDER_WRAP, 2 channel", (test_gaussian_blur<7, cv::BORDER_WRAP, 2>), (exec_gaussian_blur<7, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 7x7, BORDER_WRAP, 3 channel", (test_gaussian_blur<7, cv::BORDER_WRAP, 3>), (exec_gaussian_blur<7, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 7x7, BORDER_WRAP, 4 channel", (test_gaussian_blur<7, cv::BORDER_WRAP, 4>), (exec_gaussian_blur<7, cv::BORDER_WRAP>)),

    TEST("Gaussian blur 7x7, BORDER_REPLICATE, 1 channel", (test_gaussian_blur<7, cv::BORDER_REPLICATE, 1>), (exec_gaussian_blur<7, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 7x7, BORDER_REPLICATE, 2 channel", (test_gaussian_blur<7, cv::BORDER_REPLICATE, 2>), (exec_gaussian_blur<7, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 7x7, BORDER_REPLICATE, 3 channel", (test_gaussian_blur<7, cv::BORDER_REPLICATE, 3>), (exec_gaussian_blur<7, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 7x7, BORDER_REPLICATE, 4 channel", (test_gaussian_blur<7, cv::BORDER_REPLICATE, 4>), (exec_gaussian_blur<7, cv::BORDER_REPLICATE>)),

    TEST("Gaussian blur 9x9, BORDER_REFLECT_101, 1 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT_101, 1>), (exec_gaussian_blur<9, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 9x9, BORDER_REFLECT_101, 2 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT_101, 2>), (exec_gaussian_blur<9, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 9x9, BORDER_REFLECT_101, 3 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT_101, 3>), (exec_gaussian_blur<9, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 9x9, BORDER_REFLECT_101, 4 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT_101, 4>), (exec_gaussian_blur<9, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 9x9, BORDER_REFLECT, 1 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT, 1>), (exec_gaussian_blur<9, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 9x9, BORDER_REFLECT, 2 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT, 2>), (exec_gaussian_blur<9, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 9x9, BORDER_REFLECT, 3 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT, 3>), (exec_gaussian_blur<9, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 9x9, BORDER_REFLECT, 4 channel", (test_gaussian_blur<9, cv::BORDER_REFLECT, 4>), (exec_gaussian_blur<9, cv::BORDER_REFLECT>)),

    TEST("Gaussian blur 9x9, BORDER_WRAP, 1 channel", (test_gaussian_blur<9, cv::BORDER_WRAP, 1>), (exec_gaussian_blur<9, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 9x9, BORDER_WRAP, 2 channel", (test_gaussian_blur<9, cv::BORDER_WRAP, 2>), (exec_gaussian_blur<9, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 9x9, BORDER_WRAP, 3 channel", (test_gaussian_blur<9, cv::BORDER_WRAP, 3>), (exec_gaussian_blur<9, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 9x9, BORDER_WRAP, 4 channel", (test_gaussian_blur<9, cv::BORDER_WRAP, 4>), (exec_gaussian_blur<9, cv::BORDER_WRAP>)),

    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 1 channel", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 1>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 2 channel", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 2>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 3 channel", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 3>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 4 channel", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 4>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),

    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 1 channel, random sigma", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 1, false>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 2 channel, random sigma", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 2, false>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 3 channel, random sigma", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 3, false>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 9x9, BORDER_REPLICATE, 4 channel, random sigma", (test_gaussian_blur<9, cv::BORDER_REPLICATE, 4, false>), (exec_gaussian_blur<9, cv::BORDER_REPLICATE>)),

    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 1 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 1>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 2 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 2>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 3 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 3>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 4 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 4>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 1 channel, random sigma", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 1, false>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 2 channel, random sigma", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 2, false>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 3 channel, random sigma", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 3, false>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT_101, 4 channel, random sigma", (test_gaussian_blur<15, cv::BORDER_REFLECT_101, 4, false>), (exec_gaussian_blur<15, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 15x15, BORDER_REFLECT, 1 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT, 1>), (exec_gaussian_blur<15, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT, 2 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT, 2>), (exec_gaussian_blur<15, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT, 3 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT, 3>), (exec_gaussian_blur<15, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 15x15, BORDER_REFLECT, 4 channel", (test_gaussian_blur<15, cv::BORDER_REFLECT, 4>), (exec_gaussian_blur<15, cv::BORDER_REFLECT>)),

    TEST("Gaussian blur 15x15, BORDER_WRAP, 1 channel", (test_gaussian_blur<15, cv::BORDER_WRAP, 1>), (exec_gaussian_blur<15, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 15x15, BORDER_WRAP, 2 channel", (test_gaussian_blur<15, cv::BORDER_WRAP, 2>), (exec_gaussian_blur<15, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 15x15, BORDER_WRAP, 3 channel", (test_gaussian_blur<15, cv::BORDER_WRAP, 3>), (exec_gaussian_blur<15, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 15x15, BORDER_WRAP, 4 channel", (test_gaussian_blur<15, cv::BORDER_WRAP, 4>), (exec_gaussian_blur<15, cv::BORDER_WRAP>)),

    TEST("Gaussian blur 15x15, BORDER_REPLICATE, 1 channel", (test_gaussian_blur<15, cv::BORDER_REPLICATE, 1>), (exec_gaussian_blur<15, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 15x15, BORDER_REPLICATE, 2 channel", (test_gaussian_blur<15, cv::BORDER_REPLICATE, 2>), (exec_gaussian_blur<15, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 15x15, BORDER_REPLICATE, 3 channel", (test_gaussian_blur<15, cv::BORDER_REPLICATE, 3>), (exec_gaussian_blur<15, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 15x15, BORDER_REPLICATE, 4 channel", (test_gaussian_blur<15, cv::BORDER_REPLICATE, 4>), (exec_gaussian_blur<15, cv::BORDER_REPLICATE>)),

    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 1 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 1>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 2 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 2>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 3 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 3>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 4 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 4>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 1 channel, random sigma", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 1, false>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 2 channel, random sigma", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 2, false>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 3 channel, random sigma", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 3, false>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT_101, 4 channel, random sigma", (test_gaussian_blur<21, cv::BORDER_REFLECT_101, 4, false>), (exec_gaussian_blur<21, cv::BORDER_REFLECT_101>)),

    TEST("Gaussian blur 21x21, BORDER_REFLECT, 1 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT, 1>), (exec_gaussian_blur<21, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT, 2 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT, 2>), (exec_gaussian_blur<21, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT, 3 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT, 3>), (exec_gaussian_blur<21, cv::BORDER_REFLECT>)),
    TEST("Gaussian blur 21x21, BORDER_REFLECT, 4 channel", (test_gaussian_blur<21, cv::BORDER_REFLECT, 4>), (exec_gaussian_blur<21, cv::BORDER_REFLECT>)),

    TEST("Gaussian blur 21x21, BORDER_WRAP, 1 channel", (test_gaussian_blur<21, cv::BORDER_WRAP, 1>), (exec_gaussian_blur<21, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 21x21, BORDER_WRAP, 2 channel", (test_gaussian_blur<21, cv::BORDER_WRAP, 2>), (exec_gaussian_blur<21, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 21x21, BORDER_WRAP, 3 channel", (test_gaussian_blur<21, cv::BORDER_WRAP, 3>), (exec_gaussian_blur<21, cv::BORDER_WRAP>)),
    TEST("Gaussian blur 21x21, BORDER_WRAP, 4 channel", (test_gaussian_blur<21, cv::BORDER_WRAP, 4>), (exec_gaussian_blur<21, cv::BORDER_WRAP>)),

    TEST("Gaussian blur 21x21, BORDER_REPLICATE, 1 channel", (test_gaussian_blur<21, cv::BORDER_REPLICATE, 1>), (exec_gaussian_blur<21, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 21x21, BORDER_REPLICATE, 2 channel", (test_gaussian_blur<21, cv::BORDER_REPLICATE, 2>), (exec_gaussian_blur<21, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 21x21, BORDER_REPLICATE, 3 channel", (test_gaussian_blur<21, cv::BORDER_REPLICATE, 3>), (exec_gaussian_blur<21, cv::BORDER_REPLICATE>)),
    TEST("Gaussian blur 21x21, BORDER_REPLICATE, 4 channel", (test_gaussian_blur<21, cv::BORDER_REPLICATE, 4>), (exec_gaussian_blur<21, cv::BORDER_REPLICATE>)),
  };
  // clang-format on
  return tests;
}
