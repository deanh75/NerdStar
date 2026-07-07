// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "tests.h"

#if MANAGER
extern "C" {

typedef enum {
  KLEIDICV_OK = 0,
  KLEIDICV_ERROR_NOT_IMPLEMENTED,
} kleidicv_error_t;

typedef struct {
  size_t window_width;
  size_t window_height;
  size_t max_level;
  size_t termination_count;
  float termination_epsilon;
  float min_eig_threshold;
  int flags;
} kleidicv_optflow_lk_context_t;

enum { KLEIDICV_USE_INITIAL_FLOW = 1 << 0, KLEIDICV_GET_MIN_EIG = 1 << 1 };

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8(
    const uint8_t* prev_img, size_t prev_img_stride, const uint8_t* next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context);

}  // extern "C"
static cv::Mat to_result_mat(const std::vector<cv::Point2f>& next_points,
                             const std::vector<uint8_t>& status,
                             const std::vector<float>& err) {
  cv::Mat result(0, 1, CV_32FC1);
  for (size_t i = 0; i < next_points.size(); ++i) {
    result.push_back(next_points[i].x);
    result.push_back(next_points[i].y);
    result.push_back(static_cast<float>(status[i]));
    result.push_back(err[i]);
  }
  return result;
}

struct calc_case_t {
  std::string name;
  cv::Matx23f warp;
  int border_mode;
  cv::Scalar border_value;
};

struct calc_fixture_t {
  cv::Mat prev;
  std::vector<cv::Point2f> prev_points;
  std::vector<calc_case_t> calc_cases;
};

static cv::Mat create_synthetic_image(int width, int height, size_t channels) {
  // Build a deterministic scene with both strong corners and weak-texture
  // regions so LK can be tested on valid and difficult points.
  cv::Mat pattern(height, width, CV_8UC1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int checker = (((x / 8) + (y / 8)) & 1) != 0 ? 45 : 0;
      const int diagonal = ((x * 3) + (y * 5)) & 0xFF;
      const int ring = (((x - width / 2) * (x - width / 2)) +
                        ((y - height / 2) * (y - height / 2))) %
                       127;
      pattern.at<uint8_t>(y, x) =
          static_cast<uint8_t>((diagonal + checker + ring) & 0xFF);
    }
  }

  cv::rectangle(pattern, cv::Rect(width / 5, height / 6, width / 6, height / 5),
                cv::Scalar(245), cv::FILLED);
  cv::rectangle(pattern,
                cv::Rect(width / 2 - width / 8, height / 2 - height / 9,
                         width / 4, height / 4),
                cv::Scalar(96), cv::FILLED);
  cv::circle(pattern, cv::Point(width / 2 + width / 6, height / 3),
             std::min(width, height) / 8, cv::Scalar(18), cv::FILLED);
  cv::line(pattern, cv::Point(width / 8, height - height / 5),
           cv::Point(width - width / 7, height / 5), cv::Scalar(210), 2,
           cv::LINE_AA);
  cv::line(pattern, cv::Point(width / 10, height / 3),
           cv::Point(width - width / 8, height / 3), cv::Scalar(35), 1,
           cv::LINE_AA);

  if (channels == 1) {
    return pattern;
  }

  std::vector<cv::Mat> planes;
  planes.reserve(channels);
  for (size_t c = 0; c < channels; ++c) {
    cv::Mat shifted;
    const float sx = static_cast<float>((static_cast<int>(c) % 3) - 1);
    const float sy = static_cast<float>((static_cast<int>(c) % 2) - 1);
    const cv::Matx23f shift(1.0F, 0.0F, sx, 0.0F, 1.0F, sy);
    cv::warpAffine(pattern, shifted, cv::Mat(shift), pattern.size(),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

    cv::Mat channel;
    cv::addWeighted(shifted, 0.8, pattern, 0.2, static_cast<double>(9 * c),
                    channel);
    planes.push_back(channel);
  }

  cv::Mat image;
  cv::merge(planes, image);
  return image;
}

static cv::Point2f apply_warp_to_point(const cv::Matx23f& warp,
                                       const cv::Point2f& point) {
  return cv::Point2f(warp(0, 0) * point.x + warp(0, 1) * point.y + warp(0, 2),
                     warp(1, 0) * point.x + warp(1, 1) * point.y + warp(1, 2));
}

static std::vector<calc_case_t> build_calc_cases(int width, int height) {
  const cv::Point2f center(static_cast<float>(width) * 0.5F,
                           static_cast<float>(height) * 0.5F);
  const cv::Mat rotation =
      cv::getRotationMatrix2D(center, 2.8, 0.985);  // CV_64F 2x3

  std::vector<calc_case_t> cases;
  // Small subpixel shift checks interpolation and convergence behaviour.
  cases.push_back({"subpixel_translate",
                   cv::Matx23f(1.0F, 0.0F, 2.35F, 0.0F, 1.0F, -1.65F),
                   cv::BORDER_REFLECT_101, cv::Scalar(0)});
  // Light rotation+scale checks pyramid refinement across levels.
  cases.push_back({"small_rotation_scale",
                   cv::Matx23f(static_cast<float>(rotation.at<double>(0, 0)),
                               static_cast<float>(rotation.at<double>(0, 1)),
                               static_cast<float>(rotation.at<double>(0, 2)),
                               static_cast<float>(rotation.at<double>(1, 0)),
                               static_cast<float>(rotation.at<double>(1, 1)),
                               static_cast<float>(rotation.at<double>(1, 2))),
                   cv::BORDER_REFLECT_101, cv::Scalar(0)});
  // Larger translation with constant border introduces out-of-frame/occluded
  // tracks and exercises status handling.
  cases.push_back({"large_shift_occlusion",
                   cv::Matx23f(1.0F, 0.0F, 13.4F, 0.0F, 1.0F, -10.2F),
                   cv::BORDER_CONSTANT, cv::Scalar(0)});
  return cases;
}

static std::vector<cv::Point2f> build_test_points(const cv::Mat& image) {
  cv::Mat gray;
  cv::extractChannel(image, gray, 0);

  std::vector<cv::Point2f> points;
  // Seed with Shi-Tomasi corners (trackable points with strong gradients).
  cv::goodFeaturesToTrack(gray, points, 96, 0.01, 7.0, cv::noArray(), 5);

  const int width = image.cols;
  const int height = image.rows;
  const int flat_x0 = width / 2 - width / 8;
  const int flat_y0 = height / 2 - height / 9;
  const int flat_x1 = flat_x0 + width / 4;
  const int flat_y1 = flat_y0 + height / 4;
  // Add points from a flatter patch to exercise weak-feature rejection paths.
  for (int y = flat_y0 + 6; y <= flat_y1 - 6; y += 10) {
    for (int x = flat_x0 + 6; x <= flat_x1 - 6; x += 10) {
      points.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
  }

  // Add near-border points to stress boundary checks and status updates.
  points.emplace_back(2.0F, 2.0F);
  points.emplace_back(static_cast<float>(width - 3), 2.0F);
  points.emplace_back(2.0F, static_cast<float>(height - 3));
  points.emplace_back(static_cast<float>(width - 3),
                      static_cast<float>(height - 3));
  points.emplace_back(static_cast<float>(width - 9),
                      static_cast<float>(height / 2));
  points.emplace_back(static_cast<float>(width - 7),
                      static_cast<float>(height / 3));
  points.emplace_back(static_cast<float>(width / 2), 4.0F);
  points.emplace_back(static_cast<float>(width / 2),
                      static_cast<float>(height - 5));
  return points;
}

static std::vector<cv::Point2f> build_initial_next_points(
    const std::vector<cv::Point2f>& prev_points, const cv::Matx23f& warp,
    bool use_initial_flow) {
  std::vector<cv::Point2f> initial(prev_points.size(), cv::Point2f(0.0F, 0.0F));
  if (!use_initial_flow) {
    return initial;
  }

  // Use warped coordinates plus deterministic perturbation as initial guesses.
  // This specifically validates KLEIDICV_USE_INITIAL_FLOW behaviour.
  for (size_t i = 0; i < prev_points.size(); ++i) {
    const cv::Point2f warped = apply_warp_to_point(warp, prev_points[i]);
    const float dx = static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.13F;
    const float dy = static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.11F;
    initial[i] = warped + cv::Point2f(dx, dy);
  }
  return initial;
}

static const calc_fixture_t& get_calc_fixture(size_t channels) {
  static const std::vector<calc_fixture_t> fixtures = []() {
    constexpr int kWidth = 320;
    constexpr int kHeight = 240;

    std::vector<calc_fixture_t> result;
    result.reserve(4);
    for (size_t fixture_channels = 1; fixture_channels <= 4;
         ++fixture_channels) {
      cv::Mat prev = create_synthetic_image(kWidth, kHeight, fixture_channels);
      result.push_back(
          {prev, build_test_points(prev), build_calc_cases(kWidth, kHeight)});
    }
    return result;
  }();

  return fixtures.at(channels - 1);
}

static bool run_pyr_lk_calc_case(cv::Size win_size, size_t max_level,
                                 size_t termination_count,
                                 float min_eig_threshold, size_t channels,
                                 int flags) {
  const calc_fixture_t& fixture = get_calc_fixture(channels);
  const cv::Mat& prev = fixture.prev;
  const std::vector<cv::Point2f>& prev_points = fixture.prev_points;
  const std::vector<calc_case_t>& calc_cases = fixture.calc_cases;
  const int width = prev.cols;
  const int height = prev.rows;

  int cv_flags = 0;
  if ((flags & KLEIDICV_USE_INITIAL_FLOW) != 0) {
    cv_flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
  }
  if ((flags & KLEIDICV_GET_MIN_EIG) != 0) {
    cv_flags |= cv::OPTFLOW_LK_GET_MIN_EIGENVALS;
  }

  const cv::TermCriteria criteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
      static_cast<int>(termination_count), 0.01);
  kleidicv_optflow_lk_context_t context{};
  context.window_width = static_cast<size_t>(win_size.width);
  context.window_height = static_cast<size_t>(win_size.height);
  context.max_level = max_level;
  context.termination_count = termination_count;
  context.termination_epsilon = 0.01F;
  context.min_eig_threshold = min_eig_threshold;
  context.flags = flags;

  for (const calc_case_t& calc_case : calc_cases) {
    cv::Mat next;
    cv::warpAffine(prev, next, cv::Mat(calc_case.warp), cv::Size(width, height),
                   cv::INTER_LINEAR, calc_case.border_mode,
                   calc_case.border_value);

    const std::vector<cv::Point2f> next_initial = build_initial_next_points(
        prev_points, calc_case.warp, (flags & KLEIDICV_USE_INITIAL_FLOW) != 0);

    std::vector<cv::Point2f> expected_next = next_initial;
    std::vector<uint8_t> expected_status;
    std::vector<float> expected_err;
    // OpenCV is the reference for each synthetic motion case.
    cv::calcOpticalFlowPyrLK(prev, next, prev_points, expected_next,
                             expected_status, expected_err, win_size,
                             static_cast<int>(max_level), criteria, cv_flags,
                             static_cast<double>(min_eig_threshold));

    cv::Mat expected_mat =
        to_result_mat(expected_next, expected_status, expected_err);

    const float matrix_tolerance =
        (flags & KLEIDICV_GET_MIN_EIG) != 0 ? 0.0025F : 0.20F;

    auto run_and_check = [&](const char* path,
                             const std::vector<cv::Point2f>& initial_next,
                             auto call_fn) -> bool {
      std::vector<cv::Point2f> actual_next = initial_next;
      std::vector<uint8_t> actual_status(prev_points.size(), 0);
      std::vector<float> actual_err(prev_points.size(), 0.0F);

      kleidicv_error_t err = call_fn(actual_next, actual_status, actual_err);
      if (err != KLEIDICV_OK) {
        std::cout << path << " failed: " << err << " (case=" << calc_case.name
                  << ", channels=" << channels << ", max_level=" << max_level
                  << ", flags=" << flags << ", window=" << win_size.width << "x"
                  << win_size.height
                  << ", termination_count=" << termination_count
                  << ", min_eig_threshold=" << min_eig_threshold << ")"
                  << std::endl;
        return true;
      }

      cv::Mat actual_mat =
          to_result_mat(actual_next, actual_status, actual_err);
      if (are_float_matrices_different<float>(matrix_tolerance, actual_mat,
                                              expected_mat)) {
        std::cout << "Result mismatch (case=" << calc_case.name
                  << ", path=" << path << ", channels=" << channels
                  << ", max_level=" << max_level << ", flags=" << flags
                  << ", window=" << win_size.width << "x" << win_size.height
                  << ", termination_count=" << termination_count
                  << ", min_eig_threshold=" << min_eig_threshold << ")"
                  << std::endl;
        cv::Mat input_mat;
        cv::vconcat(prev, next, input_mat);
        fail_print_matrices(width, height, input_mat, actual_mat, expected_mat);
        return true;
      }
      return false;
    };

    if (run_and_check("raw", next_initial,
                      [&](std::vector<cv::Point2f>& actual_next,
                          std::vector<uint8_t>& actual_status,
                          std::vector<float>& actual_err) {
                        return kleidicv_optical_flow_pyr_lk_u8(
                            prev.ptr<uint8_t>(), prev.step, next.ptr<uint8_t>(),
                            next.step, static_cast<size_t>(width),
                            static_cast<size_t>(height), channels,
                            reinterpret_cast<const float*>(prev_points.data()),
                            reinterpret_cast<float*>(actual_next.data()),
                            prev_points.size(), actual_status.data(),
                            actual_err.data(), context);
                      })) {
      return true;
    }
  }

  return false;
}

template <size_t WindowSize, size_t MaxLevel, size_t TerminationCount,
          size_t Channels, int Flags>
bool test_calc_optical_flow_pyr_lk() {
  constexpr float kMinEigThreshold = 1e-4F;
  return run_pyr_lk_calc_case(cv::Size{WindowSize, WindowSize}, MaxLevel,
                              TerminationCount, kMinEigThreshold, Channels,
                              Flags);
}

std::vector<manager_only_test>& calc_optical_flow_pyr_lk_tests_get() {
  // clang-format off
  #define CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(win, lvl, term, flags, flag_text)                    \
    MANAGER_ONLY_TEST("Optical Flow PyrLK Calc window " #win "x" #win ", level " #lvl ", termination " #term            \
                      ", c1, " flag_text, (test_calc_optical_flow_pyr_lk<win, lvl, term, 1, flags>)),     \
    MANAGER_ONLY_TEST("Optical Flow PyrLK Calc window " #win "x" #win ", level " #lvl ", termination " #term            \
                      ", c2, " flag_text, (test_calc_optical_flow_pyr_lk<win, lvl, term, 2, flags>)),     \
    MANAGER_ONLY_TEST("Optical Flow PyrLK Calc window " #win "x" #win ", level " #lvl ", termination " #term            \
                      ", c3, " flag_text, (test_calc_optical_flow_pyr_lk<win, lvl, term, 3, flags>)),     \
    MANAGER_ONLY_TEST("Optical Flow PyrLK Calc window " #win "x" #win ", level " #lvl ", termination " #term            \
                      ", c4, " flag_text, (test_calc_optical_flow_pyr_lk<win, lvl, term, 4, flags>))

  static std::vector<manager_only_test> tests = {
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 0, 20, 0, "no flags"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 1, 20, 0, "no flags"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 2, 20, 0, "no flags"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 3, 20, 0, "no flags"),

    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 1, 20, KLEIDICV_USE_INITIAL_FLOW, "initial flow"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 2, 20, KLEIDICV_USE_INITIAL_FLOW, "initial flow"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 3, 20, KLEIDICV_USE_INITIAL_FLOW, "initial flow"),

    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 1, 20, KLEIDICV_GET_MIN_EIG, "min eig"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 2, 20, KLEIDICV_GET_MIN_EIG, "min eig"),
    CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS(9, 3, 20, KLEIDICV_GET_MIN_EIG, "min eig")
  };
  #undef CALC_OPTICAL_FLOW_PYR_LK_TESTS_FOR_CHANNELS
  // clang-format on
  return tests;
}

#endif
