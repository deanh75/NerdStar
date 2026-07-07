// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "opencv2/video/tracking.hpp"
#include "tests.h"

#if MANAGER
#include <cstddef>

struct BuildOpticalFlowCaseContext {
  const char* geometry_name;
  const char* layout_name;
  size_t width;
  size_t height;
  size_t channels;
  size_t max_level;
  size_t src_stride;
  bool is_contiguous;
  bool is_submatrix;
  cv::Size win_size;
  cv::Size whole_size;
  cv::Point roi_offset;
};

static void print_case_context(std::ostream& out,
                               const BuildOpticalFlowCaseContext& context) {
  out << " (geometry=" << context.geometry_name
      << ", layout=" << context.layout_name
      << ", window=" << context.win_size.width << "x" << context.win_size.height
      << ", image=" << context.width << "x" << context.height
      << ", channels=" << context.channels
      << ", max_level=" << context.max_level
      << ", src_stride=" << context.src_stride
      << ", contiguous=" << (context.is_contiguous ? "true" : "false")
      << ", submatrix=" << (context.is_submatrix ? "true" : "false")
      << ", roi_offset=" << context.roi_offset.x << "," << context.roi_offset.y
      << ", whole_size=" << context.whole_size.width << "x"
      << context.whole_size.height << ")";
}

extern "C" {

typedef enum {
  KLEIDICV_OK = 0,
} kleidicv_error_t;

typedef struct kleidicv_optical_flow_pyr_lk_pyramid_t_
    kleidicv_optical_flow_pyr_lk_pyramid_t;

kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height);

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_release(
    kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid);

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t* level_count);

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t level,
    const uint8_t** data, size_t* stride, size_t* width, size_t* height);

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t level,
    const int16_t** data, size_t* stride, size_t* width, size_t* height);

}  // extern "C"

static bool compare_pyramid_against_expected(
    kleidicv_optical_flow_pyr_lk_pyramid_t* actual_pyramid,
    const std::vector<cv::Mat>& expected_pyramid, const cv::Mat& input,
    const BuildOpticalFlowCaseContext& context) {
  size_t actual_levels = 0;
  kleidicv_error_t err = kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
      actual_pyramid, &actual_levels);
  if (err != KLEIDICV_OK) {
    std::cout << "kleidicv_optical_flow_pyr_lk_pyramid_get_level_count "
                 "failed: "
              << err;
    print_case_context(std::cout, context);
    std::cout << std::endl;
    return true;
  }

  const size_t expected_levels = expected_pyramid.size() / 2;
  if (actual_levels != expected_levels) {
    std::cout << "Level count mismatch: actual=" << actual_levels
              << " expected=" << expected_levels;
    print_case_context(std::cout, context);
    std::cout << std::endl;
    return true;
  }

  for (size_t level = 0; level < actual_levels; ++level) {
    const uint8_t* image_data = nullptr;
    size_t image_stride = 0;
    size_t image_width = 0;
    size_t image_height = 0;
    err = kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
        actual_pyramid, level, &image_data, &image_stride, &image_width,
        &image_height);
    if (err != KLEIDICV_OK) {
      std::cout << "get_image_level failed on level " << level << ": " << err;
      print_case_context(std::cout, context);
      std::cout << std::endl;
      return true;
    }

    const int image_type =
        CV_MAKETYPE(CV_8U, static_cast<int>(context.channels));
    cv::Mat actual_image(static_cast<int>(image_height),
                         static_cast<int>(image_width), image_type,
                         const_cast<uint8_t*>(image_data), image_stride);
    cv::Mat expected_image = expected_pyramid[level * 2];
    if (are_matrices_different<uint8_t>(0, actual_image, expected_image)) {
      std::cout << "Image pyramid mismatch at level " << level
                << " (image_stride=" << image_stride
                << ", expected_stride=" << expected_image.step << ")";
      print_case_context(std::cout, context);
      std::cout << std::endl;
      fail_print_matrices(context.height, context.width,
                          const_cast<cv::Mat&>(input), actual_image,
                          expected_image);
      return true;
    }

    const int16_t* scharr_data = nullptr;
    size_t scharr_stride = 0;
    size_t scharr_width = 0;
    size_t scharr_height = 0;
    err = kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
        actual_pyramid, level, &scharr_data, &scharr_stride, &scharr_width,
        &scharr_height);
    if (err != KLEIDICV_OK) {
      std::cout << "get_scharr_level failed on level " << level << ": " << err;
      print_case_context(std::cout, context);
      std::cout << std::endl;
      return true;
    }

    const int scharr_type =
        CV_MAKETYPE(CV_16S, static_cast<int>(context.channels * 2));
    cv::Mat actual_scharr(static_cast<int>(scharr_height),
                          static_cast<int>(scharr_width), scharr_type,
                          const_cast<int16_t*>(scharr_data), scharr_stride);
    cv::Mat expected_scharr = expected_pyramid[level * 2 + 1];
    if (are_matrices_different<int16_t>(0, actual_scharr, expected_scharr)) {
      std::cout << "Scharr pyramid mismatch at level " << level
                << " (scharr_stride=" << scharr_stride
                << ", expected_stride=" << expected_scharr.step << ")";
      print_case_context(std::cout, context);
      std::cout << std::endl;
      fail_print_matrices(context.height, context.width,
                          const_cast<cv::Mat&>(input), actual_scharr,
                          expected_scharr);
      return true;
    }
  }

  return false;
}

static bool run_build_optical_flow_case(cv::Size win_size, size_t max_level,
                                        size_t channels) {
  cv::RNG rng(0);

  struct GeometryCase {
    const char* name;
    size_t width;
    size_t height;
  };

  std::vector<GeometryCase> geometry_cases;
  const auto add_geometry_case = [&](const char* name, size_t width,
                                     size_t height) {
    for (const auto& geometry_case : geometry_cases) {
      if (geometry_case.width == width && geometry_case.height == height) {
        return;
      }
    }
    geometry_cases.push_back(GeometryCase{name, width, height});
  };

  for (size_t height = 16; height <= 64; height += 8) {
    for (size_t width = 16; width <= 64; width += 8) {
      add_geometry_case("grid_even", width, height);
    }
  }

  add_geometry_case("fixed_5x5", 5, 5);
  add_geometry_case("fixed_5x6", 5, 6);
  add_geometry_case("fixed_6x5", 6, 5);
  add_geometry_case("fixed_17x16", 17, 16);
  add_geometry_case("fixed_16x17", 16, 17);

  const size_t win_width = static_cast<size_t>(win_size.width);
  const size_t win_height = static_cast<size_t>(win_size.height);
  const size_t below_width = win_width > 1 ? win_width - 1 : 1;
  const size_t below_height = win_height > 1 ? win_height - 1 : 1;
  const size_t above_width = win_width + 1;
  const size_t above_height = win_height + 1;

  add_geometry_case("near_window_minus1_minus1", below_width, below_height);
  add_geometry_case("near_window_minus1_exact", below_width, win_height);
  add_geometry_case("near_window_exact_minus1", win_width, below_height);
  add_geometry_case("near_window_exact_exact", win_width, win_height);
  add_geometry_case("near_window_plus1_exact", above_width, win_height);
  add_geometry_case("near_window_exact_plus1", win_width, above_height);
  add_geometry_case("near_window_plus1_plus1", above_width, above_height);
  add_geometry_case("near_window_minus1_plus1", below_width, above_height);
  add_geometry_case("near_window_plus1_minus1", above_width, below_height);

  const auto run_single_input_case =
      [&](const cv::Mat& input, const char* geometry_name,
          const char* layout_name, size_t width, size_t height) -> bool {
    cv::Size whole_size;
    cv::Point roi_offset;
    input.locateROI(whole_size, roi_offset);
    const BuildOpticalFlowCaseContext context = {
        geometry_name,
        layout_name,
        width,
        height,
        channels,
        max_level,
        static_cast<size_t>(input.step),
        input.isContinuous(),
        input.isSubmatrix(),
        win_size,
        whole_size,
        roi_offset,
    };

    cv::Mat expected_input = input.clone();
    std::vector<cv::Mat> expected_pyramid;
    const int expected_last_level =
        cv::buildOpticalFlowPyramid(expected_input, expected_pyramid, win_size,
                                    static_cast<int>(max_level), true);

    kleidicv_optical_flow_pyr_lk_pyramid_t* actual_pyramid = nullptr;
    kleidicv_error_t err = kleidicv_build_optical_flow_pyr_lk_pyramid(
        &actual_pyramid, input.ptr<uint8_t>(0), input.step,
        static_cast<size_t>(input.cols), static_cast<size_t>(input.rows),
        channels, max_level + 1, static_cast<size_t>(win_size.width),
        static_cast<size_t>(win_size.height));
    if (err != KLEIDICV_OK) {
      std::cout << "kleidicv_build_optical_flow_pyr_lk_pyramid failed: " << err;
      print_case_context(std::cout, context);
      std::cout << std::endl;
      return true;
    }

    const size_t expected_levels = static_cast<size_t>(expected_last_level + 1);
    expected_pyramid.resize(expected_levels * 2);
    if (compare_pyramid_against_expected(actual_pyramid, expected_pyramid,
                                         input, context)) {
      (void)kleidicv_optical_flow_pyr_lk_pyramid_release(actual_pyramid);
      return true;
    }

    err = kleidicv_optical_flow_pyr_lk_pyramid_release(actual_pyramid);
    if (err != KLEIDICV_OK) {
      std::cout << "release failed: " << err;
      print_case_context(std::cout, context);
      std::cout << std::endl;
      return true;
    }

    return false;
  };

  for (const auto& geometry_case : geometry_cases) {
    const size_t width = geometry_case.width;
    const size_t height = geometry_case.height;
    const int image_type = CV_MAKETYPE(CV_8U, static_cast<int>(channels));
    cv::Mat contiguous_input(static_cast<int>(height), static_cast<int>(width),
                             image_type);
    rng.fill(contiguous_input, cv::RNG::UNIFORM, 0, 255);
    if (run_single_input_case(contiguous_input, geometry_case.name,
                              "contiguous", width, height)) {
      return true;
    }

    cv::Mat padded_storage(static_cast<int>(height + 2),
                           static_cast<int>(width + 3), image_type);
    cv::Mat roi_input = padded_storage(
        cv::Rect{1, 1, static_cast<int>(width), static_cast<int>(height)});
    rng.fill(roi_input, cv::RNG::UNIFORM, 0, 255);
    if (run_single_input_case(roi_input, geometry_case.name, "roi_padded",
                              width, height)) {
      return true;
    }
  }

  return false;
}

#define DEFINE_BUILD_OPTICAL_FLOW_TEST(win_w, win_h, lvl, ch)                \
  bool                                                                       \
  test_build_optical_flow_lk_pyramid_w##win_w##x##win_h##_l##lvl##_c##ch() { \
    return run_build_optical_flow_case(cv::Size{win_w, win_h}, lvl, ch);     \
  }

#define DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(win_w, win_h, lvl) \
  DEFINE_BUILD_OPTICAL_FLOW_TEST(win_w, win_h, lvl, 1)                 \
  DEFINE_BUILD_OPTICAL_FLOW_TEST(win_w, win_h, lvl, 2)                 \
  DEFINE_BUILD_OPTICAL_FLOW_TEST(win_w, win_h, lvl, 3)                 \
  DEFINE_BUILD_OPTICAL_FLOW_TEST(win_w, win_h, lvl, 4)

DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(3, 3, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(3, 3, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(3, 3, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(3, 3, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(3, 3, 4)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(5, 5, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(5, 5, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(5, 5, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(5, 5, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(5, 5, 4)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 6, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 6, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 6, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 6, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 6, 4)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 7, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 7, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 7, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 7, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(6, 7, 4)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(7, 6, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(7, 6, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(7, 6, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(7, 6, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(7, 6, 4)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(9, 9, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(9, 9, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(9, 9, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(9, 9, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(9, 9, 4)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(15, 15, 0)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(15, 15, 1)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(15, 15, 2)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(15, 15, 3)
DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS(15, 15, 4)

#undef DEFINE_BUILD_OPTICAL_FLOW_TEST_ALL_CHANNELS
#undef DEFINE_BUILD_OPTICAL_FLOW_TEST

std::vector<manager_only_test>& build_optical_flow_pyr_lk_pyramid_tests_get() {
  // clang-format off
  #define BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(win_w, win_h, lvl)                                           \
    MANAGER_ONLY_TEST("Build Optical Flow PyrLK Pyramid window " #win_w "x" #win_h ", level " #lvl ", 1 channel",   \
         test_build_optical_flow_lk_pyramid_w##win_w##x##win_h##_l##lvl##_c1),                                      \
    MANAGER_ONLY_TEST("Build Optical Flow PyrLK Pyramid window " #win_w "x" #win_h ", level " #lvl ", 2 channel",   \
         test_build_optical_flow_lk_pyramid_w##win_w##x##win_h##_l##lvl##_c2),                                      \
    MANAGER_ONLY_TEST("Build Optical Flow PyrLK Pyramid window " #win_w "x" #win_h ", level " #lvl ", 3 channel",   \
         test_build_optical_flow_lk_pyramid_w##win_w##x##win_h##_l##lvl##_c3),                                      \
    MANAGER_ONLY_TEST("Build Optical Flow PyrLK Pyramid window " #win_w "x" #win_h ", level " #lvl ", 4 channel",   \
         test_build_optical_flow_lk_pyramid_w##win_w##x##win_h##_l##lvl##_c4)

  static std::vector<manager_only_test> tests = {
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(3, 3, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(3, 3, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(3, 3, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(3, 3, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(3, 3, 4),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(5, 5, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(5, 5, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(5, 5, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(5, 5, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(5, 5, 4),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 6, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 6, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 6, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 6, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 6, 4),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 7, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 7, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 7, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 7, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(6, 7, 4),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(7, 6, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(7, 6, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(7, 6, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(7, 6, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(7, 6, 4),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(9, 9, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(9, 9, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(9, 9, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(9, 9, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(9, 9, 4),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(15, 15, 0),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(15, 15, 1),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(15, 15, 2),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(15, 15, 3),
    BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS(15, 15, 4)
  };
  #undef BUILD_OPTICAL_FLOW_TEST_ENTRIES_ALL_CHANNELS
  // clang-format on
  return tests;
}

#endif
