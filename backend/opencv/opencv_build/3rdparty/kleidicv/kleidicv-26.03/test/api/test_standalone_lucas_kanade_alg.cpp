// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <climits>
#include <random>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/utils.h"
#include "test_config.h"

const int DEFAULT_WINDOW_SIZE = 5;
const int DEFAULT_TERMINATION_COUNT = 30;
const float DEFAULT_MAX_ERROR = 0.2F;
const float DEFAULT_TERMINATION_EPSILON = 0.01F;
const float DEFAULT_MIN_EIGEN_VALS_THRESHOLD = 0.0001F;

struct StandaloneLucasKanadeAlgTestParams {
  std::vector<std::vector<uint8_t>> prev_image, next_image;
  std::vector<float> prev_points, next_points;
  std::vector<uint8_t> status;
  std::vector<float> err, eigenvals;
  int window_size = DEFAULT_WINDOW_SIZE;
  int termination_count = DEFAULT_TERMINATION_COUNT;
  float termination_epsilon = DEFAULT_TERMINATION_EPSILON;
  float min_eigen_vals_threshold = DEFAULT_MIN_EIGEN_VALS_THRESHOLD;
  float max_error = DEFAULT_MAX_ERROR;
  friend void PrintTo(const StandaloneLucasKanadeAlgTestParams &v,
                      std::ostream *os) {
    *os << "([\n";
    for (size_t y = 0; y < v.next_image.size(); ++y) {
      const auto &row = v.next_image[y];
      *os << "  [";
      for (size_t x = 0; x < row.size(); ++x) {
        *os << std::setw(3) << unsigned{row[x]};
        if (x + 1 != row.size()) {
          *os << ", ";
        }
      }
      *os << "]";
      if (y + 1 != v.next_image.size()) {
        *os << ",\n";
      }
    }
    *os << "], " << v.window_size << ")";
  }
};

// Like kleidicv_scharr_interleaved_s16_u8 but handles edge pixels
static void scharr_xy(const uint8_t *src, size_t src_stride, int16_t *dst,
                      size_t dst_stride, size_t width, size_t height) {
  for (size_t y = 0; y < height; y++) {
    const uint8_t *src_0 = src;
    if (y > 0) {
      src_0 += src_stride * (y - 1);
    } else if (height > 1) {
      src_0 += src_stride;
    }

    const uint8_t *src_1 = src + src_stride * y;

    const uint8_t *src_2 = src;
    if (y + 1 < height) {
      src_2 += src_stride * (y + 1);
    } else if (height > 1) {
      src_2 += src_stride * (height - 2);
    }

    auto *dst_row = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      size_t x0 = 0;
      if (x > 0) {
        x0 = x - 1;
      } else if (width > 1) {
        x0 = 1;
      }

      size_t x2 = 0;
      if (x + 1 < width) {
        x2 = x + 1;
      } else if (width > 1) {
        x2 = width - 2;
      }
      dst_row[x * 2] = static_cast<int16_t>(
          (src_0[x2] + src_2[x2] - src_0[x0] - src_2[x0]) * 3 +
          (src_1[x2] - src_1[x0]) * 10);
      dst_row[x * 2 + 1] = static_cast<int16_t>(
          (src_2[x0] + src_2[x2] - src_0[x0] - src_0[x2]) * 3 +
          (src_2[x] - src_0[x]) * 10);
    }
  }
}

template <typename T>
static std::vector<T> flatten_vector_of_vectors(
    const std::vector<std::vector<T>> &v) {
  std::vector<T> result;
  for (const auto &row : v) {
    result.insert(result.end(), row.begin(), row.end());
  }
  return result;
}

static ptrdiff_t index_for_reverse_border(ptrdiff_t x, ptrdiff_t width) {
  if (width < 2) {
    return 0;
  }
  x = abs(x);
  x %= width * 2 - 2;
  x -= width - 1;
  x = abs(x);
  x = width - x - 1;
  return x;
}

template <typename T>
static std::vector<T> pad_image_border_reverse(const T *src, size_t width,
                                               size_t height, size_t pad_left,
                                               size_t pad_right, size_t pad_y) {
  size_t dst_stride = width + pad_left + pad_right;
  std::vector<T> dst(dst_stride * (height + pad_y * 2));
  for (size_t y = 0; y < height + pad_y * 2; ++y) {
    ptrdiff_t src_y = index_for_reverse_border(
        static_cast<ptrdiff_t>(y - pad_y), static_cast<ptrdiff_t>(height));
    const T *src_row = src + src_y * width;

    T *dst_row = dst.data() + y * dst_stride;

    for (size_t x = 0; x < dst_stride; ++x) {
      dst_row[x] = src_row[index_for_reverse_border(
          static_cast<ptrdiff_t>(x - pad_left), static_cast<ptrdiff_t>(width))];
    }
  }
  return dst;
}

static void extend_images_and_test(
    const uint8_t *prev_image, const uint8_t *next_image, size_t width,
    size_t height, size_t prev_image_pad_right, size_t next_image_pad_right,
    size_t scharr_pad_right, size_t window_width, size_t window_height,
    const std::vector<float> &prev_points,
    const std::vector<float> &next_points, const std::vector<uint8_t> &status,
    const std::vector<float> &error, const std::vector<float> &eigenvals,
    float max_error, int termination_count, float termination_epsilon,
    float min_eigen_vals_threshold) {
  const size_t prev_image_stride =
      width + window_width * 2 + prev_image_pad_right;
  const size_t next_image_stride =
      width + window_width * 2 + next_image_pad_right;
  const size_t scharr_element_stride =
      (width + window_width * 2) * 2 + scharr_pad_right;
  const size_t point_count = prev_points.size() / 2;

  // In opencv/modules/video/src/lkpyramid.cpp criteria.epsilon is multiplied by
  // itself before it reaches the HAL.
  termination_epsilon *= termination_epsilon;

  ASSERT_EQ(prev_points.size(), next_points.size());

  std::vector<uint8_t> padded_prev_image = pad_image_border_reverse(
      prev_image, width, height, window_width,
      window_width + prev_image_pad_right, window_height);
  std::vector<uint8_t> padded_next_image = pad_image_border_reverse(
      next_image, width, height, window_width,
      window_width + next_image_pad_right, window_height);

  std::vector<int16_t> scharr(scharr_element_stride *
                              (height + window_height * 2));
  scharr_xy(
      padded_prev_image.data() + prev_image_stride * window_height +
          window_width,
      prev_image_stride,
      scharr.data() + window_height * scharr_element_stride + window_width * 2,
      scharr_element_stride * sizeof(int16_t), width, height);

  const std::vector<uint8_t> empty_status;
  const std::vector<float> empty_error;

  for (const auto &[expected_status, expected_err, get_min_eigen_vals] : {
           std::make_tuple(status, error, false),
           std::make_tuple(status, eigenvals, true),
           std::make_tuple(status, empty_error, false),
           std::make_tuple(empty_status, empty_error, false),
           std::make_tuple(empty_status, eigenvals, true),
       }) {
    std::vector<float> actual_next_points(prev_points);
    std::vector<uint8_t> actual_status(expected_status.size(), 1);
    std::vector<float> actual_err(expected_err.size(), -1);

    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_standalone_lucas_kanade_alg_u8(
            padded_prev_image.data() + prev_image_stride * window_height +
                window_width,
            prev_image_stride,
            scharr.data() + scharr_element_stride * window_height +
                window_width * 2,
            scharr_element_stride * sizeof(int16_t),
            padded_next_image.data() + next_image_stride * window_height +
                window_width,
            next_image_stride, width, height, 1 /*channels*/,
            prev_points.data(), actual_next_points.data(), point_count,
            actual_status.empty() ? nullptr : actual_status.data(),
            actual_err.empty() ? nullptr : actual_err.data(), window_width,
            window_height, termination_count, termination_epsilon,
            get_min_eigen_vals, min_eigen_vals_threshold));

    ASSERT_THAT(actual_status, ::testing::ElementsAreArray(expected_status));
    EXPECT_THAT(
        actual_next_points,
        ::testing::Pointwise(::testing::FloatNear(max_error), next_points));
    EXPECT_THAT(actual_err, ::testing::Pointwise(::testing::FloatNear(0.001F),
                                                 expected_err));
  }
}

class StandaloneLucasKanadeAlg
    : public testing::TestWithParam<StandaloneLucasKanadeAlgTestParams> {
 public:
  void do_test(size_t prev_image_pad_right, size_t next_image_pad_right,
               size_t scharr_pad_right) const {
    const size_t height = GetParam().prev_image.size();
    const size_t width = GetParam().prev_image.at(0).size();
    const size_t window_width = GetParam().window_size;
    const size_t window_height = GetParam().window_size;

    ASSERT_EQ(height, GetParam().next_image.size());
    for (const auto &row : GetParam().prev_image) {
      ASSERT_EQ(width, row.size());
    }
    for (const auto &row : GetParam().next_image) {
      ASSERT_EQ(width, row.size());
    }

    const std::vector<uint8_t> prev_image =
        flatten_vector_of_vectors(GetParam().prev_image);
    const std::vector<uint8_t> next_image =
        flatten_vector_of_vectors(GetParam().next_image);

    const float max_error = 0.001F;

    extend_images_and_test(
        prev_image.data(), next_image.data(), width, height,
        prev_image_pad_right, next_image_pad_right, scharr_pad_right,
        window_width, window_height, GetParam().prev_points,
        GetParam().next_points, GetParam().status, GetParam().err,
        GetParam().eigenvals, max_error, GetParam().termination_count,
        GetParam().termination_epsilon, GetParam().min_eigen_vals_threshold);
  }
};

TEST_P(StandaloneLucasKanadeAlg, NoPadding) { do_test(0, 0, 0); }
TEST_P(StandaloneLucasKanadeAlg, PadPrev) { do_test(1, 0, 0); }
TEST_P(StandaloneLucasKanadeAlg, PadNext) { do_test(0, 1, 0); }
TEST_P(StandaloneLucasKanadeAlg, PadDeriv) { do_test(0, 0, 1); }

static std::vector<std::vector<uint8_t>> make_image(size_t width, size_t height,
                                                    uint8_t gray) {
  return std::vector<std::vector<uint8_t>>(height,
                                           std::vector<uint8_t>(width, gray));
}

static std::vector<std::vector<uint8_t>> draw_rect_on_image(
    std::vector<std::vector<uint8_t>> &&image, size_t x, size_t y, size_t width,
    size_t height, uint8_t gray) {
  for (size_t i = y; i < y + height; ++i) {
    for (size_t j = x; j < x + width; ++j) {
      image[i][j] = gray;
    }
  }
  return std::move(image);
}

static std::vector<std::vector<uint8_t>> make_corner_image(size_t width,
                                                           size_t height,
                                                           size_t x, size_t y,
                                                           uint8_t background,
                                                           uint8_t foreground) {
  return draw_rect_on_image(make_image(width, height, background), 0, 0, x, y,
                            foreground);
}

// Disable check for possible exceptions thrown outside main.
// The check isn't important in a test program.
// NOLINTBEGIN(cert-err58-cpp)

// 4x4 square rotated by 15 degrees
static const std::vector<std::vector<uint8_t>> square_rotated_by_15_degrees = {
    // clang-format off
    {0,   0,   0,   0,   0,   0,   0,   0},
    {0,   0,   0,   0,  56,  49,   0,   0},
    {0,  49, 167, 239, 255, 167,   0,   0},
    {0,  56, 255, 255, 255, 231,   0,   0},
    {0,   0, 239, 255, 255, 255,  56,   0},
    {0,   0, 167, 255, 231, 167,  49,   0},
    {0,   0,  49,  56,   0,   0,   0,   0},
    {0,   0,   0,   0,   0,   0,   0,   0},
    // clang-format on
};

// 4x4 square rotated by 30 degrees and offset
static const std::vector<std::vector<uint8_t>> square_rotated_by_30_degrees = {
    // clang-format off
    {  0,   0,  24, 123,   0,   0,   0,   0},
    {  0, 112, 239, 255, 112,   0,   0,   0},
    {123, 255, 255, 255, 239,  24,   0,   0},
    { 24, 239, 255, 255, 255, 123,   0,   0},
    {  0, 112, 255, 239, 112,   0,   0,   0},
    {  0,   0, 123,  24,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0},
    {  0,   0,   0,   0,   0,   0,   0,   0},
    // clang-format on
};

// NOLINTEND(cert-err58-cpp)

static StandaloneLucasKanadeAlgTestParams make_rotating_square_test(
    int window_size, const std::vector<float> &opencv_next_points,
    const std::vector<float> &err, const std::vector<float> &eigenvals) {
  float max_error = 1.0F;
  return StandaloneLucasKanadeAlgTestParams{
      square_rotated_by_15_degrees,
      square_rotated_by_30_degrees,
      // bottom-right and bottom-left corners
      {5.95F, 4.91F, 2.09F, 5.95F},
      opencv_next_points,
      {1, 1},
      err,
      eigenvals,
      window_size,
      DEFAULT_TERMINATION_COUNT,
      DEFAULT_TERMINATION_EPSILON,
      DEFAULT_MIN_EIGEN_VALS_THRESHOLD,
      max_error};
}

using P = StandaloneLucasKanadeAlgTestParams;

INSTANTIATE_TEST_SUITE_P(
    , StandaloneLucasKanadeAlg,
    testing::Values(
        // Bottom-right corner no movement
        P{make_corner_image(4, 4, 2, 2, 0, 9),
          make_corner_image(4, 4, 2, 2, 0, 9),
          {2.0F, 2.0F},
          {2.0F, 2.0F},
          {1},
          {0.0F},
          {0.00189F}},

        // Bottom-right corner moving one pixel right
        P{make_corner_image(4, 4, 2, 2, 0, 9),
          make_corner_image(4, 4, 3, 2, 0, 9),
          {2.0F, 2.0F},
          {3.10952997F, 1.96065569F},
          {1},
          {1.655F},
          {0.00189F}},

        // Bottom-right corner moving one pixel down
        P{make_corner_image(4, 4, 2, 2, 0, 9),
          make_corner_image(4, 4, 2, 3, 0, 9),
          {2.0F, 2.0F},
          {1.96065569F, 3.10952997F},
          {1},
          {1.655F},
          {0.00189F}},

        // Bottom-right corner moving one pixel down and one pixel right
        P{make_corner_image(4, 4, 2, 2, 0, 9),
          make_corner_image(4, 4, 3, 3, 0, 9),
          {2.0F, 2.0F},
          {2.99960446F, 2.99960446F},
          {1},
          {4.32F},
          {0.00189F}},

        // Bottom-right corner moving half a pixel right
        P{make_corner_image(4, 4, 2, 2, 0, 255),
          draw_rect_on_image(make_corner_image(4, 4, 2, 2, 0, 255), 2, 0, 1, 2,
                             127),
          {2.0F, 2.0F},
          {2.5086112F, 1.902421F},
          {1},
          {33.7175F},
          {1.51807F}},

        // Some points out of bounds
        P{make_corner_image(4, 4, 2, 2, 0, 9),
          make_corner_image(4, 4, 2, 2, 0, 9),
          {2, 2, -6, 1, 1, -6, 10, 1, 1, 10},
          {2, 2, -6, 1, 1, -6, 10, 1, 1, 10},
          {1, 0, 0, 0, 0},
          {0, -1, -1, -1, -1},
          {0.00189F, -1, -1, -1, -1}},

        // Single pixel image
        P{{{0}}, {{0}}, {0.5F, 0.5F}, {0.5F, 0.5F}, {0}, {-1}, {0}},

        P{{{0}},
          {{0}},
          {0.5F, 0.5F},
          {0.5F, 0.5F},
          {0},
          {-1},
          {0},
          DEFAULT_WINDOW_SIZE,
          DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD,
          DEFAULT_MAX_ERROR},

        // Edge case of termination count 0
        P{{{0}},
          {{0}},
          {0.5F, 0.5F},
          {0.5F, 0.5F},
          {0},
          {-1},
          {0},
          DEFAULT_WINDOW_SIZE,
          0 /*termination_count*/,
          DEFAULT_TERMINATION_EPSILON,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD,
          DEFAULT_MAX_ERROR},

        // Edge case of negative termination epsilon
        P{{{0}},
          {{0}},
          {0.5F, 0.5F},
          {0.5F, 0.5F},
          {0},
          {-1},
          {0},
          DEFAULT_WINDOW_SIZE,
          DEFAULT_TERMINATION_COUNT,
          -1 /*termination_epsilon*/,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD,
          DEFAULT_MAX_ERROR},

        // Edge case of min eigen vals threshold 0
        P{{{0}},
          {{0}},
          {0.5F, 0.5F},
          {0.5F, 0.5F},
          {0},
          {-1},
          {0},
          DEFAULT_WINDOW_SIZE,
          DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON,
          0 /*min_eigen_vals_threshold*/,
          DEFAULT_MAX_ERROR},

        // Triggers a projected point going out of bounds
        P{{{201, 181, 4}, {64, 133, 70}, {35, 133, 127}},
          {{64, 242, 103}, {5, 88, 143}, {139, 219, 107}},
          {2.23284149F, 0.747504354F},
          {2.9699F, -2.02478F},
          {0},
          {-1},
          {0.03813F},
          3},

        // Triggers a projected point going out of bounds on the last iteration
        P{{{249, 6, 81}, {20, 242, 34}, {133, 143, 224}},
          {{15, 12, 160}, {150, 194, 117}, {11, 82, 246}},
          {1.85238028F, 0.249280542F},
          {4.38032F, -0.25043F},
          {0},
          {-1},
          {0.13125F},
          3},

        // Triggers successful termination due to low "velocity"
        P{{{110, 23, 46}, {72, 228, 77}, {6, 87, 244}},
          {{246, 50, 100}, {227, 18, 190}, {18, 168, 134}},
          {0.967025876F, 1.84503651F},
          {1.64828F, 2.58103F},
          {1},
          {57.18056F},
          {0.42731F},
          3},

        make_rotating_square_test(3, {5.03495F, 3.60691F, 1.39537F, 5.03591F},
                                  {10.05208F, 10.04861F}, {1.37410F, 1.37343F}),
        make_rotating_square_test(4, {5.04758F, 3.73075F, 1.12477F, 5.01238F},
                                  {9.56445F, 15.63086F}, {1.13227F, 1.14002F}),
        make_rotating_square_test(5, {5.00314F, 3.81633F, 1.16250F, 5.03388F},
                                  {12.18500F, 23.80125F}, {1.60865F, 1.62053F}),
        make_rotating_square_test(6, {4.95174F, 3.93273F, 0.77928F, 5.00313F},
                                  {18.60764F, 50.63194F}, {1.37666F, 1.38456F}),
        make_rotating_square_test(7, {4.93923F, 3.94952F, 1.01654F, 4.99058F},
                                  {26.49809F, 55.16071F}, {1.60671F, 1.61532F}),
        make_rotating_square_test(8, {4.96286F, 3.91138F, 0.94513F, 4.99251F},
                                  {34.55029F, 57.06738F}, {1.22516F, 1.22873F}),
        make_rotating_square_test(9, {4.98960F, 3.83407F, 1.07763F, 5.01845F},
                                  {47.52546F, 60.11883F}, {1.55569F, 1.55563F}),
        make_rotating_square_test(10, {4.95152F, 3.78445F, 0.96519F, 4.98130F},
                                  {53.02469F, 54.21344F}, {1.18499F, 1.18493F}),
        make_rotating_square_test(11, {4.89623F, 3.86437F, 1.06083F, 4.95983F},
                                  {60.61518F, 56.73373F}, {1.29844F, 1.29850F}),
        make_rotating_square_test(12, {4.80831F, 3.80666F, 0.94843F, 4.84862F},
                                  {66.85699F, 60.20595F}, {0.87564F, 0.87557F}),
        make_rotating_square_test(13, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {71.80584F, 66.91994F}, {0.93144F, 0.93147F}),
        make_rotating_square_test(14, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {76.03428F, 75.67442F}, {0.64332F, 0.64328F}),
        make_rotating_square_test(15, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {81.28000F, 84.23681F}, {0.69961F, 0.69964F}),
        make_rotating_square_test(16, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {84.53223F, 83.54919F}, {0.49254F, 0.49251F}),
        make_rotating_square_test(17, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {89.60305F, 83.21745F}, {0.54468F, 0.54470F}),
        make_rotating_square_test(18, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {89.92023F, 79.63406F}, {0.38917F, 0.38914F}),
        make_rotating_square_test(19, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {91.68473F, 78.35907F}, {0.43605F, 0.43606F}),
        make_rotating_square_test(20, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {91.01719F, 77.84946F}, {0.31523F, 0.31521F}),
        make_rotating_square_test(21, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {92.03614F, 80.94381F}, {0.35694F, 0.35696F}),
        make_rotating_square_test(22, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {89.82928F, 81.32819F}, {0.26052F, 0.26050F}),
        make_rotating_square_test(23, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {89.75030F, 85.61785F}, {0.29757F, 0.29758F}),
        make_rotating_square_test(24, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {85.35243F, 82.91254F}, {0.21891F, 0.21889F}),
        make_rotating_square_test(25, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {83.08310F, 83.38870F}, {0.25186F, 0.25187F}),
        make_rotating_square_test(26, {4.80835F, 3.81359F, 0.94093F, 4.84875F},
                                  {79.62486F, 79.25624F}, {0.18653F, 0.18651F}),
        make_rotating_square_test(27, {4.89751F, 3.89167F, 1.04246F, 4.96033F},
                                  {78.02598F, 78.54677F}, {0.21593F, 0.21594F}),
        make_rotating_square_test(28, {4.80836F, 3.81359F, 0.94093F, 4.84875F},
                                  {76.03428F, 75.67442F}, {0.16083F, 0.16082F}),
        make_rotating_square_test(29, {4.89743F, 3.89168F, 1.04246F, 4.96033F},
                                  {75.24889F, 75.44006F}, {0.18717F, 0.18718F}),
        make_rotating_square_test(30, {4.80836F, 3.81359F, 0.94093F, 4.84875F},
                                  {72.85309F, 72.46344F}, {0.14010F, 0.14009F}),
        make_rotating_square_test(31, {4.89743F, 3.89168F, 1.04246F, 4.96033F},
                                  {71.25634F, 71.65993F},
                                  {0.16380F, 0.16381F})));

TEST(StandaloneLKAlgTest, MultipleChannelsNotImplemented) {
  const int channels = 6;
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[1] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};
  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, sizeof(prev_image), scharr, sizeof(scharr),
                next_image, sizeof(next_image), 1, 1, channels, prev_points,
                next_points, 1, status, err, 3, 3, 30, 0.0001, false, 0.0001));
}

TEST(StandaloneLKAlgTest, NextPointOutOfRange) {
  // This test causes the predicted next point to fall outside the padded image.
  // It does this by padding the image with non-zero values, which could be
  // considered invalid input, but it's tricky to trigger otherwise.
  const ptrdiff_t width = 1, height = 1;
  const ptrdiff_t window_width = 3, window_height = 3;
  const ptrdiff_t padded_width = width + window_width * 2,
                  padded_height = height + window_height * 2;

  const uint8_t prev_image[padded_width * padded_height] = {
      201, 181, 4,   64,  133, 70,  35,  133, 127, 190, 61,  233, 62,
      183, 196, 117, 198, 243, 82,  65,  129, 23,  7,   251, 76,  123,
      73,  160, 182, 17,  22,  108, 22,  159, 57,  252, 218, 214, 239,
      125, 92,  132, 7,   28,  176, 188, 135, 212, 125};
  const uint8_t next_image[padded_width * padded_height] = {
      64,  242, 103, 5,   88,  143, 139, 219, 107, 63, 81,  42, 50,
      247, 20,  65,  149, 112, 136, 11,  178, 232, 38, 158, 92, 76,
      236, 193, 185, 124, 194, 152, 69,  56,  111, 54, 72,  93, 220,
      5,   188, 108, 232, 250, 124, 150, 116, 222, 248};

  int16_t scharr[padded_width * padded_height * 2] = {};
  scharr_xy(prev_image, padded_width, scharr,
            2 * padded_width * sizeof(int16_t), padded_width, padded_height);

  const size_t point_count = 1;
  float prev_points[point_count * 2] = {0.5F, 0.5F};
  float next_points[point_count * 2] = {0.5F, 0.5F};
  float err[point_count] = {};
  uint8_t status[point_count] = {1};

  for (uint8_t *actual_status : {status, static_cast<uint8_t *>(nullptr)}) {
    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_standalone_lucas_kanade_alg_u8(
            prev_image + padded_width * window_height + window_width,
            padded_width,
            scharr + (window_height * 2 * padded_width) + (window_width * 2),
            2 * padded_width * sizeof(int16_t),
            next_image + padded_width * window_height + window_width,
            padded_width, width, height, 1, prev_points, next_points,
            point_count, actual_status, err, window_width, window_height,
            DEFAULT_TERMINATION_COUNT,
            DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
            DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
  }
}

TEST(StandaloneLKAlgTest, Fuzz) {
  const ptrdiff_t width = 3, height = width;
  const ptrdiff_t window_width = 3, window_height = 3;
  const ptrdiff_t padded_width = width + window_width * 2,
                  padded_height = height + window_height * 2;
  const size_t point_count = width * height;
  uint8_t prev_image[width * height] = {}, next_image[width * height] = {};
  float prev_points[point_count * 2] = {};
  float next_points[point_count * 2] = {}, err[point_count] = {};
  uint8_t status[point_count] = {};

  std::mt19937_64 rng(test::Options::seed());
  std::uniform_int_distribution<uint16_t> dist(0, 255);
  std::uniform_real_distribution<float> distpt(0, width);

  for (int j = 0; j < 1000; ++j) {
    for (int i = 0; i < width * height; ++i) {
      prev_image[i] = static_cast<uint8_t>(dist(rng));
      next_image[i] = static_cast<uint8_t>(dist(rng));
    }

    const std::vector<uint8_t> padded_prev_image = pad_image_border_reverse(
        prev_image, width, height, window_width, window_width, window_height);
    const std::vector<uint8_t> padded_next_image = pad_image_border_reverse(
        next_image, width, height, window_width, window_width, window_height);

    for (ptrdiff_t y = 0; y < height; ++y) {
      for (ptrdiff_t x = 0; x < width; ++x) {
        prev_points[2 * (x + y * width)] = distpt(rng);
        prev_points[2 * (x + y * width) + 1] = distpt(rng);
        next_points[2 * (x + y * width)] = distpt(rng);
        next_points[2 * (x + y * width) + 1] = distpt(rng);
      }
    }
    memcpy(next_points, prev_points, sizeof(next_points));
    for (size_t i = 0; i < point_count; ++i) {
      status[i] = 1;
    }

    int16_t scharr[padded_width * padded_height * 2] = {};
    scharr_xy(
        padded_prev_image.data() + padded_width * window_height + window_width,
        padded_width,
        scharr + (window_height * 2 * padded_width) + (window_width * 2),
        2 * padded_width * sizeof(int16_t), width, height);

    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_standalone_lucas_kanade_alg_u8(
            padded_prev_image.data() + padded_width * window_height +
                window_width,
            padded_width,
            scharr + (window_height * 2 * padded_width) + (window_width * 2),
            2 * padded_width * sizeof(int16_t),
            padded_next_image.data() + padded_width * window_height +
                window_width,
            padded_width, width, height, 1, prev_points, next_points,
            point_count, status, err, window_width, window_height,
            DEFAULT_TERMINATION_COUNT,
            DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
            DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
  }
}

TEST(StandaloneLKAlgTest, WindowSizeTooLarge) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  ASSERT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 1, prev_points, next_points, 1, status, err,
          KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE + 1, DEFAULT_WINDOW_SIZE,
          DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  ASSERT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 1, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE + 1,
          DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, WindowBufferTotalSizeOverflow) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  ASSERT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 2, prev_points, next_points, 1, status, err,
          INT_MAX, INT_MAX, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, WindowBufferInvalidDimensions) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  ASSERT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, sizeof(prev_image), scharr, sizeof(scharr),
                next_image, sizeof(next_image), 1, 1, 1, prev_points,
                next_points, 1, status, err, 0, 5, DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  ASSERT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, sizeof(prev_image), scharr, sizeof(scharr),
                next_image, sizeof(next_image), 1, 1, 1, prev_points,
                next_points, 1, status, err, 5, 0, DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, WindowBufferPatchTimesChannelsOverflow) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, sizeof(prev_image), scharr, sizeof(scharr),
                next_image, sizeof(next_image), 1, 1, INT_MAX, prev_points,
                next_points, 1, status, err, DEFAULT_WINDOW_SIZE,
                DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, InvalidImageDimensions) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  ASSERT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 0, 1, 1, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  ASSERT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 0, 1, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, InvalidPrevNextStride) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 4;
  constexpr int kChannels = 1;
  const uint8_t prev_image[kWidth * kHeight] = {};
  const uint8_t next_image[kWidth * kHeight] = {};
  const int16_t scharr[kWidth * kHeight * 2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};
  const size_t scharr_stride_bytes =
      static_cast<size_t>(kWidth) * kChannels * 2 * sizeof(int16_t);

  ASSERT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, kWidth - 1, scharr, scharr_stride_bytes, next_image,
                kWidth, kWidth, kHeight, kChannels, prev_points, next_points, 1,
                status, err, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE,
                DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  ASSERT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, kWidth, scharr, scharr_stride_bytes, next_image,
                kWidth - 1, kWidth, kHeight, kChannels, prev_points,
                next_points, 1, status, err, DEFAULT_WINDOW_SIZE,
                DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, InvalidScharrStride) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 4;
  constexpr int kChannels = 1;
  const uint8_t prev_image[kWidth * kHeight] = {};
  const uint8_t next_image[kWidth * kHeight] = {};
  const int16_t scharr[kWidth * kHeight * 2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};
  const size_t row_bytes =
      static_cast<size_t>(kWidth) * kChannels * 2 * sizeof(int16_t);

  ASSERT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, kWidth, scharr, row_bytes - sizeof(int16_t),
                next_image, kWidth, kWidth, kHeight, kChannels, prev_points,
                next_points, 1, status, err, DEFAULT_WINDOW_SIZE,
                DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  ASSERT_EQ(
      KLEIDICV_ERROR_ALIGNMENT,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, kWidth, scharr, row_bytes + 1, next_image, kWidth, kWidth,
          kHeight, kChannels, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, InvalidScharrStrideAlignmentHeightOne) {
  constexpr int kWidth = 1;
  constexpr int kHeight = 1;
  constexpr int kChannels = 1;
  const uint8_t prev_image[kWidth * kHeight] = {};
  const uint8_t next_image[kWidth * kHeight] = {};
  const int16_t scharr[kWidth * kHeight * 2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};
  const size_t row_bytes =
      static_cast<size_t>(kWidth) * kChannels * 2 * sizeof(int16_t);

  ASSERT_EQ(
      KLEIDICV_ERROR_ALIGNMENT,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, kWidth, scharr, row_bytes + 1, next_image, kWidth, kWidth,
          kHeight, kChannels, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, NullInputPointers) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NULL_POINTER,
      kleidicv_standalone_lucas_kanade_alg_u8(
          nullptr, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 1, nullptr, nullptr, 0, nullptr, nullptr,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  EXPECT_EQ(
      KLEIDICV_ERROR_NULL_POINTER,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), nullptr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 1, nullptr, nullptr, 0, nullptr, nullptr,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  EXPECT_EQ(
      KLEIDICV_ERROR_NULL_POINTER,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), nullptr,
          sizeof(next_image), 1, 1, 1, nullptr, nullptr, 0, nullptr, nullptr,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, NullPointsWhenPointCountPositive) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NULL_POINTER,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 1, nullptr, nullptr, 1, nullptr, nullptr,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev_image, sizeof(prev_image), scharr, sizeof(scharr),
                next_image, sizeof(next_image), 1, 1, 1, prev_points, nullptr,
                1, nullptr, nullptr, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE,
                DEFAULT_TERMINATION_COUNT,
                DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON,
                false, DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, ZeroPointCountAllowsNullPoints) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};

  EXPECT_EQ(
      KLEIDICV_OK,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 1, nullptr, nullptr, 0, nullptr, nullptr,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, ImageSizeTooLarge) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  const int width = INT_MAX;
  const int height = INT_MAX;
  const size_t stride = static_cast<size_t>(INT_MAX);
  const size_t scharr_stride_bytes = static_cast<size_t>(INT_MAX) * 4U;

  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, stride, scharr, scharr_stride_bytes, next_image, stride,
          width, height, 1, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

TEST(StandaloneLKAlgTest, StatusZeroSkipsErrorWrite) {
  const ptrdiff_t width = 3, height = 3;
  const ptrdiff_t window_width = 3, window_height = 3;
  const ptrdiff_t padded_width = width + window_width * 2,
                  padded_height = height + window_height * 2;

  const uint8_t prev_image[width * height] = {10, 20, 30, 40, 50,
                                              60, 70, 80, 90};
  const uint8_t next_image[width * height] = {11, 21, 31, 41, 51,
                                              61, 71, 81, 91};

  const std::vector<uint8_t> padded_prev_image = pad_image_border_reverse(
      prev_image, width, height, window_width, window_width, window_height);
  const std::vector<uint8_t> padded_next_image = pad_image_border_reverse(
      next_image, width, height, window_width, window_width, window_height);

  int16_t scharr[padded_width * padded_height * 2] = {};
  scharr_xy(
      padded_prev_image.data() + padded_width * window_height + window_width,
      padded_width,
      scharr + (window_height * 2 * padded_width) + (window_width * 2),
      2 * padded_width * sizeof(int16_t), width, height);

  const size_t point_count = 1;
  float prev_points[point_count * 2] = {1.0F, 1.0F};
  float next_points[point_count * 2] = {1.0F, 1.0F};
  float err[point_count] = {123.0F};
  uint8_t status[point_count] = {0};

  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_standalone_lucas_kanade_alg_u8(
          padded_prev_image.data() + padded_width * window_height +
              window_width,
          padded_width,
          scharr + (window_height * 2 * padded_width) + (window_width * 2),
          2 * padded_width * sizeof(int16_t),
          padded_next_image.data() + padded_width * window_height +
              window_width,
          padded_width, width, height, 1, prev_points, next_points, point_count,
          status, err, window_width, window_height, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));

  EXPECT_EQ(status[0], 0);
  EXPECT_EQ(err[0], 123.0F);
}

TEST(StandaloneLKAlgTest, ZeroChannelsNotAllowed) {
  const uint8_t prev_image[1] = {}, next_image[1] = {};
  const int16_t scharr[2] = {};
  const float prev_points[2] = {};
  float next_points[2] = {}, err[1] = {};
  uint8_t status[1] = {};

  ASSERT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_standalone_lucas_kanade_alg_u8(
          prev_image, sizeof(prev_image), scharr, sizeof(scharr), next_image,
          sizeof(next_image), 1, 1, 0, prev_points, next_points, 1, status, err,
          DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_TERMINATION_COUNT,
          DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
          DEFAULT_MIN_EIGEN_VALS_THRESHOLD));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TEST(StandaloneLKAlgTest, CannotAllocateWindow) {
  MockMallocToFail::enable();
  const ptrdiff_t width = 1, height = 1;
  const ptrdiff_t window_width = 100, window_height = 100;
  const ptrdiff_t padded_width = width + window_width * 2,
                  padded_height = height + window_height * 2;
  const uint8_t prev_image[padded_width * padded_height] = {};
  const uint8_t next_image[padded_width * padded_height] = {};
  int16_t scharr[padded_width * padded_height * 2] = {};
  const size_t kPointCount = 1;
  float prev_points[kPointCount * 2] = {};
  float next_points[kPointCount * 2] = {};
  float err[kPointCount];
  uint8_t status[kPointCount];
  kleidicv_error_t ret = kleidicv_standalone_lucas_kanade_alg_u8(
      prev_image + padded_width * window_height + window_width, padded_width,
      scharr + (window_height * 2 * padded_width) + (window_width * 2),
      2 * padded_width * sizeof(int16_t),
      next_image + padded_width * window_height + window_width, padded_width,
      width, height, 1, prev_points, next_points, kPointCount, status, err,
      window_width, window_height, DEFAULT_TERMINATION_COUNT,
      DEFAULT_TERMINATION_EPSILON * DEFAULT_TERMINATION_EPSILON, false,
      DEFAULT_MIN_EIGEN_VALS_THRESHOLD);
  MockMallocToFail::disable();
  ASSERT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);
}
#endif
