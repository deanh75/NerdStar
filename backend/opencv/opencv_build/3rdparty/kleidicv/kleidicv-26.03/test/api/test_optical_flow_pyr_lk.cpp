// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/analysis/build_optical_flow_pyr_lk_pyramid.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

namespace {

kleidicv_optflow_lk_context_t make_default_context() {
  return {
      5,      // window_width
      5,      // window_height
      3,      // max_level
      30,     // termination_count
      0.01F,  // termination_epsilon
      1e-4F,  // min_eig_threshold
      0       // flags
  };
}

std::vector<float> make_points(size_t point_count, size_t width,
                               size_t height) {
  std::vector<float> points(point_count * 2);
  const float max_x = static_cast<float>(width > 0 ? (width - 1) : 0);
  const float max_y = static_cast<float>(height > 0 ? (height - 1) : 0);
  for (size_t i = 0; i < point_count; ++i) {
    const float x = static_cast<float>((i * 7) % (width + 1));
    const float y = static_cast<float>((i * 11) % (height + 1));
    points[i * 2] = std::min(x, max_x);
    points[i * 2 + 1] = std::min(y, max_y);
  }
  return points;
}

using KLEIDICV_TARGET_NAMESPACE::OpticalFlowLevel;
using KLEIDICV_TARGET_NAMESPACE::OpticalFlowLKPyramid;

struct PyramidPrivateLayout {
  size_t level_count;
  size_t channels;
  size_t border_width;
  size_t border_height;
  OpticalFlowLevel* levels;
};

struct FlowCallResult {
  std::vector<float> next_points;
  std::vector<uint8_t> status;
  std::vector<float> err;
};

struct FlowPathResults {
  FlowCallResult from_pyramid;
  FlowCallResult direct;
};

OpticalFlowLKPyramid* as_internal(
    kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid) {
  return reinterpret_cast<OpticalFlowLKPyramid*>(pyramid);
}

PyramidPrivateLayout* as_layout(
    kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid) {
  return reinterpret_cast<PyramidPrivateLayout*>(as_internal(pyramid));
}

void build_matching_pyramids(size_t width, size_t height, size_t channels,
                             const kleidicv_optflow_lk_context_t& context,
                             test::Array2D<uint8_t>* prev_image,
                             test::Array2D<uint8_t>* next_image,
                             kleidicv_optical_flow_pyr_lk_pyramid_t** prev,
                             kleidicv_optical_flow_pyr_lk_pyramid_t** next) {
  *prev = nullptr;
  *next = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             prev, prev_image->data(), prev_image->stride(),
                             width, height, channels, context.max_level + 1,
                             context.window_width, context.window_height));
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             next, next_image->data(), next_image->stride(),
                             width, height, channels, context.max_level + 1,
                             context.window_width, context.window_height));
}

void fill_strong_corner_image(test::Array2D<uint8_t>* image) {
  ASSERT_NE(image, nullptr);
  image->fill([](size_t row, size_t column) -> std::optional<uint8_t> {
    if ((row >= 6 && row < 10 && column >= 6 && column < 10) ||
        (row >= 6 && row < 10 && column >= 10 && column < 14) ||
        (row >= 10 && row < 14 && column >= 6 && column < 10)) {
      return 220;
    }
    return 16;
  });
}

void fill_flat_image(test::Array2D<uint8_t>* image, uint8_t value) {
  ASSERT_NE(image, nullptr);
  image->fill(value);
}

void run_matching_paths(size_t width, size_t height, size_t channels,
                        test::Array2D<uint8_t>* prev_image,
                        test::Array2D<uint8_t>* next_image,
                        const std::vector<float>& prev_points,
                        const std::vector<float>& initial_next_points,
                        kleidicv_optflow_lk_context_t context, float err_fill,
                        FlowPathResults* results) {
  ASSERT_NE(prev_image, nullptr);
  ASSERT_NE(next_image, nullptr);
  ASSERT_NE(results, nullptr);

  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  build_matching_pyramids(width, height, channels, context, prev_image,
                          next_image, &prev_pyramid, &next_pyramid);

  const size_t point_count = prev_points.size() / 2;
  results->from_pyramid.next_points = initial_next_points;
  results->direct.next_points = initial_next_points;
  results->from_pyramid.status.assign(point_count, 0);
  results->direct.status.assign(point_count, 0);
  results->from_pyramid.err.assign(point_count, err_fill);
  results->direct.err.assign(point_count, err_fill);

  ASSERT_EQ(KLEIDICV_OK, kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                             prev_pyramid, next_pyramid, prev_points.data(),
                             results->from_pyramid.next_points.data(),
                             point_count, results->from_pyramid.status.data(),
                             results->from_pyramid.err.data(), context));

  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_optical_flow_pyr_lk_u8(
          prev_image->data(), prev_image->stride(), next_image->data(),
          next_image->stride(), width, height, channels, prev_points.data(),
          results->direct.next_points.data(), point_count,
          results->direct.status.data(), results->direct.err.data(), context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

}  // namespace

TEST(CalcOpticalFlowPyrLk, FromPyramidAndDirectMatch) {
  constexpr size_t kWidth = 32;
  constexpr size_t kHeight = 24;
  constexpr size_t kChannels = 1;
  constexpr size_t kPointCount = 19;

  test::Array2D<uint8_t> prev_image(kWidth * kChannels, kHeight);
  test::Array2D<uint8_t> next_image(kWidth * kChannels, kHeight);
  test::PseudoRandomNumberGenerator<uint8_t> generator;
  prev_image.fill(generator);
  next_image.fill(generator);

  kleidicv_optflow_lk_context_t context = make_default_context();

  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &prev_pyramid, prev_image.data(), prev_image.stride(), kWidth,
                kHeight, kChannels, context.max_level + 1, context.window_width,
                context.window_height));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &next_pyramid, next_image.data(), next_image.stride(), kWidth,
                kHeight, kChannels, context.max_level + 1, context.window_width,
                context.window_height));

  const std::vector<float> prev_points =
      make_points(kPointCount, kWidth, kHeight);
  std::vector<float> from_pyramid_points(prev_points);
  std::vector<float> direct_points(prev_points);
  std::vector<uint8_t> from_pyramid_status(kPointCount, 0);
  std::vector<uint8_t> direct_status(kPointCount, 0);
  std::vector<float> from_pyramid_err(kPointCount, -1.0F);
  std::vector<float> direct_err(kPointCount, -1.0F);

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points.data(),
                from_pyramid_points.data(), kPointCount,
                from_pyramid_status.data(), from_pyramid_err.data(), context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_u8(
                prev_image.data(), prev_image.stride(), next_image.data(),
                next_image.stride(), kWidth, kHeight, kChannels,
                prev_points.data(), direct_points.data(), kPointCount,
                direct_status.data(), direct_err.data(), context));

  EXPECT_THAT(direct_status, ::testing::ElementsAreArray(from_pyramid_status));
  EXPECT_THAT(direct_points, ::testing::Pointwise(::testing::FloatNear(1e-6F),
                                                  from_pyramid_points));
  EXPECT_THAT(direct_err, ::testing::Pointwise(::testing::FloatNear(1e-5F),
                                               from_pyramid_err));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, NullPointerFromPyramid) {
  constexpr size_t kWidth = 16;
  constexpr size_t kHeight = 16;
  constexpr size_t kPointCount = 1;

  test::Array2D<uint8_t> image(kWidth, kHeight);
  ASSERT_TRUE(image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &prev_pyramid, image.data(), image.stride(),
                             kWidth, kHeight, 1, context.max_level + 1,
                             context.window_width, context.window_height));
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &next_pyramid, image.data(), image.stride(),
                             kWidth, kHeight, 1, context.max_level + 1,
                             context.window_width, context.window_height));

  float prev_points[kPointCount * 2] = {3.0F, 4.0F};
  float next_points[kPointCount * 2] = {3.0F, 4.0F};
  uint8_t status[kPointCount] = {0};

  test::test_null_args(kleidicv_optical_flow_pyr_lk_u8_from_pyramid,
                       prev_pyramid, next_pyramid, prev_points, next_points,
                       kPointCount, status, static_cast<float*>(nullptr),
                       context);

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, ErrPhotometricWrittenForSuccessfulPoints) {
  constexpr size_t kWidth = 20;
  constexpr size_t kHeight = 20;
  constexpr float kErrSentinel = -123.0F;

  test::Array2D<uint8_t> prev_image(kWidth, kHeight);
  test::Array2D<uint8_t> next_image(kWidth, kHeight);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());
  fill_strong_corner_image(&prev_image);
  next_image = prev_image;

  kleidicv_optflow_lk_context_t context = make_default_context();
  context.max_level = 0;

  const std::vector<float> prev_points = {8.0F, 8.0F};
  FlowPathResults results;
  run_matching_paths(kWidth, kHeight, 1, &prev_image, &next_image, prev_points,
                     prev_points, context, kErrSentinel, &results);

  EXPECT_THAT(results.direct.status, ::testing::ElementsAre(1));
  EXPECT_THAT(results.from_pyramid.status, ::testing::ElementsAre(1));
  EXPECT_THAT(results.direct.next_points,
              ::testing::Pointwise(::testing::FloatNear(1e-4F),
                                   results.from_pyramid.next_points));
  EXPECT_NE(results.direct.err[0], kErrSentinel);
  EXPECT_NE(results.from_pyramid.err[0], kErrSentinel);
  EXPECT_NEAR(results.direct.err[0], 0.0F, 1e-4F);
  EXPECT_NEAR(results.from_pyramid.err[0], 0.0F, 1e-4F);
  EXPECT_NEAR(results.direct.err[0], results.from_pyramid.err[0], 1e-6F);
}

TEST(CalcOpticalFlowPyrLk, ErrMinEigenModeWrittenForSuccessfulPoints) {
  constexpr size_t kWidth = 20;
  constexpr size_t kHeight = 20;
  constexpr float kErrSentinel = -123.0F;

  test::Array2D<uint8_t> prev_image(kWidth, kHeight);
  test::Array2D<uint8_t> next_image(kWidth, kHeight);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());
  fill_strong_corner_image(&prev_image);
  next_image = prev_image;

  kleidicv_optflow_lk_context_t context = make_default_context();
  context.max_level = 0;
  context.flags = KLEIDICV_GET_MIN_EIG;

  const std::vector<float> prev_points = {8.0F, 8.0F};
  FlowPathResults results;
  run_matching_paths(kWidth, kHeight, 1, &prev_image, &next_image, prev_points,
                     prev_points, context, kErrSentinel, &results);

  EXPECT_THAT(results.direct.status, ::testing::ElementsAre(1));
  EXPECT_THAT(results.from_pyramid.status, ::testing::ElementsAre(1));
  EXPECT_NE(results.direct.err[0], kErrSentinel);
  EXPECT_NE(results.from_pyramid.err[0], kErrSentinel);
  EXPECT_GT(results.direct.err[0], context.min_eig_threshold);
  EXPECT_GT(results.from_pyramid.err[0], context.min_eig_threshold);
  EXPECT_NEAR(results.direct.err[0], results.from_pyramid.err[0], 1e-6F);
}

TEST(CalcOpticalFlowPyrLk, ErrPhotometricNotWrittenForFailedPoints) {
  constexpr size_t kWidth = 20;
  constexpr size_t kHeight = 20;
  constexpr float kErrSentinel = -123.0F;

  test::Array2D<uint8_t> prev_image(kWidth, kHeight);
  test::Array2D<uint8_t> next_image(kWidth, kHeight);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());
  fill_flat_image(&prev_image, 77);
  fill_flat_image(&next_image, 77);

  kleidicv_optflow_lk_context_t context = make_default_context();
  context.max_level = 0;

  const std::vector<float> prev_points = {8.0F, 8.0F};
  FlowPathResults results;
  run_matching_paths(kWidth, kHeight, 1, &prev_image, &next_image, prev_points,
                     prev_points, context, kErrSentinel, &results);

  EXPECT_THAT(results.direct.status, ::testing::ElementsAre(0));
  EXPECT_THAT(results.from_pyramid.status, ::testing::ElementsAre(0));
  EXPECT_EQ(results.direct.err[0], kErrSentinel);
  EXPECT_EQ(results.from_pyramid.err[0], kErrSentinel);
}

TEST(CalcOpticalFlowPyrLk, NullPointerDirect) {
  uint8_t prev_image[64] = {};
  uint8_t next_image[64] = {};
  float prev_points[2] = {1.0F, 1.0F};
  float next_points[2] = {1.0F, 1.0F};
  uint8_t status[1] = {1};
  kleidicv_optflow_lk_context_t context = make_default_context();

  test::test_null_args(
      kleidicv_optical_flow_pyr_lk_u8, prev_image, static_cast<size_t>(8),
      next_image, static_cast<size_t>(8), static_cast<size_t>(8),
      static_cast<size_t>(8), static_cast<size_t>(1), prev_points, next_points,
      static_cast<size_t>(1), status, static_cast<float*>(nullptr), context);
}

TEST(CalcOpticalFlowPyrLk, ContextValidation) {
  test::Array2D<uint8_t> image(16, 16);
  ASSERT_TRUE(image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &prev_pyramid, image.data(), image.stride(), 16,
                             16, 1, context.max_level + 1, context.window_width,
                             context.window_height));
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &next_pyramid, image.data(), image.stride(), 16,
                             16, 1, context.max_level + 1, context.window_width,
                             context.window_height));

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {0};

  context.window_width = 2;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  context = make_default_context();
  context.termination_count = 0;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  context = make_default_context();
  context.termination_epsilon = -0.1F;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  context = make_default_context();
  context.flags = 1 << 7;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, MismatchedPyramidsRejected) {
  test::Array2D<uint8_t> prev_image(16, 16);
  test::Array2D<uint8_t> next_image(15, 16);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;

  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_build_optical_flow_pyr_lk_pyramid(
          &prev_pyramid, prev_image.data(), prev_image.stride(), 16, 16, 1,
          context.max_level + 1, context.window_width, context.window_height));
  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_build_optical_flow_pyr_lk_pyramid(
          &next_pyramid, next_image.data(), next_image.stride(), 15, 16, 1,
          context.max_level + 1, context.window_width, context.window_height));

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {0};

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, DirectParameterRange) {
  uint8_t prev_image[64] = {};
  uint8_t next_image[64] = {};
  float prev_points[2] = {1.0F, 1.0F};
  float next_points[2] = {1.0F, 1.0F};
  uint8_t status[1] = {0};

  kleidicv_optflow_lk_context_t context = make_default_context();

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8(prev_image, 8, next_image, 8, 0, 8,
                                            1, prev_points, next_points, 1,
                                            status, nullptr, context));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8(prev_image, 8, next_image, 8, 8, 0,
                                            1, prev_points, next_points, 1,
                                            status, nullptr, context));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8(prev_image, 8, next_image, 8, 8, 8,
                                            0, prev_points, next_points, 1,
                                            status, nullptr, context));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8(prev_image, 7, next_image, 8, 8, 8,
                                            1, prev_points, next_points, 1,
                                            status, nullptr, context));

  context.max_level = std::numeric_limits<size_t>::max();
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8(prev_image, 8, next_image, 8, 8, 8,
                                            1, prev_points, next_points, 1,
                                            status, nullptr, context));
}

TEST(CalcOpticalFlowPyrLk, PointCountZeroAllowsNullPointArrays) {
  test::Array2D<uint8_t> prev_image(16, 16);
  test::Array2D<uint8_t> next_image(16, 16);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();

  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_build_optical_flow_pyr_lk_pyramid(
          &prev_pyramid, prev_image.data(), prev_image.stride(), 16, 16, 1,
          context.max_level + 1, context.window_width, context.window_height));
  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_build_optical_flow_pyr_lk_pyramid(
          &next_pyramid, next_image.data(), next_image.stride(), 16, 16, 1,
          context.max_level + 1, context.window_width, context.window_height));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                             prev_pyramid, next_pyramid, nullptr, nullptr, 0,
                             nullptr, nullptr, context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_optical_flow_pyr_lk_u8(
                             prev_image.data(), prev_image.stride(),
                             next_image.data(), next_image.stride(), 16, 16, 1,
                             nullptr, nullptr, 0, nullptr, nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, FromPyramidUsesInitialFlowFlag) {
  constexpr size_t kWidth = 20;
  constexpr size_t kHeight = 20;
  constexpr size_t kPointCount = 4;
  test::Array2D<uint8_t> prev_image(kWidth, kHeight);
  test::Array2D<uint8_t> next_image(kWidth, kHeight);
  test::PseudoRandomNumberGenerator<uint8_t> generator;
  prev_image.fill(generator);
  next_image.fill(generator);

  kleidicv_optflow_lk_context_t context = make_default_context();
  context.flags = KLEIDICV_USE_INITIAL_FLOW;

  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &prev_pyramid, prev_image.data(), prev_image.stride(), kWidth,
                kHeight, 1, context.max_level + 1, context.window_width,
                context.window_height));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &next_pyramid, next_image.data(), next_image.stride(), kWidth,
                kHeight, 1, context.max_level + 1, context.window_width,
                context.window_height));

  const std::vector<float> prev_points =
      make_points(kPointCount, kWidth, kHeight);
  std::vector<float> next_points = prev_points;
  for (size_t i = 0; i < kPointCount; ++i) {
    next_points[i * 2] += 0.5F;
    next_points[i * 2 + 1] += 0.25F;
  }
  std::vector<uint8_t> status(kPointCount, 0);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                             prev_pyramid, next_pyramid, prev_points.data(),
                             next_points.data(), kPointCount, status.data(),
                             nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, ContextRejectsTooLargeTerminationCount) {
  test::Array2D<uint8_t> image(16, 16);
  ASSERT_TRUE(image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  const kleidicv_optflow_lk_context_t build_context = make_default_context();
  context.termination_count =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;

  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &prev_pyramid, image.data(), image.stride(), 16, 16, 1,
                build_context.max_level + 1, build_context.window_width,
                build_context.window_height));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &next_pyramid, image.data(), image.stride(), 16, 16, 1,
                build_context.max_level + 1, build_context.window_width,
                build_context.window_height));

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {0};

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, MismatchedPyramidChannelsRejected) {
  test::Array2D<uint8_t> prev_image(16, 16);
  test::Array2D<uint8_t> next_image(48, 16);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;

  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_build_optical_flow_pyr_lk_pyramid(
          &prev_pyramid, prev_image.data(), prev_image.stride(), 16, 16, 1,
          context.max_level + 1, context.window_width, context.window_height));
  ASSERT_EQ(
      KLEIDICV_OK,
      kleidicv_build_optical_flow_pyr_lk_pyramid(
          &next_pyramid, next_image.data(), next_image.stride(), 16, 16, 3,
          context.max_level + 1, context.window_width, context.window_height));

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {0};

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, ContextWindowLargerThanBuiltBorderRejected) {
  test::Array2D<uint8_t> image(16, 16);
  ASSERT_TRUE(image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &prev_pyramid, image.data(), image.stride(), 16,
                             16, 1, context.max_level + 1, context.window_width,
                             context.window_height));
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &next_pyramid, image.data(), image.stride(), 16,
                             16, 1, context.max_level + 1, context.window_width,
                             context.window_height));

  context.window_width += 1;
  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {0};

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, DirectTrackFailsWhenPyramidBuildFails) {
  uint8_t prev_image[256] = {};
  uint8_t next_image[256] = {};
  float prev_points[2] = {1.0F, 1.0F};
  float next_points[2] = {1.0F, 1.0F};
  uint8_t status[1] = {0};

  kleidicv_optflow_lk_context_t context = make_default_context();
  context.window_width = 2;

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8(prev_image, 16, next_image, 16, 16,
                                            16, 1, prev_points, next_points, 1,
                                            status, nullptr, context));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TEST(CalcOpticalFlowPyrLk, FromPyramidScratchAllocationFailure) {
  constexpr size_t kWidth = 16;
  constexpr size_t kHeight = 16;
  test::Array2D<uint8_t> prev_image(kWidth, kHeight);
  test::Array2D<uint8_t> next_image(kWidth, kHeight);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  const kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  build_matching_pyramids(kWidth, kHeight, 1, context, &prev_image, &next_image,
                          &prev_pyramid, &next_pyramid);

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {1};

  MockMallocToFail::enable();
  const kleidicv_error_t ret = kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
      prev_pyramid, next_pyramid, prev_points, next_points, 1, status, nullptr,
      context);
  MockMallocToFail::disable();
  EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}
#endif

namespace {

using FromPyramidMutationFn =
    void (*)(kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
             kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
             kleidicv_optflow_lk_context_t* context);

void run_from_pyramid_expect_range(const char* case_name, size_t width,
                                   size_t height, size_t point_count,
                                   FromPyramidMutationFn mutate = nullptr) {
  SCOPED_TRACE(case_name);
  test::Array2D<uint8_t> prev_image(width, height);
  test::Array2D<uint8_t> next_image(width, height);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  build_matching_pyramids(width, height, 1, context, &prev_image, &next_image,
                          &prev_pyramid, &next_pyramid);

  if (mutate != nullptr) {
    mutate(prev_pyramid, next_pyramid, &context);
  }

  // Keep pointers non-null for API checks; mutation/overflow should trigger
  // before point-memory usage for the large-count cases.
  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {1};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points,
                point_count, status, nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

void mutate_level_size_out_of_int_range(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*next_pyramid*/,
    kleidicv_optflow_lk_context_t* /*context*/) {
  auto* prev_internal = as_internal(prev_pyramid);
  auto& level0 = const_cast<OpticalFlowLevel&>(prev_internal->level(0));
  level0.width = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
}

void mutate_level_size_out_of_int_range_after_stride_adjustment(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  auto* prev_internal = as_internal(prev_pyramid);
  auto* next_internal = as_internal(next_pyramid);
  auto& prev_level0 = const_cast<OpticalFlowLevel&>(prev_internal->level(0));
  auto& next_level0 = const_cast<OpticalFlowLevel&>(next_internal->level(0));
  const size_t huge_width =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  prev_level0.width = huge_width;
  next_level0.width = huge_width;
  // Keep stride checks satisfied so rejection happens at the int-range guard
  // in the main LK processing loop.
  prev_level0.image_stride = std::numeric_limits<size_t>::max();
  next_level0.image_stride = std::numeric_limits<size_t>::max();
  prev_level0.scharr_stride = std::numeric_limits<size_t>::max();
  next_level0.scharr_stride = std::numeric_limits<size_t>::max();
}

void mutate_zero_level_count(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*next_pyramid*/,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->level_count = 0;
}

void mutate_channels_out_of_int_range(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->channels =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  as_layout(next_pyramid)->channels = as_layout(prev_pyramid)->channels;
}

void mutate_per_level_mismatch(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* context) {
  context->max_level = 2;
  ASSERT_GT(as_internal(prev_pyramid)->level_count(), 1U);
  auto* next_internal = as_internal(next_pyramid);
  auto& level1 = const_cast<OpticalFlowLevel&>(next_internal->level(1));
  level1.width += 1;
}

void mutate_image_stride_too_small(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*next_pyramid*/,
    kleidicv_optflow_lk_context_t* /*context*/) {
  auto* prev_internal = as_internal(prev_pyramid);
  auto& level0 = const_cast<OpticalFlowLevel&>(prev_internal->level(0));
  level0.image_stride = 0;
}

void mutate_scharr_stride_too_small(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*next_pyramid*/,
    kleidicv_optflow_lk_context_t* /*context*/) {
  auto* prev_internal = as_internal(prev_pyramid);
  auto& level0 = const_cast<OpticalFlowLevel&>(prev_internal->level(0));
  level0.scharr_stride = 0;
}

void mutate_border_overflow_prev(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*next_pyramid*/,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->border_width = std::numeric_limits<size_t>::max();
}

void mutate_border_overflow_next(
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*prev_pyramid*/,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(next_pyramid)->border_width = std::numeric_limits<size_t>::max();
}

void mutate_prev_image_stride_mul_overflow(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->channels = 3;
  as_layout(next_pyramid)->channels = 3;
  as_layout(prev_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 4;
}

void mutate_prev_image_width_add_overflow(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*next_pyramid*/,
    kleidicv_optflow_lk_context_t* /*context*/) {
  // Ensure border*2 does not overflow, but width + border*2 does.
  as_layout(prev_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 2;
}

void mutate_next_image_stride_mul_overflow(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->channels = 3;
  as_layout(next_pyramid)->channels = 3;
  as_layout(next_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 4;
}

void mutate_next_image_width_add_overflow(
    kleidicv_optical_flow_pyr_lk_pyramid_t* /*prev_pyramid*/,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  // Ensure border*2 does not overflow, but width + border*2 does.
  as_layout(next_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 2;
}

void mutate_prev_scharr_stride_mul_overflow(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->channels = 2;
  as_layout(next_pyramid)->channels = 2;
  as_layout(prev_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 5;
}

void mutate_prev_scharr_stride_mul_overflow_inner(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  // Keep image-stride multiplication in range and force Scharr-stride
  // multiplication overflow at min_prev_scharr_stride computation.
  as_layout(prev_pyramid)->channels = 2;
  as_layout(next_pyramid)->channels = 2;
  as_layout(prev_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 8;
  auto* prev_internal = as_internal(prev_pyramid);
  auto& prev_level0 = const_cast<OpticalFlowLevel&>(prev_internal->level(0));
  prev_level0.image_stride = std::numeric_limits<size_t>::max();
}

void mutate_next_scharr_stride_mul_overflow(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  as_layout(prev_pyramid)->channels = 2;
  as_layout(next_pyramid)->channels = 2;
  as_layout(next_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 5;
}

void mutate_next_scharr_stride_mul_overflow_inner(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* /*context*/) {
  // Keep image-stride multiplication in range and force Scharr-stride
  // multiplication overflow at min_next_scharr_stride computation.
  as_layout(prev_pyramid)->channels = 2;
  as_layout(next_pyramid)->channels = 2;
  as_layout(next_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 8;
  auto* next_internal = as_internal(next_pyramid);
  auto& next_level0 = const_cast<OpticalFlowLevel&>(next_internal->level(0));
  next_level0.image_stride = std::numeric_limits<size_t>::max();
}

void mutate_prev_scharr_stride_overflow_exact(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* context) {
  context->window_width = 5;
  context->window_height = 5;
  as_layout(prev_pyramid)->channels = 2;
  as_layout(next_pyramid)->channels = 2;
  as_layout(prev_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 8;
  as_layout(next_pyramid)->border_width = 5;
  auto* prev_internal = as_internal(prev_pyramid);
  auto& prev_level0 = const_cast<OpticalFlowLevel&>(prev_internal->level(0));
  prev_level0.image_stride = std::numeric_limits<size_t>::max();
}

void mutate_next_scharr_stride_overflow_exact(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* context) {
  context->window_width = 5;
  context->window_height = 5;
  as_layout(prev_pyramid)->channels = 2;
  as_layout(next_pyramid)->channels = 2;
  as_layout(prev_pyramid)->border_width = 5;
  as_layout(next_pyramid)->border_width =
      std::numeric_limits<size_t>::max() / 8;
  auto* next_internal = as_internal(next_pyramid);
  auto& next_level0 = const_cast<OpticalFlowLevel&>(next_internal->level(0));
  next_level0.image_stride = std::numeric_limits<size_t>::max();
}

void mutate_effective_max_level_out_of_int_range(
    kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    kleidicv_optflow_lk_context_t* context) {
  const size_t too_large_level =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  context->max_level = too_large_level;
  as_layout(prev_pyramid)->level_count = too_large_level + 1;
  as_layout(next_pyramid)->level_count = too_large_level + 1;
}

}  // namespace

TEST(CalcOpticalFlowPyrLk, FromPyramidPropagatesOpticalFlowKernelError) {
  constexpr size_t kWidth = 16;
  constexpr size_t kHeight = 16;
  test::Array2D<uint8_t> prev_image(kWidth, kHeight);
  test::Array2D<uint8_t> next_image(kWidth, kHeight);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  const kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  build_matching_pyramids(kWidth, kHeight, 1, context, &prev_image, &next_image,
                          &prev_pyramid, &next_pyramid);

  as_layout(prev_pyramid)->channels = 0;
  as_layout(next_pyramid)->channels = 0;

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {1};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk, FromPyramidRejectsMetadataAndStrideMutations) {
  constexpr size_t kSmallSize = 16;
  constexpr size_t kLargeSize = 64;

  run_from_pyramid_expect_range("level size out of int range", kSmallSize,
                                kSmallSize, 1,
                                mutate_level_size_out_of_int_range);
  run_from_pyramid_expect_range(
      "level size out of int range after stride fix", kSmallSize, kSmallSize, 1,
      mutate_level_size_out_of_int_range_after_stride_adjustment);
  run_from_pyramid_expect_range("zero level count", kSmallSize, kSmallSize, 1,
                                mutate_zero_level_count);
  run_from_pyramid_expect_range("channels out of int range", kSmallSize,
                                kSmallSize, 1,
                                mutate_channels_out_of_int_range);
  run_from_pyramid_expect_range("per level mismatch", kLargeSize, kLargeSize, 1,
                                mutate_per_level_mismatch);
  run_from_pyramid_expect_range("image stride too small", kSmallSize,
                                kSmallSize, 1, mutate_image_stride_too_small);
  run_from_pyramid_expect_range("scharr stride too small", kSmallSize,
                                kSmallSize, 1, mutate_scharr_stride_too_small);
}

TEST(CalcOpticalFlowPyrLk, FromPyramidRejectsBorderAndStrideOverflowMutations) {
  constexpr size_t kSize = 16;
  run_from_pyramid_expect_range("border overflow prev", kSize, kSize, 1,
                                mutate_border_overflow_prev);
  run_from_pyramid_expect_range("border overflow next", kSize, kSize, 1,
                                mutate_border_overflow_next);
  run_from_pyramid_expect_range("prev image width add overflow", kSize, kSize,
                                1, mutate_prev_image_width_add_overflow);
  run_from_pyramid_expect_range("next image width add overflow", kSize, kSize,
                                1, mutate_next_image_width_add_overflow);
  run_from_pyramid_expect_range("prev image stride mul overflow", kSize, kSize,
                                1, mutate_prev_image_stride_mul_overflow);
  run_from_pyramid_expect_range("next image stride mul overflow", kSize, kSize,
                                1, mutate_next_image_stride_mul_overflow);
  run_from_pyramid_expect_range("prev scharr stride mul overflow", kSize, kSize,
                                1, mutate_prev_scharr_stride_mul_overflow);
  run_from_pyramid_expect_range("next scharr stride mul overflow", kSize, kSize,
                                1, mutate_next_scharr_stride_mul_overflow);
  run_from_pyramid_expect_range("prev scharr inner mul overflow", kSize, kSize,
                                1,
                                mutate_prev_scharr_stride_mul_overflow_inner);
  run_from_pyramid_expect_range("next scharr inner mul overflow", kSize, kSize,
                                1,
                                mutate_next_scharr_stride_mul_overflow_inner);
}

TEST(CalcOpticalFlowPyrLk, FromPyramidRejectsPointScratchOverflows) {
  constexpr size_t kSize = 16;
  run_from_pyramid_expect_range("point scratch size overflow", kSize, kSize,
                                std::numeric_limits<size_t>::max());
  run_from_pyramid_expect_range("point scratch bytes overflow", kSize, kSize,
                                (std::numeric_limits<size_t>::max() / 8) + 1);
}

TEST(CalcOpticalFlowPyrLk, FromPyramidRejectsEffectiveMaxLevelOutOfIntRange) {
  constexpr size_t kSize = 16;
  run_from_pyramid_expect_range("effective max level out of int range", kSize,
                                kSize, 1,
                                mutate_effective_max_level_out_of_int_range);
}

TEST(CalcOpticalFlowPyrLk,
     FromPyramidRejectsPrevScharrStrideMulOverflowExactBranch) {
  constexpr size_t kSize = 16;
  constexpr size_t kChannels = 2;
  test::Array2D<uint8_t> prev_image(kSize * kChannels, kSize);
  test::Array2D<uint8_t> next_image(kSize * kChannels, kSize);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  build_matching_pyramids(kSize, kSize, kChannels, context, &prev_image,
                          &next_image, &prev_pyramid, &next_pyramid);

  mutate_prev_scharr_stride_overflow_exact(prev_pyramid, next_pyramid,
                                           &context);

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {1};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}

TEST(CalcOpticalFlowPyrLk,
     FromPyramidRejectsNextScharrStrideMulOverflowExactBranch) {
  constexpr size_t kSize = 16;
  constexpr size_t kChannels = 2;
  test::Array2D<uint8_t> prev_image(kSize * kChannels, kSize);
  test::Array2D<uint8_t> next_image(kSize * kChannels, kSize);
  ASSERT_TRUE(prev_image.valid());
  ASSERT_TRUE(next_image.valid());

  kleidicv_optflow_lk_context_t context = make_default_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  build_matching_pyramids(kSize, kSize, kChannels, context, &prev_image,
                          &next_image, &prev_pyramid, &next_pyramid);

  mutate_next_scharr_stride_overflow_exact(prev_pyramid, next_pyramid,
                                           &context);

  float prev_points[2] = {4.0F, 4.0F};
  float next_points[2] = {4.0F, 4.0F};
  uint8_t status[1] = {1};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid, next_pyramid, prev_points, next_points, 1, status,
                nullptr, context));

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid));
}
