// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "framework/array.h"
#include "kleidicv/kleidicv.h"

namespace {

template <typename T>
constexpr bool kIsFloatLike =
    std::is_floating_point_v<T> || std::is_same_v<T, float16_t>;

template <typename T>
T sample_value(size_t row, size_t col) {
  if constexpr (kIsFloatLike<T>) {
    const int v = static_cast<int>((row * 13 + col * 7) % 19) - 9;
    return static_cast<T>(0.25F * static_cast<float>(v));
  } else if constexpr (std::is_signed_v<T>) {
    const int v = static_cast<int>((row * 17 + col * 11) % 101) - 50;
    return static_cast<T>(v);
  } else {
    return static_cast<T>((row * 17 + col * 11) % 251);
  }
}

template <typename T>
void fill_array(test::Array2D<T>& array) {
  for (size_t row = 0; row < array.height(); ++row) {
    for (size_t col = 0; col < array.width(); ++col) {
      *array.at(row, col) = sample_value<T>(row, col);
    }
  }
}

template <typename T>
void expect_same_array(const test::Array2D<T>& lhs,
                       const test::Array2D<T>& rhs) {
  if constexpr (kIsFloatLike<T>) {
    EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(static_cast<T>(0.001F), lhs, rhs);
  } else {
    EXPECT_EQ_ARRAY2D(lhs, rhs);
  }
}

template <typename T>
using UnaryFn = kleidicv_error_t (*)(const T*, size_t, T*, size_t, size_t,
                                     size_t);

template <typename T>
using BinaryFn = kleidicv_error_t (*)(const T*, size_t, const T*, size_t, T*,
                                      size_t, size_t, size_t);

template <typename T>
using BinaryScaleFn = kleidicv_error_t (*)(const T*, size_t, const T*, size_t,
                                           T*, size_t, size_t, size_t, double);

template <typename InT, typename OutT, typename... ExtraArgs>
using UnaryExtraFn = kleidicv_error_t (*)(const InT*, size_t, OutT*, size_t,
                                          size_t, size_t, ExtraArgs...);

template <typename T>
void run_unary_pair(const char* name, UnaryFn<T> cpu_fn, UnaryFn<T> sme_fn,
                    size_t channels = 1) {
  constexpr size_t kWidth = 33;
  constexpr size_t kHeight = 5;
  test::Array2D<T> src{kWidth * channels, kHeight, 3};
  test::Array2D<T> cpu_dst{kWidth * channels, kHeight, 1};
  test::Array2D<T> sme_dst{kWidth * channels, kHeight, 1};
  fill_array(src);
  cpu_dst.fill(static_cast<T>(0));
  sme_dst.fill(static_cast<T>(0));

  EXPECT_EQ(KLEIDICV_OK, cpu_fn(src.data(), src.stride(), cpu_dst.data(),
                                cpu_dst.stride(), kWidth, kHeight))
      << name;
  EXPECT_EQ(KLEIDICV_OK, sme_fn(src.data(), src.stride(), sme_dst.data(),
                                sme_dst.stride(), kWidth, kHeight))
      << name;
  expect_same_array(cpu_dst, sme_dst);
}

template <typename T>
void run_binary_pair(const char* name, BinaryFn<T> cpu_fn, BinaryFn<T> sme_fn) {
  constexpr size_t kWidth = 33;
  constexpr size_t kHeight = 5;
  test::Array2D<T> src_a{kWidth, kHeight, 3};
  test::Array2D<T> src_b{kWidth, kHeight, 7};
  test::Array2D<T> cpu_dst{kWidth, kHeight, 1};
  test::Array2D<T> sme_dst{kWidth, kHeight, 1};
  fill_array(src_a);
  fill_array(src_b);
  cpu_dst.fill(static_cast<T>(0));
  sme_dst.fill(static_cast<T>(0));

  EXPECT_EQ(KLEIDICV_OK,
            cpu_fn(src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                   cpu_dst.data(), cpu_dst.stride(), kWidth, kHeight))
      << name;
  EXPECT_EQ(KLEIDICV_OK,
            sme_fn(src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                   sme_dst.data(), sme_dst.stride(), kWidth, kHeight))
      << name;
  expect_same_array(cpu_dst, sme_dst);
}

template <typename T>
void run_binary_with_scale_pair(const char* name, BinaryScaleFn<T> cpu_fn,
                                BinaryScaleFn<T> sme_fn) {
  constexpr size_t kWidth = 33;
  constexpr size_t kHeight = 5;
  constexpr double kScale = 0.75;
  test::Array2D<T> src_a{kWidth, kHeight, 3};
  test::Array2D<T> src_b{kWidth, kHeight, 7};
  test::Array2D<T> cpu_dst{kWidth, kHeight, 1};
  test::Array2D<T> sme_dst{kWidth, kHeight, 1};
  fill_array(src_a);
  fill_array(src_b);
  cpu_dst.fill(static_cast<T>(0));
  sme_dst.fill(static_cast<T>(0));

  EXPECT_EQ(KLEIDICV_OK,
            cpu_fn(src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                   cpu_dst.data(), cpu_dst.stride(), kWidth, kHeight, kScale))
      << name;
  EXPECT_EQ(KLEIDICV_OK,
            sme_fn(src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                   sme_dst.data(), sme_dst.stride(), kWidth, kHeight, kScale))
      << name;
  expect_same_array(cpu_dst, sme_dst);
}

// Source and destination might have different number of channels
template <typename InT, typename OutT, typename... ExtraArgs>
void run_diff_channels_op_pair(const char* name,
                               UnaryExtraFn<InT, OutT, ExtraArgs...> cpu_fn,
                               UnaryExtraFn<InT, OutT, ExtraArgs...> sme_fn,
                               size_t src_channels, size_t dst_channels,
                               ExtraArgs... extra_args) {
  constexpr size_t kWidth = 33;
  constexpr size_t kHeight = 6;
  test::Array2D<InT> src{kWidth * src_channels, kHeight, 3};
  test::Array2D<OutT> cpu_dst{kWidth * dst_channels, kHeight, 1};
  test::Array2D<OutT> sme_dst{kWidth * dst_channels, kHeight, 1};
  fill_array(src);
  cpu_dst.fill(static_cast<OutT>(0));
  sme_dst.fill(static_cast<OutT>(0));

  EXPECT_EQ(KLEIDICV_OK,
            cpu_fn(src.data(), src.stride(), cpu_dst.data(), cpu_dst.stride(),
                   kWidth, kHeight, extra_args...))
      << name;
  EXPECT_EQ(KLEIDICV_OK,
            sme_fn(src.data(), src.stride(), sme_dst.data(), sme_dst.stride(),
                   kWidth, kHeight, extra_args...))
      << name;
  expect_same_array(cpu_dst, sme_dst);
}

template <typename T>
using MinMaxFn = kleidicv_error_t (*)(const T*, size_t, size_t, size_t, T*, T*);

template <typename T>
void run_min_max_pair(const char* name, MinMaxFn<T> cpu_fn,
                      MinMaxFn<T> sme_fn) {
  constexpr size_t kWidth = 37;
  constexpr size_t kHeight = 4;
  test::Array2D<T> src{kWidth, kHeight, 3};
  fill_array(src);
  T min_cpu{}, max_cpu{}, min_sme{}, max_sme{};
  EXPECT_EQ(KLEIDICV_OK, cpu_fn(src.data(), src.stride(), kWidth, kHeight,
                                &min_cpu, &max_cpu))
      << name;
  EXPECT_EQ(KLEIDICV_OK, sme_fn(src.data(), src.stride(), kWidth, kHeight,
                                &min_sme, &max_sme))
      << name;
  if constexpr (kIsFloatLike<T>) {
    EXPECT_NEAR(min_cpu, min_sme, 1e-6F);
    EXPECT_NEAR(max_cpu, max_sme, 1e-6F);
  } else {
    EXPECT_EQ(min_cpu, min_sme);
    EXPECT_EQ(max_cpu, max_sme);
  }
}

struct OpticalFlowPyramidDeleter {
  void operator()(kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid) const {
    if (pyramid != nullptr) {
      static_cast<void>(kleidicv_optical_flow_pyr_lk_pyramid_release(pyramid));
    }
  }
};

using OpticalFlowPyramidPtr =
    std::unique_ptr<kleidicv_optical_flow_pyr_lk_pyramid_t,
                    OpticalFlowPyramidDeleter>;

void expect_same_optical_flow_pyr_lk_pyramid(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* cpu_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t* sme_pyramid,
    size_t channels) {
  size_t cpu_level_count = 0;
  size_t sme_level_count = 0;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
                             cpu_pyramid, &cpu_level_count));
  ASSERT_EQ(KLEIDICV_OK, kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
                             sme_pyramid, &sme_level_count));
  ASSERT_EQ(cpu_level_count, sme_level_count);

  for (size_t level = 0; level < cpu_level_count; ++level) {
    SCOPED_TRACE(::testing::Message() << "level=" << level);

    const uint8_t* cpu_image_data = nullptr;
    size_t cpu_image_stride = 0;
    size_t cpu_image_width = 0;
    size_t cpu_image_height = 0;
    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
                  cpu_pyramid, level, &cpu_image_data, &cpu_image_stride,
                  &cpu_image_width, &cpu_image_height));

    const uint8_t* sme_image_data = nullptr;
    size_t sme_image_stride = 0;
    size_t sme_image_width = 0;
    size_t sme_image_height = 0;
    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
                  sme_pyramid, level, &sme_image_data, &sme_image_stride,
                  &sme_image_width, &sme_image_height));

    ASSERT_EQ(cpu_image_stride, sme_image_stride);
    ASSERT_EQ(cpu_image_width, sme_image_width);
    ASSERT_EQ(cpu_image_height, sme_image_height);
    for (size_t y = 0; y < cpu_image_height; ++y) {
      const uint8_t* cpu_row = cpu_image_data + y * cpu_image_stride;
      const uint8_t* sme_row = sme_image_data + y * sme_image_stride;
      for (size_t x = 0; x < cpu_image_width * channels; ++x) {
        ASSERT_EQ(cpu_row[x], sme_row[x]);
      }
    }

    const int16_t* cpu_scharr_data = nullptr;
    size_t cpu_scharr_stride = 0;
    size_t cpu_scharr_width = 0;
    size_t cpu_scharr_height = 0;
    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
                  cpu_pyramid, level, &cpu_scharr_data, &cpu_scharr_stride,
                  &cpu_scharr_width, &cpu_scharr_height));

    const int16_t* sme_scharr_data = nullptr;
    size_t sme_scharr_stride = 0;
    size_t sme_scharr_width = 0;
    size_t sme_scharr_height = 0;
    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
                  sme_pyramid, level, &sme_scharr_data, &sme_scharr_stride,
                  &sme_scharr_width, &sme_scharr_height));

    ASSERT_EQ(cpu_scharr_stride, sme_scharr_stride);
    ASSERT_EQ(cpu_scharr_width, sme_scharr_width);
    ASSERT_EQ(cpu_scharr_height, sme_scharr_height);
    const size_t scharr_row_elements = cpu_scharr_width * channels * 2;
    for (size_t y = 0; y < cpu_scharr_height; ++y) {
      const int16_t* cpu_row = reinterpret_cast<const int16_t*>(
          reinterpret_cast<const uint8_t*>(cpu_scharr_data) +
          y * cpu_scharr_stride);
      const int16_t* sme_row = reinterpret_cast<const int16_t*>(
          reinterpret_cast<const uint8_t*>(sme_scharr_data) +
          y * sme_scharr_stride);
      for (size_t x = 0; x < scharr_row_elements; ++x) {
        ASSERT_EQ(cpu_row[x], sme_row[x]);
      }
    }
  }
}

kleidicv_optflow_lk_context_t make_default_optical_flow_pyr_lk_context() {
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

std::vector<float> make_optical_flow_points(size_t point_count, size_t width,
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

void expect_same_optical_flow_result(const std::vector<float>& cpu_next_points,
                                     const std::vector<float>& sme_next_points,
                                     const std::vector<uint8_t>& cpu_status,
                                     const std::vector<uint8_t>& sme_status,
                                     const std::vector<float>& cpu_err,
                                     const std::vector<float>& sme_err) {
  ASSERT_EQ(cpu_next_points.size(), sme_next_points.size());
  ASSERT_EQ(cpu_status.size(), sme_status.size());
  ASSERT_EQ(cpu_err.size(), sme_err.size());
  EXPECT_EQ(cpu_status, sme_status);

  for (size_t i = 0; i < cpu_next_points.size(); ++i) {
    EXPECT_NEAR(cpu_next_points[i], sme_next_points[i], 1e-6F)
        << "next_points index=" << i;
  }
  for (size_t i = 0; i < cpu_err.size(); ++i) {
    EXPECT_NEAR(cpu_err[i], sme_err[i], 1e-5F) << "err index=" << i;
  }
}

}  // namespace

TEST(SmeApi, BinaryOpParity) {
  run_binary_pair<int8_t>("saturating_add_s8", kleidicv_saturating_add_s8,
                          kleidicv_saturating_add_s8_sme);
  run_binary_pair<uint8_t>("saturating_add_u8", kleidicv_saturating_add_u8,
                           kleidicv_saturating_add_u8_sme);
  run_binary_pair<int16_t>("saturating_add_s16", kleidicv_saturating_add_s16,
                           kleidicv_saturating_add_s16_sme);
  run_binary_pair<uint16_t>("saturating_add_u16", kleidicv_saturating_add_u16,
                            kleidicv_saturating_add_u16_sme);
  run_binary_pair<int32_t>("saturating_add_s32", kleidicv_saturating_add_s32,
                           kleidicv_saturating_add_s32_sme);
  run_binary_pair<uint32_t>("saturating_add_u32", kleidicv_saturating_add_u32,
                            kleidicv_saturating_add_u32_sme);
  run_binary_pair<int64_t>("saturating_add_s64", kleidicv_saturating_add_s64,
                           kleidicv_saturating_add_s64_sme);
  run_binary_pair<uint64_t>("saturating_add_u64", kleidicv_saturating_add_u64,
                            kleidicv_saturating_add_u64_sme);

  run_binary_pair<int8_t>("saturating_sub_s8", kleidicv_saturating_sub_s8,
                          kleidicv_saturating_sub_s8_sme);
  run_binary_pair<uint8_t>("saturating_sub_u8", kleidicv_saturating_sub_u8,
                           kleidicv_saturating_sub_u8_sme);
  run_binary_pair<int16_t>("saturating_sub_s16", kleidicv_saturating_sub_s16,
                           kleidicv_saturating_sub_s16_sme);
  run_binary_pair<uint16_t>("saturating_sub_u16", kleidicv_saturating_sub_u16,
                            kleidicv_saturating_sub_u16_sme);
  run_binary_pair<int32_t>("saturating_sub_s32", kleidicv_saturating_sub_s32,
                           kleidicv_saturating_sub_s32_sme);
  run_binary_pair<uint32_t>("saturating_sub_u32", kleidicv_saturating_sub_u32,
                            kleidicv_saturating_sub_u32_sme);
  run_binary_pair<int64_t>("saturating_sub_s64", kleidicv_saturating_sub_s64,
                           kleidicv_saturating_sub_s64_sme);
  run_binary_pair<uint64_t>("saturating_sub_u64", kleidicv_saturating_sub_u64,
                            kleidicv_saturating_sub_u64_sme);

  run_binary_pair<uint8_t>("saturating_absdiff_u8",
                           kleidicv_saturating_absdiff_u8,
                           kleidicv_saturating_absdiff_u8_sme);
  run_binary_pair<int8_t>("saturating_absdiff_s8",
                          kleidicv_saturating_absdiff_s8,
                          kleidicv_saturating_absdiff_s8_sme);
  run_binary_pair<uint16_t>("saturating_absdiff_u16",
                            kleidicv_saturating_absdiff_u16,
                            kleidicv_saturating_absdiff_u16_sme);
  run_binary_pair<int16_t>("saturating_absdiff_s16",
                           kleidicv_saturating_absdiff_s16,
                           kleidicv_saturating_absdiff_s16_sme);
  run_binary_pair<int32_t>("saturating_absdiff_s32",
                           kleidicv_saturating_absdiff_s32,
                           kleidicv_saturating_absdiff_s32_sme);

  run_binary_pair<uint8_t>("bitwise_and", kleidicv_bitwise_and,
                           kleidicv_bitwise_and_sme);
  run_binary_pair<uint8_t>("compare_equal_u8", kleidicv_compare_equal_u8,
                           kleidicv_compare_equal_u8_sme);
  run_binary_pair<uint8_t>("compare_greater_u8", kleidicv_compare_greater_u8,
                           kleidicv_compare_greater_u8_sme);
}

TEST(SmeApi, MultiplyParity) {
  run_binary_with_scale_pair<uint8_t>("saturating_multiply_u8",
                                      kleidicv_saturating_multiply_u8,
                                      kleidicv_saturating_multiply_u8_sme);
  run_binary_with_scale_pair<int8_t>("saturating_multiply_s8",
                                     kleidicv_saturating_multiply_s8,
                                     kleidicv_saturating_multiply_s8_sme);
  run_binary_with_scale_pair<uint16_t>("saturating_multiply_u16",
                                       kleidicv_saturating_multiply_u16,
                                       kleidicv_saturating_multiply_u16_sme);
  run_binary_with_scale_pair<int16_t>("saturating_multiply_s16",
                                      kleidicv_saturating_multiply_s16,
                                      kleidicv_saturating_multiply_s16_sme);
  run_binary_with_scale_pair<int32_t>("saturating_multiply_s32",
                                      kleidicv_saturating_multiply_s32,
                                      kleidicv_saturating_multiply_s32_sme);
}

TEST(SmeApi, MiscArithmeticConversionParity) {
  constexpr size_t kWidth = 33;
  constexpr size_t kHeight = 6;
  test::Array2D<int16_t> src_a{kWidth, kHeight, 3};
  test::Array2D<int16_t> src_b{kWidth, kHeight, 1};
  test::Array2D<int16_t> cpu_dst{kWidth, kHeight, 1};
  test::Array2D<int16_t> sme_dst{kWidth, kHeight, 1};
  fill_array(src_a);
  fill_array(src_b);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                cpu_dst.data(), cpu_dst.stride(), kWidth, kHeight, 27));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_saturating_add_abs_with_threshold_s16_sme(
                src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                sme_dst.data(), sme_dst.stride(), kWidth, kHeight, 27));
  expect_same_array(cpu_dst, sme_dst);

  run_unary_pair<float>("exp_f32", kleidicv_exp_f32, kleidicv_exp_f32_sme);

  run_diff_channels_op_pair<uint8_t, uint8_t, uint8_t, uint8_t>(
      "threshold_binary_u8", kleidicv_threshold_binary_u8,
      kleidicv_threshold_binary_u8_sme, 1, 1, 100, 200);
  run_diff_channels_op_pair<uint8_t, uint8_t, uint8_t, uint8_t>(
      "in_range_u8", kleidicv_in_range_u8, kleidicv_in_range_u8_sme, 1, 1, 32,
      180);
  run_diff_channels_op_pair<float, uint8_t, float, float>(
      "in_range_f32", kleidicv_in_range_f32, kleidicv_in_range_f32_sme, 1, 1,
      -1.0F, 1.0F);

  run_diff_channels_op_pair<uint8_t, float16_t, double, double>(
      "scale_u8_f16", kleidicv_scale_u8_f16, kleidicv_scale_u8_f16_sme, 1, 1,
      0.75, 3.0);
  run_diff_channels_op_pair<float, float, double, double>(
      "scale_f32", kleidicv_scale_f32, kleidicv_scale_f32_sme, 1, 1, 1.125,
      -0.25);

  run_diff_channels_op_pair<float, int8_t>("f32_to_s8", kleidicv_f32_to_s8,
                                           kleidicv_f32_to_s8_sme, 1, 1);
  run_diff_channels_op_pair<float, uint8_t>("f32_to_u8", kleidicv_f32_to_u8,
                                            kleidicv_f32_to_u8_sme, 1, 1);
  run_diff_channels_op_pair<int8_t, float>("s8_to_f32", kleidicv_s8_to_f32,
                                           kleidicv_s8_to_f32_sme, 1, 1);
  run_diff_channels_op_pair<uint8_t, float>("u8_to_f32", kleidicv_u8_to_f32,
                                            kleidicv_u8_to_f32_sme, 1, 1);
}

TEST(SmeApi, GrayRgbConversionParity) {
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "gray_to_rgb_u8", kleidicv_gray_to_rgb_u8, kleidicv_gray_to_rgb_u8_sme, 1,
      3);
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "gray_to_rgba_u8", kleidicv_gray_to_rgba_u8, kleidicv_gray_to_rgba_u8_sme,
      1, 4);

  run_diff_channels_op_pair<uint8_t, uint8_t>("rgb_to_bgr_u8",
                                              kleidicv_rgb_to_bgr_u8,
                                              kleidicv_rgb_to_bgr_u8_sme, 3, 3);
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "rgba_to_bgra_u8", kleidicv_rgba_to_bgra_u8, kleidicv_rgba_to_bgra_u8_sme,
      4, 4);
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "rgb_to_bgra_u8", kleidicv_rgb_to_bgra_u8, kleidicv_rgb_to_bgra_u8_sme, 3,
      4);
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "rgb_to_rgba_u8", kleidicv_rgb_to_rgba_u8, kleidicv_rgb_to_rgba_u8_sme, 3,
      4);
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "rgba_to_bgr_u8", kleidicv_rgba_to_bgr_u8, kleidicv_rgba_to_bgr_u8_sme, 4,
      3);
  run_diff_channels_op_pair<uint8_t, uint8_t>(
      "rgba_to_rgb_u8", kleidicv_rgba_to_rgb_u8, kleidicv_rgba_to_rgb_u8_sme, 4,
      3);
}

TEST(SmeApi, YuvRgbConversionParity) {
  constexpr size_t kWidth = 34;
  constexpr size_t kHeight = 8;

  test::Array2D<uint8_t> yuv444{kWidth * 3, kHeight, 3};
  test::Array2D<uint8_t> rgb_cpu{kWidth * 3, kHeight, 1};
  test::Array2D<uint8_t> rgb_sme{kWidth * 3, kHeight, 1};
  fill_array(yuv444);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_to_rgb_u8(yuv444.data(), yuv444.stride(),
                                   rgb_cpu.data(), rgb_cpu.stride(), kWidth,
                                   kHeight, KLEIDICV_YUV444_TO_RGB));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_to_rgb_u8_sme(yuv444.data(), yuv444.stride(),
                                       rgb_sme.data(), rgb_sme.stride(), kWidth,
                                       kHeight, KLEIDICV_YUV444_TO_RGB));
  expect_same_array(rgb_cpu, rgb_sme);

  // Drive YUV420P branch in yuv_to_rgb_api.cpp
  test::Array2D<uint8_t> yuv420p{kWidth, (kHeight * 3 + 1) / 2, 3};
  test::Array2D<uint8_t> rgb420p_cpu{kWidth * 3, kHeight, 1};
  test::Array2D<uint8_t> rgb420p_sme{kWidth * 3, kHeight, 1};
  fill_array(yuv420p);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_to_rgb_u8(yuv420p.data(), yuv420p.stride(),
                                   rgb420p_cpu.data(), rgb420p_cpu.stride(),
                                   kWidth, kHeight, KLEIDICV_IYUV_TO_RGB));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_to_rgb_u8_sme(yuv420p.data(), yuv420p.stride(),
                                       rgb420p_sme.data(), rgb420p_sme.stride(),
                                       kWidth, kHeight, KLEIDICV_IYUV_TO_RGB));
  expect_same_array(rgb420p_cpu, rgb420p_sme);

  // Drive YUV422 branch in yuv_to_rgb_api.cpp
  test::Array2D<uint8_t> yuv422{kWidth * 2, kHeight, 3};
  test::Array2D<uint8_t> rgb422_cpu{kWidth * 3, kHeight, 1};
  test::Array2D<uint8_t> rgb422_sme{kWidth * 3, kHeight, 1};
  fill_array(yuv422);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_to_rgb_u8(yuv422.data(), yuv422.stride(),
                                   rgb422_cpu.data(), rgb422_cpu.stride(),
                                   kWidth, kHeight, KLEIDICV_YUYV_TO_RGB));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_to_rgb_u8_sme(yuv422.data(), yuv422.stride(),
                                       rgb422_sme.data(), rgb422_sme.stride(),
                                       kWidth, kHeight, KLEIDICV_YUYV_TO_RGB));
  expect_same_array(rgb422_cpu, rgb422_sme);

  test::Array2D<uint8_t> y_plane{kWidth, kHeight, 1};
  test::Array2D<uint8_t> uv_plane{kWidth, kHeight / 2, 1};
  test::Array2D<uint8_t> rgb_nv_cpu{kWidth * 3, kHeight, 1};
  test::Array2D<uint8_t> rgb_nv_sme{kWidth * 3, kHeight, 1};
  fill_array(y_plane);
  fill_array(uv_plane);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_semiplanar_to_rgb_u8(
                y_plane.data(), y_plane.stride(), uv_plane.data(),
                uv_plane.stride(), rgb_nv_cpu.data(), rgb_nv_cpu.stride(),
                kWidth, kHeight, KLEIDICV_NV12_TO_RGB));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_yuv_semiplanar_to_rgb_u8_sme(
                y_plane.data(), y_plane.stride(), uv_plane.data(),
                uv_plane.stride(), rgb_nv_sme.data(), rgb_nv_sme.stride(),
                kWidth, kHeight, KLEIDICV_NV12_TO_RGB));
  expect_same_array(rgb_nv_cpu, rgb_nv_sme);

  test::Array2D<uint8_t> rgb_src{kWidth * 3, kHeight, 3};
  test::Array2D<uint8_t> yuv_cpu{kWidth * 3, kHeight, 1};
  test::Array2D<uint8_t> yuv_sme{kWidth * 3, kHeight, 1};
  fill_array(rgb_src);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_rgb_to_yuv_u8(rgb_src.data(), rgb_src.stride(),
                                   yuv_cpu.data(), yuv_cpu.stride(), kWidth,
                                   kHeight, KLEIDICV_RGB_TO_YUV444));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_rgb_to_yuv_u8_sme(rgb_src.data(), rgb_src.stride(),
                                       yuv_sme.data(), yuv_sme.stride(), kWidth,
                                       kHeight, KLEIDICV_RGB_TO_YUV444));
  expect_same_array(yuv_cpu, yuv_sme);

  test::Array2D<uint8_t> y_dst_cpu{kWidth, kHeight, 1};
  test::Array2D<uint8_t> y_dst_sme{kWidth, kHeight, 1};
  test::Array2D<uint8_t> uv_dst_cpu{kWidth, kHeight / 2, 1};
  test::Array2D<uint8_t> uv_dst_sme{kWidth, kHeight / 2, 1};
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_rgb_to_yuv_semiplanar_u8(
                rgb_src.data(), rgb_src.stride(), y_dst_cpu.data(),
                y_dst_cpu.stride(), uv_dst_cpu.data(), uv_dst_cpu.stride(),
                kWidth, kHeight, KLEIDICV_RGB_TO_NV12));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_rgb_to_yuv_semiplanar_u8_sme(
                rgb_src.data(), rgb_src.stride(), y_dst_sme.data(),
                y_dst_sme.stride(), uv_dst_sme.data(), uv_dst_sme.stride(),
                kWidth, kHeight, KLEIDICV_RGB_TO_NV12));
  expect_same_array(y_dst_cpu, y_dst_sme);
  expect_same_array(uv_dst_cpu, uv_dst_sme);
}

TEST(SmeApi, MorphologyAndFilterParity) {
  constexpr size_t kWidth = 17;
  constexpr size_t kHeight = 10;
  constexpr size_t kChannels = 3;

  test::Array2D<uint8_t> src{kWidth * kChannels, kHeight, 3};
  test::Array2D<uint8_t> dilate_cpu{kWidth * kChannels, kHeight, 1};
  test::Array2D<uint8_t> dilate_sme{kWidth * kChannels, kHeight, 1};
  fill_array(src);
  const std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT> border_value{17, 31,
                                                                         55, 0};

  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_dilate_u8(src.data(), src.stride(), dilate_cpu.data(),
                               dilate_cpu.stride(), kWidth, kHeight, kChannels,
                               3, 3, 1, 1, KLEIDICV_BORDER_TYPE_REPLICATE,
                               border_value.data(), 1));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_dilate_u8_sme(
                src.data(), src.stride(), dilate_sme.data(),
                dilate_sme.stride(), kWidth, kHeight, kChannels, 3, 3, 1, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, border_value.data(), 1));
  expect_same_array(dilate_cpu, dilate_sme);

  test::Array2D<uint8_t> erode_cpu{kWidth * kChannels, kHeight, 1};
  test::Array2D<uint8_t> erode_sme{kWidth * kChannels, kHeight, 1};
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_erode_u8(src.data(), src.stride(), erode_cpu.data(),
                              erode_cpu.stride(), kWidth, kHeight, kChannels, 3,
                              3, 1, 1, KLEIDICV_BORDER_TYPE_REPLICATE,
                              border_value.data(), 1));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_erode_u8_sme(
                src.data(), src.stride(), erode_sme.data(), erode_sme.stride(),
                kWidth, kHeight, kChannels, 3, 3, 1, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, border_value.data(), 1));
  expect_same_array(erode_cpu, erode_sme);

  constexpr std::array<uint8_t, 5> kKernelU8{5, 0, 1, 2, 2};
  test::Array2D<uint8_t> sep_u8_cpu{kWidth, kHeight, 1};
  test::Array2D<uint8_t> sep_u8_sme{kWidth, kHeight, 1};
  test::Array2D<uint8_t> sep_u8_src{kWidth, kHeight, 3};
  fill_array(sep_u8_src);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_separable_filter_2d_u8(
                sep_u8_src.data(), sep_u8_src.stride(), sep_u8_cpu.data(),
                sep_u8_cpu.stride(), kWidth, kHeight, 1, kKernelU8.data(),
                kKernelU8.size(), kKernelU8.data(), kKernelU8.size(),
                KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_separable_filter_2d_u8_sme(
                sep_u8_src.data(), sep_u8_src.stride(), sep_u8_sme.data(),
                sep_u8_sme.stride(), kWidth, kHeight, 1, kKernelU8.data(),
                kKernelU8.size(), kKernelU8.data(), kKernelU8.size(),
                KLEIDICV_BORDER_TYPE_REPLICATE));
  expect_same_array(sep_u8_cpu, sep_u8_sme);

  constexpr std::array<uint16_t, 5> kKernelU16{5, 0, 1, 2, 2};
  test::Array2D<uint16_t> sep_u16_src{kWidth, kHeight, 3};
  test::Array2D<uint16_t> sep_u16_cpu{kWidth, kHeight, 1};
  test::Array2D<uint16_t> sep_u16_sme{kWidth, kHeight, 1};
  fill_array(sep_u16_src);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_separable_filter_2d_u16(
                sep_u16_src.data(), sep_u16_src.stride(), sep_u16_cpu.data(),
                sep_u16_cpu.stride(), kWidth, kHeight, 1, kKernelU16.data(),
                kKernelU16.size(), kKernelU16.data(), kKernelU16.size(),
                KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_separable_filter_2d_u16_sme(
                sep_u16_src.data(), sep_u16_src.stride(), sep_u16_sme.data(),
                sep_u16_sme.stride(), kWidth, kHeight, 1, kKernelU16.data(),
                kKernelU16.size(), kKernelU16.data(), kKernelU16.size(),
                KLEIDICV_BORDER_TYPE_REPLICATE));
  expect_same_array(sep_u16_cpu, sep_u16_sme);

  test::Array2D<uint8_t> gauss_cpu{kWidth, kHeight, 1};
  test::Array2D<uint8_t> gauss_sme{kWidth, kHeight, 1};
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_gaussian_blur_u8(sep_u8_src.data(), sep_u8_src.stride(),
                                      gauss_cpu.data(), gauss_cpu.stride(),
                                      kWidth, kHeight, 1, 3, 3, 1.0F, 1.0F,
                                      KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_gaussian_blur_u8_sme(
                sep_u8_src.data(), sep_u8_src.stride(), gauss_sme.data(),
                gauss_sme.stride(), kWidth, kHeight, 1, 3, 3, 1.0F, 1.0F,
                KLEIDICV_BORDER_TYPE_REPLICATE));
  expect_same_array(gauss_cpu, gauss_sme);

  constexpr size_t kDownsampleSrcWidth = 45;
  constexpr size_t kDownsampleSrcHeight = 12;
  test::Array2D<uint8_t> down_src{kDownsampleSrcWidth, kDownsampleSrcHeight, 3};
  test::Array2D<uint8_t> down_cpu{(kDownsampleSrcWidth + 1) / 2,
                                  (kDownsampleSrcHeight + 1) / 2, 1};
  test::Array2D<uint8_t> down_sme{(kDownsampleSrcWidth + 1) / 2,
                                  (kDownsampleSrcHeight + 1) / 2, 1};
  fill_array(down_src);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_blur_and_downsample_u8(
                down_src.data(), down_src.stride(), kDownsampleSrcWidth,
                kDownsampleSrcHeight, down_cpu.data(), down_cpu.stride(), 1,
                KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_blur_and_downsample_u8_sme(
                down_src.data(), down_src.stride(), kDownsampleSrcWidth,
                kDownsampleSrcHeight, down_sme.data(), down_sme.stride(), 1,
                KLEIDICV_BORDER_TYPE_REPLICATE));
  expect_same_array(down_cpu, down_sme);
}

TEST(SmeApi, ResizeParity) {
  auto run_resize_u8_case = [](const char* name, size_t src_width,
                               size_t src_height, size_t dst_width,
                               size_t dst_height, size_t channels) {
    test::Array2D<uint8_t> src{src_width * channels, src_height, 3};
    test::Array2D<uint8_t> cpu_dst{dst_width * channels, dst_height, 1};
    test::Array2D<uint8_t> sme_dst{dst_width * channels, dst_height, 1};
    fill_array(src);

    EXPECT_EQ(KLEIDICV_OK, kleidicv_resize_linear_u8(
                               src.data(), src.stride(), src_width, src_height,
                               cpu_dst.data(), cpu_dst.stride(), dst_width,
                               dst_height, channels))
        << name;
    EXPECT_EQ(KLEIDICV_OK, kleidicv_resize_linear_u8_sme(
                               src.data(), src.stride(), src_width, src_height,
                               sme_dst.data(), sme_dst.stride(), dst_width,
                               dst_height, channels))
        << name;
    expect_same_array(cpu_dst, sme_dst);
  };

  // Drive all kUseSME branches in resize_linear_api.cpp
  run_resize_u8_case("ch1_quarter", 10, 10, 5, 5, 1);
  run_resize_u8_case("ch1_2x2", 9, 7, 18, 14, 1);
  run_resize_u8_case("ch1_4x4", 8, 8, 32, 32, 1);
  run_resize_u8_case("ch1_r2", 16, 16, 9, 9, 1);
  run_resize_u8_case("ch1_r3", 32, 16, 12, 9, 1);
  run_resize_u8_case("ch2_r2", 16, 16, 9, 9, 2);
  run_resize_u8_case("ch2_r3", 32, 16, 12, 9, 2);
  run_resize_u8_case("ch3_r2", 32, 20, 18, 12, 3);
  run_resize_u8_case("ch3_r3", 32, 20, 14, 12, 3);

  constexpr size_t kSrcWidth = 4;
  constexpr size_t kSrcHeight = 4;
  constexpr size_t kDstWidth = 8;
  constexpr size_t kDstHeight = 8;
  test::Array2D<float> resize_src_f32{kSrcWidth, kSrcHeight, 3};
  test::Array2D<float> resize_cpu_f32{kDstWidth, kDstHeight, 1};
  test::Array2D<float> resize_sme_f32{kDstWidth, kDstHeight, 1};
  fill_array(resize_src_f32);
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear_f32(
                resize_src_f32.data(), resize_src_f32.stride(), kSrcWidth,
                kSrcHeight, resize_cpu_f32.data(), resize_cpu_f32.stride(),
                kDstWidth, kDstHeight, 1));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear_f32_sme(
                resize_src_f32.data(), resize_src_f32.stride(), kSrcWidth,
                kSrcHeight, resize_sme_f32.data(), resize_sme_f32.stride(),
                kDstWidth, kDstHeight, 1));
  expect_same_array(resize_cpu_f32, resize_sme_f32);
}

TEST(SmeApi, GradientParity) {
  constexpr size_t kSobelWidth = 43;
  constexpr size_t kSobelHeight = 9;
  constexpr size_t kSobelChannels = 2;
  test::Array2D<uint8_t> grad_src{kSobelWidth * kSobelChannels, kSobelHeight,
                                  3};
  test::Array2D<int16_t> sobel_v_cpu{kSobelWidth * kSobelChannels, kSobelHeight,
                                     1};
  test::Array2D<int16_t> sobel_v_sme{kSobelWidth * kSobelChannels, kSobelHeight,
                                     1};
  test::Array2D<int16_t> sobel_h_cpu{kSobelWidth * kSobelChannels, kSobelHeight,
                                     1};
  test::Array2D<int16_t> sobel_h_sme{kSobelWidth * kSobelChannels, kSobelHeight,
                                     1};
  fill_array(grad_src);
  EXPECT_EQ(KLEIDICV_OK, kleidicv_sobel_3x3_vertical_s16_u8(
                             grad_src.data(), grad_src.stride(),
                             sobel_v_cpu.data(), sobel_v_cpu.stride(),
                             kSobelWidth, kSobelHeight, kSobelChannels));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_sobel_3x3_vertical_s16_u8_sme(
                             grad_src.data(), grad_src.stride(),
                             sobel_v_sme.data(), sobel_v_sme.stride(),
                             kSobelWidth, kSobelHeight, kSobelChannels));
  expect_same_array(sobel_v_cpu, sobel_v_sme);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sobel_3x3_horizontal_s16_u8(
                             grad_src.data(), grad_src.stride(),
                             sobel_h_cpu.data(), sobel_h_cpu.stride(),
                             kSobelWidth, kSobelHeight, kSobelChannels));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_sobel_3x3_horizontal_s16_u8_sme(
                             grad_src.data(), grad_src.stride(),
                             sobel_h_sme.data(), sobel_h_sme.stride(),
                             kSobelWidth, kSobelHeight, kSobelChannels));
  expect_same_array(sobel_h_cpu, sobel_h_sme);

  constexpr size_t kScharrWidth = 45;
  constexpr size_t kScharrHeight = 7;
  constexpr size_t kScharrChannels = 2;
  test::Array2D<uint8_t> scharr_src{kScharrWidth * kScharrChannels,
                                    kScharrHeight, 3};
  // I case of kleidicv_scharr_interleaved_s16_u8 the result's width and height
  // are smaller than the input.
  test::Array2D<int16_t> scharr_cpu{(kScharrWidth - 2) * kScharrChannels * 2,
                                    kScharrHeight - 2, 1};
  test::Array2D<int16_t> scharr_sme{(kScharrWidth - 2) * kScharrChannels * 2,
                                    kScharrHeight - 2, 1};
  fill_array(scharr_src);
  EXPECT_EQ(KLEIDICV_OK, kleidicv_scharr_interleaved_s16_u8(
                             scharr_src.data(), scharr_src.stride(),
                             kScharrWidth, kScharrHeight, kScharrChannels,
                             scharr_cpu.data(), scharr_cpu.stride()));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_scharr_interleaved_s16_u8_sme(
                             scharr_src.data(), scharr_src.stride(),
                             kScharrWidth, kScharrHeight, kScharrChannels,
                             scharr_sme.data(), scharr_sme.stride()));
  expect_same_array(scharr_cpu, scharr_sme);
}

TEST(SmeApi, MinMaxSumParity) {
  run_min_max_pair<uint8_t>("min_max_u8", kleidicv_min_max_u8,
                            kleidicv_min_max_u8_sme);
  run_min_max_pair<int8_t>("min_max_s8", kleidicv_min_max_s8,
                           kleidicv_min_max_s8_sme);
  run_min_max_pair<uint16_t>("min_max_u16", kleidicv_min_max_u16,
                             kleidicv_min_max_u16_sme);
  run_min_max_pair<int16_t>("min_max_s16", kleidicv_min_max_s16,
                            kleidicv_min_max_s16_sme);
  run_min_max_pair<int32_t>("min_max_s32", kleidicv_min_max_s32,
                            kleidicv_min_max_s32_sme);
  run_min_max_pair<float>("min_max_f32", kleidicv_min_max_f32,
                          kleidicv_min_max_f32_sme);

  constexpr size_t kWidth = 45;
  constexpr size_t kHeight = 7;
  test::Array2D<float> sum_src{kWidth, kHeight, 3};
  fill_array(sum_src);
  float cpu_sum = 0.F;
  float sme_sum = 0.F;
  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(sum_src.data(), sum_src.stride(),
                                          kWidth, kHeight, &cpu_sum));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32_sme(sum_src.data(), sum_src.stride(),
                                              kWidth, kHeight, &sme_sum));
  EXPECT_NEAR(cpu_sum, sme_sum, 1e-5F);
}

TEST(SmeApi, MedianParity) {
  auto run_median = [](auto cpu_fn, auto sme_fn, auto sample_fn) {
    using T = decltype(sample_fn(size_t{0}, size_t{0}));
    constexpr size_t kMedianWidth = 11;
    constexpr size_t kMedianHeight = 8;
    test::Array2D<T> src{kMedianWidth, kMedianHeight, 3};
    test::Array2D<T> cpu_dst{kMedianWidth, kMedianHeight, 1};
    test::Array2D<T> sme_dst{kMedianWidth, kMedianHeight, 1};
    for (size_t row = 0; row < src.height(); ++row) {
      for (size_t col = 0; col < src.width(); ++col) {
        *src.at(row, col) = sample_fn(row, col);
      }
    }
    ASSERT_EQ(KLEIDICV_OK, cpu_fn(src.data(), src.stride(), cpu_dst.data(),
                                  cpu_dst.stride(), kMedianWidth, kMedianHeight,
                                  1, 3, 3, KLEIDICV_BORDER_TYPE_REPLICATE));
    ASSERT_EQ(KLEIDICV_OK, sme_fn(src.data(), src.stride(), sme_dst.data(),
                                  sme_dst.stride(), kMedianWidth, kMedianHeight,
                                  1, 3, 3, KLEIDICV_BORDER_TYPE_REPLICATE));
    expect_same_array(cpu_dst, sme_dst);
  };

  run_median(kleidicv_median_blur_u8, kleidicv_median_blur_u8_sme,
             sample_value<uint8_t>);
  run_median(kleidicv_median_blur_s16, kleidicv_median_blur_s16_sme,
             sample_value<int16_t>);
  run_median(kleidicv_median_blur_u16, kleidicv_median_blur_u16_sme,
             sample_value<uint16_t>);
  run_median(kleidicv_median_blur_f32, kleidicv_median_blur_f32_sme,
             sample_value<float>);
  run_median(kleidicv_median_blur_s8, kleidicv_median_blur_s8_sme,
             sample_value<int8_t>);
  run_median(kleidicv_median_blur_s32, kleidicv_median_blur_s32_sme,
             sample_value<int32_t>);
  run_median(kleidicv_median_blur_u32, kleidicv_median_blur_u32_sme,
             sample_value<uint32_t>);
}

TEST(SmeApi, OpticalFlowPyrLkFromPyramidParity) {
  constexpr size_t kWidth = 45;
  constexpr size_t kHeight = 24;
  constexpr size_t kChannels = 1;
  constexpr size_t kPointCount = 19;

  test::Array2D<uint8_t> prev_image{kWidth * kChannels, kHeight, 3};
  test::Array2D<uint8_t> next_image{kWidth * kChannels, kHeight, 5};
  fill_array(prev_image);
  fill_array(next_image);

  const kleidicv_optflow_lk_context_t context =
      make_default_optical_flow_pyr_lk_context();
  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_raw = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_raw = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &prev_raw, prev_image.data(), prev_image.stride(),
                             kWidth, kHeight, kChannels, context.max_level + 1,
                             context.window_width, context.window_height));
  ASSERT_EQ(KLEIDICV_OK, kleidicv_build_optical_flow_pyr_lk_pyramid(
                             &next_raw, next_image.data(), next_image.stride(),
                             kWidth, kHeight, kChannels, context.max_level + 1,
                             context.window_width, context.window_height));
  ASSERT_NE(nullptr, prev_raw);
  ASSERT_NE(nullptr, next_raw);

  const OpticalFlowPyramidPtr prev_pyramid{prev_raw};
  const OpticalFlowPyramidPtr next_pyramid{next_raw};
  const std::vector<float> prev_points =
      make_optical_flow_points(kPointCount, kWidth, kHeight);
  std::vector<float> cpu_next_points(prev_points);
  std::vector<float> sme_next_points(prev_points);
  std::vector<uint8_t> cpu_status(kPointCount, 0);
  std::vector<uint8_t> sme_status(kPointCount, 0);
  std::vector<float> cpu_err(kPointCount, -1.0F);
  std::vector<float> sme_err(kPointCount, -1.0F);

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
                prev_pyramid.get(), next_pyramid.get(), prev_points.data(),
                cpu_next_points.data(), kPointCount, cpu_status.data(),
                cpu_err.data(), context));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_u8_from_pyramid_sme(
                prev_pyramid.get(), next_pyramid.get(), prev_points.data(),
                sme_next_points.data(), kPointCount, sme_status.data(),
                sme_err.data(), context));

  expect_same_optical_flow_result(cpu_next_points, sme_next_points, cpu_status,
                                  sme_status, cpu_err, sme_err);
}

TEST(SmeApi, OpticalFlowPyrLkFromImageParity) {
  constexpr size_t kWidth = 45;
  constexpr size_t kHeight = 24;
  constexpr size_t kChannels = 1;
  constexpr size_t kPointCount = 19;

  test::Array2D<uint8_t> prev_image{kWidth * kChannels, kHeight, 3};
  test::Array2D<uint8_t> next_image{kWidth * kChannels, kHeight, 5};
  fill_array(prev_image);
  fill_array(next_image);

  const kleidicv_optflow_lk_context_t context =
      make_default_optical_flow_pyr_lk_context();
  const std::vector<float> prev_points =
      make_optical_flow_points(kPointCount, kWidth, kHeight);
  std::vector<float> cpu_next_points(prev_points);
  std::vector<float> sme_next_points(prev_points);
  std::vector<uint8_t> cpu_status(kPointCount, 0);
  std::vector<uint8_t> sme_status(kPointCount, 0);
  std::vector<float> cpu_err(kPointCount, -1.0F);
  std::vector<float> sme_err(kPointCount, -1.0F);

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_u8(
                prev_image.data(), prev_image.stride(), next_image.data(),
                next_image.stride(), kWidth, kHeight, kChannels,
                prev_points.data(), cpu_next_points.data(), kPointCount,
                cpu_status.data(), cpu_err.data(), context));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_optical_flow_pyr_lk_u8_sme(
                prev_image.data(), prev_image.stride(), next_image.data(),
                next_image.stride(), kWidth, kHeight, kChannels,
                prev_points.data(), sme_next_points.data(), kPointCount,
                sme_status.data(), sme_err.data(), context));

  expect_same_optical_flow_result(cpu_next_points, sme_next_points, cpu_status,
                                  sme_status, cpu_err, sme_err);
}

TEST(SmeApi, BuildOpticalFlowPyrLkPyramidParity) {
  constexpr size_t kWidth = 43;
  constexpr size_t kHeight = 29;
  constexpr size_t kChannels = 3;
  constexpr size_t kRequestedLevels = 4;
  constexpr size_t kWindowWidth = 5;
  constexpr size_t kWindowHeight = 5;

  test::Array2D<uint8_t> src{kWidth * kChannels, kHeight, 3};
  fill_array(src);

  kleidicv_optical_flow_pyr_lk_pyramid_t* cpu_raw = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* sme_raw = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid(
                &cpu_raw, src.data(), src.stride(), kWidth, kHeight, kChannels,
                kRequestedLevels, kWindowWidth, kWindowHeight));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_build_optical_flow_pyr_lk_pyramid_sme(
                &sme_raw, src.data(), src.stride(), kWidth, kHeight, kChannels,
                kRequestedLevels, kWindowWidth, kWindowHeight));
  ASSERT_NE(nullptr, cpu_raw);
  ASSERT_NE(nullptr, sme_raw);

  const OpticalFlowPyramidPtr cpu_pyramid{cpu_raw};
  const OpticalFlowPyramidPtr sme_pyramid{sme_raw};
  expect_same_optical_flow_pyr_lk_pyramid(cpu_pyramid.get(), sme_pyramid.get(),
                                          kChannels);
}

TEST(SmeApi, StandaloneLucasKanadeParity) {
  constexpr size_t kWidth = 45;
  constexpr size_t kHeight = 12;
  test::Array2D<uint8_t> prev{kWidth, kHeight, 3};
  test::Array2D<uint8_t> next{kWidth, kHeight, 3};
  test::Array2D<int16_t> prev_deriv{kWidth * 2, kHeight, 3};
  fill_array(prev);
  fill_array(next);
  fill_array(prev_deriv);

  std::array<float, 2> prev_points{0.0F, 0.0F};
  std::array<float, 2> next_points_cpu{0.0F, 0.0F};
  std::array<float, 2> next_points_sme{0.0F, 0.0F};

  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_standalone_lucas_kanade_alg_u8(
                prev.data(), prev.stride(), prev_deriv.data(),
                prev_deriv.stride(), next.data(), next.stride(), kWidth,
                kHeight, 1, prev_points.data(), next_points_cpu.data(), 0,
                nullptr, nullptr, 5, 5, 10, 0.01, false, 0.0001F));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_standalone_lucas_kanade_alg_u8_sme(
                prev.data(), prev.stride(), prev_deriv.data(),
                prev_deriv.stride(), next.data(), next.stride(), kWidth,
                kHeight, 1, prev_points.data(), next_points_sme.data(), 0,
                nullptr, nullptr, 5, 5, 10, 0.01, false, 0.0001F));
  EXPECT_EQ(next_points_cpu, next_points_sme);
}
