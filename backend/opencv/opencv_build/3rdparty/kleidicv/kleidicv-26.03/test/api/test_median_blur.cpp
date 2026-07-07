// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_MEDIAN_BLUR(type, type_suffix) \
  KLEIDICV_API(median_blur, kleidicv_median_blur_##type_suffix, type)

KLEIDICV_MEDIAN_BLUR(int8_t, s8);
KLEIDICV_MEDIAN_BLUR(uint8_t, u8);
KLEIDICV_MEDIAN_BLUR(uint16_t, u16);
KLEIDICV_MEDIAN_BLUR(int16_t, s16);
KLEIDICV_MEDIAN_BLUR(uint32_t, u32);
KLEIDICV_MEDIAN_BLUR(int32_t, s32);
KLEIDICV_MEDIAN_BLUR(float, f32);

template <typename ElementType>
class MedianBlurTest : public testing::Test {
 public:
  struct TestParams {
    size_t width;
    size_t src_padding;
    size_t dst_padding;
    size_t height;
    size_t channels;
    size_t filter_size;
    kleidicv_border_type_t border_type;
  };

  static std::vector<TestParams> generate_test_cases(
      const std::vector<size_t>& widths,
      const std::vector<size_t>& src_paddings,
      const std::vector<size_t>& dst_paddings,
      const std::vector<size_t>& heights, const std::vector<size_t>& channels,
      const std::vector<size_t>& filter_sizes,
      const std::vector<kleidicv_border_type_t>& border_types) {
    std::vector<TestParams> cases;

    for (size_t w : widths) {
      for (size_t src_pad : src_paddings) {
        for (size_t dst_pad : dst_paddings) {
          for (size_t h : heights) {
            for (size_t c : channels) {
              for (size_t f : filter_sizes) {
                for (auto b : border_types) {
                  cases.push_back({w, src_pad, dst_pad, h, c, f, b});
                }
              }
            }
          }
        }
      }
    }

    return cases;
  }

  static std::vector<TestParams> get_unpadded_test_cases(size_t filter_size) {
    std::vector<size_t> widths = {2 * filter_size + 16};
    std::vector<size_t> src_paddings = {0};
    std::vector<size_t> dst_paddings = {0};
    std::vector<size_t> heights = {2 * filter_size + 16};
    std::vector<size_t> channels = {1, 2, 3, 4};
    std::vector<size_t> filter_sizes = {filter_size};
    std::vector<kleidicv_border_type_t> border_types = {
        KLEIDICV_BORDER_TYPE_REPLICATE, KLEIDICV_BORDER_TYPE_REFLECT,
        KLEIDICV_BORDER_TYPE_WRAP, KLEIDICV_BORDER_TYPE_REVERSE};

    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, filter_sizes, border_types);
  }

  static std::vector<TestParams> get_padded_test_cases(size_t filter_size) {
    std::vector<size_t> widths = {2 * filter_size + 16};
    std::vector<size_t> src_paddings = {5};
    std::vector<size_t> dst_paddings = {13};
    std::vector<size_t> heights = {2 * filter_size + 16};
    std::vector<size_t> channels = {1, 2, 3, 4};
    std::vector<size_t> filter_sizes = {filter_size};
    std::vector<kleidicv_border_type_t> border_types = {
        KLEIDICV_BORDER_TYPE_REPLICATE};

    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, filter_sizes, border_types);
  }

  static std::vector<TestParams> get_small_range_filter_test_cases(
      size_t filter_size) {
    std::vector<size_t> widths = {25, filter_size - 1};
    std::vector<size_t> src_paddings = {0};
    std::vector<size_t> dst_paddings = {3};
    std::vector<size_t> heights = {filter_size, filter_size - 1};
    std::vector<size_t> channels = {1, 2, 3, 4};
    std::vector<size_t> filter_sizes = {filter_size};
    std::vector<kleidicv_border_type_t> border_types = {
        KLEIDICV_BORDER_TYPE_REPLICATE, KLEIDICV_BORDER_TYPE_REFLECT,
        KLEIDICV_BORDER_TYPE_WRAP, KLEIDICV_BORDER_TYPE_REVERSE};

    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, filter_sizes, border_types);
  }

  static std::vector<TestParams> get_mid_range_filter_test_cases() {
    std::vector<size_t> widths = {50};
    std::vector<size_t> src_paddings = {0};
    std::vector<size_t> dst_paddings = {5};
    std::vector<size_t> heights = {20};
    std::vector<size_t> channels = {1, 4};
    std::vector<size_t> filter_sizes = {9, 15};
    std::vector<kleidicv_border_type_t> border_types = {
        KLEIDICV_BORDER_TYPE_REPLICATE, KLEIDICV_BORDER_TYPE_REFLECT,
        KLEIDICV_BORDER_TYPE_WRAP, KLEIDICV_BORDER_TYPE_REVERSE};
    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, filter_sizes, border_types);
  }

  static std::vector<TestParams> get_large_range_filter_test_cases(
      size_t filter_size) {
    std::vector<size_t> widths = {60};
    std::vector<size_t> src_paddings = {0};
    std::vector<size_t> dst_paddings = {5};
    std::vector<size_t> heights = {60, filter_size - 1};
    std::vector<size_t> channels = {1, 3};
    std::vector<size_t> filter_sizes = {filter_size};
    std::vector<kleidicv_border_type_t> border_types = {
        KLEIDICV_BORDER_TYPE_REPLICATE, KLEIDICV_BORDER_TYPE_REFLECT,
        KLEIDICV_BORDER_TYPE_WRAP, KLEIDICV_BORDER_TYPE_REVERSE};
    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, filter_sizes, border_types);
  }

  void run_test_case(const TestParams& params) {
    test::Array2D<ElementType> src{params.width * params.channels,
                                   params.height, params.src_padding,
                                   params.channels};
    test::Array2D<ElementType> expected_dst{params.width * params.channels,
                                            params.height, 0, params.channels};
    test::Array2D<ElementType> dst{params.width * params.channels,
                                   params.height, params.dst_padding,
                                   params.channels};

    test::PseudoRandomNumberGenerator<ElementType> input_value_random_range;
    src.fill(input_value_random_range);

    calculate_reference(src, expected_dst, params.filter_size,
                        params.border_type);

    auto status = median_blur<ElementType>()(
        src.data(), src.stride(), dst.data(), dst.stride(), params.width,
        params.height, params.channels, params.filter_size, params.filter_size,
        params.border_type);

    EXPECT_EQ(KLEIDICV_OK, status);
    EXPECT_EQ_ARRAY2D(expected_dst, dst);
  }

 private:
  int get_physical_index(int index, int limit,
                         kleidicv_border_type_t border_type) {
    int result = 0;

    if (index >= 0 && index < limit) {
      result = index;
    } else {
      switch (border_type) {
        case KLEIDICV_BORDER_TYPE_REPLICATE: {
          result = std::clamp(index, 0, limit - 1);
          break;
        }
        case KLEIDICV_BORDER_TYPE_REFLECT: {
          if (index < 0) {
            result = -index - 1;
          } else {
            result = 2 * limit - index - 1;
          }
          break;
        }

        case KLEIDICV_BORDER_TYPE_WRAP: {
          result = (index + limit) % limit;
          break;
        }

        case KLEIDICV_BORDER_TYPE_REVERSE: {
          if (index < 0) {
            result = -index;
          } else {
            result = 2 * limit - index - 2;
          }
          break;
        }

        default:
          result = 0;
          break;
      }
    }
    return result;
  }

  void calculate_reference(const test::Array2D<ElementType>& src,
                           test::Array2D<ElementType>& dst, size_t filter_size,
                           kleidicv_border_type_t border_type) {
    const int half_kernel_size = static_cast<int>(filter_size) / 2;
    const int height = static_cast<int>(src.height());
    const int width = static_cast<int>(src.width() / src.channels());

    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        for (size_t channel = 0; channel < src.channels(); ++channel) {
          std::vector<ElementType> window;

          for (int window_row = -half_kernel_size;
               window_row <= half_kernel_size; ++window_row) {
            for (int window_col = -half_kernel_size;
                 window_col <= half_kernel_size; ++window_col) {
              int row_after_border_handling =
                  get_physical_index(row + window_row, height, border_type);
              int col_after_border_handling =
                  get_physical_index(col + window_col, width, border_type);

              window.push_back(*src.at(
                  row_after_border_handling,
                  col_after_border_handling * src.channels() + channel));
            }
          }

          std::sort(window.begin(), window.end());
          dst.set(row, col * dst.channels() + channel,
                  {window[window.size() / 2]});
        }
      }
    }
  }
};

using ElementTypes = ::testing::Types<uint8_t, int8_t, int16_t, uint16_t,
                                      int32_t, uint32_t, float>;
TYPED_TEST_SUITE(MedianBlurTest, ElementTypes);

TYPED_TEST(MedianBlurTest,
           RunAllParamCombinationsWithoutPaddingWithSmallFilterSize) {
  if (test::Options::are_long_running_tests_skipped()) {
    GTEST_SKIP()
        << "Long running test "
           "MedianBlurTest::"
           "RunAllParamCombinationsWithoutPaddingWithSmallFilterSize skipped";
  }

  for (auto ksize : {3, 5, 7}) {
    for (const auto& params : TestFixture::get_unpadded_test_cases(ksize)) {
      this->run_test_case(params);
    }
  }
}

TYPED_TEST(MedianBlurTest,
           RunAllParamCombinationsWithPaddingWithSmallFilterSize) {
  if (test::Options::are_long_running_tests_skipped()) {
    GTEST_SKIP() << "Long running test "
                    "MedianBlurTest::"
                    "RunAllParamCombinationsWithPaddingWithSmallFilterSize "
                    "skipped";
  }

  for (auto ksize : {3, 5, 7}) {
    for (const auto& params : TestFixture::get_padded_test_cases(ksize)) {
      this->run_test_case(params);
    }
  }
}

TYPED_TEST(MedianBlurTest, RunAllParamCombinationsWithSmallImageAndFilterSize) {
  for (auto ksize : {3, 5, 7}) {
    for (const auto& params :
         TestFixture::get_small_range_filter_test_cases(ksize)) {
      this->run_test_case(params);
    }
  }
}

TYPED_TEST(MedianBlurTest, ChannelNumber) {
  test::Array2D<TypeParam> src{25, 25, 0, KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1};
  test::Array2D<TypeParam> dst{25, 25, 0, KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 25, 25,
                                     KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, 5, 5,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(MedianBlurTest, BorderNotImplemented) {
  test::Array2D<TypeParam> src{25, 25};
  test::Array2D<TypeParam> dst{25, 25};
  auto status = median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                         dst.stride(), 25, 25, 1, 5, 5,
                                         KLEIDICV_BORDER_TYPE_TRANSPARENT);
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, status);
}

TYPED_TEST(MedianBlurTest, HeightTooSmall) {
  test::Array2D<TypeParam> src{100, 4};
  test::Array2D<TypeParam> dst{100, 4};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 100, 3, 1, 7, 7,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(MedianBlurTest, WidthTooSmall) {
  test::Array2D<TypeParam> src{4, 100};
  test::Array2D<TypeParam> dst{4, 100};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 3, 100, 1, 7, 7,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(MedianBlurTest, NullPointer) {
  test::Array2D<TypeParam> src{5, 5};
  test::Array2D<TypeParam> dst{5, 5};
  test::test_null_args(median_blur<TypeParam>(), src.data(), src.stride(),
                       dst.data(), dst.stride(), 5, 5, 1, 5, 5,
                       KLEIDICV_BORDER_TYPE_REPLICATE);
}

TYPED_TEST(MedianBlurTest, OversizeImage) {
  test::Array2D<TypeParam> src{5, 5};
  test::Array2D<TypeParam> dst{5, 5};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), KLEIDICV_MAX_IMAGE_PIXELS,
                                     KLEIDICV_MAX_IMAGE_PIXELS, 1, 5, 5,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            median_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, KLEIDICV_MAX_IMAGE_PIXELS, 1, 5,
                5, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), KLEIDICV_MAX_IMAGE_PIXELS,
                                     KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 5, 5,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                               dst.stride(), 10, KLEIDICV_MAX_IMAGE_PIXELS, 1,
                               5, 5, KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(MedianBlurTest, UnsupportedFilterSizes) {
  test::Array2D<TypeParam> src{1000, 1000};
  test::Array2D<TypeParam> dst{1000, 1000};

  // Test unsupported large square filter
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 1000, 1000, 1, 257, 257,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));

  // Test non-square filter with valid height
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 1000, 1000, 1, 100, 5,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));

  // Test non-square filter with valid width
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 1000, 1000, 1, 5, 100,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));

  // Test unsupported small filter
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 1000, 1000, 1, 1, 1,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));

  // Test unsupported even filter
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 1000, 1000, 1, 4, 4,
                                     KLEIDICV_BORDER_TYPE_REPLICATE));

  // Test mid-range square filters that are not implemented
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                     dst.stride(), 1000, 1000, 1, 9, 9,
                                     KLEIDICV_BORDER_TYPE_TRANSPARENT));

  if (!std::is_same_v<TypeParam, uint8_t>) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              median_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                       dst.stride(), 1000, 1000, 1, 9, 9,
                                       KLEIDICV_BORDER_TYPE_REPLICATE));
  }
}

TYPED_TEST(MedianBlurTest, SrcDstChannelCombinations) {
  using ElementType = TypeParam;
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  constexpr size_t filter_size = 5;
  constexpr kleidicv_border_type_t border_type = KLEIDICV_BORDER_TYPE_REPLICATE;

  {
    test::Array2D<ElementType> buffer{width, height, 0, 1};
    auto status = median_blur<ElementType>()(
        buffer.data(), buffer.stride(), buffer.data(), buffer.stride(), width,
        height, 1, filter_size, filter_size, border_type);
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, status);
  }

  {
    test::Array2D<ElementType> buffer{width * KLEIDICV_MAXIMUM_CHANNEL_COUNT,
                                      height, 0,
                                      KLEIDICV_MAXIMUM_CHANNEL_COUNT};
    auto status = median_blur<ElementType>()(
        buffer.data(), buffer.stride(), buffer.data(), buffer.stride(), width,
        height, KLEIDICV_MAXIMUM_CHANNEL_COUNT, filter_size, filter_size,
        border_type);
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, status);
  }

  {
    test::Array2D<ElementType> src{width, height, 0, 1};
    test::Array2D<ElementType> dst{width, height, 0, 1};
    auto status = median_blur<ElementType>()(
        src.data(), src.stride(), dst.data(), dst.stride(), width, height, 1,
        filter_size, filter_size, border_type);
    EXPECT_EQ(KLEIDICV_OK, status);
  }

  {
    test::Array2D<ElementType> src{width * KLEIDICV_MAXIMUM_CHANNEL_COUNT,
                                   height, 0, KLEIDICV_MAXIMUM_CHANNEL_COUNT};
    test::Array2D<ElementType> dst{width * KLEIDICV_MAXIMUM_CHANNEL_COUNT,
                                   height, 0, KLEIDICV_MAXIMUM_CHANNEL_COUNT};
    auto status = median_blur<ElementType>()(
        src.data(), src.stride(), dst.data(), dst.stride(), width, height,
        KLEIDICV_MAXIMUM_CHANNEL_COUNT, filter_size, filter_size, border_type);
    EXPECT_EQ(KLEIDICV_OK, status);
  }
}
template <typename ElementType>
class MedianBlurMidAndLargeRangeTest : public MedianBlurTest<ElementType> {};
using ByteType = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(MedianBlurMidAndLargeRangeTest, ByteType);
TYPED_TEST(MedianBlurMidAndLargeRangeTest,
           RunAllParamCombinationsWithSmallImageAndMidRangeFilterSize) {
  for (const auto& params : TestFixture::get_mid_range_filter_test_cases()) {
    this->run_test_case(params);
  }
}

TYPED_TEST(MedianBlurMidAndLargeRangeTest,
           RunAllParamCombinationsWithSmallImageAndLargeRangeFilterSize) {
  for (auto ksize : {17, 35}) {
    for (const auto& params :
         TestFixture::get_large_range_filter_test_cases(ksize)) {
      this->run_test_case(params);
    }
  }
}

TYPED_TEST(MedianBlurMidAndLargeRangeTest,
           RunAllParamCombinationsWithoutPaddingWithMidAndLargeFilterSize) {
  if (test::Options::are_long_running_tests_skipped()) {
    GTEST_SKIP()
        << "Long running test "
           "MedianBlurMidAndLargeRangeTest::"
           "RunAllParamCombinationsWithoutPaddingWithMidAndLargeFilterSize "
           "skipped";
  }

  for (auto ksize : {9, 15, 17, 255}) {
    for (const auto& params : TestFixture::get_unpadded_test_cases(ksize)) {
      this->run_test_case(params);
    }
  }
}

TYPED_TEST(MedianBlurMidAndLargeRangeTest,
           RunAllParamCombinationsWithPaddingWithMidAndLargeFilterSize) {
  if (test::Options::are_long_running_tests_skipped()) {
    GTEST_SKIP()
        << "Long running test "
           "MedianBlurMidAndLargeRangeTest::"
           "RunAllParamCombinationsWithPaddingWithMidAndLargeFilterSize "
           "skipped";
  }

  for (auto ksize : {9, 15, 17, 255}) {
    for (const auto& params : TestFixture::get_padded_test_cases(ksize)) {
      this->run_test_case(params);
    }
  }
}
