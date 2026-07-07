// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

#define KLEIDICV_REMAP_S16(type, type_suffix) \
  KLEIDICV_API(remap_s16, kleidicv_remap_s16_##type_suffix, type)

KLEIDICV_REMAP_S16(uint8_t, u8);
KLEIDICV_REMAP_S16(uint16_t, u16);

#define KLEIDICV_REMAP_S16POINT5(type, type_suffix) \
  KLEIDICV_API(remap_s16point5, kleidicv_remap_s16point5_##type_suffix, type)

KLEIDICV_REMAP_S16POINT5(uint8_t, u8);
KLEIDICV_REMAP_S16POINT5(uint16_t, u16);

#define KLEIDICV_REMAP_F32(type, type_suffix) \
  KLEIDICV_API(remap_f32, kleidicv_remap_f32_##type_suffix, type)

KLEIDICV_REMAP_F32(uint8_t, u8);
KLEIDICV_REMAP_F32(uint16_t, u16);

template <typename ScalarType>
static const ScalarType *get_array2d_element_or_border(
    const test::Array2D<ScalarType> &src, ptrdiff_t x, ptrdiff_t y,
    ptrdiff_t ch, kleidicv_border_type_t border_type,
    const ScalarType *border_value) {
  // Width is the number of pixels in a row, but Array2D does not handle that
  const ptrdiff_t src_width =
      static_cast<ptrdiff_t>(src.width() / src.channels());
  const ptrdiff_t src_height = static_cast<ptrdiff_t>(src.height());

  if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
    x = std::clamp<ptrdiff_t>(x, 0, src_width - 1);
    y = std::clamp<ptrdiff_t>(y, 0, src_height - 1);
  } else {
    assert(border_type == KLEIDICV_BORDER_TYPE_CONSTANT);
    if (x >= src_width || y >= src_height || x < 0 || y < 0) {
      return border_value + ch;
    }
  }
  return src.at(y, x * src.channels() + ch);
}

template <class ScalarType>
class RemapS16 : public testing::Test {
 public:
  static void test_random(size_t src_w, size_t src_h, size_t dst_w,
                          size_t dst_h, size_t channels,
                          kleidicv_border_type_t border_type,
                          const ScalarType *border_value, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::PseudoRandomNumberGenerator<int16_t> coord_generator;
    mapxy.fill(coord_generator);
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, border_type,
                 border_value, padding);
  }

  static void test_outside_random(size_t src_w, size_t src_h, size_t dst_w,
                                  size_t dst_h, size_t channels,
                                  kleidicv_border_type_t border_type,
                                  const ScalarType *border_value,
                                  size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::PseudoRandomNumberGeneratorIntRange<int16_t> coord_generator{
        static_cast<int16_t>(-src_w), static_cast<int16_t>(2 * src_w)};
    mapxy.fill(coord_generator);
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, border_type,
                 border_value, padding);
  }

  static void test_blend(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
                         size_t channels, kleidicv_border_type_t border_type,
                         const ScalarType *border_value, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        // Use a second degree function to add a nonlinear blend to the image
        *mapxy.at(row, column * 2) = std::max<int16_t>(
            0, std::min(
                   static_cast<int16_t>(src_w - 1),
                   static_cast<int16_t>(column * 2 - column * column / dst_w)));
        *mapxy.at(row, column * 2 + 1) = std::max<int16_t>(
            0, std::min(static_cast<int16_t>(src_h - 1),
                        static_cast<int16_t>(row * (dst_w - column) / dst_w +
                                             4 * row / dst_h)));
      }
    }
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, border_type,
                 border_value, padding);
  }

  // Test coordinates with edge values that may easily overflow
  static void test_corner_cases(size_t src_w, size_t src_h, size_t dst_w,
                                size_t dst_h, size_t channels,
                                kleidicv_border_type_t border_type,
                                const ScalarType *border_value,
                                size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    const int16_t corner_x_values[] = {-32768,
                                       -1,
                                       0,
                                       static_cast<int16_t>(src_w - 2),
                                       static_cast<int16_t>(src_w - 1),
                                       static_cast<int16_t>(src_w),
                                       32767};
    const int16_t corner_y_values[] = {-32768,
                                       -1,
                                       0,
                                       1,
                                       static_cast<int16_t>(src_h - 2),
                                       static_cast<int16_t>(src_h - 1),
                                       static_cast<int16_t>(src_h),
                                       32767};
    // Number of elements in corner_x_values and corner_y_values
    // One more y than x, so it will cycle through all combinations
    static const size_t nx = sizeof(corner_x_values) / sizeof(int16_t);
    static const size_t ny = sizeof(corner_y_values) / sizeof(int16_t);
    size_t counter = 0;
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        *mapxy.at(row, column * 2) = corner_x_values[counter % nx];
        *mapxy.at(row, column * 2 + 1) = corner_y_values[counter % ny];
        ++counter;
      }
    }

    // This part is the same as execute_test() but without initializing source.
    // Corner Cases use the biggest possible source.
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    actual.fill(42);

    calculate_expected(source, mapxy, border_type, border_value, expected);

    ASSERT_EQ(KLEIDICV_OK, remap_s16<ScalarType>()(
                               source.data(), source.stride(), src_w,
                               source.height(), actual.data(), actual.stride(),
                               dst_w, actual.height(), channels, mapxy.data(),
                               mapxy.stride(), border_type, border_value));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

 private:
  static void execute_test(test::Array2D<int16_t> &mapxy, size_t src_w,
                           size_t src_h, size_t dst_w, size_t dst_h,
                           size_t channels, kleidicv_border_type_t border_type,
                           const ScalarType *border_value, size_t padding) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    test::PseudoRandomNumberGenerator<ScalarType> generator;
    source.fill(generator);
    actual.fill(42);

    calculate_expected(source, mapxy, border_type, border_value, expected);

    ASSERT_EQ(KLEIDICV_OK, remap_s16<ScalarType>()(
                               source.data(), source.stride(), src_w,
                               source.height(), actual.data(), actual.stride(),
                               dst_w, actual.height(), channels, mapxy.data(),
                               mapxy.stride(), border_type, border_value));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }
  static void calculate_expected(test::Array2D<ScalarType> &src,
                                 test::Array2D<int16_t> &mapxy,
                                 kleidicv_border_type_t border_type,
                                 const ScalarType *border_value,
                                 test::Array2D<ScalarType> &expected) {
    auto get_src = [&](ptrdiff_t x, ptrdiff_t y, size_t ch) {
      return get_array2d_element_or_border(src, x, y, ptrdiff_t(ch),
                                           border_type, border_value);
    };

    for (size_t row = 0; row < expected.height(); row++) {
      for (size_t column = 0; column < expected.width() / src.channels();
           ++column) {
        for (size_t ch = 0; ch < src.channels(); ++ch) {
          const int16_t *coords = mapxy.at(row, column * 2);
          int16_t x = coords[0], y = coords[1];
          *expected.at(row, column * src.channels() + ch) = *get_src(x, y, ch);
        }
      }
    }
  }
};

using RemapElementTypes = ::testing::Types<uint8_t, uint16_t>;
TYPED_TEST_SUITE(RemapS16, RemapElementTypes);

template <typename T>
static const auto &get_borders() {
  using P = std::pair<kleidicv_border_type_t, const T *>;
  static const T border_value[KLEIDICV_MAXIMUM_CHANNEL_COUNT] = {4, 5, 6, 7};
  static const std::array borders{
      P{KLEIDICV_BORDER_TYPE_REPLICATE, nullptr},
      P{KLEIDICV_BORDER_TYPE_CONSTANT, border_value},
  };
  return borders;
}

TYPED_TEST(RemapS16, RandomNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t channels = 1;
  size_t padding = 0;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_random(src_w, src_h, dst_w, dst_h, channels, border_type,
                             border_value, padding);
  }
}

TYPED_TEST(RemapS16, OutsideRandomPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t channels = 1;
  size_t padding = 13;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_outside_random(src_w, src_h, dst_w, dst_h, channels,
                                     border_type, border_value, padding);
  }
}

TYPED_TEST(RemapS16, BlendPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t channels = 1;
  size_t padding = 13;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_blend(src_w, src_h, dst_w, dst_h, channels, border_type,
                            border_value, padding);
  }
}

TYPED_TEST(RemapS16, BlendBigStride) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 16;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t channels = 1;
  size_t padding = std::numeric_limits<uint16_t>::max() - src_w;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_blend(src_w, src_h, dst_w, dst_h, channels, border_type,
                            border_value, padding);
  }
}

TYPED_TEST(RemapS16, CornerCases) {
  size_t src_w = std::numeric_limits<int16_t>::max() + 1;
  size_t src_h = std::numeric_limits<int16_t>::max() + 1;
  size_t dst_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t dst_h = 4;
  size_t channels = 1;
  size_t padding = 17;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, channels,
                                   border_type, border_value, padding);
  }
}

TYPED_TEST(RemapS16, NullPointer) {
  const TypeParam src[4] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  test::test_null_args(remap_s16<TypeParam>(), src, 2 * sizeof(TypeParam), 2, 2,
                       dst, 1 * sizeof(TypeParam), 1, 1, 1, mapxy, 4,
                       KLEIDICV_BORDER_TYPE_CONSTANT, src);
}

TYPED_TEST(RemapS16, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 0, 0, 1, dst, 0, 0, 1, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 1 * sizeof(TypeParam), 1, 0, dst,
                                   1 * sizeof(TypeParam), 1, 0, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  // Source too wide
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 1 * sizeof(TypeParam),
                                   std::numeric_limits<int16_t>::max() + 2, 1,
                                   dst, 8, 8, 1, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source too high
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 1 * sizeof(TypeParam), 1,
                                   std::numeric_limits<int16_t>::max() + 2, dst,
                                   8, 8, 1, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source extremely wide
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16<TypeParam>()(
                src, (KLEIDICV_MAX_IMAGE_PIXELS + 1) * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, dst, 8 * sizeof(TypeParam), 8,
                1, 1, mapxy, 4, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source extremely high
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16<TypeParam>()(src, sizeof(TypeParam), 1,
                                   KLEIDICV_MAX_IMAGE_PIXELS + 1, dst,
                                   8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source too large
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16<TypeParam>()(
                src, KLEIDICV_MAX_IMAGE_PIXELS * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, dst,
                8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Destination too wide
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16<TypeParam>()(
                src, 1 * sizeof(TypeParam), 1, 1, dst,
                (KLEIDICV_MAX_IMAGE_PIXELS + 1) * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, mapxy, 4,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Destination too large
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16<TypeParam>()(
                src, 1 * sizeof(TypeParam), 1, 1, dst,
                KLEIDICV_MAX_IMAGE_PIXELS * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 1, mapxy,
                4, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16, UnsupportedBigStride) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_s16<TypeParam>()(
          src, (std::numeric_limits<uint16_t>::max() + 1L) * sizeof(TypeParam),
          1, 1, dst, 8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4,
          KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_OK,
            remap_s16<TypeParam>()(
                src, std::numeric_limits<uint16_t>::max() * sizeof(TypeParam),
                1, 1, dst, 8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16, UnsupportedTwoChannels) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 1 * sizeof(TypeParam), 1, 1, dst,
                                   8 * sizeof(TypeParam), 8, 1, 2, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16, UnsupportedBorderType) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 1 * sizeof(TypeParam), 1, 1, dst,
                                   8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REFLECT, src));
}

TYPED_TEST(RemapS16, UnsupportedTooSmallImage) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16<TypeParam>()(src, 1 * sizeof(TypeParam), 1, 1, dst,
                                   8 * sizeof(TypeParam), 7, 1, 1, mapxy, 4,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

template <class ScalarType>
class RemapS16Point5 : public testing::Test {
 public:
  static const uint16_t FRAC_BITS = 5;
  static const uint16_t FRAC_MAX = 1 << FRAC_BITS;
  static const uint16_t FRAC_MAX_SQUARE = FRAC_MAX * FRAC_MAX;

  static void test_random(size_t src_w, size_t src_h, size_t dst_w,
                          size_t dst_h, size_t channels,
                          kleidicv_border_type_t border_type,
                          const ScalarType *border_value, size_t padding) {
    test::Array2D<int16_t> mapxy(2 * dst_w, dst_h, padding, 2);
    test::PseudoRandomNumberGenerator<int16_t> coord_generator;
    mapxy.fill(coord_generator);
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    test::PseudoRandomNumberGenerator<uint16_t> frac_generator;
    mapfrac.fill(frac_generator);
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels,
                 border_type, border_value, padding);
  }

  static void test_outside_random(size_t src_w, size_t src_h, size_t dst_w,
                                  size_t dst_h, size_t channels,
                                  kleidicv_border_type_t border_type,
                                  const ScalarType *border_value,
                                  size_t padding) {
    test::Array2D<int16_t> mapxy(2 * dst_w, dst_h, padding, 2);
    test::PseudoRandomNumberGeneratorIntRange<int16_t> coord_generator{
        static_cast<int16_t>(-src_w), static_cast<int16_t>(2 * src_w)};
    mapxy.fill(coord_generator);
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    test::PseudoRandomNumberGeneratorIntRange<uint16_t> frac_generator(
        0, static_cast<uint16_t>(FRAC_MAX_SQUARE * 3 / 2));
    mapfrac.fill(frac_generator);
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels,
                 border_type, border_value, padding);
  }

  static void test_blend(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
                         size_t channels, kleidicv_border_type_t border_type,
                         const ScalarType *border_value, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        // Use a second degree function to add a nonlinear blend to the image
        auto r = static_cast<double>(row), c = static_cast<double>(column),
             w = static_cast<double>(dst_w), h = static_cast<double>(dst_h);
        double x = c * 2 - c * c / w;
        double y = r * (w - c) / w + 4 * r / h;
        *mapxy.at(row, column * 2) = std::max<int16_t>(
            0,
            std::min(static_cast<int16_t>(src_w - 1), static_cast<int16_t>(x)));
        *mapxy.at(row, column * 2 + 1) = std::max<int16_t>(
            0,
            std::min(static_cast<int16_t>(src_h - 1), static_cast<int16_t>(y)));
        *mapfrac.at(row, column) =
            static_cast<uint16_t>(FRAC_MAX * (x - static_cast<int16_t>(x))) |
            (static_cast<uint16_t>(FRAC_MAX * (y - static_cast<int16_t>(y)))
             << FRAC_BITS);
      }
    }
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels,
                 border_type, border_value, padding);
  }

  // Test coordinates with edge values that may easily overflow
  static void test_corner_cases(size_t src_w, size_t src_h, size_t dst_w,
                                size_t dst_h, size_t channels,
                                kleidicv_border_type_t border_type,
                                const ScalarType *border_value,
                                size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    // One more y than x so we'll see many combinations
    const int16_t corner_x_values[] = {-32768,
                                       -1,
                                       0,
                                       static_cast<int16_t>(src_w - 2),
                                       static_cast<int16_t>(src_w - 1),
                                       static_cast<int16_t>(src_w),
                                       32767};
    const int16_t corner_y_values[] = {-32768,
                                       -1,
                                       0,
                                       1,
                                       static_cast<int16_t>(src_h - 2),
                                       static_cast<int16_t>(src_h - 1),
                                       static_cast<int16_t>(src_h),
                                       32767};
    static const size_t nx = sizeof(corner_x_values) / sizeof(int16_t);
    static const size_t ny = sizeof(corner_y_values) / sizeof(int16_t);
    size_t counter = 0;
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        *mapxy.at(row, column * 2) = corner_x_values[counter % nx];
        *mapxy.at(row, column * 2 + 1) = corner_y_values[counter % ny];
        *mapfrac.at(row, column) =
            static_cast<uint16_t>((counter * 3) % FRAC_MAX) |
            (static_cast<uint16_t>((counter * 5) % FRAC_MAX) << FRAC_BITS);
        ++counter;
      }
    }

    // This part is the same as execute_test() except source initialization.
    // Corner Cases use the biggest possible source, so it is only initializing
    // the edges.
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    const int64_t kMaxVal = std::numeric_limits<ScalarType>::max();
    auto generateSource = [&](size_t x, size_t y) {
      return static_cast<ScalarType>((x + y) % 2 ? kMaxVal : 0);
    };
    for (size_t y = 0; y < src_h; ++y) {
      *source.at(y, 0) = generateSource(y, 0);
      *source.at(y, 1) = generateSource(y, 1);
      *source.at(y, 2) = generateSource(y, 2);
      *source.at(y, src_w - 3) = generateSource(y, src_w - 3);
      *source.at(y, src_w - 2) = generateSource(y, src_w - 2);
      *source.at(y, src_w - 1) = generateSource(y, src_w - 1);
    }
    for (size_t x = 0; x < src_w; ++x) {
      *source.at(0, x) = generateSource(0, x);
      *source.at(1, x) = generateSource(1, x);
      *source.at(2, x) = generateSource(2, x);
      *source.at(src_h - 3, x) = generateSource(src_h - 3, x);
      *source.at(src_h - 2, x) = generateSource(src_h - 2, x);
      *source.at(src_h - 1, x) = generateSource(src_h - 1, x);
    }
    actual.fill(42);

    calculate_expected(source, mapxy, mapfrac, border_type, border_value,
                       expected);

    ASSERT_EQ(KLEIDICV_OK,
              remap_s16point5<ScalarType>()(
                  source.data(), source.stride(), src_w, source.height(),
                  actual.data(), actual.stride(), dst_w, actual.height(),
                  channels, mapxy.data(), mapxy.stride(), mapfrac.data(),
                  mapfrac.stride(), border_type, border_value));

    if (expected.compare_to(actual, 1)) {
      if (source.width() < 100 && source.height() < 100) {
        std::cout << "source:\n";
        dump(&source);
      }
      std::cout << "mapxy:\n";
      dump(&mapxy);
      std::cout << "mapfrac:\n";
      dump(&mapfrac);
      std::cout << "expected:\n";
      dump(&expected);
      std::cout << "actual:\n";
      dump(&actual);
    }

    EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, actual, expected);
  }

 private:
  static void execute_test(test::Array2D<int16_t> &mapxy,
                           test::Array2D<uint16_t> &mapfrac, size_t src_w,
                           size_t src_h, size_t dst_w, size_t dst_h,
                           size_t channels, kleidicv_border_type_t border_type,
                           const ScalarType *border_value, size_t padding) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};
    ScalarType counter = 0;
    for (size_t y = 0; y < src_h; ++y) {
      for (size_t x = 0; x < src_total_width; ++x) {
        *source.at(y, x) = ++counter;
      }
    }
    actual.fill(42);

    calculate_expected(source, mapxy, mapfrac, border_type, border_value,
                       expected);

    ASSERT_EQ(KLEIDICV_OK,
              remap_s16point5<ScalarType>()(
                  source.data(), source.stride(), src_w, source.height(),
                  actual.data(), actual.stride(), dst_w, actual.height(),
                  channels, mapxy.data(), mapxy.stride(), mapfrac.data(),
                  mapfrac.stride(), border_type, border_value));

    if (expected.compare_to(actual)) {
      if (source.width() < 100 && source.height() < 100) {
        std::cout << "source:\n";
        dump(&source);
      }
      std::cout << "mapxy:\n";
      dump(&mapxy);
      std::cout << "mapfrac:\n";
      dump(&mapfrac);
      std::cout << "expected:\n";
      dump(&expected);
      std::cout << "actual:\n";
      dump(&actual);
    }

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

  static ScalarType lerp_2d(size_t cx, size_t cy, ScalarType a, ScalarType b,
                            ScalarType c, ScalarType d) {
    size_t inv_cx = FRAC_MAX - cx, inv_cy = FRAC_MAX - cy;
    ScalarType r = static_cast<ScalarType>((inv_cx * inv_cy * a +
                                            cx * inv_cy * b + inv_cx * cy * c +
                                            cx * cy * d + FRAC_MAX_SQUARE / 2) /
                                           FRAC_MAX_SQUARE);
    return r;
  }

  static void calculate_expected(test::Array2D<ScalarType> &src,
                                 test::Array2D<int16_t> &mapxy,
                                 test::Array2D<uint16_t> &mapfrac,
                                 kleidicv_border_type_t border_type,
                                 const ScalarType *border_value,
                                 test::Array2D<ScalarType> &expected) {
    auto get_src = [&](ptrdiff_t x, ptrdiff_t y, size_t ch) {
      return get_array2d_element_or_border(src, x, y, ptrdiff_t(ch),
                                           border_type, border_value);
    };

    for (size_t row = 0; row < expected.height(); row++) {
      for (size_t column = 0; column < expected.width() / src.channels();
           ++column) {
        // Clang-tidy thinks mapfrac may contain garbage, but it is fully
        // initialized at all code paths and the map size always equals dst
        // map pixel size
        // NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult)
        uint8_t x_frac = *mapfrac.at(row, column) & (FRAC_MAX - 1);
        uint8_t y_frac =
            (*mapfrac.at(row, column) >> FRAC_BITS) & (FRAC_MAX - 1);
        // NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
        const int16_t *coords = mapxy.at(row, column * 2);
        ptrdiff_t x = coords[0], y = coords[1];
        for (size_t ch = 0; ch < src.channels(); ++ch) {
          *expected.at(row, column * src.channels() + ch) = lerp_2d(
              x_frac, y_frac, *get_src(x, y, ch), *get_src(x + 1, y, ch),
              *get_src(x, y + 1, ch), *get_src(x + 1, y + 1, ch));
        }
      }
    }
  }
};

using RemapS16Point5ElementTypes = ::testing::Types<uint8_t, uint16_t>;
TYPED_TEST_SUITE(RemapS16Point5, RemapS16Point5ElementTypes);

template <typename T>
size_t defaultWidth() {
  return 3 * test::Options::vector_lanes<T>() - 1;
}

size_t defaultHeight() { return 4; }

TYPED_TEST(RemapS16Point5, RandomNoPadding) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_random(w, h, w, h, 1, border_type, border_value, 0);
  }
}

TYPED_TEST(RemapS16Point5, RandomNoPadding4ch) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  size_t channels = 4;
  size_t padding = 0;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_random(w, h, w, h, channels, border_type, border_value,
                             padding);
  }
}

TYPED_TEST(RemapS16Point5, BlendPadding) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_blend(w, h, w, h, 1, border_type, border_value, 13);
  }
}

TYPED_TEST(RemapS16Point5, BlendPadding4ch) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  size_t channels = 4;
  size_t padding = 7;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_blend(w, h, w, h, channels, border_type, border_value,
                            padding);
  }
}

TYPED_TEST(RemapS16Point5, OutsideRandomPadding) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_outside_random(w, h, w, h, 1, border_type, border_value,
                                     13);
  }
}

TYPED_TEST(RemapS16Point5, OutsideRandomPadding4ch) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  size_t channels = 4;
  size_t padding = 11;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_outside_random(w, h, w, h, channels, border_type,
                                     border_value, padding);
  }
}

TYPED_TEST(RemapS16Point5, BlendBigStride) {
  size_t src_w = defaultWidth<TypeParam>();
  size_t src_h = defaultHeight();
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t channels = 1;
  size_t padding = std::numeric_limits<uint16_t>::max() - src_w;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_blend(src_w, src_h, dst_w, dst_h, channels, border_type,
                            border_value, padding);
  }
}

TYPED_TEST(RemapS16Point5, BlendBigStride4ch) {
  size_t w = defaultWidth<TypeParam>();
  size_t h = defaultHeight();
  size_t channels = 4;
  size_t padding =
      std::numeric_limits<uint16_t>::max() / channels - w * channels;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_blend(w, h, w, h, channels, border_type, border_value,
                            padding);
  }
}

TYPED_TEST(RemapS16Point5, CornerCases) {
  size_t src_w = std::numeric_limits<int16_t>::max() + 1;
  size_t src_h = std::numeric_limits<int16_t>::max() + 1;
  size_t dst_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t dst_h = 4;
  size_t channels = 1;
  size_t padding = 17;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, channels,
                                   border_type, border_value, padding);
  }
}

TYPED_TEST(RemapS16Point5, CornerCases4ch) {
  size_t src_w = 100;
  size_t src_h = 8;
  size_t dst_w = defaultWidth<TypeParam>();
  size_t dst_h = defaultHeight();
  size_t channels = 4;
  size_t padding = 17;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, channels,
                                   border_type, border_value, padding);
  }
}

TYPED_TEST(RemapS16Point5, NullPointer) {
  const TypeParam src[4] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  uint16_t mapfrac[1] = {};
  test::test_null_args(remap_s16point5<TypeParam>(), src, 2 * sizeof(TypeParam),
                       2, 2, dst, 1 * sizeof(TypeParam), 1, 1, 1, mapxy, 4,
                       mapfrac, 2, KLEIDICV_BORDER_TYPE_CONSTANT, src);
}

TYPED_TEST(RemapS16Point5, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  uint16_t mapfrac[1] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16point5<TypeParam>()(
                src, 0, 0, 1, dst, 0, 0, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_s16point5<TypeParam>()(
          src, 1 * sizeof(TypeParam), 1, 0, dst, 1 * sizeof(TypeParam), 1, 0, 1,
          mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16Point5, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  uint16_t mapfrac[1] = {};

  const size_t kTooBig = std::numeric_limits<int16_t>::max() + 2L;
  // Source too wide
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16point5<TypeParam>()(
                src, kTooBig * sizeof(TypeParam), kTooBig, 1, dst,
                8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source too high
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_s16point5<TypeParam>()(
          src, sizeof(TypeParam), 1, kTooBig, dst, 8 * sizeof(TypeParam), 8, 1,
          1, mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source extremely wide
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      remap_s16point5<TypeParam>()(
          src, (KLEIDICV_MAX_IMAGE_PIXELS + 1) * sizeof(TypeParam),
          KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, dst, 8 * sizeof(TypeParam), 8, 1, 1,
          mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source extremely high
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16point5<TypeParam>()(
                src, sizeof(TypeParam), 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, dst,
                8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Source too large
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16point5<TypeParam>()(
                src, KLEIDICV_MAX_IMAGE_PIXELS * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, dst,
                8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Destination too wide
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16point5<TypeParam>()(
                src, sizeof(TypeParam), 1, 1, dst,
                (KLEIDICV_MAX_IMAGE_PIXELS + 1) * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  // Destination too large
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_s16point5<TypeParam>()(
                src, 1 * sizeof(TypeParam), 1, 1, dst,
                (KLEIDICV_MAX_IMAGE_PIXELS + 1) * sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 1, mapxy,
                4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16Point5, UnsupportedBigStride) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_s16point5<TypeParam>()(
          src, (std::numeric_limits<uint16_t>::max() + 1L) * sizeof(TypeParam),
          1, 1, dst, 8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4, mapfrac, 2,
          KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_OK,
            remap_s16point5<TypeParam>()(
                src, std::numeric_limits<uint16_t>::max() * sizeof(TypeParam),
                1, 1, dst, 8 * sizeof(TypeParam), 8, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16Point5, UnsupportedChannels) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16point5<TypeParam>()(
                src, 1, 1, 1, dst, 8, 8, 1, 2, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16point5<TypeParam>()(
                src, 1, 1, 1, dst, 8, 8, 1, 3, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapS16Point5, UnsupportedBorderType) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_s16point5<TypeParam>()(
                src, 1 * sizeof(TypeParam), 1, 1, dst, 8 * sizeof(TypeParam), 8,
                1, 1, mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REFLECT, src));
}

TYPED_TEST(RemapS16Point5, UnsupportedBigStride4ch) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_s16point5<TypeParam>()(
          src,
          (std::numeric_limits<uint16_t>::max() / 4 + 1L) * sizeof(TypeParam),
          1, 1, dst, 8, 8, 1, 4, mapxy, 4, mapfrac, 2,
          KLEIDICV_BORDER_TYPE_CONSTANT, src));
}

TYPED_TEST(RemapS16Point5, UnsupportedTooSmallImage) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_s16point5<TypeParam>()(
          src, 1 * sizeof(TypeParam), 1, 1, dst, 8 * sizeof(TypeParam), 7, 1, 1,
          mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

template <class ScalarType>
class RemapF32 : public testing::Test {
 public:
  static void test_random(size_t src_w, size_t src_h, size_t dst_w,
                          size_t dst_h, size_t channels,
                          kleidicv_border_type_t border_type,
                          const ScalarType *border_value,
                          kleidicv_interpolation_type_t interpolation,
                          size_t padding) {
    test::PseudoRandomNumberGenerator<float> coord_generator;
    test::Array2D<float> mapx(dst_w, dst_h, padding);
    test::Array2D<float> mapy(dst_w, dst_h, padding);
    mapx.fill(coord_generator);
    mapy.fill(coord_generator);
    execute_test(mapx, mapy, src_w, src_h, dst_w, dst_h, channels, border_type,
                 border_value, interpolation, padding);
  }

  static void test_outside_random(size_t src_w, size_t src_h, size_t dst_w,
                                  size_t dst_h, size_t channels,
                                  kleidicv_border_type_t border_type,
                                  const ScalarType *border_value,
                                  kleidicv_interpolation_type_t interpolation,
                                  size_t padding) {
    test::Array2D<float> mapx(dst_w, dst_h, padding);
    test::PseudoRandomNumberGeneratorFloatRange<float> xcoord_generator{
        // static_cast<float>(-src_w), static_cast<float>(2 * src_w)}; -src_w
        // overflow
        static_cast<float>(-static_cast<int64_t>(src_w)),
        static_cast<float>(2 * src_w)};
    mapx.fill(xcoord_generator);
    test::Array2D<float> mapy(dst_w, dst_h, padding);
    test::PseudoRandomNumberGeneratorFloatRange<float> ycoord_generator{
        static_cast<float>(-static_cast<int64_t>(src_h)),
        static_cast<float>(2 * src_h)};
    mapy.fill(ycoord_generator);
    execute_test(mapx, mapy, src_w, src_h, dst_w, dst_h, channels, border_type,
                 border_value, interpolation, padding);
  }

  static void test_blend(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
                         size_t channels, kleidicv_border_type_t border_type,
                         const ScalarType *border_value,
                         kleidicv_interpolation_type_t interpolation,
                         size_t padding) {
    test::Array2D<float> mapx(dst_w, dst_h, padding);
    test::Array2D<float> mapy(dst_w, dst_h, padding);
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        // Use a second degree function to add a nonlinear blend to the image
        auto r = static_cast<double>(row), c = static_cast<double>(column),
             w = static_cast<double>(dst_w), h = static_cast<double>(dst_h);
        double x = c * 2 - c * c / w;
        double y = r * (w - c) / w + 4 * r / h;
        *mapx.at(row, column) = std::max<float>(
            0, std::min(static_cast<float>(src_w - 1), static_cast<float>(x)));
        *mapy.at(row, column) = std::max<float>(
            0, std::min(static_cast<float>(src_h - 1), static_cast<float>(y)));
      }
    }
    execute_test(mapx, mapy, src_w, src_h, dst_w, dst_h, channels, border_type,
                 border_value, interpolation, padding);
  }

  // Test coordinates with edge values that may easily overflow
  static void test_corner_cases(size_t src_w, size_t src_h, size_t dst_w,
                                size_t dst_h, size_t channels,
                                kleidicv_border_type_t border_type,
                                const ScalarType *border_value,
                                kleidicv_interpolation_type_t interpolation,
                                size_t padding) {
    test::Array2D<float> mapx(dst_w, dst_h, padding);
    test::Array2D<float> mapy(dst_w, dst_h, padding);
    const float corner_x_values[] = {std::numeric_limits<float>::min(),
                                     -1.8,
                                     -0.3,
                                     0.2,
                                     static_cast<float>(src_w) - 1.5F,
                                     static_cast<float>(src_w) - 0.93F,
                                     static_cast<float>(src_w) + 0.1F,
                                     static_cast<float>(src_w) + 1.2F,
                                     std::numeric_limits<float>::max()};
    const float corner_y_values[] = {std::numeric_limits<float>::min(),
                                     -1.3,
                                     -0.7,
                                     0.33,
                                     1.1,
                                     static_cast<float>(src_h) - 1.8F,
                                     static_cast<float>(src_h) - 0.77F,
                                     static_cast<float>(src_h) + 0.17F,
                                     static_cast<float>(src_h) + 1.06F,
                                     std::numeric_limits<float>::max()};
    const size_t nx = sizeof(corner_x_values) / sizeof(float);
    const size_t ny = sizeof(corner_y_values) / sizeof(float);
    size_t counter = 0;
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        *mapx.at(row, column) = corner_x_values[counter % nx];
        *mapy.at(row, column) = corner_y_values[counter % ny];
        ++counter;
      }
    }

    // This part is the same as execute_test() but without initializing the
    // whole source. Corner Cases use the biggest possible source.
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    // Initalize the four corners only
    const int64_t kMaxVal = std::numeric_limits<ScalarType>::max() * 3 / 4;
    const int64_t kMinVal =
        std::numeric_limits<ScalarType>::lowest() + kMaxVal / 3;
    auto generateSource = [&](size_t x, size_t y) {
      return static_cast<ScalarType>((x + y) % 2 ? kMaxVal : kMinVal);
    };
    for (size_t y = 0; y < 2; ++y) {
      *source.at(y, 0) = generateSource(y, 0);
      *source.at(y, 1) = generateSource(y, 1);
      *source.at(y, 2) = generateSource(y, 2);
      *source.at(y, src_w - 3) = generateSource(y, src_w - 3);
      *source.at(y, src_w - 2) = generateSource(y, src_w - 2);
      *source.at(y, src_w - 1) = generateSource(y, src_w - 1);
      *source.at(src_h - y - 1, 0) = generateSource(src_h - y - 1, 0);
      *source.at(src_h - y - 1, 1) = generateSource(src_h - y - 1, 1);
      *source.at(src_h - y - 1, 2) = generateSource(src_h - y - 1, 2);
      *source.at(src_h - y - 1, src_w - 3) =
          generateSource(src_h - y - 1, src_w - 3);
      *source.at(src_h - y - 1, src_w - 2) =
          generateSource(src_h - y - 1, src_w - 2);
      *source.at(src_h - y - 1, src_w - 1) =
          generateSource(src_h - y - 1, src_w - 1);
    }

    test::PseudoRandomNumberGenerator<ScalarType> generator;
    actual.fill(42);

    calculate_expected(source, mapx, mapy, border_type, border_value,
                       interpolation, expected);

    ASSERT_EQ(KLEIDICV_OK,
              remap_f32<ScalarType>()(
                  source.data(), source.stride(), src_w, source.height(),
                  actual.data(), actual.stride(), dst_w, actual.height(),
                  channels, mapx.data(), mapx.stride(), mapy.data(),
                  mapy.stride(), interpolation, border_type, border_value));

    if (expected.compare_to(actual, 1)) {
      if (source.width() < 100 && source.height() < 100) {
        std::cout << "source:\n";
        dump(&source);
      }
      std::cout << "mapx:\n";
      dump(&mapx);
      std::cout << "mapy:\n";
      dump(&mapy);
      std::cout << "expected:\n";
      dump(&expected);
      std::cout << "actual:\n";
      dump(&actual);
    }

    EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, actual, expected);
  }

 private:
  static void execute_test(test::Array2D<float> &mapx,
                           test::Array2D<float> &mapy, size_t src_w,
                           size_t src_h, size_t dst_w, size_t dst_h,
                           size_t channels, kleidicv_border_type_t border_type,
                           const ScalarType *border_value,
                           kleidicv_interpolation_type_t interpolation,
                           size_t padding) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};
    test::PseudoRandomNumberGenerator<ScalarType> generator;
    source.fill(generator);
    actual.fill(42);

    calculate_expected(source, mapx, mapy, border_type, border_value,
                       interpolation, expected);

    ASSERT_EQ(KLEIDICV_OK,
              remap_f32<ScalarType>()(
                  source.data(), source.stride(), src_w, source.height(),
                  actual.data(), actual.stride(), dst_w, actual.height(),
                  channels, mapx.data(), mapx.stride(), mapy.data(),
                  mapy.stride(), interpolation, border_type, border_value));

    if (expected.compare_to(actual, 1)) {
      if (source.width() < 100 && source.height() < 100) {
        std::cout << "source:\n";
        dump(&source);
      }
      std::cout << "mapx:\n";
      dump(&mapx);
      std::cout << "mapy:\n";
      dump(&mapy);
      std::cout << "expected:\n";
      dump(&expected);
      std::cout << "actual:\n";
      dump(&actual);
    }

    EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, actual, expected);
  }

  static void calculate_expected(test::Array2D<ScalarType> &src,
                                 test::Array2D<float> &mapx,
                                 test::Array2D<float> &mapy,
                                 kleidicv_border_type_t border_type,
                                 const ScalarType *border_value,
                                 kleidicv_interpolation_type_t interpolation,
                                 test::Array2D<ScalarType> &expected) {
    auto get_src = [&](ptrdiff_t x, ptrdiff_t y, size_t ch) {
      return get_array2d_element_or_border(src, x, y, ptrdiff_t(ch),
                                           border_type, border_value);
    };

    for (size_t row = 0; row < expected.height(); row++) {
      for (size_t column = 0; column < expected.width() / src.channels();
           ++column) {
        float x = *mapx.at(row, column);
        float y = *mapy.at(row, column);

        if (interpolation == KLEIDICV_INTERPOLATION_LINEAR) {
          ptrdiff_t ix = static_cast<ptrdiff_t>(std::max<float>(
              INT_MIN,
              std::min<float>(std::floor(x),
                              static_cast<float>(KLEIDICV_MAX_IMAGE_PIXELS))));
          ptrdiff_t iy = static_cast<ptrdiff_t>(std::max<float>(
              INT_MIN,
              std::min<float>(std::floor(y),
                              static_cast<float>(KLEIDICV_MAX_IMAGE_PIXELS))));
          float xfrac = x - std::floor(x);
          float yfrac = y - std::floor(y);
          for (size_t ch = 0; ch < src.channels(); ++ch) {
            float a = *get_src(ix, iy, ch);
            float b = *get_src(ix + 1, iy, ch);
            float c = *get_src(ix, iy + 1, ch);
            float d = *get_src(ix + 1, iy + 1, ch);
            float line1 = (b - a) * xfrac + a;
            float line2 = (d - c) * xfrac + c;
            float float_result = (line2 - line1) * yfrac + line1;
            *expected.at(row, column * src.channels() + ch) =
                static_cast<ScalarType>(std::lround(float_result));
          }
        } else {
          assert(interpolation == KLEIDICV_INTERPOLATION_NEAREST);
          ptrdiff_t ix = static_cast<ptrdiff_t>(std::max<float>(
              INT_MIN,
              std::min<float>(std::round(x),
                              static_cast<float>(KLEIDICV_MAX_IMAGE_PIXELS))));
          ptrdiff_t iy = static_cast<ptrdiff_t>(std::max<float>(
              INT_MIN,
              std::min<float>(std::round(y),
                              static_cast<float>(KLEIDICV_MAX_IMAGE_PIXELS))));
          for (size_t ch = 0; ch < src.channels(); ++ch) {
            *expected.at(row, column * src.channels() + ch) =
                *get_src(ix, iy, ch);
          }
        }
      }
    }
  }
};

using RemapF32ElementTypes = ::testing::Types<uint8_t, uint16_t>;
TYPED_TEST_SUITE(RemapF32, RemapF32ElementTypes);

TYPED_TEST(RemapF32, RandomNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t padding = 0;
  for (size_t channels = 1; channels <= 2; ++channels) {
    for (auto [border_type, border_value] : get_borders<TypeParam>()) {
      for (auto interpolation :
           {KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_INTERPOLATION_NEAREST}) {
        TestFixture::test_random(src_w, src_h, dst_w, dst_h, channels,
                                 border_type, border_value, interpolation,
                                 padding);
      }
    }
  }
}

TYPED_TEST(RemapF32, BlendPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t padding = 13;
  for (size_t channels = 2; channels <= 2; ++channels) {
    for (auto [border_type, border_value] : get_borders<TypeParam>()) {
      for (auto interpolation :
           {KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_INTERPOLATION_NEAREST}) {
        TestFixture::test_blend(src_w, src_h, dst_w, dst_h, channels,
                                border_type, border_value, interpolation,
                                padding);
      }
    }
  }
}

TYPED_TEST(RemapF32, OutsideRandomPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t padding = 13;
  for (size_t channels = 1; channels <= 2; ++channels) {
    for (auto [border_type, border_value] : get_borders<TypeParam>()) {
      for (auto interpolation :
           {KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_INTERPOLATION_NEAREST}) {
        TestFixture::test_outside_random(src_w, src_h, dst_w, dst_h, channels,
                                         border_type, border_value,
                                         interpolation, padding);
      }
    }
  }
}

TYPED_TEST(RemapF32, BlendBigStride) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 2;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  size_t padding = 1 << 16;
  for (size_t channels = 1; channels <= 2; ++channels) {
    for (auto [border_type, border_value] : get_borders<TypeParam>()) {
      for (auto interpolation :
           {KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_INTERPOLATION_NEAREST}) {
        TestFixture::test_blend(src_w, src_h, dst_w, dst_h, channels,
                                border_type, border_value, interpolation,
                                padding);
      }
    }
  }
}

TYPED_TEST(RemapF32, CornerCases) {
  size_t src_w = (1ULL << 12) - 1;
  size_t src_h = (1ULL << 12) - 1;
  size_t dst_w = 4;
  size_t dst_h = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t padding = 17;
  for (size_t channels = 1; channels <= 2; ++channels) {
    for (auto [border_type, border_value] : get_borders<TypeParam>()) {
      for (auto interpolation :
           {KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_INTERPOLATION_NEAREST}) {
        TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, channels,
                                       border_type, border_value, interpolation,
                                       padding);
      }
    }
  }
}

// Test for load_src_into_floats_large
TYPED_TEST(RemapF32, CornerCasesLargeLoad) {
  // TODO: It takes long to run!
  size_t src_w = 1ULL << 18;
  size_t src_h = 1ULL << 14;
  size_t dst_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t dst_h = 4;
  size_t padding = 1;
  for (size_t channels = 1; channels <= 2; ++channels) {
    for (auto [border_type, border_value] : get_borders<TypeParam>()) {
      for (auto interpolation :
           {KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_INTERPOLATION_NEAREST}) {
        TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, channels,
                                       border_type, border_value, interpolation,
                                       padding);
      }
    }
  }
}

TYPED_TEST(RemapF32, NullPointer) {
  const size_t element_size = sizeof(TypeParam);
  const size_t src_width = 2;
  const size_t src_height = 2;
  const size_t src_stride = src_width * element_size;
  const size_t dst_width = 1;
  const size_t dst_height = 1;
  const size_t dst_stride = dst_width * element_size;
  const TypeParam src[4] = {};
  TypeParam dst[1];
  float mapx[1] = {};
  const size_t mapx_stride = dst_width * sizeof(float);
  float mapy[1] = {};
  const size_t mapy_stride = dst_width * sizeof(float);
  const TypeParam border_value[1] = {};
  for (size_t channels = 1; channels <= 2; ++channels) {
    test::test_null_args(remap_f32<TypeParam>(), src, src_stride, src_width,
                         src_height, dst, dst_stride, dst_width, dst_height,
                         channels, mapx, mapx_stride, mapy, mapy_stride,
                         KLEIDICV_INTERPOLATION_LINEAR,
                         KLEIDICV_BORDER_TYPE_CONSTANT, border_value);
  }
}

TYPED_TEST(RemapF32, ZeroHeightImage) {
  const size_t kW = 4;
  const TypeParam src[kW] = {};
  TypeParam dst[kW];
  const size_t src_stride = kW * sizeof(TypeParam);
  const size_t big_stride = (1UL << 32UL) - sizeof(TypeParam);
  const size_t dst_stride = kW * sizeof(TypeParam);
  float mapx[kW] = {-0.2, 0.3, 1.4, 2.5};
  float mapy[kW] = {-1.8, -0.7, 0.6, 1.3};
  const size_t mapx_stride = kW * sizeof(float);
  const size_t mapy_stride = kW * sizeof(float);

  // TODO: Why these sets of parameters?
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    EXPECT_EQ(KLEIDICV_OK,
              remap_f32<TypeParam>()(src, src_stride, kW, 1, dst, dst_stride,
                                     kW, 0, 1, mapx, mapx_stride, mapy,
                                     mapy_stride, KLEIDICV_INTERPOLATION_LINEAR,
                                     border_type, border_value));
    EXPECT_EQ(KLEIDICV_OK,
              remap_f32<TypeParam>()(src, big_stride, kW, 2, dst, dst_stride,
                                     kW, 0, 1, mapx, mapx_stride, mapy,
                                     mapy_stride, KLEIDICV_INTERPOLATION_LINEAR,
                                     border_type, border_value));
  }
  const TypeParam border_value[1] = {0};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_f32<TypeParam>()(
                src, src_stride, kW, 0, dst, dst_stride, kW, 1, 1, mapx,
                mapx_stride, mapy, mapy_stride, KLEIDICV_INTERPOLATION_LINEAR,
                KLEIDICV_BORDER_TYPE_CONSTANT, border_value));
}

TYPED_TEST(RemapF32, InvalidImageSize) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[1];
  float mapx[1] = {};
  float mapy[1] = {};

  // TODO: Why these sets of parameters?
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      remap_f32<TypeParam>()(src, element_size, KLEIDICV_MAX_IMAGE_PIXELS + 1,
                             1, dst, element_size, 1, 1, 1, mapx, sizeof(float),
                             mapy, sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_f32<TypeParam>()(src, element_size, KLEIDICV_MAX_IMAGE_PIXELS,
                                   KLEIDICV_MAX_IMAGE_PIXELS, dst, element_size,
                                   1, 1, 1, mapx, sizeof(float), mapy,
                                   sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            remap_f32<TypeParam>()(src, element_size, 1, 1, dst, element_size,
                                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, mapx,
                                   sizeof(float), mapy, sizeof(float),
                                   KLEIDICV_INTERPOLATION_LINEAR,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      remap_f32<TypeParam>()(src, element_size, 1, 1, dst, element_size,
                             KLEIDICV_MAX_IMAGE_PIXELS,
                             KLEIDICV_MAX_IMAGE_PIXELS, 1, mapx, sizeof(float),
                             mapy, sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedThreeChannels) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};
  const size_t channels = 3;

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_f32<TypeParam>()(src, element_size, 1, 1, dst, 16 * element_size,
                             16, 1, channels, mapx, 16 * sizeof(float), mapy,
                             16 * sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedTooSmallImage) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_f32<TypeParam>()(src, element_size, 1, 1, dst, 16 * element_size, 3,
                             1, 1, mapx, 16 * sizeof(float), mapy,
                             16 * sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedBigStride) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};
  const uint64_t src_stride =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_f32<TypeParam>()(src, src_stride, 1, 1, dst, 16 * element_size, 16,
                             1, 1, mapx, 16 * sizeof(float), mapy,
                             16 * sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedBigSourceWidth) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_f32<TypeParam>()(src, element_size, 1ULL << 24, 1, dst,
                                   16 * element_size, 16, 1, 1, mapx,
                                   16 * sizeof(float), mapy, 16 * sizeof(float),
                                   KLEIDICV_INTERPOLATION_LINEAR,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_f32<TypeParam>()(src, element_size, (1ULL << 32) + 1, 1, dst,
                                   16 * element_size, 16, 1, 1, mapx,
                                   16 * sizeof(float), mapy, 16 * sizeof(float),
                                   KLEIDICV_INTERPOLATION_LINEAR,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedBigSourceHeight) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_f32<TypeParam>()(src, element_size, 1, 1ULL << 24, dst,
                                   16 * element_size, 16, 1, 1, mapx,
                                   16 * sizeof(float), mapy, 16 * sizeof(float),
                                   KLEIDICV_INTERPOLATION_LINEAR,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            remap_f32<TypeParam>()(src, element_size, 1, (1ULL << 32) + 1, dst,
                                   16 * element_size, 16, 1, 1, mapx,
                                   16 * sizeof(float), mapy, 16 * sizeof(float),
                                   KLEIDICV_INTERPOLATION_LINEAR,
                                   KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedBigDestinationWidth) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_f32<TypeParam>()(src, element_size, 1, 1, dst, 16 * element_size,
                             1ULL << 24, 1, 1, mapx, 16 * sizeof(float), mapy,
                             16 * sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, UnsupportedBigDestinationHeight) {
  const size_t element_size = sizeof(TypeParam);
  const TypeParam src[1] = {};
  TypeParam dst[16];
  float mapx[16] = {};
  float mapy[16] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      remap_f32<TypeParam>()(src, element_size, 1, 1, dst, 16 * element_size,
                             16, 1ULL << 24, 1, mapx, 16 * sizeof(float), mapy,
                             16 * sizeof(float), KLEIDICV_INTERPOLATION_LINEAR,
                             KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

TYPED_TEST(RemapF32, Misalignment) {
  const TypeParam src[8] = {};
  TypeParam dst[8];
  const size_t data_stride_ok = sizeof(TypeParam);
  const size_t data_stride_nok = sizeof(TypeParam) + 1;
  float mapx[8] = {};
  float mapy[8] = {};
  const size_t map_stride_ok = sizeof(float);
  const size_t map_stride_nok = sizeof(float) + 1;
  auto I = KLEIDICV_INTERPOLATION_LINEAR;
  auto B = KLEIDICV_BORDER_TYPE_REPLICATE;

  // Is misalignment of the data possible?
  if (data_stride_ok != 1) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              remap_f32<TypeParam>()(
                  src, data_stride_nok, 4, 2, dst, data_stride_ok, 4, 2, 1,
                  mapx, map_stride_ok, mapy, map_stride_ok, I, B, nullptr));
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              remap_f32<TypeParam>()(
                  src, data_stride_ok, 4, 2, dst, data_stride_nok, 4, 2, 1,
                  mapx, map_stride_ok, mapy, map_stride_ok, I, B, nullptr));
  }
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            remap_f32<TypeParam>()(
                src, data_stride_nok, 4, 2, dst, data_stride_ok, 4, 2, 1, mapx,
                map_stride_nok, mapy, map_stride_ok, I, B, nullptr));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            remap_f32<TypeParam>()(src, data_stride_ok, 4, 2, dst,
                                   data_stride_ok, 4, 2, 1, mapx, map_stride_ok,
                                   mapy, map_stride_nok, I, B, nullptr));
}
