// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/warp_perspective.h"

// clang-format off
static const float transform_identity[] = {
  1.0, 0, 0,
  0, 1.0, 0,
  0, 0, 1.0
};

static const float transform_transpose[] = {
  0, 1.0, 0,
  1.0, 0, 0,
  0, 0, 1.0
};

static const float transform_small[] = {
  0.8, 0.1, 2,
  0.1, 0.8, -2,
  0.001, 0.001, 1.7
};
// clang-format on

// Perspective calculation in float greatly amplifies any small errors coming
// from precision innaccuracies, let it be Fused-Multiply-Add or just the
// limitation of the single precision float.
// Taking the fractional part of a value in the thousands' range decreases
// the precision by 3 decimal digits, and the single precision float only has
// 7 decimal digits. Doing a linear interpolation between 0 and 255 would
// decrease it even more, but ensuring that neighbouring pixels have
// neighbouring values decreases this effect by more than 2 decimal digits so it
// can be expected that the error won't be bigger than 1.
template <class ScalarType>
static void sequential_initializer(test::Array2D<ScalarType> &source) {
  for (int64_t y = 0; y < static_cast<int64_t>(source.height()); ++y) {
    for (int64_t x = 0; x < static_cast<int64_t>(source.width()); ++x) {
      const int64_t kMaxVal = std::numeric_limits<ScalarType>::max();
      *source.at(y, x) = abs((x + y) % (2 * kMaxVal + 1) - kMaxVal);
    }
  }
}

template <typename ScalarType>
static const ScalarType *get_array2d_element_or_border(
    const test::Array2D<ScalarType> &src, ptrdiff_t x, ptrdiff_t y,
    kleidicv_border_type_t border_type, const ScalarType *border_value) {
  if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
    x = std::clamp<ptrdiff_t>(x, 0, static_cast<ptrdiff_t>(src.width()) - 1);
    y = std::clamp<ptrdiff_t>(y, 0, static_cast<ptrdiff_t>(src.height()) - 1);
  } else {
    assert(border_type == KLEIDICV_BORDER_TYPE_CONSTANT);
    if (x >= static_cast<ptrdiff_t>(src.width()) ||
        y >= static_cast<ptrdiff_t>(src.height()) || x < 0 || y < 0) {
      return border_value;
    }
  }
  return src.at(y, x * src.channels());
}

template <typename T>
static const auto &get_borders() {
  using P = std::pair<kleidicv_border_type_t, const T *>;
  static const T border_value[KLEIDICV_MAXIMUM_CHANNEL_COUNT] = {4, 3, 2, 1};
  static const std::array borders{
      P{KLEIDICV_BORDER_TYPE_REPLICATE, nullptr},
      P{KLEIDICV_BORDER_TYPE_REPLICATE, border_value},
      P{KLEIDICV_BORDER_TYPE_CONSTANT, border_value},
  };
  return borders;
}

template <class ScalarType, kleidicv_interpolation_type_t Interpolation>
class WarpPerspectiveBase : public testing::Test {
 public:
  static void test_stripe(size_t src_w, size_t src_h, size_t dst_w,
                          size_t dst_h, size_t y_begin, size_t y_end,
                          const float transform[9], size_t channels,
                          kleidicv_border_type_t border_type,
                          const ScalarType *border_value, size_t padding,
                          void (*initializer)(test::Array2D<ScalarType> &) =
                              sequential_initializer) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    initializer(source);
    for (size_t row = y_begin; row < y_end; ++row) {
      for (size_t column = 0; column < actual.width(); ++column) {
        *actual.at(row, column) = 42;
      }
    }

    calculate_expected(source, y_begin, y_end, transform, border_type,
                       border_value, expected);

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_warp_perspective_stripe_u8(
                  source.data(), source.stride(), source.width(),
                  source.height(), actual.data(), actual.stride(),
                  actual.width(), actual.height(), y_begin, y_end, transform,
                  channels, Interpolation, border_type, border_value));

    ScalarType threshold = 1;
    for (size_t row = y_begin; row < y_end; ++row) {
      for (size_t column = 0; column < expected.width(); ++column) {
        const ScalarType *lhs = expected.at(row, column);
        const ScalarType *rhs = actual.at(row, column);
        if (!lhs || !rhs) {
          GTEST_FAIL() << "Nullptr at (row=" << row << ", column=" << column
                       << "): " << reinterpret_cast<const void *>(lhs) << " , "
                       << reinterpret_cast<const void *>(lhs) << std::endl;
        }
        ScalarType error;
        if constexpr (std::is_integral<ScalarType>::value) {
          error = static_cast<ScalarType>(
              abs(static_cast<int64_t>(lhs[0]) - static_cast<int64_t>(rhs[0])));
        } else {
          error = std::abs(lhs[0] - rhs[0]);
        }
        if (error > threshold) {
          if (source.width() < 100 && source.height() < 100) {
            std::cout << "source:\n";
            dump(&source);
          }
          std::cout << "transform:\n";
          std::cout << transform[0] << "  " << transform[1] << "  "
                    << transform[2] << "\n";
          std::cout << transform[3] << "  " << transform[4] << "  "
                    << transform[5] << "\n";
          std::cout << transform[6] << "  " << transform[7] << "  "
                    << transform[8] << "\n\n";
          std::cout << "expected:\n";
          dump(&expected);
          std::cout << "actual:\n";
          dump(&actual);

          GTEST_FAIL() << "Mismatch at (row=" << row << ", col=" << column
                       << "): " << +(expected).at(row, column)[0] << " vs "
                       << +(actual).at(row, column)[0] << "." << std::endl;
        }
      }
    }
  }

  static void test(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
                   const float transform[9], size_t channels,
                   kleidicv_border_type_t border_type,
                   const ScalarType *border_value, size_t padding,
                   void (*initializer)(test::Array2D<ScalarType> &) =
                       sequential_initializer) {
    test_stripe(src_w, src_h, dst_w, dst_h, 0, dst_h, transform, channels,
                border_type, border_value, padding, initializer);
  }

  static void test_special_source(
      size_t src_w, size_t src_h, size_t src_stride, size_t dst_w, size_t dst_h,
      const float transform[9], size_t channels,
      kleidicv_border_type_t border_type, const ScalarType *border_value,
      void (*initializer)(test::Array2D<ScalarType> &) =
          sequential_initializer) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    // Use fake parameters to avoid large filling of padding
    size_t total_width_bytes = src_total_width * sizeof(ScalarType);
    size_t physical_stride = std::max(src_stride, total_width_bytes);
    size_t physical_width = physical_stride / sizeof(ScalarType);

    test::Array2D<ScalarType> source{physical_width, src_h, 0, channels};
    initializer(source);

    test::Array2D<ScalarType> actual{dst_total_width, dst_h, 0, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, 0, channels};

    actual.fill(42);

    calculate_expected(source, 0, expected.height(), transform, border_type,
                       border_value, expected);

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_warp_perspective_u8(
                  source.data(), src_stride, src_w, src_h, actual.data(),
                  actual.stride(), actual.width(), actual.height(), transform,
                  channels, Interpolation, border_type, border_value));

    EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, actual, expected);
  }

  static void test_random_transforms() {
    float transform[9];
    // Not entirely random, as very small and very big floats (in absolute
    // value) cause too big errors and they are far from being valid use cases
    // anyway
    test::PseudoRandomNumberGeneratorIntRange<int16_t> exponentGenerator(-10,
                                                                         10);
    test::PseudoRandomNumberGeneratorFloatRange<float> mantissaGenerator(-1.0,
                                                                         1.0);

    for (size_t cc = 0; cc < 100; ++cc) {
      for (size_t i = 0; i < 9; ++i) {
        transform[i] =
            mantissaGenerator.next().value_or(1.0) *
            static_cast<float>(
                exp(static_cast<double>(exponentGenerator.next().value_or(1))));
      }

      size_t src_w = 3 * test::Options::vector_lanes<ScalarType>() - 1;
      size_t src_h = 4;
      size_t dst_w = src_w;
      size_t dst_h = src_h;
      for (auto [border_type, border_value] : get_borders<ScalarType>()) {
        test(src_w, src_h, dst_w, dst_h, transform, 1, border_type,
             border_value, 19);
      }
    }
  }

 private:
  static void calculate_expected(test::Array2D<ScalarType> &src, size_t y_begin,
                                 size_t y_end, const float transform[9],
                                 kleidicv_border_type_t border_type,
                                 const ScalarType *border_value,
                                 test::Array2D<ScalarType> &expected) {
    auto get_src = [&](ptrdiff_t x, ptrdiff_t y) {
      return get_array2d_element_or_border(src, x, y, border_type,
                                           border_value);
    };

    for (size_t y = y_begin; y < y_end; ++y) {
      for (size_t x = 0; x < expected.width() / src.channels(); ++x) {
        if constexpr (Interpolation == KLEIDICV_INTERPOLATION_NEAREST) {
          float dx = static_cast<float>(x), dy = static_cast<float>(y);
          float dw = transform[6] * dx + transform[7] * dy + transform[8];
          float inv_w = 1.F / dw;
          float fx =
              inv_w * (transform[0] * dx + transform[1] * dy + transform[2]);
          float fy =
              inv_w * (transform[3] * dx + transform[4] * dy + transform[5]);
          ptrdiff_t ix = static_cast<ptrdiff_t>(std::max<double>(
              INT_MIN,
              std::min<double>(std::round(fx), KLEIDICV_MAX_IMAGE_PIXELS)));
          ptrdiff_t iy = static_cast<ptrdiff_t>(std::max<double>(
              INT_MIN,
              std::min<double>(std::round(fy), KLEIDICV_MAX_IMAGE_PIXELS)));
          for (size_t ch = 0; ch < src.channels(); ++ch) {
            *expected.at(y, x * src.channels() + ch) = get_src(ix, iy)[ch];
          }
        }
        if constexpr (Interpolation == KLEIDICV_INTERPOLATION_LINEAR) {
          double dx = static_cast<double>(x), dy = static_cast<double>(y);
          double tw = transform[6] * dx + transform[7] * dy + transform[8];
          double inv_w = 1.F / tw;
          double tx = transform[0] * dx + transform[1] * dy + transform[2];
          double ty = transform[3] * dx + transform[4] * dy + transform[5];
          double fx = inv_w * tx;
          double fy = inv_w * ty;
          ptrdiff_t ix0 = static_cast<ptrdiff_t>(std::max<double>(
              INT_MIN, std::min<double>(floor(fx), KLEIDICV_MAX_IMAGE_PIXELS)));
          ptrdiff_t iy0 = static_cast<ptrdiff_t>(std::max<double>(
              INT_MIN, std::min<double>(floor(fy), KLEIDICV_MAX_IMAGE_PIXELS)));
          ptrdiff_t ix1 = ix0 + 1;
          ptrdiff_t iy1 = iy0 + 1;
          double xfrac = fx - floor(fx);
          double yfrac = fy - floor(fy);
          for (size_t ch = 0; ch < src.channels(); ++ch) {
            double a = get_src(ix0, iy0)[ch];
            double b = get_src(ix1, iy0)[ch];
            double c = get_src(ix0, iy1)[ch];
            double d = get_src(ix1, iy1)[ch];
            double line1 = (b - a) * xfrac + a;
            double line2 = (d - c) * xfrac + c;
            double double_result = (line2 - line1) * yfrac + line1;
            *expected.at(y, x * src.channels() + ch) =
                static_cast<ScalarType>(lround(double_result));
          }
        }
      }
    }
  }
};

template <class ScalarType>
class WarpPerspectiveNearest
    : public WarpPerspectiveBase<ScalarType, KLEIDICV_INTERPOLATION_NEAREST> {};

using WarpPerspectiveElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(WarpPerspectiveNearest, WarpPerspectiveElementTypes);

TYPED_TEST(WarpPerspectiveNearest, IdentityNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_identity, 1,
                      border_type, border_value, 0);
  }
}

TYPED_TEST(WarpPerspectiveNearest, TransposeNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_transpose, 1,
                      border_type, border_value, 0);
  }
}

TYPED_TEST(WarpPerspectiveNearest, SmallPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_small, 1,
                      border_type, border_value, 13);
  }
}

TYPED_TEST(WarpPerspectiveNearest, Upscale) {
  // clang-format off
  const float transform_upscale[] = {
    0.2, 0.05, 0.1,
    0.02, 0.2, -0.1,
    0.0001, 0.0001, 1.1
  };
  // clang-format on

  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w * 3;
  size_t dst_h = src_h * 2;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_upscale, 1,
                      border_type, border_value, 3);
  }
}

TYPED_TEST(WarpPerspectiveNearest, RandomTransform) {
  TestFixture::test_random_transforms();
}

TYPED_TEST(WarpPerspectiveNearest, DivisionByZero) {
  // clang-format off
  const float transform_div_by_zero[] = {
    1, 0, 0,
    0, 1, 0,
    1, 1, -3
  };
  // clang-format on

  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_div_by_zero, 1,
                      border_type, border_value, 3);
  }
}

static const size_t kBigWidth = 1ULL << 17, kBigHeight = 1ULL << 17;
static const size_t kPartWidth = 21, kPartHeight = 21;

template <class ScalarType>
static void part_initializer(test::Array2D<ScalarType> &source) {
  ScalarType counter = 0;
  for (size_t y = kBigHeight; y < kBigHeight + kPartHeight; ++y) {
    for (size_t x = kBigWidth; x < kBigWidth + kPartWidth; ++x) {
      *source.at(y, x) = ++counter;
    }
  }
}

TYPED_TEST(WarpPerspectiveNearest, BigSourceImage) {
  // clang-format off
  const float transform_cut[] = {
    1, 0, kBigWidth + kPartWidth * 0.5F,
    0, 1, kBigHeight + kPartHeight * 0.5F,
    0, 0, 1
  };
  // clang-format on

  // Let stride * height be bigger than 1 << 32
  size_t dst_w = kPartWidth;
  size_t dst_h = kPartHeight;
  size_t src_w = kBigWidth + kPartWidth;
  size_t src_h = kBigHeight + kPartHeight;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_cut, 1, border_type,
                      border_value, 0, part_initializer);
  }
}

TYPED_TEST(WarpPerspectiveNearest, BigWidthDestination) {
  // clang-format off
  const float transform_upscale[] = {
    0.0011, 0, 0.3,
    0, 1.02, -0.1,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = kBigWidth;
  size_t dst_h = 1;
  size_t src_w = dst_w / 1000;
  size_t src_h = dst_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_upscale, 1,
                      border_type, border_value, 3);
  }
}

static const size_t kHugeHeight = (1ULL << 24) - 1;
TYPED_TEST(WarpPerspectiveNearest, BigHeightDestination) {
  // clang-format off
  const float transform_upscale[] = {
    1.03, 0, 0.1,
    0, 0.00011, -0.3,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = 17;
  size_t dst_h = kHugeHeight;
  size_t src_w = dst_w;
  size_t src_h = dst_h / 10000;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_stripe(src_w, src_h, dst_w, dst_h, dst_h - kPartHeight,
                             dst_h, transform_upscale, 1, border_type,
                             border_value, 0);
  }
}

template <class ScalarType>
static void huge_height_part_initializer(test::Array2D<ScalarType> &source) {
  ScalarType counter = 0;
  for (size_t y = kHugeHeight - kPartHeight; y < kHugeHeight; ++y) {
    for (size_t x = 0; x < 17; ++x) {
      *source.at(y, x) = ++counter;
    }
  }
}

TYPED_TEST(WarpPerspectiveNearest, HugeHeightSourceAndDestination) {
  // clang-format off
  const float transform[] = {
    1.03, 0, 0.1,
    0, 0.9993, -0.3,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = 17;
  size_t dst_h = (1ULL << 24) - 1;
  size_t src_w = dst_w;
  size_t src_h = dst_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_stripe(src_w, src_h, dst_w, dst_h, dst_h - 16, dst_h,
                             transform, 1, border_type, border_value, 0,
                             huge_height_part_initializer<TypeParam>);
  }
}

static const size_t oneline_big_width = 1ULL << 23;
static const size_t oneline_part_width = 16, oneline_part_offset = 1ULL << 17,
                    oneline_dst_height = 16;

template <class ScalarType>
static void oneline_part_initializer(test::Array2D<ScalarType> &source) {
  ScalarType counter = 0;
  for (size_t x = oneline_part_offset;
       x < oneline_part_offset + oneline_part_width; ++x) {
    *source.at(0, x) = ++counter;
  }
}

TYPED_TEST(WarpPerspectiveNearest, OneLineBigSourceImage) {
  // clang-format off
  const float tr[] = {
    1, 0, oneline_part_offset,
    0, 0, 0,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = oneline_part_width;
  size_t dst_h = oneline_dst_height;
  size_t src_w = oneline_big_width + oneline_part_width;
  size_t src_h = 1;

  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_special_source(src_w, src_h, 0, dst_w, dst_h, tr, 1,
                                     border_type, border_value,
                                     oneline_part_initializer);

    TestFixture::test_special_source(src_w, src_h, (1ULL << 32) - 1, dst_w,
                                     dst_h, tr, 1, border_type, border_value,
                                     oneline_part_initializer);
  }
}

TYPED_TEST(WarpPerspectiveNearest, OneLineSmallSourceImage) {
  // clang-format off
  const float tr[] = {
    -1, 0, 1,
    0, 0, 0,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = oneline_part_width;
  size_t dst_h = oneline_dst_height;
  size_t src_w = 2;
  size_t src_h = 1;

  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_special_source(src_w, src_h, src_h, dst_w, dst_h, tr, 1,
                                     border_type, border_value);
  }
}

TYPED_TEST(WarpPerspectiveNearest, NullPointer) {
  const TypeParam src[4] = {};
  const TypeParam border_value[KLEIDICV_MAXIMUM_CHANNEL_COUNT] = {};
  TypeParam dst[8];
  test::test_null_args(kleidicv_warp_perspective_u8, src, 2, 2, 2, dst, 8, 8, 1,
                       transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_CONSTANT, border_value);
}

TYPED_TEST(WarpPerspectiveNearest, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 0, 1, dst, 1, 0, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 0, dst, 1, 1, 0, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, dst, 8, 8, 1,
                transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_warp_perspective_u8(
          src, 1, KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, dst, 8,
          8, 1, transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
          KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1,
                transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedTwoChannels) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 8, 1, transform_identity, 2,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedBorderType) {
  const TypeParam src[1] = {};
  const TypeParam border_value[KLEIDICV_MAXIMUM_CHANNEL_COUNT] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REFLECT,
                border_value));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedTooSmallImage) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 7, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedBigStride) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1UL << 32, 1, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedBigSourceHeight) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1UL << 24, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedBigSourceWidth) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1UL << 24, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedBigDestinationHeight) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 8, 1UL << 24, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                nullptr));
}

TYPED_TEST(WarpPerspectiveNearest, UnsupportedBigDestinationWidth) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 1UL << 24, 1UL << 24, 1, transform_identity,
                1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));
}

/////////////////////////////////////////////////////////////
//       WarpPerspective with Linear Interpolation
/////////////////////////////////////////////////////////////

template <class ScalarType>
class WarpPerspectiveLinear
    : public WarpPerspectiveBase<ScalarType, KLEIDICV_INTERPOLATION_LINEAR> {};

TYPED_TEST_SUITE(WarpPerspectiveLinear, WarpPerspectiveElementTypes);

TYPED_TEST(WarpPerspectiveLinear, SimpleResize) {
  const size_t src_w = 16;
  const size_t src_h = 4;
  const size_t dst_w = 17;
  const size_t dst_h = 5;
  const size_t padding = 1;

  test::Array2D<TypeParam> source(src_w, src_h, padding, 1);
  source.fill(0);
  test::Array2D<TypeParam> actual(dst_w, dst_h, padding, 1);
  actual.fill(0);
  test::Array2D<TypeParam> expected(dst_w, dst_h, padding, 1);
  expected.fill(0);

  // clang-format off
  source.set(0, 0, { 0, 100,   0, 100,   0, 100,   0, 100,   0, 100,   0, 100,   0, 100,   0, 100 });
  source.set(1, 0, { 0, 100,   0, 100,   0, 100,   0, 100,   0, 100,   0, 100,   0, 100,   0, 100});
  source.set(2, 0, { 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100});

  // first line: y = -0.3, so only the first line counts. x = [0, 0.9, 1.8, 2.7, 3.6, ...]
  expected.set(0, 0, {  0,  90,  20,  70,  40,  50,  60,  30,  80,  10, 100,  10,  80,  30,  60,  50,  40 });
  // y = 1.2,  x = [-0.1, 0.8, 1.7, 2.6, 3.5, ...]
  expected.set(1, 0, { 20,  84,  44,  68,  60,  52,  76,  36,  92,  20,  92,  36,  76,  52,  60,  68,  44 });
  // y = 2.7,  x = [-0.2, 0.7, 1.6, 2.5, ...]
  expected.set(2, 0, { 30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30 });
  // y = 4.2 and 5.7: all zeroes, x does not count

  const float little_scale[] = {
    0.9, -0.1, 0,
    0, 1.5, -0.3,
    0, 0, 1
  };
  // clang-format on

  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_warp_perspective_u8(
                source.data(), source.stride(), source.width(), source.height(),
                actual.data(), actual.stride(), actual.width(), actual.height(),
                little_scale, 1, KLEIDICV_INTERPOLATION_LINEAR,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ_ARRAY2D(actual, expected);
}

TYPED_TEST(WarpPerspectiveLinear, IdentityNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_identity, 1,
                      border_type, border_value, 0);
  }
}

TYPED_TEST(WarpPerspectiveLinear, TransposeNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_transpose, 1,
                      border_type, border_value, 0);
  }
}

TYPED_TEST(WarpPerspectiveLinear, SmallPadding) {
  size_t src_w = 5 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_small, 1,
                      border_type, border_value, 13);
  }
}

TYPED_TEST(WarpPerspectiveLinear, Upscale) {
  // clang-format off
  const float transform_upscale[] = {
    0.2, 0.05, 0.1,
    0.02, 0.2, -0.1,
    0.0001, 0.0001, 1.1
  };
  // clang-format on

  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w * 3;
  size_t dst_h = src_h * 2;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_upscale, 1,
                      border_type, border_value, 3);
  }
}

TYPED_TEST(WarpPerspectiveLinear, RandomTransform) {
  TestFixture::test_random_transforms();
}

TYPED_TEST(WarpPerspectiveLinear, DivisionByZero) {
  // clang-format off
  const float transform_div_by_zero[] = {
    1, 0, 0,
    0, 1, 0,
    1, 1, -3
  };
  // clang-format on

  size_t kW = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t kH = 2;

  test::Array2D<TypeParam> source{kW, kH, 1, 1};
  test::Array2D<TypeParam> dst{kW, kH, 1, 1};

  for (int64_t y = 0; y < static_cast<int64_t>(source.height()); ++y) {
    for (int64_t x = 0; x < static_cast<int64_t>(source.width()); ++x) {
      const int64_t kMaxVal = std::numeric_limits<TypeParam>::max() / 2;
      *source.at(y, x) =
          kMaxVal / 4 + abs((x + y) % (2 * kMaxVal + 1) - kMaxVal);
    }
  }

  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    EXPECT_EQ(KLEIDICV_OK,
              kleidicv_warp_perspective_u8(
                  source.data(), source.stride(), kW, kH, dst.data(),
                  dst.stride(), kW, kH, transform_div_by_zero, 1,
                  KLEIDICV_INTERPOLATION_LINEAR, border_type, border_value));
  }
}

template <class ScalarType>
static void crisscross_initializer(test::Array2D<ScalarType> &source) {
  for (int64_t y = 0; y < static_cast<int64_t>(source.height()); ++y) {
    for (int64_t x = 0; x < static_cast<int64_t>(source.width()); ++x) {
      *source.at(y, x) =
          ((x % 10 == 0 && x > 0) || (y % 10 == 0 && y > 0)) ? 0 : 255;
    }
  }
}

TYPED_TEST(WarpPerspectiveLinear, RoughSource) {
  // clang-format off
  const float transformation[] = {
  -0.14721069682646492627, 0.0039219946625178928046, -0.23197875743643098234,
   -9.6823083601201584969, 0.27678993497443726834, -24.722070405199925602,
   0.043781423781792741523, 0.0021065152608927420821, -0.5757716849053092778
  };
  // clang-format on

  size_t src_w = 13;
  size_t src_h = 634;
  size_t dst_w = 17;
  size_t dst_h = 852;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transformation, 1,
                      border_type, border_value, 13, crisscross_initializer);
  }
}

TYPED_TEST(WarpPerspectiveLinear, BigSourceImage) {
  // clang-format off
  const float transform_cut[] = {
    1, 0, 1ULL<<17,
    0, 1, 1ULL<<17,
    0, 0, 1
  };
  // clang-format on

  // Let stride * height be bigger than 1 << 32
  size_t dst_w = kPartWidth;
  size_t dst_h = kPartHeight;
  size_t src_w = kBigWidth + kPartWidth + 3;
  size_t src_h = kBigHeight + kPartHeight;

  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_cut, 1, border_type,
                      border_value, 0, part_initializer);
  }
}

TYPED_TEST(WarpPerspectiveLinear, BigWidthDestination) {
  // clang-format off
  const float transform_upscale[] = {
    0.0011, 0, 0.3,
    0, 1.02, -0.1,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = kBigWidth;
  size_t dst_h = 1;
  size_t src_w = dst_w / 1000;
  size_t src_h = dst_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test(src_w, src_h, dst_w, dst_h, transform_upscale, 1,
                      border_type, border_value, 3);
  }
}

TYPED_TEST(WarpPerspectiveLinear, BigHeightDestination) {
  // clang-format off
  const float transform_upscale[] = {
    1.03, 0, 0.1,
    0, 0.00011, -0.3,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = 17;
  size_t dst_h = kHugeHeight;
  size_t src_w = dst_w;
  size_t src_h = dst_h / 10000;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_stripe(src_w, src_h, dst_w, dst_h, dst_h - kPartHeight,
                             dst_h, transform_upscale, 1, border_type,
                             border_value, 0);
  }
}

TYPED_TEST(WarpPerspectiveLinear, HugeHeightSourceAndDestination) {
  // clang-format off
  const float transform[] = {
    1.03, 0, 0.1,
    0, 0.9993, -0.3,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = 17;
  size_t dst_h = (1ULL << 24) - 1;
  size_t src_w = dst_w;
  size_t src_h = dst_h;
  for (auto [border_type, border_value] : get_borders<TypeParam>()) {
    TestFixture::test_stripe(src_w, src_h, dst_w, dst_h, dst_h - kPartHeight,
                             dst_h, transform, 1, border_type, border_value, 0,
                             huge_height_part_initializer<TypeParam>);
  }
}
