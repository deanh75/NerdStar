// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/generator.h"
#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

template <typename ElementType>
class SaturatingAddAbsWithThresholdTestBase
    : public BinaryOperationTest<ElementType> {
 protected:
  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return kleidicv_saturating_add_abs_with_threshold_s16(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->inputs_[1].data(), this->inputs_[1].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height(), this->threshold());
  }

  virtual ElementType threshold() = 0;
};  // end of class SaturatingAddAbsWithThresholdTestBase<ElementType>

template <typename ElementType>
class SaturatingAddAbsWithThresholdTestPositive final
    : public SaturatingAddAbsWithThresholdTestBase<ElementType> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  ElementType threshold() override { return 50; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {10, 39,  0},
      {10, 40,  0},
      {10, 41, 51},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class SaturatingAddAbsWithThresholdTestNegative final
    : public SaturatingAddAbsWithThresholdTestBase<ElementType> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  ElementType threshold() override { return 50; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {-10, -39,  0},
      {-10, -40,  0},
      {-10, -41, 51},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class SaturatingAddAbsWithThresholdTestMin final
    : public SaturatingAddAbsWithThresholdTestBase<ElementType> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::lowest;
  using BinaryOperationTest<ElementType>::max;

  ElementType threshold() override { return lowest(); }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { lowest(), lowest(), max()},
      { lowest(),     0, max()},
      { lowest(),     1, max()},
      { lowest(), max(), max()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class SaturatingAddAbsWithThresholdTestZero final
    : public SaturatingAddAbsWithThresholdTestBase<ElementType> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  ElementType threshold() override { return 1; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 0, 0, 0},
      { 0, 1, 0},
      { 0, 2, 2},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class SaturatingAddAbsWithThresholdTestMax final
    : public SaturatingAddAbsWithThresholdTestBase<ElementType> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::max;

  ElementType threshold() override { return max() - 1; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {0, max() - 2,     0},
      {0, max() - 1,     0},
      {0,     max(), max()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class SaturatingAddAbsWithThresholdTest : public testing::Test {};

using ElementTypes = ::testing::Types<int16_t>;

// Tests saturating_add_abs_with_threshold<type> API.
TYPED_TEST_SUITE(SaturatingAddAbsWithThresholdTest, ElementTypes);

TYPED_TEST(SaturatingAddAbsWithThresholdTest, TestPositive) {
  SaturatingAddAbsWithThresholdTestPositive<TypeParam>{}.test();
  SaturatingAddAbsWithThresholdTestPositive<TypeParam>{}.with_padding(1).test();
  SaturatingAddAbsWithThresholdTestPositive<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, TestNegative) {
  SaturatingAddAbsWithThresholdTestNegative<TypeParam>{}.test();
  SaturatingAddAbsWithThresholdTestNegative<TypeParam>{}.with_padding(1).test();
  SaturatingAddAbsWithThresholdTestNegative<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, TestMin) {
  SaturatingAddAbsWithThresholdTestMin<TypeParam>{}.test();
  SaturatingAddAbsWithThresholdTestMin<TypeParam>{}.with_padding(1).test();
  SaturatingAddAbsWithThresholdTestMin<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, TestZero) {
  SaturatingAddAbsWithThresholdTestZero<TypeParam>{}.test();
  SaturatingAddAbsWithThresholdTestZero<TypeParam>{}.with_padding(1).test();
  SaturatingAddAbsWithThresholdTestZero<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, TestMax) {
  SaturatingAddAbsWithThresholdTestMax<TypeParam>{}.test();
  SaturatingAddAbsWithThresholdTestMax<TypeParam>{}.with_padding(1).test();
  SaturatingAddAbsWithThresholdTestMax<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, InPlaceOperation) {
  const size_t kWidth = test::Options::vector_length() + 1;
  constexpr size_t kHeight = 19;
  constexpr size_t kPadding = 3;
  constexpr TypeParam kThreshold = 31;

  test::PseudoRandomNumberGenerator<TypeParam> generator;
  test::Array2D<TypeParam> src_a{kWidth, kHeight, kPadding};
  test::Array2D<TypeParam> src_b{kWidth, kHeight, kPadding};
  test::Array2D<TypeParam> in_place{kWidth, kHeight, kPadding};
  test::Array2D<TypeParam> out_of_place{kWidth, kHeight, kPadding};

  src_a.fill(generator);
  src_b.fill(generator);
  in_place = src_a;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_saturating_add_abs_with_threshold_s16(
                             in_place.data(), in_place.stride(), src_b.data(),
                             src_b.stride(), in_place.data(), in_place.stride(),
                             kWidth, kHeight, kThreshold));
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
                out_of_place.data(), out_of_place.stride(), kWidth, kHeight,
                kThreshold));
  EXPECT_EQ_ARRAY2D(in_place, out_of_place);

  in_place = src_b;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_saturating_add_abs_with_threshold_s16(
                             src_a.data(), src_a.stride(), in_place.data(),
                             in_place.stride(), in_place.data(),
                             in_place.stride(), kWidth, kHeight, kThreshold));
  EXPECT_EQ_ARRAY2D(in_place, out_of_place);
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, NullPointer) {
  TypeParam src[1] = {}, dst[1];
  test::test_null_args(kleidicv_saturating_add_abs_with_threshold_s16, src,
                       sizeof(TypeParam), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), 1, 1, 1);
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2] = {};
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src, sizeof(TypeParam) + 1, src, sizeof(TypeParam), dst,
                sizeof(TypeParam), 1, 2, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src, sizeof(TypeParam), src, sizeof(TypeParam) + 1, dst,
                sizeof(TypeParam), 1, 2, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam) + 1, 1, 2, 1));
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, kleidicv_saturating_add_abs_with_threshold_s16(
                             src, sizeof(TypeParam), src, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 0, 1, 1));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_saturating_add_abs_with_threshold_s16(
                             src, sizeof(TypeParam), src, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 1, 0, 1));
}

TYPED_TEST(SaturatingAddAbsWithThresholdTest, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_saturating_add_abs_with_threshold_s16(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, 1));
}
