// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/generator.h"
#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_THRESHOLD_BINARY(type, suffix) \
  KLEIDICV_API(threshold_binary, kleidicv_threshold_binary_##suffix, type)

KLEIDICV_THRESHOLD_BINARY(uint8_t, u8);

template <typename ElementType>
class ThresholdBinaryTestBase : public UnaryOperationTest<ElementType> {
 protected:
  // Needed to initialize value()
  using UnaryOperationTest<ElementType>::max;

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return kleidicv_threshold_binary_u8(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height(), this->threshold(), this->value());
  }

  virtual ElementType threshold() = 0;
  constexpr ElementType value() { return max(); }
};  // end of class ThresholdBinaryTestBase<ElementType>

template <typename ElementType>
class ThresholdBinaryTest final : public ThresholdBinaryTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;

  ElementType threshold() override { return 13; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 12,             0},
      { 13,             0},
      { 14, this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class ThresholdBinaryTestMin final
    : public ThresholdBinaryTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<ElementType>::lowest;

  ElementType threshold() override { return lowest(); }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {    lowest(),             0},
      {lowest() + 1, this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class ThresholdBinaryTestZero final
    : public ThresholdBinaryTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;

  ElementType threshold() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 0,             0},
      { 1, this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class ThresholdBinaryTestMax final
    : public ThresholdBinaryTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<ElementType>::max;

  ElementType threshold() override { return max() - 1; }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {max() - 1,             0},
      {    max(), this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class ThresholdBinary : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t>;

TYPED_TEST_SUITE(ThresholdBinary, ElementTypes);

TYPED_TEST(ThresholdBinary, Test) { ThresholdBinaryTest<TypeParam>{}.test(); }

// Tests various padding combinations.
TYPED_TEST(ThresholdBinary, Padding) {
  ThresholdBinaryTest<TypeParam>{}.with_paddings({0}, {0}).test();
  ThresholdBinaryTest<TypeParam>{}.with_paddings({0}, {1}).test();
  ThresholdBinaryTest<TypeParam>{}.with_paddings({1}, {0}).test();
  ThresholdBinaryTest<TypeParam>{}.with_paddings({1}, {1}).test();
}

TYPED_TEST(ThresholdBinary, InPlaceOperation) {
  const size_t kWidth = test::Options::vector_length() + 1;
  constexpr size_t kHeight = 19;
  constexpr size_t kPadding = 3;
  constexpr TypeParam kThreshold = 113;
  constexpr TypeParam kValue = 201;

  test::PseudoRandomNumberGenerator<TypeParam> generator;
  test::Array2D<TypeParam> source{kWidth, kHeight, kPadding};
  test::Array2D<TypeParam> in_place{kWidth, kHeight, kPadding};
  test::Array2D<TypeParam> out_of_place{kWidth, kHeight, kPadding};

  source.fill(generator);
  in_place = source;

  ASSERT_EQ(KLEIDICV_OK,
            threshold_binary<TypeParam>()(in_place.data(), in_place.stride(),
                                          in_place.data(), in_place.stride(),
                                          kWidth, kHeight, kThreshold, kValue));
  ASSERT_EQ(KLEIDICV_OK,
            threshold_binary<TypeParam>()(
                source.data(), source.stride(), out_of_place.data(),
                out_of_place.stride(), kWidth, kHeight, kThreshold, kValue));
  EXPECT_EQ_ARRAY2D(in_place, out_of_place);
}

TYPED_TEST(ThresholdBinary, TestMin) {
  ThresholdBinaryTestMin<TypeParam>{}.test();
}

TYPED_TEST(ThresholdBinary, TestZero) {
  ThresholdBinaryTestZero<TypeParam>{}.test();
}

TYPED_TEST(ThresholdBinary, TestMax) {
  ThresholdBinaryTestMax<TypeParam>{}.test();
}

TYPED_TEST(ThresholdBinary, NullPointer) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  test::test_null_args(threshold_binary<TypeParam>(), src, sizeof(TypeParam),
                       dst, sizeof(TypeParam), 1, 1, 1, 1);
}

TYPED_TEST(ThresholdBinary, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            threshold_binary<TypeParam>()(src, sizeof(TypeParam) + 1, dst,
                                          sizeof(TypeParam), 1, 1, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            threshold_binary<TypeParam>()(src, sizeof(TypeParam), dst,
                                          sizeof(TypeParam) + 1, 1, 1, 1, 1));
}

TYPED_TEST(ThresholdBinary, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK,
            threshold_binary<TypeParam>()(src, sizeof(TypeParam), dst,
                                          sizeof(TypeParam), 0, 1, 1, 1));
  EXPECT_EQ(KLEIDICV_OK,
            threshold_binary<TypeParam>()(src, sizeof(TypeParam), dst,
                                          sizeof(TypeParam), 1, 0, 1, 1));
}

TYPED_TEST(ThresholdBinary, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            threshold_binary<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            threshold_binary<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 1, 1));
}
