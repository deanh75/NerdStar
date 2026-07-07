// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

template <typename DestinationType, typename SourceType>
static DestinationType saturating_cast(SourceType value) {
  if (value >
      static_cast<SourceType>(std::numeric_limits<DestinationType>::max())) {
    return std::numeric_limits<DestinationType>::max();
  }
  if (value < std::numeric_limits<DestinationType>::lowest()) {
    return std::numeric_limits<DestinationType>::lowest();
  }
  return static_cast<DestinationType>(value);
}

uint8_t scalar_scale_u8_u8(uint8_t x, double scale, double shift) {
  float result = static_cast<float>(x * scale + shift);
  if (result < std::numeric_limits<uint8_t>::lowest()) {
    return std::numeric_limits<uint8_t>::lowest();
  }
  if (result > std::numeric_limits<uint8_t>::max()) {
    return std::numeric_limits<uint8_t>::max();
  }
  return static_cast<uint8_t>(lrintf(result));
}

float scalar_scale_f32_f32(float x, double scale, double shift) {
  return x * static_cast<float>(scale) + static_cast<float>(shift);
}

float16_t scalar_scale_u8_f16(uint8_t x, double scale, double shift) {
  return static_cast<float16_t>(x * scale + shift);
}

#define KLEIDICV_SCALE_API_SAMETYPE(type, suffix)                           \
  KLEIDICV_API_DIFFERENT_IO_TYPES(scale_api, kleidicv_scale_##suffix, type, \
                                  type)

#define KLEIDICV_SCALE_API(src_type, dst_type, src_suffix, dst_suffix)        \
  KLEIDICV_API_DIFFERENT_IO_TYPES(scale_api,                                  \
                                  kleidicv_scale_##src_suffix##_##dst_suffix, \
                                  src_type, dst_type)

#define KLEIDICV_SCALE_OPERATION(src_type, dst_type, src_suffix, dst_suffix) \
  KLEIDICV_API_DIFFERENT_IO_TYPES(scale_operation,                           \
                                  &scalar_scale_##src_suffix##_##dst_suffix, \
                                  src_type, dst_type)

KLEIDICV_SCALE_API_SAMETYPE(uint8_t, u8);
KLEIDICV_SCALE_OPERATION(uint8_t, uint8_t, u8, u8);
KLEIDICV_SCALE_API_SAMETYPE(float, f32);
KLEIDICV_SCALE_OPERATION(float, float, f32, f32);
KLEIDICV_SCALE_API(uint8_t, float16_t, u8, f16);
KLEIDICV_SCALE_OPERATION(uint8_t, float16_t, u8, f16);

template <typename InputType, typename OutputType>
class ScaleTestBase : public OperationTest<InputType, 1, 1, OutputType> {
 protected:
  using OperationTest<InputType, 1, 1, OutputType>::lowest;
  using OperationTest<InputType, 1, 1, OutputType>::max;
  using typename OperationTest<InputType, 1, 1, OutputType>::Elements;

  static constexpr InputType
  input_min_if_integral_else_smallest_positive_normalized() {
    return std::numeric_limits<InputType>::min();
  }

  static constexpr InputType input_lowest() {
    return std::numeric_limits<InputType>::lowest();
  }

  static constexpr InputType input_max() {
    return std::numeric_limits<InputType>::max();
  }

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return scale_api<InputType, OutputType>()(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height(), this->scale(), this->shift());
  }
  virtual double scale() = 0;
  virtual double shift() = 0;

  // Prepares expected outputs for the operation.
  void setup() override {
    this->expected_[0].fill(
        scale_operation<InputType, OutputType>()(0, scale(), shift()));
    OperationTest<InputType, 1, 1, OutputType>::setup();
  }

  void fill_expected(std::vector<Elements>& elements) {
    for (size_t i = 0; i < elements.size(); ++i) {
      std::get<1>(elements[i]) = scale_operation<InputType, OutputType>()(
          std::get<0>(elements[i]), scale(), shift());
    }
  }
};  // end of class ScaleTestBase<InputType, OutputType>

template <typename InputType, typename OutputType>
class ScaleTestLinearBase {
 public:
  // Sets the number of padding bytes at the end of rows.
  ScaleTestLinearBase& with_padding(size_t padding) {
    padding_ = padding;
    return *this;
  }

  // minimum_size set by caller to trigger the 'big' scale path.
  void test_scalar(size_t minimum_size = 1, bool in_place = false) {
    size_t width = test::Options::vector_lanes<InputType>() - 1;
    test_linear(width, minimum_size, in_place);
  }

  void test_vector(size_t minimum_size = 1, bool in_place = false) {
    size_t width = test::Options::vector_lanes<InputType>() * 2;
    test_linear(width, minimum_size, in_place);
  }

 protected:
  static constexpr OutputType
  min_if_integral_else_smallest_positive_normalized() {
    return std::numeric_limits<OutputType>::min();
  }

  static constexpr OutputType lowest() {
    return std::numeric_limits<OutputType>::lowest();
  }

  static constexpr OutputType max() {
    return std::numeric_limits<OutputType>::max();
  }

  static constexpr InputType
  input_min_if_integral_else_smallest_positive_normalized() {
    return std::numeric_limits<InputType>::min();
  }

  static constexpr InputType input_lowest() {
    return std::numeric_limits<InputType>::lowest();
  }

  static constexpr InputType input_max() {
    return std::numeric_limits<InputType>::max();
  }

  virtual double scale() = 0;
  virtual double shift() = 0;

 private:
  class GenerateLinearSeries : public test::Generator<InputType> {
   public:
    explicit GenerateLinearSeries(InputType start_from, InputType step)
        : counter_{start_from}, step_{step} {}

    std::optional<InputType> next() override { return counter_ + step_; }

   private:
    InputType counter_, step_;
  };  // end of class GenerateLinearSeries

  // Number of padding bytes at the end of rows.
  size_t padding_{0};

  void test_linear(size_t width, size_t minimum_size, bool in_place) {
    size_t image_size =
        std::max(minimum_size, std::min(saturating_cast<size_t, InputType>(
                                            input_max() - input_lowest()),
                                        static_cast<size_t>(10000ULL)));
    size_t step = std::max(
        static_cast<InputType>(image_size / (input_max() - input_lowest())),
        static_cast<InputType>(1));
    size_t height = image_size / width + 1;
    test::Array2D<InputType> source(width, height, padding_, 1);
    test::Array2D<OutputType> expected(width, height, padding_, 1);
    test::Array2D<OutputType> actual(width, height, padding_, 1);

    GenerateLinearSeries generator(
        input_min_if_integral_else_smallest_positive_normalized(), step);

    source.fill(generator);

    calculate_expected(source, expected);

    if constexpr (std::is_same_v<InputType, OutputType>) {
      if (in_place) {
        actual = source;
        ASSERT_EQ(KLEIDICV_OK,
                  (scale_api<InputType, OutputType>()(
                      actual.data(), actual.stride(), actual.data(),
                      actual.stride(), width, height, scale(), shift())));
        EXPECT_EQ_ARRAY2D(expected, actual);
        return;
      }
    } else {
      ASSERT_EQ(KLEIDICV_OK,
                (scale_api<InputType, OutputType>()(
                    source.data(), source.stride(), actual.data(),
                    actual.stride(), width, height, scale(), shift())));
      EXPECT_EQ_ARRAY2D(expected, actual);
    }
  }

 protected:
  void calculate_expected(const test::Array2D<InputType>& source,
                          test::Array2D<OutputType>& expected) {
    for (size_t hindex = 0; hindex < source.height(); ++hindex) {
      for (size_t vindex = 0; vindex < source.width(); ++vindex) {
        // clang-tidy cannot detect that source is fully initialized
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign,clang-analyzer-core.CallAndMessage)
        *expected.at(hindex, vindex) = scale_operation<InputType, OutputType>()(
            *source.at(hindex, vindex), scale(), shift());
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign,clang-analyzer-core.CallAndMessage)
      }
    }
  }
};  // end of class ScaleTestLinearBase

template <typename InputType, typename OutputType>
class ScaleTestLinear1 final
    : public ScaleTestLinearBase<InputType, OutputType> {
  double scale() override { return 10; };
  double shift() override { return 13; };
};
template <typename InputType, typename OutputType>
class ScaleTestLinear2 final
    : public ScaleTestLinearBase<InputType, OutputType> {
  double scale() override { return 14.69; };
  double shift() override { return 10.13; };
};

template <typename InputType, typename OutputType>
class ScaleTestLinear3 final
    : public ScaleTestLinearBase<InputType, OutputType> {
  double scale() override { return 0.18; };
  double shift() override { return 1.41; };
};

template <typename InputType, typename OutputType>
class ScaleTestLinear4 final
    : public ScaleTestLinearBase<InputType, OutputType> {
  double scale() override { return 1.0; };
  double shift() override { return -17.7; };
};

template <typename InputType, typename OutputType>
class ScaleTestAdd final : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;

  double scale() override { return 1.0; }
  double shift() override { return 6.0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { 8, 14},
      {12, 18},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestSubtract final : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;

  double scale() override { return 8; }
  double shift() override { return -3; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { 6, 45},
      {20, 157},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestDivide final : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;

  double scale() override { return 0.25; }
  double shift() override { return 3; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      {252, 0},
      {255, 0},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestMultiply final : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;

  double scale() override { return 3.14; }
  double shift() override { return 2.72; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      {{60}, {0}},
      {{45}, {0}},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestZero final : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;

  using ScaleTestBase<InputType, OutputType>::
      input_min_if_integral_else_smallest_positive_normalized;
  using ScaleTestBase<InputType, OutputType>::input_lowest;
  using ScaleTestBase<InputType, OutputType>::input_max;

  double scale() override { return 0; }
  double shift() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
        {input_lowest(), 0},
        {input_min_if_integral_else_smallest_positive_normalized(), 0},
        {    0, 0},
        {input_max(), 0},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestUnderflowByShift final
    : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;
  using ScaleTestBase<InputType, OutputType>::
      input_min_if_integral_else_smallest_positive_normalized;
  using ScaleTestBase<InputType, OutputType>::input_lowest;

  double scale() override { return 1; }
  double shift() override { return -1; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
        {input_lowest() + 1, 0},
        {    input_lowest(), 0},
        {input_min_if_integral_else_smallest_positive_normalized() + 1, 0},
        {    input_min_if_integral_else_smallest_positive_normalized(), 0},
        {          0, 0},
        {          1, 0},
        {          2, 0},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestOverflowByShift final
    : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;
  using ScaleTestBase<InputType, OutputType>::input_max;

  double scale() override { return 1; }
  double shift() override { return 1; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      {input_max() - 1, 0},
      {    input_max(), 0},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestUnderflowByScale final
    : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;
  using ScaleTestBase<InputType, OutputType>::input_max;

  double scale() override { return -2; }
  double shift() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { input_max(), 0},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename InputType, typename OutputType>
class ScaleTestOverflowByScale final
    : public ScaleTestBase<InputType, OutputType> {
  using Elements =
      typename OperationTest<InputType, 1, 1, OutputType>::Elements;
  using ScaleTestBase<InputType, OutputType>::input_max;

  double scale() override { return 2; }
  double shift() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { input_max(), 0},
        // clang-format on
    };
    ScaleTestBase<InputType, OutputType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

class ScaleU8Params {
 public:
  using InputType = uint8_t;
  using OutputType = uint8_t;
};

class ScaleFloatParams {
 public:
  using InputType = float;
  using OutputType = float;
};

class ScaleU8ToFloat16Params {
 public:
  using InputType = uint8_t;
  using OutputType = float16_t;
};

template <typename ElementType>
class ScaleTest : public testing::Test {};

using ElementTypes =
    ::testing::Types<ScaleU8Params, ScaleFloatParams, ScaleU8ToFloat16Params>;

TYPED_TEST_SUITE(ScaleTest, ElementTypes);

TYPED_TEST(ScaleTest, TestScalar1) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear1<InputType, OutputType>{}.test_scalar();
  ScaleTestLinear1<InputType, OutputType>{}.with_padding(1).test_scalar();
}
TYPED_TEST(ScaleTest, TestVector1) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear1<InputType, OutputType>{}.test_vector();
  ScaleTestLinear1<InputType, OutputType>{}.with_padding(1).test_vector();
}

TYPED_TEST(ScaleTest, TestScalar1Tbx) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear1<InputType, OutputType>{}.test_scalar(700);
  ScaleTestLinear1<InputType, OutputType>{}.with_padding(1).test_scalar(700);
}
TYPED_TEST(ScaleTest, TestVector1Tbx) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear1<InputType, OutputType>{}.test_vector(700);
  ScaleTestLinear1<InputType, OutputType>{}.with_padding(1).test_vector(700);
}

TYPED_TEST(ScaleTest, TestScalar2) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear2<InputType, OutputType>{}.test_scalar();
}
TYPED_TEST(ScaleTest, TestVector2) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear2<InputType, OutputType>{}.test_vector();
}

TYPED_TEST(ScaleTest, TestScalar2Tbx) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear2<InputType, OutputType>{}.test_scalar(700);
}
TYPED_TEST(ScaleTest, TestVector2Tbx) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear2<InputType, OutputType>{}.test_vector(700);
}

TYPED_TEST(ScaleTest, TestScalar3) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear3<InputType, OutputType>{}.test_scalar();
}
TYPED_TEST(ScaleTest, TestVector3) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear3<InputType, OutputType>{}.test_vector();
}

TYPED_TEST(ScaleTest, TestScalar3Tbx) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear3<InputType, OutputType>{}.test_scalar(700);
}
TYPED_TEST(ScaleTest, TestVector3Tbx) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear3<InputType, OutputType>{}.test_vector(700);
}

TYPED_TEST(ScaleTest, InPlaceOperation) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;

  // In-place operation is only supported for same-type scale variants
  if constexpr (std::is_same_v<InputType, OutputType>) {
    const size_t kWidth = test::Options::vector_length() + 1;
    constexpr size_t kHeight = 19;
    constexpr size_t kPadding = 3;
    constexpr double kScale = 14.69;
    constexpr double kShift = 10.13;

    test::PseudoRandomNumberGenerator<InputType> generator;
    test::Array2D<InputType> source{kWidth, kHeight, kPadding};
    test::Array2D<OutputType> in_place{kWidth, kHeight, kPadding};
    test::Array2D<OutputType> out_of_place{kWidth, kHeight, kPadding};

    source.fill(generator);
    in_place = source;

    auto out_of_place_err = scale_api<InputType, OutputType>()(
        source.data(), source.stride(), out_of_place.data(),
        out_of_place.stride(), kWidth, kHeight, kScale, kShift);
    ASSERT_EQ(KLEIDICV_OK, out_of_place_err);

    auto in_place_err = scale_api<InputType, OutputType>()(
        in_place.data(), in_place.stride(), in_place.data(), in_place.stride(),
        kWidth, kHeight, kScale, kShift);
    ASSERT_EQ(KLEIDICV_OK, in_place_err);
    EXPECT_EQ_ARRAY2D(in_place, out_of_place);
  }
}

TYPED_TEST(ScaleTest, InPlaceScalar2) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear2<InputType, OutputType>{}.test_scalar(1, true);
}
TYPED_TEST(ScaleTest, InPlaceVector2) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear2<InputType, OutputType>{}.test_vector(1, true);
}
TYPED_TEST(ScaleTest, InPlaceAddScalar) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear4<InputType, OutputType>{}.test_scalar(1, true);
}
TYPED_TEST(ScaleTest, InPlaceAddVector) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestLinear4<InputType, OutputType>{}.test_vector(1, true);
}

TYPED_TEST(ScaleTest, TestAdd) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestAdd<InputType, OutputType>{}.test();
  ScaleTestAdd<InputType, OutputType>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<InputType>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestSubtract) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestSubtract<InputType, OutputType>{}.test();
  ScaleTestSubtract<InputType, OutputType>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<InputType>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestDivide) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestDivide<InputType, OutputType>{}.test();
  ScaleTestDivide<InputType, OutputType>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<InputType>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestMultiply) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestMultiply<InputType, OutputType>{}.test();
  ScaleTestMultiply<InputType, OutputType>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<InputType>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestZero) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestZero<InputType, OutputType>{}.test();
}

TYPED_TEST(ScaleTest, TestUnderflowByShift) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestUnderflowByShift<InputType, OutputType>{}.test();
}

TYPED_TEST(ScaleTest, TestOverflowByShift) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestOverflowByShift<InputType, OutputType>{}.test();
}

TYPED_TEST(ScaleTest, TestUnderflowByScale) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestUnderflowByScale<InputType, OutputType>{}.test();
}

TYPED_TEST(ScaleTest, TestOverflowByScale) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  ScaleTestOverflowByScale<InputType, OutputType>{}.test();
}

TYPED_TEST(ScaleTest, NullPointer) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  InputType src[1] = {};
  OutputType dst[1];
  test::test_null_args(scale_api<InputType, OutputType>(), src,
                       sizeof(InputType), dst, sizeof(OutputType), 1, 1, 2, 0);
}

TYPED_TEST(ScaleTest, Misalignment) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  if (sizeof(InputType) == 1) {
    // misalignment impossible
    return;
  }
  InputType src[2] = {};
  OutputType dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT, (scale_api<InputType, OutputType>()(
                                          src, sizeof(InputType) + 1, dst,
                                          sizeof(OutputType), 1, 2, 2, 0)));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT, (scale_api<InputType, OutputType>()(
                                          src, sizeof(InputType), dst,
                                          sizeof(OutputType) + 1, 1, 2, 2, 0)));
}

TYPED_TEST(ScaleTest, ZeroImageSize) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  InputType src[1] = {};
  OutputType dst[1];
  EXPECT_EQ(KLEIDICV_OK,
            (scale_api<InputType, OutputType>()(
                src, sizeof(InputType), dst, sizeof(OutputType), 0, 1, 2, 0)));
  EXPECT_EQ(KLEIDICV_OK,
            (scale_api<InputType, OutputType>()(
                src, sizeof(InputType), dst, sizeof(OutputType), 1, 0, 2, 0)));
}

TYPED_TEST(ScaleTest, OversizeImage) {
  using InputType = typename TypeParam::InputType;
  using OutputType = typename TypeParam::OutputType;
  InputType src[1] = {};
  OutputType dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            (scale_api<InputType, OutputType>()(
                src, sizeof(InputType), dst, sizeof(OutputType),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 2, 0)));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            (scale_api<InputType, OutputType>()(
                src, sizeof(InputType), dst, sizeof(OutputType),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 2, 0)));
}
