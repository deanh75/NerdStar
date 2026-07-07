// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/kernel.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_GAUSSIAN_BLUR(type, type_suffix) \
  KLEIDICV_API(gaussian_blur, kleidicv_gaussian_blur_##type_suffix, type)

KLEIDICV_GAUSSIAN_BLUR(uint8_t, u8);

// Implements KernelTestParams for Gaussian Blur operators.
template <typename ElementType, size_t KernelSize>
struct GaussianBlurKernelTestParams;

template <size_t KernelSize>
struct GaussianBlurKernelTestParams<uint8_t, KernelSize> {
  using InputType = uint8_t;
  using IntermediateType = uint64_t;
  using OutputType = uint8_t;

  static constexpr size_t kKernelSize = KernelSize;
};  // end of struct GaussianBlurKernelTestParams<uint8_t, KernelSize>

static constexpr std::array<kleidicv_border_type_t, 1> kReplicateBorder = {
    KLEIDICV_BORDER_TYPE_REPLICATE};

static constexpr std::array<kleidicv_border_type_t, 4> kAllBorders = {
    KLEIDICV_BORDER_TYPE_REPLICATE,
    KLEIDICV_BORDER_TYPE_REFLECT,
    KLEIDICV_BORDER_TYPE_WRAP,
    KLEIDICV_BORDER_TYPE_REVERSE,
};

static constexpr size_t kToleranceOne = 1;

// Test for GaussianBlur operator.
template <class KernelTestParams,
          typename ArrayLayoutsGetterType = decltype(test::small_array_layouts),
          typename BorderContainerType = decltype(kAllBorders)>
class GaussianBlurTest : public test::KernelTest<KernelTestParams> {
  using Base = test::KernelTest<KernelTestParams>;
  using typename test::KernelTest<KernelTestParams>::InputType;
  using typename test::KernelTest<KernelTestParams>::IntermediateType;
  using typename test::KernelTest<KernelTestParams>::OutputType;
  using ArrayContainerType =
      std::invoke_result_t<ArrayLayoutsGetterType, size_t, size_t>;
  static constexpr size_t kKernelSize = KernelTestParams::kKernelSize;

 public:
  explicit GaussianBlurTest(
      KernelTestParams,
      ArrayLayoutsGetterType array_layouts_getter = test::small_array_layouts,
      BorderContainerType border_types = kAllBorders, size_t tolerance = 0)
      : test::KernelTest<KernelTestParams>{tolerance},
        array_layouts_{array_layouts_getter(KernelTestParams::kKernelSize - 1,
                                            KernelTestParams::kKernelSize - 1)},
        border_types_{border_types},
        array_layout_generator_{array_layouts_},
        border_type_generator_{border_types_},
        sigma_{0.0} {}

  GaussianBlurTest &with_sigma(float sigma) {
    sigma_ = sigma;
    return *this;
  }

  void test(const test::Array2D<IntermediateType> &mask) {
    test::Kernel kernel{mask};
    // Create generators and execute test.
    test::SequenceGenerator tested_border_values{
        test::default_border_values<InputType>()};
    test::PseudoRandomNumberGenerator<InputType> element_generator;
    Base::test(kernel, array_layout_generator_, border_type_generator_,
               tested_border_values, element_generator);
  }

  void test_with_generated_mask() {
    test::Array2D<IntermediateType> mask{kKernelSize, kKernelSize};
    calculate_mask(mask);
    test(mask);
  }

  void calculate_mask(test::Array2D<IntermediateType> &mask) {
    constexpr size_t kHalfKernelSize =
        kleidicv::get_half_kernel_size(kKernelSize);
    std::vector<uint8_t> half_kernel(kHalfKernelSize);
    bool success = kleidicv::generate_gaussian_half_kernel(
        half_kernel.data(), kHalfKernelSize, sigma_);
    if (success) {
      for (size_t row = 0; row < kKernelSize; ++row) {
        for (size_t column = 0; column < kKernelSize; ++column) {
          *mask.at(row, column) =
              half_kernel[row >= kHalfKernelSize ? kKernelSize - 1 - row
                                                 : row] *
              half_kernel[column >= kHalfKernelSize ? kKernelSize - 1 - column
                                                    : column];
        }
      }
    } else {
      // When all the kernel's energy is in the middle, the returned kernel is
      // invalid, so it is filled in a different way
      mask.fill(0);
      *mask.at(kHalfKernelSize - 1, kHalfKernelSize - 1) = 256 * 256;
    }
  }

 private:
  kleidicv_error_t call_api(const test::Array2D<InputType> *input,
                            test::Array2D<OutputType> *output,
                            kleidicv_border_type_t border_type,
                            const InputType *) override {
    return gaussian_blur<InputType>()(
        input->data(), input->stride(), output->data(), output->stride(),
        input->width() / input->channels(), input->height(), input->channels(),
        KernelTestParams::kKernelSize, KernelTestParams::kKernelSize, sigma_,
        sigma_, border_type);
  }

  // Apply rounding to nearest integer division.
  IntermediateType scale_result(const test::Kernel<IntermediateType> &,
                                IntermediateType result) override {
    if (sigma_ == 0.0) {
      if constexpr (KernelTestParams::kKernelSize == 3) {
        return (result + 8) / 16;
      }
      if constexpr (KernelTestParams::kKernelSize == 5) {
        return (result + 128) / 256;
      }
      if constexpr (KernelTestParams::kKernelSize == 7) {
        return (result + 2048) / 4096;
      }
    }
    return (result + 32768) / 65536;
  }

  const ArrayContainerType array_layouts_;
  const BorderContainerType border_types_;
  test::SequenceGenerator<ArrayContainerType> array_layout_generator_;
  test::SequenceGenerator<BorderContainerType> border_type_generator_;
  float sigma_;
};  // end of class GaussianBlurTest<KernelTestParams, ArrayLayoutsGetterType,
    // BorderContainerType>

using ElementTypes = ::testing::Types<uint8_t>;

template <typename ElementType>
class GaussianBlur : public testing::Test {};

TYPED_TEST_SUITE(GaussianBlur, ElementTypes);

// Tests gaussian_blur_3x3_<input_type> API.
TYPED_TEST(GaussianBlur, 3x3Small) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;

  // 3x3 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{3, 3};
  // clang-format off
  mask.set(0, 0, { 1, 2, 1});
  mask.set(1, 0, { 2, 4, 2});
  mask.set(2, 0, { 1, 2, 1});
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

TYPED_TEST(GaussianBlur, 3x3Default) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  // 3x3 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{3, 3};
  // clang-format off
  mask.set(0, 0, { 1, 2, 1});
  mask.set(1, 0, { 2, 4, 2});
  mask.set(2, 0, { 1, 2, 1});
  // clang-format on
  GaussianBlurTest{KernelTestParams{}, test::default_array_layouts,
                   kReplicateBorder}
      .test(mask);
}

// Tests gaussian_blur_5x5_<input_type> API.
TYPED_TEST(GaussianBlur, 5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;
  // 5x5 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{5, 5};
  // clang-format off
  mask.set(0, 0, { 1,  4,  6,  4, 1});
  mask.set(1, 0, { 4, 16, 24, 16, 4});
  mask.set(2, 0, { 6, 24, 36, 24, 6});
  mask.set(3, 0, { 4, 16, 24, 16, 4});
  mask.set(4, 0, { 1,  4,  6,  4, 1});
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

// Tests gaussian_blur_7x7_<input_type> API.
TYPED_TEST(GaussianBlur, 7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;
  // 7x7 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{7, 7};
  // clang-format off
  mask.set(0, 0, {  4,  14,  28,  36,  28,  14,  4 });
  mask.set(1, 0, { 14,  49,  98, 126,  98,  49, 14 });
  mask.set(2, 0, { 28,  98, 196, 252, 196,  98, 28 });
  mask.set(3, 0, { 36, 126, 252, 324, 252, 126, 36 });
  mask.set(4, 0, { 28,  98, 196, 252, 196,  98, 28 });
  mask.set(5, 0, { 14,  49,  98, 126,  98,  49, 14 });
  mask.set(6, 0, {  4,  14,  28,  36,  28,  14,  4 });
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

// Tests gaussian_blur_9x9_<input_type> API.
TYPED_TEST(GaussianBlur, 9x9) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 9>;
  // 9x9 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{9, 9};
  // clang-format off
  mask.set(0, 0, {  16,  52, 120, 204, 240, 204, 120,  52,  16 });
  mask.set(1, 0, {  52, 169, 390, 663, 780, 663, 390, 169,  52 });
  mask.set(2, 0, { 120, 390, 900, 1530, 1800, 1530, 900, 390, 120 });
  mask.set(3, 0, { 204, 663, 1530, 2601, 3060, 2601, 1530, 663, 204 });
  mask.set(4, 0, { 240, 780, 1800, 3060, 3600, 3060, 1800, 780, 240 });
  mask.set(5, 0, { 204, 663, 1530, 2601, 3060, 2601, 1530, 663, 204 });
  mask.set(6, 0, { 120, 390, 900, 1530, 1800, 1530, 900, 390, 120 });
  mask.set(7, 0, {  52, 169, 390, 663, 780, 663, 390, 169,  52 });
  mask.set(8, 0, {  16,  52, 120, 204, 240, 204, 120,  52,  16 });
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

const auto minimal_array_layouts_for_fixed = [](size_t min_w, size_t min_h) {
  // Number of 16-bit elements in a SIMD vector (maximum border length)
  size_t margin = min_w / 2;
  // minimum allowed width, needed for the fixed size kernels to activate the
  // NEON scalar path
  size_t min_width = 2 * margin + 1;
  // two borders + unrolltwice + unrollonce + one for the tail
  size_t big_width = 2 * margin + (2 + 1) * test::Options::vector_length();
  return std::array<test::ArrayLayout, 4>{{
      {1 * min_width, min_h + 1, 1, 1},
      {2 * min_width, min_h, 1, 2},
      {3 * min_width, min_h, 1, 3},
      {big_width, min_h, 1, 1},
  }};
};

const auto minimal_array_layouts_for_arbitrary = [](size_t min_w,
                                                    size_t min_h) {
  // Number of 16-bit elements in a SIMD vector (maximum border length)
  size_t mbl = test::Options::vector_length() / 2;
  size_t margin = min_w / 2;
  size_t min_width = (margin + mbl - 1) / mbl * mbl + margin;
  // two borders + one for the tail, so the NEON scalar path activates
  size_t small_width = 2 * ((margin + mbl - 1) / mbl * mbl) + 1;
  // two borders + unrolltwice + unrollonce + one for the tail
  size_t big_width = small_width + (2 + 1) * test::Options::vector_length();
  return std::array<test::ArrayLayout, 5>{{
      {1 * min_width, min_h + 1, 1, 1},
      {1 * small_width, min_h + 1, 1, 1},
      {2 * min_width, min_h, 1, 2},
      {3 * min_width, min_h, 1, 3},
      {big_width, min_h, 1, 1},
  }};
};

size_t minimumValidWidth(size_t kernel_size, size_t vector_length) {
  if (kernel_size <= 9 || kernel_size == 15 || kernel_size == 21) {
    return kernel_size - 1;
  }
  size_t margin = kernel_size / 2;
  // Maximum Border Length
  size_t bl = vector_length / 2;
  return (margin + bl - 1) / bl * bl + margin;
}

TYPED_TEST(GaussianBlur, 3x3_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

TYPED_TEST(GaussianBlur, 5x5_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

TYPED_TEST(GaussianBlur, 7x7_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

TYPED_TEST(GaussianBlur, 9x9_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 9>;
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

// 11x11 use the generic solution.
TYPED_TEST(GaussianBlur, 11x11_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 11>;
  // TODO kReplicateBorder is temporary until we implement all borders
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_arbitrary,
                   kReplicateBorder, kToleranceOne}
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_arbitrary,
                   kReplicateBorder, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_arbitrary,
                   kReplicateBorder, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

// Tests gaussian_blur_15x15_<input_type> API. It always uses CustomSigma.
TYPED_TEST(GaussianBlur, 15x15_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

// Tests gaussian_blur_21x21_<input_type> API. It always uses CustomSigma.
TYPED_TEST(GaussianBlur, 21x21_CustomSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 21>;
  // TODO kReplicateBorder is temporary until we implement all borders
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(2.2)
      .test_with_generated_mask();
  GaussianBlurTest{KernelTestParams{}, minimal_array_layouts_for_fixed,
                   kAllBorders, kToleranceOne}
      .with_sigma(0.01)
      .test_with_generated_mask();
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType3x3) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                         sizeof(TypeParam), validSize,
                                         validSize, 1, 3, 3, 0.0, 0.0, border));
  }
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                         sizeof(TypeParam), validSize,
                                         validSize, 1, 5, 5, 0.0, 0.0, border));
  }
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                         sizeof(TypeParam), validSize,
                                         validSize, 1, 7, 7, 0.0, 0.0, border));
  }
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType9x9) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 9>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                         sizeof(TypeParam), validSize,
                                         validSize, 1, 9, 9, 0.0, 0.0, border));
  }
}

// Arbitrary kernel size algorithm only supports REPLICATED borders
TYPED_TEST(GaussianBlur, UnsupportedBorderType11x11) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 11>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_WRAP,
           KLEIDICV_BORDER_TYPE_REVERSE,
           KLEIDICV_BORDER_TYPE_REFLECT,
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 11, 11, 0.0, 0.0, border));
  }
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType15x15) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 15, 15, 0.0, 0.0, border));
  }
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType21x21) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 21>;
  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 21, 21, 0.0, 0.0, border));
  }
}

TYPED_TEST(GaussianBlur, DifferentKernelSize) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                validSize, 1, 5, 15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(GaussianBlur, NonZeroSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 1.23, 4.56,
                                 KLEIDICV_BORDER_TYPE_REPLICATE));

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 1.23, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE));

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 0.0, 4.56,
                                 KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(GaussianBlur, UnsupportedKernelSize) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 257>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                validSize, 1, 1, 1, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 257, 257, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(GaussianBlur, NullPointer) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 3, 3, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE);
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 5, 5, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE);
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 7, 7, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE);
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 15, 15, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE);
}

TYPED_TEST(GaussianBlur, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 3, 3, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 3, 3, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 5, 5, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 5, 5, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 7, 7, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 7, 7, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));

  EXPECT_EQ(
      KLEIDICV_ERROR_ALIGNMENT,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam) + 1, dst,
                                 sizeof(TypeParam), validSize, validSize, 1, 15,
                                 15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam) + 1, validSize,
                                       validSize, 1, 15, 15, 0.0, 0.0,
                                       KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(GaussianBlur, ZeroImageSize3x3) {
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 0, 3, 1, 3, 3, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 3, 0, 1, 3, 3, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, ZeroImageSize5x5) {
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 0, 5, 1, 5, 5, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 5, 0, 1, 5, 5, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, ZeroImageSize7x7) {
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 0, 7, 1, 7, 7, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 7, 0, 1, 7, 7, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, ZeroImageSize9x9) {
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 0, 9, 1, 9, 9, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 9, 0, 1, 9, 9, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, ZeroImageSize15x15) {
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 0, 15, 1, 15, 15, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), 15, 0, 1, 15, 15, 0.0,
                                       0.0, KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, ValidImageSize3x3) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());
  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2});
  src.set(1, 0, {4, 3});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 3, 3, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {3, 3});
  expected.set(1, 0, {3, 3});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 3, 3, 2.25, 2.25, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {3, 3});
  expected.set(1, 0, {2, 2});
  EXPECT_EQ_ARRAY2D(expected, dst);
}

TYPED_TEST(GaussianBlur, ValidImageSize5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2, 3, 4});
  src.set(1, 0, {8, 7, 6, 5});
  src.set(2, 0, {1, 2, 3, 4});
  src.set(3, 0, {16, 27, 38, 49});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 5, 5, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {5, 5, 5, 5});
  expected.set(1, 0, {6, 6, 6, 7});
  expected.set(2, 0, {9, 10, 12, 13});
  expected.set(3, 0, {11, 13, 16, 18});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 5, 5, 2.25, 2.25, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {4, 4, 4, 4});
  expected.set(1, 0, {8, 9, 9, 10});
  expected.set(2, 0, {9, 9, 10, 11});
  expected.set(3, 0, {10, 11, 12, 12});
  EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, expected, dst);
}

TYPED_TEST(GaussianBlur, ValidImageSize7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2, 3, 4, 5, 6});
  src.set(1, 0, {12, 11, 10, 9, 8, 7});
  src.set(2, 0, {1, 2, 3, 4, 5, 6});
  src.set(3, 0, {1, 2, 3, 4, 5, 6});
  src.set(4, 0, {11, 22, 33, 44, 55, 66});
  src.set(5, 0, {127, 67, 37, 27, 17, 7});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 7, 7, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {6, 6, 6, 6, 6, 6});
  expected.set(1, 0, {6, 6, 7, 7, 8, 8});
  expected.set(2, 0, {9, 9, 10, 10, 11, 12});
  expected.set(3, 0, {16, 16, 16, 17, 18, 19});
  expected.set(4, 0, {26, 26, 25, 26, 27, 27});
  expected.set(5, 0, {32, 31, 29, 29, 30, 30});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 7, 7, 2.25, 2.25, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {5, 5, 6, 6, 6, 6});
  expected.set(1, 0, {7, 7, 8, 9, 9, 10});
  expected.set(2, 0, {13, 13, 13, 13, 13, 13});
  expected.set(3, 0, {18, 18, 19, 19, 19, 19});
  expected.set(4, 0, {22, 22, 23, 23, 23, 23});
  expected.set(5, 0, {24, 24, 24, 24, 24, 24});
  EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, expected, dst);
}

TYPED_TEST(GaussianBlur, ValidImageSize9x9) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 9>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2, 3, 4, 5, 6, 7, 8});
  src.set(1, 0, {16, 15, 14, 13, 12, 11, 10, 9});
  src.set(2, 0, {1, 2, 3, 4, 5, 6, 7, 8});
  src.set(3, 0, {1, 2, 3, 4, 5, 6, 7, 8});
  src.set(4, 0, {11, 22, 33, 44, 55, 66, 77, 88});
  src.set(5, 0, {127, 67, 37, 27, 17, 7, 5, 3});
  src.set(6, 0, {1, 2, 3, 4, 5, 6, 7, 8});
  src.set(7, 0, {8, 7, 6, 5, 4, 3, 2, 1});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 9, 9, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {8, 8, 8, 9, 9, 10, 10, 10});
  expected.set(1, 0, {9, 9, 9, 10, 10, 11, 11, 11});
  expected.set(2, 0, {11, 12, 12, 12, 13, 14, 15, 15});
  expected.set(3, 0, {16, 16, 16, 17, 18, 19, 20, 20});
  expected.set(4, 0, {22, 21, 20, 20, 20, 21, 22, 22});
  expected.set(5, 0, {24, 23, 21, 20, 19, 19, 19, 20});
  expected.set(6, 0, {23, 22, 19, 17, 15, 15, 15, 15});
  expected.set(7, 0, {21, 20, 18, 15, 14, 13, 13, 13});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), validSize,
                validSize, 1, 9, 9, 2.25, 2.25, KLEIDICV_BORDER_TYPE_REVERSE));
  expected.set(0, 0, {9, 9, 9, 10, 11, 12, 12, 13});
  expected.set(1, 0, {10, 10, 11, 11, 12, 12, 12, 12});
  expected.set(2, 0, {13, 13, 13, 14, 14, 15, 15, 16});
  expected.set(3, 0, {16, 16, 16, 16, 17, 17, 18, 18});
  expected.set(4, 0, {18, 18, 18, 18, 18, 18, 19, 19});
  expected.set(5, 0, {21, 21, 20, 19, 18, 17, 17, 17});
  expected.set(6, 0, {22, 21, 20, 19, 18, 17, 17, 17});
  expected.set(7, 0, {22, 21, 20, 18, 17, 16, 16, 16});
  EXPECT_EQ_ARRAY2D_WITH_TOLERANCE(1, expected, dst);
}

template <typename TypeParam, size_t kKernelSize>
void test_undersize_image() {
  // 8 is the number of 16-bit elements in a 128-bit vector.
  size_t validWidth = minimumValidWidth(kKernelSize, 8);
  size_t underWidth = validWidth - 1;
  size_t validHeight = kKernelSize - 1;
  size_t underHeight = validHeight - 1;
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), underWidth,
                underHeight, 1, kKernelSize, kKernelSize, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), underWidth,
                validHeight, 1, kKernelSize, kKernelSize, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validWidth,
                underHeight, 1, kKernelSize, kKernelSize, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(GaussianBlur, UndersizeImage3x3) {
  test_undersize_image<TypeParam, 3>();
}

TYPED_TEST(GaussianBlur, UndersizeImage5x5) {
  test_undersize_image<TypeParam, 5>();
}

TYPED_TEST(GaussianBlur, UndersizeImage7x7) {
  test_undersize_image<TypeParam, 7>();
}

TYPED_TEST(GaussianBlur, UndersizeImage9x9) {
  test_undersize_image<TypeParam, 9>();
}

TYPED_TEST(GaussianBlur, UndersizeImage15x15) {
  test_undersize_image<TypeParam, 15>();
}

TYPED_TEST(GaussianBlur, UndersizeImage21x21) {
  test_undersize_image<TypeParam, 21>();
}

TYPED_TEST(GaussianBlur, OversizeImage) {
  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 KLEIDICV_MAX_IMAGE_PIXELS + 1, 14, 1, 15, 15,
                                 0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 14, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 15, 15,
                                 0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 1, 11, 11,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(GaussianBlur, ChannelNumber) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                validSize, KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, 15, 15, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, InvalidKernelSize) {
  size_t kernel_size = 16;
  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), kernel_size,
                                       kernel_size, 1, kernel_size, kernel_size,
                                       0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst,
                                       sizeof(TypeParam), kernel_size,
                                       kernel_size, 1, kernel_size, kernel_size,
                                       1.0, 1.0, KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(GaussianBlur, InvalidBorderType) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;

  size_t validSize = minimumValidWidth(KernelTestParams::kKernelSize,
                                       test::Options::vector_length());

  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                validSize, 1, 15, 15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_CONSTANT));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_TRANSPARENT));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                validSize, 1, 15, 15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_NONE));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TYPED_TEST(GaussianBlur, Allocation) {
  constexpr size_t max_size = 21;
  uint8_t src[max_size * max_size] = {};
  uint8_t dst[max_size * max_size] = {};

  constexpr size_t fixed_sizes[] = {3, 5, 7, 9};
  constexpr float sigma_binomial = 0.0F;
  constexpr float sigma_non_binomial = 1.0F;

  for (size_t kernel_size : fixed_sizes) {
    const size_t stride = kernel_size * sizeof(uint8_t);
    MockMallocToFail::enable();
    auto ret = gaussian_blur<TypeParam>()(
        src, stride, dst, stride, kernel_size, kernel_size, /*ch*/ 1,
        kernel_size, kernel_size, sigma_binomial, sigma_binomial,
        KLEIDICV_BORDER_TYPE_REPLICATE);
    MockMallocToFail::disable();
    EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);
  }

  for (size_t kernel_size : fixed_sizes) {
    const size_t stride = kernel_size * sizeof(uint8_t);
    MockMallocToFail::enable();
    auto ret = gaussian_blur<TypeParam>()(
        src, stride, dst, stride, kernel_size, kernel_size, /*ch*/ 1,
        kernel_size, kernel_size, sigma_non_binomial, sigma_non_binomial,
        KLEIDICV_BORDER_TYPE_REPLICATE);
    MockMallocToFail::disable();
    EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);
  }

  constexpr size_t non_binomial_sizes[] = {15, 21};
  for (size_t kernel_size : non_binomial_sizes) {
    const size_t stride = kernel_size * sizeof(uint8_t);
    MockMallocToFail::enable();
    auto ret = gaussian_blur<TypeParam>()(
        src, stride, dst, stride, kernel_size, kernel_size, /*ch*/ 1,
        kernel_size, kernel_size, sigma_non_binomial, sigma_non_binomial,
        KLEIDICV_BORDER_TYPE_REPLICATE);
    MockMallocToFail::disable();
    EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);
  }

  MockMallocToFail::enable();
  auto ret11 =
      gaussian_blur<TypeParam>()(src, 13, dst, 13, 13, 13, /*ch*/ 1, 11, 11,
                                 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE);
  MockMallocToFail::disable();
  EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret11);
}
#endif

static std::vector<uint8_t> generate_reference_kernel(size_t half_size,
                                                      float sigma) {
  std::vector<float> float_kernel(half_size);

  for (size_t i = 0; i < half_size; ++i) {
    float_kernel[i] = static_cast<float>(
        std::exp(-std::pow(i, 2) / (2 * std::pow(sigma, 2))));
  }

  float sum = 0;
  for (float val : float_kernel) {
    sum += val;
  }

  // Sum needs to be corrected to contain all kernel values
  sum = (sum * 2) - 1;

  for (auto &val : float_kernel) {
    val = val / sum;
    // Multiplication is needed as the results are fixed point values on
    // uint16_t type
    val *= 256;
  }

  std::vector<uint8_t> kernel_to_return(half_size);
  // Conversion with rounding error diffusion
  float last_rounding_error = 0.0;
  for (size_t i = 0; i < half_size; ++i) {
    float corrected_value =
        float_kernel[half_size - 1 - i] - last_rounding_error;
    float rounded_value = std::round(corrected_value);
    last_rounding_error = rounded_value - corrected_value;
    kernel_to_return[i] = static_cast<uint8_t>(rounded_value);
  }

  return kernel_to_return;
}

template <size_t Size>
void test_sigma() {
  const std::vector<uint8_t> expected_half_kernel =
      generate_reference_kernel(Size, 3.0);
  std::vector<uint8_t> actual_half_kernel(Size);
  kleidicv::generate_gaussian_half_kernel(actual_half_kernel.data(), Size, 3.0);

  EXPECT_EQ(expected_half_kernel, actual_half_kernel);

  const std::vector<uint8_t> expected_half_kernel1 =
      generate_reference_kernel(Size, ((Size * 2) - 1) * 0.15 + 0.35);
  kleidicv::generate_gaussian_half_kernel(actual_half_kernel.data(), Size, 0.0);

  EXPECT_EQ(expected_half_kernel1, actual_half_kernel);
}

TYPED_TEST(GaussianBlur, KernelGenerationFromSigma) {
  test_sigma<2>();
  test_sigma<3>();
  test_sigma<4>();
  test_sigma<8>();
}
