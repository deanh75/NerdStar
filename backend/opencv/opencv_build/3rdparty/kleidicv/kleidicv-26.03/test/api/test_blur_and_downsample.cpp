// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <type_traits>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/kernel.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

static constexpr std::array kAllBorders = {
    KLEIDICV_BORDER_TYPE_REPLICATE,
    KLEIDICV_BORDER_TYPE_REFLECT,
    KLEIDICV_BORDER_TYPE_WRAP,
    KLEIDICV_BORDER_TYPE_REVERSE,
};

static constexpr size_t kKernelSize = 5;
static constexpr size_t kMinWidthHeight = kKernelSize - 1;

#define KLEIDICV_BLUR_AND_DOWNSAMPLE(type, type_suffix) \
  KLEIDICV_API(blur_and_downsample,                     \
               kleidicv_blur_and_downsample_##type_suffix, type)

KLEIDICV_BLUR_AND_DOWNSAMPLE(uint8_t, u8);

template <typename ElementType>
struct BlurAndDownsampleKernelTestParams;

template <>
struct BlurAndDownsampleKernelTestParams<uint8_t> {
  using InputType = uint8_t;
  using IntermediateType = uint64_t;
  using OutputType = uint8_t;
};  // end of struct BlurAndDownsampleKernelTestParams<uint8_t>

// Test for kleidicv_blur_and_downsample_
template <class KernelTestParams, typename ArrayLayoutsGetterType,
          typename BorderTcontainerType>
class BlurAndDownsampleTest : public test::KernelTest<KernelTestParams> {
  using Base = test::KernelTest<KernelTestParams>;
  using typename Base::InputType;
  using typename Base::IntermediateType;
  using typename Base::OutputType;
  using ArrayContainerType =
      std::invoke_result_t<ArrayLayoutsGetterType, size_t, size_t>;

 public:
  BlurAndDownsampleTest(KernelTestParams,
                        ArrayLayoutsGetterType array_layouts_getter,
                        BorderTcontainerType border_types)
      : array_layouts_{array_layouts_getter(kMinWidthHeight, kMinWidthHeight)},
        border_types_{border_types},
        array_layout_generator_{array_layouts_},
        border_type_generator_{border_types_} {}

  void test(const test::Array2D<IntermediateType> &mask) {
    test::Kernel kernel{mask};
    // Create generators and execute test.
    test::SequenceGenerator tested_border_values{
        test::default_border_values<InputType>()};
    test::PseudoRandomNumberGenerator<InputType> element_generator;
    Base::test(kernel, array_layout_generator_, border_type_generator_,
               tested_border_values, element_generator);
  }

 private:
  kleidicv_error_t call_api(const test::Array2D<InputType> *input,
                            test::Array2D<OutputType> *output,
                            kleidicv_border_type_t border_type,
                            const InputType *) override {
    auto ret = blur_and_downsample<InputType>()(
        input->data(), input->stride(), input->width() / input->channels(),
        input->height(), output->data(), output->stride(), input->channels(),
        border_type);
    return ret;
  }

  // Base class' functionality is overridden as pixels in odd rows and columns
  // are dropped.
  void calculate_expected(
      const test::Kernel<IntermediateType> &kernel,
      const test::TwoDimensional<InputType> &source) override {
    const size_t channels = Base::expected_.channels();
    const size_t expected_pixel_width = Base::expected_.width() / channels;
    for (size_t row = 0; row < Base::expected_.height(); ++row) {
      for (size_t pixel_column = 0; pixel_column < expected_pixel_width;
           ++pixel_column) {
        const size_t src_column = pixel_column * 2 * channels;
        for (size_t channel = 0; channel < channels; ++channel) {
          IntermediateType result = Base::calculate_expected_at(
              kernel, source, row * 2, src_column + channel);
          Base::expected_.at(row, pixel_column * channels + channel)[0] =
              static_cast<OutputType>(scale_result(kernel, result));
        }
      }
    }
  }

  // Base class' functionality is overridden as the output has half the width
  // and height compared to the input in case of blur_and_downsample
  void create_arrays(const test::Kernel<IntermediateType> &kernel,
                     const test::ArrayLayout &array_layout) override {
    Base::input_ = test::Array2D<InputType>{array_layout};
    ASSERT_TRUE(Base::input_.valid());

    const size_t input_pixel_width = array_layout.width / array_layout.channels;
    test::ArrayLayout output_array_layout{
        ((input_pixel_width + 1) / 2) * array_layout.channels,
        (array_layout.height + 1) / 2, array_layout.padding,
        array_layout.channels};

    Base::expected_ = test::Array2D<OutputType>{output_array_layout};
    ASSERT_TRUE(Base::expected_.valid());

    Base::actual_ = test::Array2D<OutputType>{output_array_layout};
    ASSERT_TRUE(Base::actual_.valid());

    Base::input_with_borders_ = test::Array2D<InputType>{
        array_layout.width +
            (kernel.left() + kernel.right()) * array_layout.channels,
        array_layout.height + kernel.top() + kernel.bottom(), 0,
        array_layout.channels};
    ASSERT_TRUE(Base::input_with_borders_.valid());
  }

  // Apply rounding to nearest integer division.
  IntermediateType scale_result(const test::Kernel<IntermediateType> &,
                                IntermediateType result) override {
    return (result + 128) / 256;
  }

  const ArrayContainerType array_layouts_;
  const BorderTcontainerType border_types_;
  test::SequenceGenerator<ArrayContainerType> array_layout_generator_;
  test::SequenceGenerator<BorderTcontainerType> border_type_generator_;
};  // end of class BlurAndDownsampleTest<KernelTestParams,
    // ArrayLayoutsGetterType, BorderTcontainerType>

using ElementTypes = ::testing::Types<uint8_t>;

template <typename ElementType>
class BlurAndDownsample : public testing::Test {};

TYPED_TEST_SUITE(BlurAndDownsample, ElementTypes);

TYPED_TEST(BlurAndDownsample, API) {
  using KernelTestParams = BlurAndDownsampleKernelTestParams<TypeParam>;
  test::Array2D<typename KernelTestParams::IntermediateType> mask{kKernelSize,
                                                                  kKernelSize};
  const auto larger_layouts = [](size_t, size_t) {
    return std::array{
        // width (interleaved), height, padding, channels
        test::ArrayLayout{160, 120, 0, 1}, test::ArrayLayout{320, 120, 16, 2},
        test::ArrayLayout{480, 128, 16, 3}, test::ArrayLayout{640, 128, 16, 4},
        // Odd pixel widths to exercise scalar tail paths.
        test::ArrayLayout{322, 121, 0, 2},  // 161 px wide
        test::ArrayLayout{644, 123, 0, 4},  // 161 px wide
    };
  };
  // clang-format off
  mask.set(0, 0, { 1,  4,  6,  4, 1});
  mask.set(1, 0, { 4, 16, 24, 16, 4});
  mask.set(2, 0, { 6, 24, 36, 24, 6});
  mask.set(3, 0, { 4, 16, 24, 16, 4});
  mask.set(4, 0, { 1,  4,  6,  4, 1});
  // clang-format on
  BlurAndDownsampleTest{KernelTestParams{}, larger_layouts, kAllBorders}.test(
      mask);
}

// A simple test suite to test functionality without the kernel test framework
TEST(BlurAndDownsample, Minimal_u8) {
  using TypeParam = uint8_t;

  test::Array2D<TypeParam> input{5, 5};
  // clang-format off
  input.set(0, 0, { 1,  5,  17,  38, 89});
  input.set(1, 0, { 5, 17, 38, 89, 171});
  input.set(2, 0, { 17, 38, 89, 171, 250});
  input.set(3, 0, { 38, 89, 171, 250, 101});
  input.set(4, 0, { 89,  171,  250,  101, 1});
  // clang-format on

  test::Array2D<TypeParam> expected{3, 3};
  // clang-format off
  expected.set(0, 0, { 6,  35,  99});
  expected.set(1, 0, { 35, 103, 161});
  expected.set(2, 0, { 99, 161, 78});
  // clang-format on

  test::Array2D<TypeParam> actual{3, 3};

  EXPECT_EQ(KLEIDICV_OK,
            blur_and_downsample<TypeParam>()(
                input.data(), input.stride(), input.width() / input.channels(),
                input.height(), actual.data(), actual.stride(),
                input.channels(), KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ_ARRAY2D(expected, actual);
}

TYPED_TEST(BlurAndDownsample, UnsupportedBorderType) {
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              blur_and_downsample<TypeParam>()(
                  src, sizeof(TypeParam), kMinWidthHeight, kMinWidthHeight, dst,
                  sizeof(TypeParam), 1, border));
  }
}

TYPED_TEST(BlurAndDownsample, NullPointer) {
  TypeParam src[1] = {}, dst[1];
  test::test_null_args(blur_and_downsample<TypeParam>(), src, sizeof(TypeParam),
                       kMinWidthHeight, kMinWidthHeight, dst, sizeof(TypeParam),
                       1, KLEIDICV_BORDER_TYPE_REPLICATE);
}

TYPED_TEST(BlurAndDownsample, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam) + 1, kMinWidthHeight, kMinWidthHeight,
                dst, sizeof(TypeParam), 1, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), kMinWidthHeight, kMinWidthHeight, dst,
                sizeof(TypeParam) + 1, 1, KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(BlurAndDownsample, UndersizeImage) {
  const size_t underSize = kKernelSize - 2;
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), underSize, underSize, dst,
                sizeof(TypeParam), 1, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), underSize, kMinWidthHeight, dst,
                sizeof(TypeParam), 1, KLEIDICV_BORDER_TYPE_REPLICATE));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), kMinWidthHeight, underSize, dst,
                sizeof(TypeParam), 1, KLEIDICV_BORDER_TYPE_REPLICATE));
}

TYPED_TEST(BlurAndDownsample, OversizeImage) {
  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), (KLEIDICV_MAX_IMAGE_PIXELS / 4) + 1, 4,
                dst, sizeof(TypeParam), 1, KLEIDICV_BORDER_TYPE_REFLECT));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, dst, sizeof(TypeParam), 1,
                KLEIDICV_BORDER_TYPE_REFLECT));
}

TYPED_TEST(BlurAndDownsample, ChannelNumber) {
  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), kMinWidthHeight, kMinWidthHeight, dst,
                sizeof(TypeParam), 5, KLEIDICV_BORDER_TYPE_REFLECT));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            blur_and_downsample<TypeParam>()(
                src, sizeof(TypeParam), kMinWidthHeight, kMinWidthHeight, dst,
                sizeof(TypeParam), 0, KLEIDICV_BORDER_TYPE_REFLECT));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TYPED_TEST(BlurAndDownsample, Allocation) {
  constexpr size_t width = kMinWidthHeight;
  constexpr size_t height = kMinWidthHeight;
  TypeParam src[width * height] = {};
  TypeParam dst[width * height] = {};

  MockMallocToFail::enable();
  auto ret = blur_and_downsample<TypeParam>()(
      src, sizeof(TypeParam) * width, width, height, dst, sizeof(TypeParam), 1,
      KLEIDICV_BORDER_TYPE_REPLICATE);
  MockMallocToFail::disable();
  EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);
}
#endif
