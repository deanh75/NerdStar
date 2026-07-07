// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/kernel.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/morphology/workspace.h"
#include "test_config.h"

#define KLEIDICV_DILATE(type, type_suffix) \
  KLEIDICV_API(dilate, kleidicv_dilate_##type_suffix, type)

KLEIDICV_DILATE(uint8_t, u8);

template <typename ElementType>
class DilateParams {
 public:
  using InputType = ElementType;
  using IntermediateType = ElementType;
  using OutputType = ElementType;

  static decltype(auto) api() { return dilate<ElementType>(); }
  static decltype(auto) operation() {
    return [](ElementType a, ElementType b) { return std::max(a, b); };
  }
};  // end of class DilateParams<ElementType>

#define KLEIDICV_ERODE(type, type_suffix) \
  KLEIDICV_API(erode, kleidicv_erode_##type_suffix, type)

KLEIDICV_ERODE(uint8_t, u8);
template <typename ElementType>

class ErodeParams {
 public:
  using InputType = ElementType;
  using IntermediateType = ElementType;
  using OutputType = ElementType;

  static decltype(auto) api() { return erode<ElementType>(); }
  static decltype(auto) operation() {
    return [](ElementType a, ElementType b) { return std::min(a, b); };
  }
};  // end of class ErodeParams<ElementType>

static constexpr std::array<kleidicv_border_type_t, 1> kDefaultBorder = {
    KLEIDICV_BORDER_TYPE_REPLICATE};

static constexpr std::array<kleidicv_border_type_t, 1> kConstantBorder = {
    KLEIDICV_BORDER_TYPE_CONSTANT};

template <typename ElementType>
static const std::array<std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT>,
                        4> &
more_border_values() {
  using limit = std::numeric_limits<ElementType>;
  static const std::array<std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT>,
                          4>
      values = {{{0, 0, 0, 0},  // default
                 {7, 42, 99, 9},
                 {limit::min(), limit::max(), limit::min(), limit::max()},
                 {0, limit::min(), limit::max(), 0}}};
  return values;
}

template <class MorphologyKernelTestParams,
          typename ArrayLayoutsGetterType = decltype(test::small_array_layouts),
          typename BorderContainerType = decltype(kDefaultBorder),
          typename BorderValuesContainerType = std::array<
              std::array<uint8_t, KLEIDICV_MAXIMUM_CHANNEL_COUNT>, 1>>
class MorphologyTest : public test::KernelTest<MorphologyKernelTestParams> {
  using Base = test::KernelTest<MorphologyKernelTestParams>;
  using typename Base::InputType;
  using typename Base::IntermediateType;
  using typename Base::OutputType;
  using ArrayContainerType =
      std::invoke_result_t<ArrayLayoutsGetterType, size_t, size_t>;

 public:
  MorphologyTest(
      MorphologyKernelTestParams, size_t kernel_width, size_t kernel_height,
      ArrayLayoutsGetterType array_layouts_getter = test::small_array_layouts,
      BorderContainerType border_types = kDefaultBorder,
      BorderValuesContainerType border_values =
          test::default_border_values<InputType>())
      : kernel_width_{kernel_width},
        kernel_height_{kernel_height},
        mask_{kernel_width, kernel_height},
        kernel_{mask_},
        iterations_{1},
        array_layouts_{
            array_layouts_getter(std::max<size_t>(kernel_width - 1, 1),
                                 std::max<size_t>(kernel_height - 1, 1))},
        border_types_{border_types},
        border_values_{border_values},
        array_layout_generator_{array_layouts_},
        border_type_generator_{border_types_},
        border_values_generator_{border_values_} {}

  MorphologyTest &with_anchor(test::Point anchor) {
    kernel_ = test::Kernel(mask_, anchor);
    return *this;
  }

  MorphologyTest &with_iterations(size_t iter) {
    iterations_ = iter;
    return *this;
  }

  void test() {
    test::PseudoRandomNumberGenerator<InputType> element_generator;
    Base::test(kernel_, array_layout_generator_, border_type_generator_,
               border_values_generator_, element_generator);
  }

 private:
  kleidicv_error_t call_api(const test::Array2D<InputType> *input,
                            test::Array2D<OutputType> *output,
                            kleidicv_border_type_t border_type,
                            const InputType *border_value) override {
    return MorphologyKernelTestParams::api()(
        input->data(), input->stride(), output->data(), output->stride(),
        input->width() / input->channels(), input->height(), input->channels(),
        kernel_width_, kernel_height_, static_cast<size_t>(kernel_.anchor().x),
        static_cast<size_t>(kernel_.anchor().y), border_type,
        reinterpret_cast<const uint8_t *>(border_value), iterations_);
  }

  void prepare_expected(const test::Kernel<IntermediateType> &kernel,
                        const test::ArrayLayout &array_layout,
                        kleidicv_border_type_t border_type,
                        const InputType *border_value) override {
    Base::prepare_expected(kernel, array_layout, border_type, border_value);
    if (iterations_ > 1) {
      test::Array2D<InputType> saved_input = this->input_;
      for (size_t i = 1; i < iterations_; ++i) {
        this->input_ = this->expected_;
        Base::prepare_expected(kernel, array_layout, border_type, border_value);
      }
      this->input_ = saved_input;
    }
  }

  IntermediateType calculate_expected_at(
      const test::Kernel<IntermediateType> &kernel,
      const test::TwoDimensional<InputType> &source, size_t row,
      size_t column) override {
    IntermediateType result = source.at(row, column)[0];
    for (size_t height = 0; height < kernel.height(); ++height) {
      for (size_t width = 0; width < kernel.width(); ++width) {
        result = MorphologyKernelTestParams::operation()(
            result,
            source.at(row + height, column + width * source.channels())[0]);
      }
    }
    return result;
  }

  const size_t kernel_width_;
  const size_t kernel_height_;
  const test::Array2D<InputType> mask_;
  test::Kernel<InputType> kernel_;
  size_t iterations_;
  const ArrayContainerType array_layouts_;
  const BorderContainerType border_types_;
  const BorderValuesContainerType border_values_;
  test::SequenceGenerator<ArrayContainerType> array_layout_generator_;
  test::SequenceGenerator<BorderContainerType> border_type_generator_;
  test::SequenceGenerator<BorderValuesContainerType> border_values_generator_;
};  // end of class MorphologyTest<MorphologyKernelTestParams,
    // ArrayLayoutsGetterType, BorderContainerType, BorderValuesContainerType>

template <typename TypeParam>
class Morphology : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t>;

TYPED_TEST_SUITE(Morphology, ElementTypes);

TYPED_TEST(Morphology, 1xN) {
  MorphologyTest{DilateParams<TypeParam>{}, 1, 1}.test();
  MorphologyTest{ErodeParams<TypeParam>{}, 1, 1}.test();

  MorphologyTest{DilateParams<TypeParam>{}, 1, 2, test::default_array_layouts}
      .test();
  MorphologyTest{ErodeParams<TypeParam>{}, 1, 2, test::default_array_layouts}
      .test();
  MorphologyTest{DilateParams<TypeParam>{}, 3, 1}.test();
  MorphologyTest{ErodeParams<TypeParam>{}, 3, 1}.test();
}

std::array<test::ArrayLayout, 4> get_large_array_layouts(size_t min_width,
                                                         size_t min_height) {
  size_t vl = test::Options::vector_length();
  size_t big_height = std::max(2 * vl + 1, min_height * 4);

  return {{
      // clang-format off
      //         width,         height,  padding, channels
      {  min_width * 4,     min_height,        0,        4},
      {  min_width * 4,     min_height,       vl,        4},
      {  min_width * 2,     big_height,        0,        1},
      {  min_width * 2,     big_height,       vl,        1},
      // clang-format on
  }};
}

TYPED_TEST(Morphology, LargeArrays) {
  MorphologyTest{DilateParams<TypeParam>{}, 3, 3, get_large_array_layouts}
      .test();
  MorphologyTest{ErodeParams<TypeParam>{}, 3, 3, get_large_array_layouts}
      .test();

  MorphologyTest{DilateParams<TypeParam>{}, 3, 3, get_large_array_layouts,
                 kConstantBorder}
      .test();
  MorphologyTest{ErodeParams<TypeParam>{}, 3, 3, get_large_array_layouts,
                 kConstantBorder}
      .test();
}

TYPED_TEST(Morphology, MediumArrays) {
  MorphologyTest{DilateParams<TypeParam>{}, 3, 3, test::default_array_layouts}
      .test();
  MorphologyTest{ErodeParams<TypeParam>{}, 3, 3, test::default_array_layouts}
      .test();

  MorphologyTest{DilateParams<TypeParam>{}, 5, 5, test::default_array_layouts}
      .test();
  MorphologyTest{ErodeParams<TypeParam>{}, 5, 5, test::default_array_layouts}
      .test();
}

TYPED_TEST(Morphology, BorderValues) {
  MorphologyTest{DilateParams<TypeParam>{},
                 3,
                 3,
                 test::small_array_layouts,
                 kConstantBorder,
                 more_border_values<TypeParam>()}
      .test();
  MorphologyTest{ErodeParams<TypeParam>{},
                 3,
                 3,
                 test::small_array_layouts,
                 kConstantBorder,
                 more_border_values<TypeParam>()}
      .test();
}

TYPED_TEST(Morphology, UnortodoxSizes) {
  MorphologyTest{DilateParams<TypeParam>{}, 4, 4}.test();
  MorphologyTest{ErodeParams<TypeParam>{}, 7, 5}.test();

  MorphologyTest{DilateParams<TypeParam>{}, 8, 4}.test();
  MorphologyTest{DilateParams<TypeParam>{}, 6, 10}.test();
  MorphologyTest{ErodeParams<TypeParam>{}, 12, 4}.test();
}

TYPED_TEST(Morphology, Iterations) {
  MorphologyTest{DilateParams<TypeParam>{}, 3, 3}.with_iterations(2).test();
  MorphologyTest{ErodeParams<TypeParam>{}, 6, 4}.with_iterations(3).test();
  MorphologyTest{DilateParams<TypeParam>{}, 2, 7}.with_iterations(4).test();
}

TYPED_TEST(Morphology, InPlaceOperation) {
  constexpr size_t kWidth = 47;
  constexpr size_t kHeight = 96;
  constexpr size_t kChannels = 3;
  constexpr size_t kKernelWidth = 5;
  constexpr size_t kKernelHeight = 5;
  constexpr size_t kAnchorX = 2;
  constexpr size_t kAnchorY = 2;
  constexpr size_t kIterations = 2;

  test::PseudoRandomNumberGenerator<TypeParam> generator;
  test::Array2D<TypeParam> src{kWidth * kChannels, kHeight,
                               test::Options::vector_length(), kChannels};
  test::Array2D<TypeParam> in_place{kWidth * kChannels, kHeight,
                                    test::Options::vector_length(), kChannels};
  test::Array2D<TypeParam> out_of_place_src{
      kWidth * kChannels, kHeight, test::Options::vector_length(), kChannels};
  test::Array2D<TypeParam> out_of_place_dst{
      kWidth * kChannels, kHeight, test::Options::vector_length(), kChannels};

  src.fill(generator);
  in_place = src;
  out_of_place_src = src;

  const uint8_t border_value[] = {7, 42, 99, 9};

  ASSERT_EQ(KLEIDICV_OK,
            DilateParams<TypeParam>::api()(
                in_place.data(), in_place.stride(), in_place.data(),
                in_place.stride(), kWidth, kHeight, kChannels, kKernelWidth,
                kKernelHeight, kAnchorX, kAnchorY,
                KLEIDICV_BORDER_TYPE_REPLICATE, border_value, kIterations));
  ASSERT_EQ(
      KLEIDICV_OK,
      DilateParams<TypeParam>::api()(
          out_of_place_src.data(), out_of_place_src.stride(),
          out_of_place_dst.data(), out_of_place_dst.stride(), kWidth, kHeight,
          kChannels, kKernelWidth, kKernelHeight, kAnchorX, kAnchorY,
          KLEIDICV_BORDER_TYPE_REPLICATE, border_value, kIterations));
  EXPECT_EQ_ARRAY2D(in_place, out_of_place_dst);

  in_place = src;
  out_of_place_src = src;
  out_of_place_dst.fill(static_cast<TypeParam>(0));

  ASSERT_EQ(KLEIDICV_OK,
            ErodeParams<TypeParam>::api()(
                in_place.data(), in_place.stride(), in_place.data(),
                in_place.stride(), kWidth, kHeight, kChannels, kKernelWidth,
                kKernelHeight, kAnchorX, kAnchorY,
                KLEIDICV_BORDER_TYPE_REPLICATE, border_value, kIterations));
  ASSERT_EQ(
      KLEIDICV_OK,
      ErodeParams<TypeParam>::api()(
          out_of_place_src.data(), out_of_place_src.stride(),
          out_of_place_dst.data(), out_of_place_dst.stride(), kWidth, kHeight,
          kChannels, kKernelWidth, kKernelHeight, kAnchorX, kAnchorY,
          KLEIDICV_BORDER_TYPE_REPLICATE, border_value, kIterations));
  EXPECT_EQ_ARRAY2D(in_place, out_of_place_dst);
}

TYPED_TEST(Morphology, Anchors) {
  MorphologyTest{ErodeParams<TypeParam>{}, 3, 5}.with_anchor({0, 0}).test();

  MorphologyTest{DilateParams<TypeParam>{}, 3, 5, test::small_array_layouts,
                 kConstantBorder}
      .with_anchor({2, 0})
      .test();

  MorphologyTest{ErodeParams<TypeParam>{}, 3, 5, test::small_array_layouts,
                 kConstantBorder}
      .with_anchor({0, 4})
      .test();

  MorphologyTest{DilateParams<TypeParam>{}, 3, 5}.with_anchor({2, 4}).test();
}

template <typename ElementType>
static void test_valid_image_size(kleidicv_rectangle_t kernel,
                                  test::Array2D<ElementType> src) {
  size_t validSize = kernel.width - 1;
  const uint8_t border_value[] = {0, 0, 1, 1};

  test::Array2D<ElementType> dst{validSize, validSize,
                                 test::Options::vector_length()};

  for (size_t x = 0; x < kernel.width; x += kernel.width - 1) {
    for (size_t y = 0; y < kernel.width; y += kernel.width - 1) {
      kleidicv_point_t anchor{x, y};
      for (kleidicv_border_type_t border : {
               KLEIDICV_BORDER_TYPE_REPLICATE,
               KLEIDICV_BORDER_TYPE_CONSTANT,
           }) {
        EXPECT_EQ(KLEIDICV_OK,
                  ErodeParams<ElementType>::api()(
                      src.data(), src.stride(), dst.data(), dst.stride(),
                      validSize, validSize, 1, kernel.width, kernel.height,
                      anchor.x, anchor.y, border, border_value, 1));
        EXPECT_EQ(KLEIDICV_OK,
                  DilateParams<ElementType>::api()(
                      src.data(), src.stride(), dst.data(), dst.stride(),
                      validSize, validSize, 1, kernel.width, kernel.height,
                      anchor.x, anchor.y, border, border_value, 1));
      }
    }
  }
}

template <typename ElementType>
static void test_undersize_image(kleidicv_rectangle_t kernel) {
  size_t underSize = kernel.width - 2;
  size_t validWidth = kernel.width + 10;
  size_t validHeight = kernel.height + 5;
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};
  kleidicv_point_t anchor{1, 1};
  ElementType src[1] = {}, dst[1] = {};
  auto expect_not_implemented = [&](size_t width, size_t height) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              ErodeParams<ElementType>::api()(
                  src, sizeof(ElementType), dst, sizeof(ElementType), width,
                  height, 1, kernel.width, kernel.height, anchor.x, anchor.y,
                  border, border_value, 1));
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              DilateParams<ElementType>::api()(
                  src, sizeof(ElementType), dst, sizeof(ElementType), width,
                  height, 1, kernel.width, kernel.height, anchor.x, anchor.y,
                  border, border_value, 1));
  };
  expect_not_implemented(underSize, underSize);
  expect_not_implemented(underSize, validHeight);
  expect_not_implemented(validWidth, underSize);
}

TYPED_TEST(Morphology, UnsupportedBorderType) {
  TypeParam src[1] = {}, dst[1] = {};
  kleidicv_rectangle_t kernel{1, 1};
  kleidicv_point_t anchor{0, 0};
  const uint8_t border_value[] = {0, 0, 1, 1};
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_REFLECT,
           KLEIDICV_BORDER_TYPE_WRAP,
           KLEIDICV_BORDER_TYPE_REVERSE,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              DilateParams<TypeParam>::api()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1, 1,
                  kernel.width, kernel.height, anchor.x, anchor.y, border,
                  border_value, 1));
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              ErodeParams<TypeParam>::api()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1, 1,
                  kernel.width, kernel.height, anchor.x, anchor.y, border,
                  border_value, 1));
  }
}

TYPED_TEST(Morphology, UnsupportedSize) {
  TypeParam src[1] = {}, dst[1] = {};
  kleidicv_rectangle_t small_rect{1, 1};
  kleidicv_point_t anchor{0, 0};
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};

  for (kleidicv_rectangle_t bad_rect : {
           kleidicv_rectangle_t{KLEIDICV_MAX_IMAGE_PIXELS + 1, 1},
           kleidicv_rectangle_t{KLEIDICV_MAX_IMAGE_PIXELS,
                                KLEIDICV_MAX_IMAGE_PIXELS},
       }) {
    auto expect_range = [&](kleidicv_error_t status, size_t width,
                            size_t height, size_t kernel_width,
                            size_t kernel_height) {
      EXPECT_EQ(status, DilateParams<TypeParam>::api()(
                            src, sizeof(TypeParam), dst, sizeof(TypeParam),
                            width, height, 1, kernel_width, kernel_height,
                            anchor.x, anchor.y, border, border_value, 1));
      EXPECT_EQ(status, ErodeParams<TypeParam>::api()(
                            src, sizeof(TypeParam), dst, sizeof(TypeParam),
                            width, height, 1, kernel_width, kernel_height,
                            anchor.x, anchor.y, border, border_value, 1));
    };
    expect_range(KLEIDICV_ERROR_RANGE, 1, 1, bad_rect.width, bad_rect.height);
    expect_range(KLEIDICV_ERROR_RANGE, bad_rect.width, bad_rect.height,
                 small_rect.width, small_rect.height);
  }

  const size_t image_width = KLEIDICV_MAX_IMAGE_PIXELS;
  const size_t image_height = 1;
  const size_t kernel_width = image_width + 1;
  const size_t kernel_height = 1;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), image_width,
                image_height, 1, kernel_width, kernel_height, anchor.x,
                anchor.y, border, border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), image_width,
                image_height, 1, kernel_width, kernel_height, anchor.x,
                anchor.y, border, border_value, 1));
}

TYPED_TEST(Morphology, UnsupportedChannels) {
  TypeParam src[1] = {}, dst[1] = {};
  kleidicv_rectangle_t small_rect{1, 1};
  kleidicv_point_t anchor{0, 0};
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      DilateParams<TypeParam>::api()(
          src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1,
          KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, small_rect.width,
          small_rect.height, anchor.x, anchor.y, border, border_value, 1));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      ErodeParams<TypeParam>::api()(
          src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1,
          KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, small_rect.width,
          small_rect.height, anchor.x, anchor.y, border, border_value, 1));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TYPED_TEST(Morphology, CannotAllocateImage) {
  MockMallocToFail::enable();
  kleidicv_rectangle_t kernel{3, 3}, image{3072, 2048};
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};
  kleidicv_point_t anchor{1, 1};
  TypeParam src[1] = {}, dst[1] = {};

  EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), image.width,
                image.height, 1, kernel.width, kernel.height, anchor.x,
                anchor.y, border, border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), image.width,
                image.height, 1, kernel.width, kernel.height, anchor.x,
                anchor.y, border, border_value, 1));
  MockMallocToFail::disable();
}
#endif

TYPED_TEST(Morphology, OversizeImage) {
  kleidicv_rectangle_t kernel{3, 1UL << 33},
      image{1UL << 33, KLEIDICV_MAX_IMAGE_PIXELS};
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};
  kleidicv_point_t anchor{1, 1};
  TypeParam src[1] = {}, dst[1] = {};

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), image.width,
                image.height, 1, kernel.width, kernel.height, anchor.x,
                anchor.y, border, border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), image.width,
                image.height, 1, kernel.width, kernel.height, anchor.x,
                anchor.y, border, border_value, 1));
}

TYPED_TEST(Morphology, WorkspaceBufferOverflow) {
  using Workspace = ::KLEIDICV_TARGET_NAMESPACE::MorphologyWorkspace;

  kleidicv::Rectangle kernel{1, 32};
  kleidicv::Point anchor{0, 0};
  constexpr size_t kBufferRowsHeight = 128;
  // Choose a width that makes buffer_rows_stride * buffer_rows_height overflow.
  const size_t element_stride =
      static_cast<size_t>(KLEIDICV_MAXIMUM_TYPE_SIZE) *
      static_cast<size_t>(KLEIDICV_MAXIMUM_CHANNEL_COUNT);
  const size_t image_width =
      (std::numeric_limits<size_t>::max() / kBufferRowsHeight) /
          element_stride +
      1;
  const kleidicv::Rectangle image{image_width, size_t{1}};

  auto workspace_variant = Workspace::create(
      kernel, anchor, Workspace::BorderType::REPLICATE, nullptr,
      KLEIDICV_MAXIMUM_CHANNEL_COUNT, KLEIDICV_MAXIMUM_TYPE_SIZE, image);
  auto *error = std::get_if<kleidicv_error_t>(&workspace_variant);
  ASSERT_NE(nullptr, error);
  EXPECT_EQ(KLEIDICV_ERROR_RANGE, *error);
}

TYPED_TEST(Morphology, WorkspaceBufferOverflowRows) {
  using Workspace = ::KLEIDICV_TARGET_NAMESPACE::MorphologyWorkspace;

  const kleidicv::Rectangle kernel{size_t{1},
                                   size_t{KLEIDICV_MAX_IMAGE_PIXELS}};
  const kleidicv::Point anchor{0, 0};
  const kleidicv::Rectangle image{1024, 1};

  size_t kernel_pixels = 0;
  EXPECT_FALSE(
      __builtin_mul_overflow(kernel.width(), kernel.height(), &kernel_pixels));
  EXPECT_LE(kernel_pixels, KLEIDICV_MAX_IMAGE_PIXELS);

  size_t image_pixels = 0;
  EXPECT_FALSE(
      __builtin_mul_overflow(image.width(), image.height(), &image_pixels));
  EXPECT_LE(image_pixels, KLEIDICV_MAX_IMAGE_PIXELS);

  const size_t rows_per_iteration =
      std::max(2 * kernel.height(), static_cast<size_t>(32ULL));
  const size_t buffer_rows_height = 2 * rows_per_iteration;
  const size_t buffer_rows_stride = KLEIDICV_MAXIMUM_TYPE_SIZE * image.width() *
                                    KLEIDICV_MAXIMUM_CHANNEL_COUNT;
  size_t buffer_rows_size = 0;
  EXPECT_TRUE(__builtin_mul_overflow(buffer_rows_stride, buffer_rows_height,
                                     &buffer_rows_size));

  auto workspace_variant = Workspace::create(
      kernel, anchor, Workspace::BorderType::REPLICATE, nullptr,
      KLEIDICV_MAXIMUM_CHANNEL_COUNT, KLEIDICV_MAXIMUM_TYPE_SIZE, image);
  auto *error = std::get_if<kleidicv_error_t>(&workspace_variant);
  ASSERT_NE(nullptr, error);
  EXPECT_EQ(KLEIDICV_ERROR_RANGE, *error);
}

TYPED_TEST(Morphology, NullBorderValue) {
  TypeParam src[1] = {}, dst[1] = {};

  EXPECT_EQ(KLEIDICV_BORDER_TYPE_CONSTANT,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
  EXPECT_EQ(KLEIDICV_BORDER_TYPE_CONSTANT,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
}

TYPED_TEST(Morphology, InvalidAnchors) {
  kleidicv_rectangle_t kernel1{1, 1}, kernel2{6, 4}, image{20, 20};
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};
  kleidicv_point_t anchor1{1, 0}, anchor2{6, 3}, anchor3{5, 4};
  test::Array2D<TypeParam> src{20, 20};
  test::Array2D<TypeParam> dst{20, 20};

  auto expect_range = [&](kleidicv_rectangle_t kernel,
                          kleidicv_point_t anchor) {
    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              DilateParams<TypeParam>::api()(
                  src.data(), sizeof(TypeParam) * image.width, dst.data(),
                  sizeof(TypeParam) * image.width, image.width, image.height, 1,
                  kernel.width, kernel.height, anchor.x, anchor.y, border,
                  border_value, 1));
    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              ErodeParams<TypeParam>::api()(
                  src.data(), sizeof(TypeParam) * image.width, dst.data(),
                  sizeof(TypeParam) * image.width, image.width, image.height, 1,
                  kernel.width, kernel.height, anchor.x, anchor.y, border,
                  border_value, 1));
  };
  expect_range(kernel1, anchor1);
  expect_range(kernel2, anchor2);
  expect_range(kernel2, anchor3);
}

TYPED_TEST(Morphology, InvalidChannelNumber) {
  TypeParam src[1] = {}, dst[1] = {};
  const uint8_t border_value[] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1,
                KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, 1, 1, 0, 0,
                KLEIDICV_BORDER_TYPE_REPLICATE, border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1,
                KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, 1, 1, 0, 0,
                KLEIDICV_BORDER_TYPE_REPLICATE, border_value, 1));
}

TYPED_TEST(Morphology, DilateNullPointer) {
  TypeParam src[1] = {}, dst[1] = {};
  const uint8_t border_value[] = {0, 0, 1, 1};
  test::test_null_args(DilateParams<TypeParam>::api(), src, sizeof(TypeParam),
                       dst, sizeof(TypeParam), 1, 1, 1, 1, 1, 0, 0,
                       KLEIDICV_BORDER_TYPE_CONSTANT, border_value, 1);
}

TYPED_TEST(Morphology, ErodeNullPointer) {
  TypeParam src[1] = {}, dst[1] = {};
  const uint8_t border_value[] = {0, 0, 1, 1};
  test::test_null_args(ErodeParams<TypeParam>::api(), src, sizeof(TypeParam),
                       dst, sizeof(TypeParam), 1, 1, 1, 1, 1, 0, 0,
                       KLEIDICV_BORDER_TYPE_CONSTANT, border_value, 1);
}

TYPED_TEST(Morphology, DilateMisalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2];
  const uint8_t border_value[] = {0, 0, 1, 1};
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), 1, 2, 1, 1,
                1, 0, 0, KLEIDICV_BORDER_TYPE_CONSTANT, border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, 1, 2, 1, 1,
                1, 0, 0, KLEIDICV_BORDER_TYPE_CONSTANT, border_value, 1));
}

TYPED_TEST(Morphology, ErodeMisalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2];
  const uint8_t border_value[] = {0, 0, 1, 1};
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), 1, 2, 1, 1,
                1, 0, 0, KLEIDICV_BORDER_TYPE_CONSTANT, border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, 1, 2, 1, 1,
                1, 0, 0, KLEIDICV_BORDER_TYPE_CONSTANT, border_value, 1));
}

TYPED_TEST(Morphology, DilateZeroImageSize) {
  TypeParam src[1] = {}, dst[1] = {};
  const uint8_t border_value[] = {0, 0, 1, 1};
  EXPECT_EQ(KLEIDICV_OK,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 0, 1, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, border_value, 1));
  EXPECT_EQ(KLEIDICV_OK,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 0, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, border_value, 1));
}

TYPED_TEST(Morphology, ErodeZeroImageSize) {
  TypeParam src[1] = {}, dst[1] = {};
  const uint8_t border_value[] = {0, 0, 1, 1};
  EXPECT_EQ(KLEIDICV_OK,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 0, 1, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, border_value, 1));
  EXPECT_EQ(KLEIDICV_OK,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 0, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, border_value, 1));
}

TYPED_TEST(Morphology, ValidImageSize) {
  kleidicv_rectangle_t kernel3x3{3, 3};
  kleidicv_rectangle_t kernel5x5{5, 5};
  test::Array2D<TypeParam> src2x2{kernel3x3.width - 1, kernel3x3.width - 1,
                                  test::Options::vector_length()};
  src2x2.set(0, 0, {1, 2});
  src2x2.set(1, 0, {1, 2});
  test::Array2D<TypeParam> src4x4{kernel5x5.width - 1, kernel5x5.width - 1,
                                  test::Options::vector_length()};
  src4x4.set(0, 0, {1, 2, 3, 4});
  src4x4.set(1, 0, {1, 2, 3, 4});
  src4x4.set(2, 0, {1, 2, 3, 4});
  src4x4.set(3, 0, {1, 2, 3, 4});
  test_valid_image_size<TypeParam>(kernel3x3, src2x2);
  test_valid_image_size<TypeParam>(kernel5x5, src4x4);
}

TYPED_TEST(Morphology, UndersizeImage) {
  kleidicv_rectangle_t kernel3x3{3, 3};
  kleidicv_rectangle_t kernel5x5{5, 5};
  test_undersize_image<TypeParam>(kernel3x3);
  test_undersize_image<TypeParam>(kernel5x5);
}

TYPED_TEST(Morphology, ImageSmallerThanKernel) {
  kleidicv_rectangle_t kernel{3, 3};
  kleidicv_point_t anchor{1, 1};
  kleidicv_border_type_t border = KLEIDICV_BORDER_TYPE_REPLICATE;
  const uint8_t border_value[] = {0, 0, 1, 1};
  TypeParam src[1] = {}, dst[1] = {};
  const size_t width = kernel.width - 2;
  const size_t height = kernel.height - 2;

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), width, height,
                1, kernel.width, kernel.height, anchor.x, anchor.y, border,
                border_value, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            ErodeParams<TypeParam>::api()(src, sizeof(TypeParam), dst,
                                          sizeof(TypeParam), width, height, 1,
                                          kernel.width, kernel.height, anchor.x,
                                          anchor.y, border, border_value, 1));
}

TYPED_TEST(Morphology, DilateOversizeImage) {
  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, 1, 1, 0, 0,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            DilateParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
}

TYPED_TEST(Morphology, ErodeOversizeImage) {
  TypeParam src[1] = {}, dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, 1, 1, 0, 0,
                KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            ErodeParams<TypeParam>::api()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 1, 1, 1,
                0, 0, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
}
