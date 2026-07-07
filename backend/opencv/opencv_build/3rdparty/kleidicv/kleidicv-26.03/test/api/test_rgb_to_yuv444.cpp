// SPDX-FileCopyrightText: 2024 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"
#include "test_config.h"

class RgbToYuvTest final {
 public:
  RgbToYuvTest(size_t channels, bool switch_blue)
      : src_channels_(channels), switch_blue_(switch_blue) {}

  template <typename F>
  void execute_scalar_test(F impl, kleidicv_color_conversion_t color_format) {
    size_t scalar_path_width = 5;
    execute_test(impl, scalar_path_width, 0, color_format);
    // Padding version
    execute_test(impl, scalar_path_width, test::Options::vector_length(),
                 color_format);
  }

  template <typename F>
  void execute_vector_test(F impl, kleidicv_color_conversion_t color_format) {
    size_t vector_path_width = (3 * test::Options::vector_lanes<uint8_t>()) - 3;
    execute_test(impl, vector_path_width, 0, color_format);
    // Padding version
    execute_test(impl, vector_path_width, test::Options::vector_length(),
                 color_format);
  }

 private:
  static const size_t kDstChannels = 3;
  template <typename F>
  void execute_test(F impl, size_t logical_width, size_t padding,
                    kleidicv_color_conversion_t color_format) {
    test::Array2D<uint8_t> src{logical_width * src_channels_, 5, padding};
    src.fill(0);
    src.set(0, 0, {0, 22, 4, 0, 27, 9, 255, 125, 255, 255, 60, 255});
    src.set(1, 0, {0, 154, 0, 0, 154, 0, 61, 255, 11, 47, 255, 0});
    src.set(2, 0, {0, 22, 4, 76, 143, 125, 203, 0, 255, 204, 0, 255});
    src.set(4, 0, {0, 145, 0, 0, 145, 0, 0, 255, 0, 0, 255, 0});

    test::Array2D<uint8_t> expected{logical_width * kDstChannels, src.height(),
                                    padding};
    expected.fill(0);
    calculate_expected(src, expected);

    test::Array2D<uint8_t> actual{logical_width * kDstChannels, src.height(),
                                  padding};
    actual.fill(42);
    auto err = impl(src.data(), src.stride(), actual.data(), actual.stride(),
                    logical_width, expected.height(), color_format);

    ASSERT_EQ(KLEIDICV_OK, err);
    EXPECT_EQ_ARRAY2D(expected, actual);

    test::test_null_args(impl, src.data(), src.stride(), actual.data(),
                         actual.stride(), logical_width, expected.height(),
                         color_format);

    EXPECT_EQ(KLEIDICV_OK, impl(src.data(), src.stride(), actual.data(),
                                actual.stride(), 0, 1, color_format));
    EXPECT_EQ(KLEIDICV_OK, impl(src.data(), src.stride(), actual.data(),
                                actual.stride(), 1, 0, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), actual.data(), actual.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), actual.data(), actual.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS,
                   color_format));

    EXPECT_EQ(
        KLEIDICV_ERROR_NOT_IMPLEMENTED,
        impl(src.data(), src.stride(), actual.data(), actual.stride(),
             logical_width, expected.height(), kleidicv_color_conversion_t{}));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), actual.data(), actual.stride(),
                   actual.width(), actual.height(),
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV444 |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST)));
  }

  void calculate_expected(test::Array2D<uint8_t> &src_arr,
                          test::Array2D<uint8_t> &exp_arr) const {
    for (size_t vindex = 0; vindex < exp_arr.height(); vindex++) {
      for (size_t hindex = 0; hindex < exp_arr.width() / kDstChannels;
           hindex++) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        int32_t r = *src_arr.at(
            vindex, hindex * src_channels_ + (switch_blue_ ? 2 : 0));
        int32_t g = *src_arr.at(vindex, hindex * src_channels_ + 1);
        int32_t b = *src_arr.at(
            vindex, hindex * src_channels_ + (switch_blue_ ? 0 : 2));
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)

        static const int32_t R2Y = 4899, G2Y = 9617, B2Y = 1868;
        static const int32_t R2V = 14369, B2U = 8061;

        int32_t y = (r * R2Y + g * G2Y + b * B2Y + (1 << 13)) >> 14;
        int32_t u = (((b - y) * B2U + (1 << 13)) >> 14) + 128;
        int32_t v = (((r - y) * R2V + (1 << 13)) >> 14) + 128;

        uint8_t y_u8 = saturate_cast_s32_to_u8(y);
        uint8_t u_u8 = saturate_cast_s32_to_u8(u);
        uint8_t v_u8 = saturate_cast_s32_to_u8(v);

        exp_arr.set(vindex, hindex * kDstChannels, {y_u8, u_u8, v_u8});
      }
    }
  }

  static uint8_t saturate_cast_s32_to_u8(int32_t rhs) {
    return static_cast<uint8_t>(
        std::min(std::max(0, rhs),
                 static_cast<int32_t>(std::numeric_limits<uint8_t>::max())));
  }

  size_t src_channels_;
  bool switch_blue_;
};

TEST(RgbToYuv, RGB_TO_YUV444_SCALAR) {
  RgbToYuvTest rgb2yuv_test(3, false);
  rgb2yuv_test.execute_scalar_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_RGB_TO_YUV444);
}

TEST(RgbToYuv, RGB_TO_YUV444_VECTOR) {
  RgbToYuvTest rgb2yuv_test(3, false);
  rgb2yuv_test.execute_vector_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_RGB_TO_YUV444);
}

TEST(RgbToYuv, RGBA_TO_YUV444_SCALAR) {
  RgbToYuvTest rgb2yuv_test(4, false);
  rgb2yuv_test.execute_scalar_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_RGBA_TO_YUV444);
}

TEST(RgbToYuv, RGBA_TO_YUV44_VECTOR) {
  RgbToYuvTest rgb2yuv_test(4, false);
  rgb2yuv_test.execute_vector_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_RGBA_TO_YUV444);
}

TEST(RgbToYuv, BGR_TO_YUV444_SCALAR) {
  RgbToYuvTest rgb2yuv_test(3, true);
  rgb2yuv_test.execute_scalar_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_BGR_TO_YUV444);
}

TEST(RgbToYuv, BGR_TO_YUV444_VECTOR) {
  RgbToYuvTest rgb2yuv_test(3, true);
  rgb2yuv_test.execute_vector_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_BGR_TO_YUV444);
}

TEST(RgbToYuv, BGRA_TO_YUV444_SCALAR) {
  RgbToYuvTest rgb2yuv_test(4, true);
  rgb2yuv_test.execute_scalar_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_BGRA_TO_YUV444);
}

TEST(RgbToYuv, BGRA_TO_YUV444_VECTOR) {
  RgbToYuvTest rgb2yuv_test(4, true);
  rgb2yuv_test.execute_vector_test(kleidicv_rgb_to_yuv_u8,
                                   KLEIDICV_BGRA_TO_YUV444);
}
