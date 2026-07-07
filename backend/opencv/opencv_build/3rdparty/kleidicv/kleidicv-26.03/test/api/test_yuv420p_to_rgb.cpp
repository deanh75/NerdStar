// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

class YUV420p2RGBTest : public testing::Test {
 public:
  static inline int get_dcn(kleidicv_color_conversion_t color) {
    return (color & KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA) ? 4 : 3;
  }
  static inline kleidicv_color_conversion_t make_color_conversion_type(
      size_t num_channels, bool is_bgr, bool is_vu, bool is_uyvy,
      kleidicv_color_conversion_t base_fmt) {
    unsigned v = static_cast<unsigned>(base_fmt);
    if (is_vu) {
      v |= KLEIDICV_COLOR_CONVERSION_FLAG_VU;
    }
    if (is_bgr) {
      v |= KLEIDICV_COLOR_CONVERSION_FLAG_BGR;
    }
    if (num_channels == 4) {
      v |= KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA;
    }
    // is_uyvy only meaningful for YUV422; harmless otherwise
    if (is_uyvy) {
      v |= KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST;
    }
    return static_cast<kleidicv_color_conversion_t>(v);
  }
  struct TestParams {
    size_t width;
    size_t src_padding;
    size_t dst_padding;
    size_t height;
    size_t channels;
    bool is_yv12;
    bool is_bgr;
  };

  static std::vector<TestParams> generate_test_cases(
      const std::vector<size_t>& widths,
      const std::vector<size_t>& src_paddings,
      const std::vector<size_t>& dst_paddings,
      const std::vector<size_t>& heights, const std::vector<size_t>& channels,
      const std::vector<bool>& uv_cases,
      const std::vector<bool>& output_image_case) {
    std::vector<TestParams> cases;

    for (size_t w : widths) {
      for (size_t src_pad : src_paddings) {
        for (size_t dst_pad : dst_paddings) {
          for (size_t h : heights) {
            for (size_t c : channels) {
              for (bool uv_case : uv_cases) {
                for (bool is_bgr : output_image_case) {
                  cases.push_back({w, src_pad, dst_pad, h, c, uv_case, is_bgr});
                }
              }
            }
          }
        }
      }
    }

    return cases;
  }

  static std::vector<TestParams> get_test_cases() {
    std::vector<size_t> widths = {2, 4, 6, 18,
                                  test::Options::vector_length() * 2 + 1};
    std::vector<size_t> src_paddings = {4};
    std::vector<size_t> dst_paddings = {0};
    std::vector<size_t> heights = {2, 5, 11, 16};
    std::vector<size_t> channels = {3, 4};
    std::vector<bool> uv_cases = {true, false};
    std::vector<bool> output_image_case = {true, false};
    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, uv_cases, output_image_case);
  }

  void run_test_case(const TestParams& params) {
    test::Array2D<uint8_t> src{params.width, (params.height * 3 + 1) / 2,
                               params.src_padding};

    test::Array2D<uint8_t> expected_dst{params.width * params.channels,
                                        params.height, params.dst_padding,
                                        params.channels};

    test::Array2D<uint8_t> dst{params.width * params.channels, params.height,
                               params.dst_padding, params.channels};

    test::PseudoRandomNumberGenerator<uint8_t> input_value_random_range;
    src.fill(input_value_random_range);

    calculate_referenc(src.data(), src.stride(), expected_dst.data(),
                       expected_dst.stride(), params.width, params.height,
                       params.is_yv12, params.is_bgr, params.channels);

    auto status = KLEIDICV_OK;

    const kleidicv_color_conversion_t color_format = make_color_conversion_type(
        params.channels, params.is_bgr, params.is_yv12, false,
        KLEIDICV_COLOR_CONVERSION_FMT_YUV420P);

    status = kleidicv_yuv_to_rgb_u8(src.data(), src.stride(), dst.data(),
                                    dst.stride(), params.width, params.height,
                                    color_format);

    EXPECT_EQ(KLEIDICV_OK, status);
    EXPECT_EQ_ARRAY2D(expected_dst, dst);
  }

  template <typename Func>
  void run_unsupported(Func impl, kleidicv_color_conversion_t color_format) {
    test::Array2D<uint8_t> src{20, (10 * 3 + 1) / 2};
    size_t channels = get_dcn(color_format);
    test::Array2D<uint8_t> dst{20 * channels, 10, 0, channels};

    test::test_null_args(impl, src.data(), src.stride(), dst.data(),
                         dst.stride(), dst.width(), dst.height(), color_format);

    EXPECT_EQ(KLEIDICV_OK, impl(src.data(), src.stride(), dst.data(),
                                dst.stride(), 0, 1, color_format));

    EXPECT_EQ(KLEIDICV_OK, impl(src.data(), src.stride(), dst.data(),
                                dst.stride(), 1, 0, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS,
                   color_format));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   dst.width(), dst.height(), kleidicv_color_conversion_t{}));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   dst.width(), dst.height(),
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST)));
  }

 private:
  static uint8_t saturate_cast_s32_to_u8(int32_t rhs) {
    return static_cast<uint8_t>(
        std::min(std::max(0, rhs),
                 static_cast<int32_t>(std::numeric_limits<uint8_t>::max())));
  }
  void calculate_referenc(const uint8_t* src, size_t src_stride, uint8_t* dst,
                          size_t dst_stride, size_t width, size_t height,
                          bool is_yv12, bool BGR, size_t channels) {
    // this will the pointer to u plane
    const uint8_t* u = src + src_stride * height;
    // this will the pointer to v plane
    const uint8_t* v = src + src_stride * (height + height / 4) +
                       (width / 2) * ((height % 4) / 2);
    size_t ustepIdx = 0;
    size_t vstepIdx = height % 4 == 2 ? 1 : 0;
    size_t uvsteps[2] = {width / 2, static_cast<int>(src_stride) - width / 2};
    size_t usIdx = ustepIdx, vsIdx = vstepIdx;
    const uint8_t* y1 = src;
    const uint8_t* u1 = u;
    const uint8_t* v1 = v;
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        int32_t y = y1[h * src_stride + w];
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
        y = std::max(0, y - 16);
        int32_t u = u1[w >> 1];
        u -= 128;
        int32_t v = v1[w >> 1];
        v -= 128;
        if (is_yv12) {
          std::swap(u, v);
        }
        int32_t r = ((1220542 * y) + (1673527 * v) + (1 << 19)) >> 20;
        int32_t g =
            ((1220542 * y) - (409993 * u) - (852492 * v) + (1 << 19)) >> 20;
        int32_t b = ((1220542 * y) + (2116026 * u) + (1 << 19)) >> 20;

        uint8_t r_u8 = saturate_cast_s32_to_u8(r);
        uint8_t g_u8 = saturate_cast_s32_to_u8(g);
        uint8_t b_u8 = saturate_cast_s32_to_u8(b);

        if (BGR) {
          std::swap(b_u8, r_u8);
        }
        dst[h * dst_stride + w * channels + 0] = r_u8;
        dst[h * dst_stride + w * channels + 1] = g_u8;
        dst[h * dst_stride + w * channels + 2] = b_u8;
        if (channels == 4) {
          dst[h * dst_stride + w * channels + 3] = 0xff;
        }
      }
      if ((h % 2) == 1) {
        u1 += uvsteps[(usIdx++) & 1];
        v1 += uvsteps[(vsIdx++) & 1];
      }
    }
  }
};

TEST_F(YUV420p2RGBTest, ConvertspaddedInputsWithAllParamCombinations) {
  for (const auto& params : get_test_cases()) {
    run_test_case(params);
  }
}

TEST_F(YUV420p2RGBTest, ReturnsErrorForUnsupportedCombinations) {
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_YV12_TO_BGR);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_YV12_TO_RGB);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_YV12_TO_BGRA);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_YV12_TO_RGBA);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_IYUV_TO_BGR);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_IYUV_TO_RGB);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_IYUV_TO_BGRA);
  run_unsupported(kleidicv_yuv_to_rgb_u8, KLEIDICV_IYUV_TO_RGBA);
}
