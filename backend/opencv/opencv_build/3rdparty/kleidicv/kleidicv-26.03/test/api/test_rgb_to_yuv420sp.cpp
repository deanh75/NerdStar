// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"
#include "test_config.h"

class RGB2YUV420SpTest : public testing::Test {
 public:
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

  static inline int get_scn(kleidicv_color_conversion_t color) {
    return (color & KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA) ? 4 : 3;
  }
  struct TestParams {
    size_t width;
    size_t src_padding;
    size_t dst_padding;
    size_t height;
    size_t channels;
    bool is_nv21;
    bool is_rgb;
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
                for (bool is_rgb : output_image_case) {
                  cases.push_back({w, src_pad, dst_pad, h, c, uv_case, is_rgb});
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
    std::vector<size_t> widths = {3, 27, 32, 770};
    std::vector<size_t> src_paddings = {2};
    std::vector<size_t> dst_paddings = {3};
    std::vector<size_t> heights = {11, 16};
    std::vector<size_t> channels = {3, 4};
    std::vector<bool> uv_cases = {true, false};
    std::vector<bool> output_image_case = {true, false};
    return generate_test_cases(widths, src_paddings, dst_paddings, heights,
                               channels, uv_cases, output_image_case);
  }

  void run_test_case(const TestParams& params) {
    test::Array2D<uint8_t> src{params.width * params.channels, params.height,
                               params.src_padding, params.channels};

    test::Array2D<uint8_t> expected_y_dst{params.width, params.height,
                                          params.dst_padding};

    test::Array2D<uint8_t> expected_uv_dst{
        KLEIDICV_TARGET_NAMESPACE::align_up(params.width, 2),
        (params.height + 1) / 2, params.dst_padding};

    test::Array2D<uint8_t> y_dst{params.width, params.height,
                                 params.dst_padding};

    test::Array2D<uint8_t> uv_dst{
        KLEIDICV_TARGET_NAMESPACE::align_up(params.width, 2),
        (params.height + 1) / 2, params.dst_padding};

    test::PseudoRandomNumberGenerator<uint8_t> input_value_random_range;
    src.fill(input_value_random_range);

    calculate_reference(src.data(), src.stride(), expected_y_dst.data(),
                        expected_y_dst.stride(), expected_uv_dst.data(),
                        expected_uv_dst.stride(), params.width, params.height,
                        params.is_nv21, params.is_rgb, params.channels);

    auto status = KLEIDICV_OK;

    const kleidicv_color_conversion_t color_format = make_color_conversion_type(
        params.channels, !params.is_rgb, params.is_nv21, false,
        KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP);

    status = kleidicv_rgb_to_yuv_semiplanar_u8(
        src.data(), src.stride(), y_dst.data(), y_dst.stride(), uv_dst.data(),
        uv_dst.stride(), params.width, params.height, color_format);

    EXPECT_EQ(KLEIDICV_OK, status);
    EXPECT_EQ_ARRAY2D(expected_y_dst, y_dst);
    EXPECT_EQ_ARRAY2D(expected_uv_dst, uv_dst);
  }

  template <typename Func>
  void run_unsupported(Func impl, kleidicv_color_conversion_t color_format) {
    size_t channels = get_scn(color_format);
    test::Array2D<uint8_t> src{20 * channels, 10, 0, channels};
    test::Array2D<uint8_t> y_dst{20, 10};
    test::Array2D<uint8_t> uv_dst{20, 5};

    test::test_null_args(impl, src.data(), src.stride(), y_dst.data(),
                         y_dst.stride(), uv_dst.data(), uv_dst.stride(),
                         src.width(), src.height(), color_format);

    EXPECT_EQ(KLEIDICV_OK,
              impl(src.data(), src.stride(), y_dst.data(), y_dst.stride(),
                   uv_dst.data(), uv_dst.stride(), 0, 1, color_format));

    EXPECT_EQ(KLEIDICV_OK,
              impl(src.data(), src.stride(), y_dst.data(), y_dst.stride(),
                   uv_dst.data(), uv_dst.stride(), 1, 0, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), y_dst.data(), y_dst.stride(),
                   uv_dst.data(), uv_dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), y_dst.data(), y_dst.stride(),
                   uv_dst.data(), uv_dst.stride(), KLEIDICV_MAX_IMAGE_PIXELS,
                   KLEIDICV_MAX_IMAGE_PIXELS, color_format));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), y_dst.data(), y_dst.stride(),
                   uv_dst.data(), uv_dst.stride(), src.width(), src.height(),
                   kleidicv_color_conversion_t{}));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), y_dst.data(), y_dst.stride(),
                   uv_dst.data(), uv_dst.stride(), src.width(), src.height(),
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST)));
  }

 private:
  // Coefficients for RGB to YUV420p conversion
  static const int kWeightScale = 20;
  static const int kRYWeight =
      269484;  // 0.299055 * (236-16)/256 * (1 << kWeightScale)
  static const int kGYWeight =
      528482;  // 0.586472 * (236-16)/256 * (1 << kWeightScale)
  static const int kBYWeight =
      102760;  // 0.114035 * (236-16)/256 * (1 << kWeightScale)
  static const int kRUWeight = -155188;  // -0.148 * (1 << (kWeightScale-1))
  static const int kGUWeight = -305135;  // -0.291 * (1 << (kWeightScale-1))
  static const int kBUWeight = 460324;   //  0.439 * (1 << (kWeightScale-1))
  static const int kGVWeight = -385875;  // -0.368 * (1 << (kWeightScale-1))
  static const int kBVWeight = -74448;   // -0.071 * (1 << (kWeightScale-1))
  static uint8_t saturate_cast_s32_to_u8(int32_t rhs) {
    return static_cast<uint8_t>(
        std::min(std::max(0, rhs),
                 static_cast<int32_t>(std::numeric_limits<uint8_t>::max())));
  }
  uint8_t rgb_to_y(uint8_t r, uint8_t g, uint8_t b) {
    const int kShifted16 = (16 << kWeightScale);
    const int kHalfShift = (1 << (kWeightScale - 1));
    int yy =
        kRYWeight * r + kGYWeight * g + kBYWeight * b + kHalfShift + kShifted16;

    return std::clamp(yy >> kWeightScale, 0, 0xff);
  }

  static void rgb_to_uv(uint8_t r, uint8_t g, uint8_t b, uint8_t& u,
                        uint8_t& v) {
    const int kHalfShift = (1 << (kWeightScale - 1));
    const int kShifted128 = (128 << kWeightScale);
    int uu = kRUWeight * r + kGUWeight * g + kBUWeight * b + kHalfShift +
             kShifted128;
    int vv = kBUWeight * r + kGVWeight * g + kBVWeight * b + kHalfShift +
             kShifted128;

    u = std::clamp(uu >> kWeightScale, 0, 0xff);
    v = std::clamp(vv >> kWeightScale, 0, 0xff);
  }
  void calculate_reference(const uint8_t* src, size_t src_stride,
                           uint8_t* y_dst, size_t y_stride, uint8_t* uv_dst,
                           size_t uv_stride, size_t width, size_t height,
                           bool is_nv21, bool RGB, size_t channels) {
    const uint8_t* src_row = nullptr;
    uint8_t* y_row = nullptr;
    uint8_t* u_row = nullptr;

    for (size_t h = 0; h < height; h++) {
      src_row = src + src_stride * h;
      y_row = y_dst + y_stride * h;

      bool evenRow = (h % 2) == 0;
      if (evenRow) {
        u_row = uv_dst + uv_stride * (h / 2);
      }

      for (size_t w = 0; w < width; w++) {
        uint8_t b0{}, g0{}, r0{};
        b0 = src_row[w * channels + 0];
        g0 = src_row[w * channels + 1];
        r0 = src_row[w * channels + 2];
        if (RGB) {
          std::swap(b0, r0);
        }
        uint8_t y0 = rgb_to_y(r0, g0, b0);
        y_row[w] = y0;
        bool evenCol = (w % 2) == 0;
        if (evenRow && evenCol) {
          uint8_t uu{}, vv{};
          rgb_to_uv(r0, g0, b0, uu, vv);
          if (is_nv21) {
            std::swap(uu, vv);
          }
          u_row[w + 0] = uu;
          u_row[w + 1] = vv;
        }
      }
    }
  }
};

TEST_F(RGB2YUV420SpTest, ConvertspaddedInputsWithAllParamCombinations) {
  for (const auto& params : get_test_cases()) {
    run_test_case(params);
  }
}

TEST_F(RGB2YUV420SpTest, ReturnsErrorForUnsupportedCombinations) {
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_BGR_TO_NV21);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_RGB_TO_NV21);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_BGRA_TO_NV21);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_RGBA_TO_NV21);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_BGR_TO_NV12);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_RGB_TO_NV12);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_BGRA_TO_NV12);
  run_unsupported(kleidicv_rgb_to_yuv_semiplanar_u8, KLEIDICV_RGBA_TO_NV12);
}
