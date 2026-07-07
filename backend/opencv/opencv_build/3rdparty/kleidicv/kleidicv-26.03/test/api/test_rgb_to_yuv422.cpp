// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"
#include "test_config.h"

class RGB8toYUV422Ref {
 public:
  RGB8toYUV422Ref() = default;

  void operator()(const uint8_t* src_data, size_t src_step, uint8_t* dst_data,
                  size_t dst_step, size_t width, size_t height,
                  size_t src_channels, bool is_bgr, bool is_vu, bool is_uyvy) {
    const size_t bIdx = is_bgr ? 0 : 2;
    const size_t rIdx = 2 - bIdx;

    for (size_t h = 0; h < height;
         ++h, src_data += src_step, dst_data += dst_step) {
      for (size_t w = 0; w < width; w += 2) {
        size_t src_w = w * src_channels;
        const uint8_t r1 = src_data[src_w + rIdx], g1 = src_data[src_w + 1],
                      b1 = src_data[src_w + bIdx];
        const uint8_t r2 = src_data[src_w + src_channels + rIdx],
                      g2 = src_data[src_w + src_channels + 1],
                      b2 = src_data[src_w + src_channels + bIdx];

        size_t yidx{0}, uidx{0}, vidx{0};
        if (is_uyvy) {
          yidx = 1;
          uidx = 0;
          vidx = 2;
        } else {
          yidx = 0;
          if (is_vu) {
            uidx = 3;
            vidx = 1;
          } else {
            uidx = 1;
            vidx = 3;
          }
        }

        size_t dst_w = w * 2;  // Storing 4 elements to 2 input pixels
        RGB82Yuv422(r1, g1, b1, r2, g2, b2, &dst_data[dst_w], yidx, uidx, vidx);
      }
    }
  }

 private:
  static constexpr int RGB2YUV422_SHIFT = 14;
  static constexpr int R2Y422 = 4211;   // 0.299077 * (236 - 16) / 256 * 16384
  static constexpr int G2Y422 = 8258;   // 0.586506 * (236 - 16) / 256 * 16384
  static constexpr int B2Y422 = 1606;   // 0.114062 * (236 - 16) / 256 * 16384
  static constexpr int R2U422 = -1212;  // -0.148 * 8192
  static constexpr int G2U422 = -2384;  // -0.291 * 8192
  static constexpr int B2U422 = 3596;   //  0.439 * 8192
  static constexpr int G2V422 = -3015;  // -0.368 * 8192
  static constexpr int B2V422 = -582;   // -0.071 * 8192

  static uint8_t sat_cast_s32_to_u8(int32_t x) {
    return static_cast<uint8_t>(std::min<int32_t>(
        std::max<int32_t>(0, x), std::numeric_limits<uint8_t>::max()));
  }

  static void RGB2Y(const uint8_t r, const uint8_t g, const uint8_t b,
                    uint8_t& y) {
    int y_ =
        r * R2Y422 + g * G2Y422 + b * B2Y422 + (1 << RGB2YUV422_SHIFT) * 16;
    y = sat_cast_s32_to_u8(((1 << (RGB2YUV422_SHIFT - 1)) + y_) >>
                           RGB2YUV422_SHIFT);
  }

  static void RGB2UV(const uint8_t r1, const uint8_t g1, const uint8_t b1,
                     const uint8_t r2, const uint8_t g2, const uint8_t b2,
                     uint8_t& u, uint8_t& v) {
    int sr = r1 + r2, sg = g1 + g2, sb = b1 + b2;

    int u_ = sr * R2U422 + sg * G2U422 + sb * B2U422 +
             (1 << (RGB2YUV422_SHIFT - 1)) * 256;
    u = sat_cast_s32_to_u8(((1 << (RGB2YUV422_SHIFT - 1)) + u_) >>
                           RGB2YUV422_SHIFT);

    int v_ = sr * B2U422 + sg * G2V422 + sb * B2V422 +
             (1 << (RGB2YUV422_SHIFT - 1)) * 256;
    v = sat_cast_s32_to_u8(((1 << (RGB2YUV422_SHIFT - 1)) + v_) >>
                           RGB2YUV422_SHIFT);
  }

  static void RGB82Yuv422(const uint8_t r1, const uint8_t g1, const uint8_t b1,
                          const uint8_t r2, const uint8_t g2, const uint8_t b2,
                          uint8_t* dst_pixel, size_t yidx, size_t uidx,
                          size_t vidx) {
    uint8_t &u = dst_pixel[uidx], &v = dst_pixel[vidx], &y1 = dst_pixel[yidx],
            &y2 = dst_pixel[yidx + 2];

    RGB2Y(r1, g1, b1, y1);
    RGB2Y(r2, g2, b2, y2);

    RGB2UV(r1, g1, b1, r2, g2, b2, u, v);
  }
};

// --- END: tiny helpers ------------------------------------------------------

class RGB2YUV422iTest : public testing::Test {
 public:
  struct TestParams {
    size_t src_channels;  // 3 or 4 (RGB/BGR or RGBA/BGRA)
    bool is_bgr;          // true = BGR/BGRA input, false = RGB/RGBA input
    std::string yuv_name;
    bool is_vu;
    bool is_uyvy;
  };

  static std::vector<TestParams> get_test_cases() {
    std::vector<size_t> src_channels = {3, 4};
    std::vector<bool> bgr_cases = {false, true};  // RGB*, then BGR*
    std::vector<std::tuple<std::string, bool, bool>> yuv_layouts = {
        {"YUYV", /*is_vu=*/false, /*is_uyvy=*/false},
        {"UYVY", /*is_vu=*/false, /*is_uyvy=*/true},
        {"YVYU", /*is_vu=*/true, /*is_uyvy=*/false},
    };  // RGB*, then BGR*

    std::vector<TestParams> cases;
    for (auto c : src_channels) {
      for (auto bgr : bgr_cases) {
        for (const auto& [yuv_name, is_vu, is_uyvy] : yuv_layouts) {
          cases.push_back({c, bgr, yuv_name, is_vu, is_uyvy});
        }
      }
    }

    return cases;
  }

  void run_case(const TestParams& p) {
    // src: scn bytes per pixel
    size_t width = 130;
    size_t height = 8;
    size_t src_padding = 0;
    size_t dst_padding = 0;
    test::Array2D<uint8_t> src{width * p.src_channels, height, src_padding,
                               p.src_channels};
    src.fill(0);
    src.set(0, 0,
            {0, 0,   0, 255, 255, 255, 255, 255, 128, 128, 128, 255, 255, 0,
             0, 255, 0, 255, 0,   255, 0,   0,   255, 255, 255, 255, 0,   255});
    src.set(1, 0,
            {
                0,   255, 255, 255, 255, 0,   255, 255, 16,  128,
                128, 255, 235, 128, 128, 255, 16,  255, 128, 255,
                16,  0,   128, 255, 16,  128, 255, 255,
            });
    src.set(2, 0, {2, 3, 240, 228});
    src.set(4, 0, {7, 11, 128, 129});

    // dst: 2 bytes per pixel (YUV422 interleaved)
    test::Array2D<uint8_t> expected{width * 2, height, dst_padding};
    test::Array2D<uint8_t> dst{width * 2, height, dst_padding};

    // Reference
    RGB8toYUV422Ref()(src.data(), src.stride(), expected.data(),
                      expected.stride(), width, height, p.src_channels,
                      p.is_bgr, p.is_vu, p.is_uyvy);

    // DUT
    const auto fmt =
        make_color_conversion_type(p.src_channels, p.is_bgr, p.is_vu, p.is_uyvy,
                                   KLEIDICV_COLOR_CONVERSION_FMT_YUV422);

    auto err = kleidicv_rgb_to_yuv_u8(src.data(), src.stride(), dst.data(),
                                      dst.stride(), width, height, fmt);

    ASSERT_EQ(KLEIDICV_OK, err) << "format=" << fmt << " layout=" << p.yuv_name
                                << " scn=" << p.src_channels
                                << " is_bgr=" << (p.is_bgr ? "BGR" : "RGB");
    EXPECT_EQ_ARRAY2D(expected, dst);
  }

  template <typename F>
  void run_unsupported(F impl) {
    // Build some small scratch buffers.
    test::Array2D<uint8_t> src_rgb{60, 10, test::Options::vector_length(), 3};
    test::Array2D<uint8_t> dst_yuv{40, 10, 0};  // 2Bpp * 20px = 40 bytes/row
    test::PseudoRandomNumberGenerator<uint8_t> prng;
    src_rgb.fill(prng);

    // Null args
    test::test_null_args(impl, src_rgb.data(), src_rgb.stride(), dst_yuv.data(),
                         dst_yuv.stride(), src_rgb.width() / 3,
                         src_rgb.height(), KLEIDICV_RGB_TO_YUYV);

    // Range checks
    EXPECT_EQ(
        KLEIDICV_ERROR_RANGE,
        impl(src_rgb.data(), src_rgb.stride(), dst_yuv.data(), dst_yuv.stride(),
             KLEIDICV_MAX_IMAGE_PIXELS + 2, 1, KLEIDICV_RGB_TO_YUYV));
    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src_rgb.data(), src_rgb.stride(), dst_yuv.data(),
                   dst_yuv.stride(), KLEIDICV_MAX_IMAGE_PIXELS,
                   KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_RGB_TO_YUYV));

    EXPECT_EQ(
        KLEIDICV_ERROR_NOT_IMPLEMENTED,
        impl(src_rgb.data(), src_rgb.stride(), dst_yuv.data(), dst_yuv.stride(),
             /*width=*/1, src_rgb.height(), KLEIDICV_RGB_TO_YUYV));
    EXPECT_EQ(
        KLEIDICV_ERROR_NOT_IMPLEMENTED,
        impl(src_rgb.data(), src_rgb.stride(), dst_yuv.data(), dst_yuv.stride(),
             /*width=*/3, src_rgb.height(), KLEIDICV_RGB_TO_YUYV));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src_rgb.data(), src_rgb.stride(), dst_yuv.data(),
                   dst_yuv.stride(), src_rgb.width() / 3, src_rgb.height(),
                   kleidicv_color_conversion_t{}));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src_rgb.data(), src_rgb.stride(), dst_yuv.data(),
                   dst_yuv.stride(), dst_yuv.width(), dst_yuv.height(),
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST |
                       KLEIDICV_COLOR_CONVERSION_FLAG_VU)));
  }

 private:
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
};

TEST_F(RGB2YUV422iTest, ReturnsErrorForUnsupportedCombinations) {
  run_unsupported(kleidicv_rgb_to_yuv_u8);
}

TEST_F(RGB2YUV422iTest, ConvertsPaddedInputs_AllParamCombinations) {
  for (const auto& params : get_test_cases()) {
    run_case(params);
  }
}
