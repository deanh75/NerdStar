// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"
#include "test_config.h"

class YUV422i2RGBTest : public testing::Test {
 public:
  struct Yuv422Layout {
    const char* name;
    int uIdx;  // 0: U next to Y, 1: V next to Y (controls YUYV/YVYU)
    int ycn;   // 0: Y starts at index 0 (YUYV/YVYU), 1: UYVY
  };
  struct TestParams {
    size_t channels;  // 3 or 4
    bool is_bgr;      // true = BGR/BGRA, false = RGB/RGBA
    Yuv422Layout layout;
  };

  static std::vector<TestParams> get_test_cases() {
    std::vector<size_t> channels = {3, 4};
    std::vector<bool> bgr_cases = {false, true};

    std::vector<TestParams> cases;
    for (auto c : channels) {
      for (auto bgr : bgr_cases) {
        for (const auto& L : kLayouts) {
          cases.push_back({c, bgr, L});
        }
      }
    }

    return cases;
  }

  void run_case(const TestParams& p) {
    // YUV422 interleaved = 2 bytes per pixel => row bytes = width * 2 + pad
    size_t width = 130;
    size_t height = 8;
    size_t src_padding = 0;
    size_t dst_padding = 0;

    test::Array2D<uint8_t> src{width * 2, height, src_padding};
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

    test::Array2D<uint8_t> expected{width * p.channels, height, dst_padding,
                                    p.channels};
    test::Array2D<uint8_t> dst{width * p.channels, height, dst_padding,
                               p.channels};

    yuv422_interleaved_ref(
        src.data(), src.stride(), expected.data(), expected.stride(),
        static_cast<int>(width), static_cast<int>(height), p.is_bgr,
        static_cast<int>(p.channels), p.layout.uIdx, p.layout.ycn);

    kleidicv_color_conversion_t fmt = make_color_conversion_type(
        p.channels, p.is_bgr, p.layout.uIdx, p.layout.ycn,
        KLEIDICV_COLOR_CONVERSION_FMT_YUV422);

    auto err = kleidicv_yuv_to_rgb_u8(src.data(), src.stride(), dst.data(),
                                      dst.stride(), static_cast<int>(width),
                                      static_cast<int>(height), fmt);

    ASSERT_EQ(KLEIDICV_OK, err);
    EXPECT_EQ_ARRAY2D(expected, dst);
  }

  template <typename F>
  void run_unsupported(F impl) {
    test::Array2D<uint8_t> src{40, 10, test::Options::vector_length()};
    test::Array2D<uint8_t> dst{60, 10, 0, 3};
    test::test_null_args(impl, src.data(), src.stride(), dst.data(),
                         dst.stride(), dst.width() / 3, dst.height(),
                         KLEIDICV_YVYU_TO_RGBA);

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, KLEIDICV_YVYU_TO_RGBA));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS,
                   KLEIDICV_YVYU_TO_RGBA));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 1,
                   dst.height(), KLEIDICV_YVYU_TO_RGBA));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 3,
                   dst.height(), KLEIDICV_YVYU_TO_RGBA));

    EXPECT_EQ(
        KLEIDICV_ERROR_NOT_IMPLEMENTED,
        impl(src.data(), src.stride(), dst.data(), dst.stride(),
             dst.width() / 3, dst.height(), kleidicv_color_conversion_t{}));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   dst.width() / 3, dst.height(),
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST |
                       KLEIDICV_COLOR_CONVERSION_FLAG_VU)));
  }

 private:
  static constexpr Yuv422Layout kLayouts[] = {
      {"YUYV", /*uIdx=*/0, /*ycn=*/0},
      {"UYVY", /*uIdx=*/0, /*ycn=*/1},
      {"YVYU", /*uIdx=*/1, /*ycn=*/0},
  };
  static constexpr int32_t KY = 1220542;
  static constexpr int32_t KRV = 1673527;
  static constexpr int32_t KGU = -409993;
  static constexpr int32_t KGV = -852492;
  static constexpr int32_t KBU = 2116026;
  static constexpr int KSHIFT = 20;

  static inline uint8_t sat_cast_s32_to_u8(int32_t x) {
    return static_cast<uint8_t>(std::min<int32_t>(
        std::max<int32_t>(0, x), std::numeric_limits<uint8_t>::max()));
  }

  // Minimal, scalar reference that matches your ref harness math and layout
  // (kept local to the test). It writes RGB/BGR or RGBA/BGRA depending on args.
  static void yuv422_interleaved_ref(const uint8_t* src, size_t src_stride,
                                     uint8_t* dst, size_t dst_stride, int width,
                                     int height, bool bgr, int dcn, int uIdx,
                                     int ycn) {
    // Byte offsets for the chroma samples inside the 4-byte YUV422 tuple
    // (Y0 U Y1 V). uidx stays within {0, 1, 3}, while vidx lands on the
    // companion chroma slot {1, 2, 3}, mirroring the production kernels.
    const int uidx = chroma_u_offset(ycn, uIdx);
    const int vidx = chroma_v_offset(uidx);
    const int y0idx = ycn;  // 0 if Y*.., 1 if U first (UYVY)
    const int y1idx = y0idx + 2;

    for (int y = 0; y < height; ++y) {
      const uint8_t* srow = src + y * src_stride;
      uint8_t* drow = dst + y * dst_stride;

      for (int x = 0; x < width; x += 2) {
        const uint8_t* p = srow + static_cast<ptrdiff_t>(x * 2);

        const int32_t U = int32_t(p[uidx]) - 128;
        const int32_t V = int32_t(p[vidx]) - 128;

        const int32_t addR = (1 << (KSHIFT - 1)) + KRV * V;
        const int32_t addG = (1 << (KSHIFT - 1)) + KGU * U + KGV * V;
        const int32_t addB = (1 << (KSHIFT - 1)) + KBU * U;

        auto emit = [&](int ybyte, uint8_t* out) {
          int32_t Y = std::max(0, int32_t(ybyte) - 16);
          int32_t r = ((KY * Y) + addR) >> KSHIFT;
          int32_t g = ((KY * Y) + addG) >> KSHIFT;
          int32_t b = ((KY * Y) + addB) >> KSHIFT;
          uint8_t R = sat_cast_s32_to_u8(r);
          uint8_t G = sat_cast_s32_to_u8(g);
          uint8_t B = sat_cast_s32_to_u8(b);
          if (bgr) {
            std::swap(R, B);
          }
          out[0] = R;
          out[1] = G;
          out[2] = B;
          if (dcn == 4) {
            out[3] = 0xFF;
          }
        };

        const uint8_t Y0 = p[y0idx];
        const uint8_t Y1 = p[y1idx];

        emit(Y0, drow + static_cast<ptrdiff_t>(x * dcn));
        emit(Y1, drow + static_cast<ptrdiff_t>((x + 1) * dcn));
      }
    }
  }

  // Map (channels, bgr?, layout) to kleidicv_color_conversion_t values you
  // used.
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

  static constexpr int chroma_u_offset(int y_component_idx,
                                       int chroma_selector) {
    return 1 - y_component_idx + chroma_selector * 2;
  }

  static constexpr int chroma_v_offset(int chroma_u_idx) {
    return (2 + chroma_u_idx) % 4;
  }
};

TEST_F(YUV422i2RGBTest, ConvertsPaddedInputs_AllParamCombinations) {
  for (const auto& params : get_test_cases()) {
    run_case(params);
  }
}

TEST_F(YUV422i2RGBTest, ReturnsErrorForUnsupportedCombinations) {
  run_unsupported(kleidicv_yuv_to_rgb_u8);
}
