// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "multithreading_fake.h"

// Tuple of width, height, thread count.
typedef std::tuple<unsigned, unsigned, unsigned> P;

class Yuv422Thread : public testing::TestWithParam<P> {
 public:
  static inline int get_dcn(kleidicv_color_conversion_t color) {
    return (color & KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA) ? 4 : 3;
  }

  template <typename SingleThreadedFunc, typename MultithreadedFunc>
  void check(SingleThreadedFunc single_threaded_func,
             MultithreadedFunc multithreaded_func,
             kleidicv_color_conversion_t color_format) {
    unsigned width = 0, height = 0, thread_count = 0;
    std::tie(width, height, thread_count) = GetParam();

    // YUV422 interleaved: 2 bytes per pixel (e.g., YUYV / UYVY / YVYU).
    test::Array2D<uint8_t> src(size_t{width} * 2, height);
    auto channel = get_dcn(color_format);
    test::Array2D<uint8_t> dst_single(size_t{width} * channel, height);
    test::Array2D<uint8_t> dst_multi(size_t{width} * channel, height);

    test::PseudoRandomNumberGenerator<uint8_t> generator;
    src.fill(generator);

    kleidicv_error_t single_result =
        single_threaded_func(src.data(), src.stride(), dst_single.data(),
                             dst_single.stride(), width, height, color_format);

    kleidicv_error_t multi_result = multithreaded_func(
        src.data(), src.stride(), dst_multi.data(), dst_multi.stride(), width,
        height, color_format, get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }

  template <typename Func>
  void run_unsupported(Func impl, kleidicv_color_conversion_t color_format) {
    // 20 pixels wide, 40 bytes per row in YUV422.
    test::Array2D<uint8_t> src{40, 10};
    size_t channel = get_dcn(color_format);
    test::Array2D<uint8_t> dst{60, 10, 0, channel};

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 1, 1,
                   color_format, get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 5, 1,
                   color_format, get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_OK,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 2, 0,
                   color_format, get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, color_format,
                   get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS,
                   color_format, get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 2, 2,
                   kleidicv_color_conversion_t{}, get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 2, 2,
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST |
                       KLEIDICV_COLOR_CONVERSION_FLAG_VU),
                   get_multithreading_fake(2)));
  }
};

TEST_P(Yuv422Thread, yuyv_to_bgr) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YUYV_TO_BGR);
}

TEST_P(Yuv422Thread, yuyv_to_rgb) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YUYV_TO_RGB);
}

TEST_P(Yuv422Thread, uyvy_to_rgb) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_UYVY_TO_RGB);
}

TEST_P(Yuv422Thread, yvyu_to_rgb) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YVYU_TO_RGB);
}

TEST_P(Yuv422Thread, yuyv_to_bgra) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YUYV_TO_BGRA);
}

TEST_P(Yuv422Thread, uyvy_to_bgra) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_UYVY_TO_BGRA);
}

TEST_P(Yuv422Thread, yvyu_to_bgra) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YVYU_TO_BGRA);
}

TEST_P(Yuv422Thread, yuyv_to_rgba) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YUYV_TO_RGBA);
}

TEST_P(Yuv422Thread, uyvy_to_rgba) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_UYVY_TO_RGBA);
}

TEST_P(Yuv422Thread, yvyu_to_rgba) {
  check(kleidicv_yuv_to_rgb_u8, kleidicv_thread_yuv_to_rgb_u8,
        KLEIDICV_YVYU_TO_RGBA);
}

TEST_F(Yuv422Thread, ReturnsErrorForUnsupportedCombinations) {
  run_unsupported(kleidicv_thread_yuv_to_rgb_u8, KLEIDICV_YUYV_TO_BGR);
}

INSTANTIATE_TEST_SUITE_P(, Yuv422Thread,
                         testing::Values(P{2, 1, 1}, P{2, 2, 1}, P{2, 2, 2},
                                         P{2, 1, 2}, P{2, 2, 1}, P{2, 3, 2},
                                         P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2},
                                         P{2, 6, 3}, P{2, 7, 4}, P{12, 34, 5},
                                         P{12, 37, 5}, P{16, 37, 5},
                                         P{2, 1000, 2}));
