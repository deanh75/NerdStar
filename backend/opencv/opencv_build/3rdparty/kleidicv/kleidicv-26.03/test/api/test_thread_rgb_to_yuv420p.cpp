// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

class RgbToYuv420Thread : public testing::TestWithParam<P> {
 public:
  static inline int get_scn(kleidicv_color_conversion_t color) {
    return (color & KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA) ? 4 : 3;
  }
  template <typename SingleThreadedFunc, typename MultithreadedFunc>
  void check(SingleThreadedFunc single_threaded_func,
             MultithreadedFunc multithreaded_func,
             kleidicv_color_conversion_t color_format) {
    unsigned width = 0, height = 0, thread_count = 0;
    std::tie(width, height, thread_count) = GetParam();
    size_t channels = get_scn(color_format);
    test::Array2D<uint8_t> src(size_t{width} * channels, height, channels),
        dst_single(width, (height * 3 + 1) / 2),
        dst_multi(width, (height * 3 + 1) / 2);

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
    size_t channels = get_scn(color_format);
    test::Array2D<uint8_t> src{20 * channels, 10, 0, channels};
    test::Array2D<uint8_t> dst(20, (10 * 3 + 1) / 2);

    EXPECT_EQ(KLEIDICV_OK,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 0, 1,
                   color_format, get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_OK,
              impl(src.data(), src.stride(), dst.data(), dst.stride(), 1, 0,
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
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   dst.width(), dst.height(), kleidicv_color_conversion_t{},
                   get_multithreading_fake(2)));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              impl(src.data(), src.stride(), dst.data(), dst.stride(),
                   dst.width(), dst.height(),
                   static_cast<kleidicv_color_conversion_t>(
                       KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                       KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST),
                   get_multithreading_fake(2)));
  }
};

TEST_P(RgbToYuv420Thread, BGR_TO_YV12) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_BGR_TO_YV12);
}

TEST_P(RgbToYuv420Thread, RGB_TO_YV12) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_RGB_TO_YV12);
}

TEST_P(RgbToYuv420Thread, BGR_TO_IYUV) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_BGR_TO_IYUV);
}

TEST_P(RgbToYuv420Thread, RGB_TO_IYUV) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_RGB_TO_IYUV);
}

TEST_P(RgbToYuv420Thread, BGRA_TO_YV12) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_BGRA_TO_YV12);
}

TEST_P(RgbToYuv420Thread, RGBA_TO_YV12) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_RGBA_TO_YV12);
}

TEST_P(RgbToYuv420Thread, BGRA_TO_IYUV) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_BGRA_TO_IYUV);
}

TEST_P(RgbToYuv420Thread, RGBA_TO_IYUV) {
  check(kleidicv_rgb_to_yuv_u8, kleidicv_thread_rgb_to_yuv_u8,
        KLEIDICV_RGBA_TO_IYUV);
}

TEST_F(RgbToYuv420Thread, ReturnsErrorForUnsupportedCombinations) {
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_RGB_TO_IYUV);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_BGR_TO_IYUV);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_RGBA_TO_IYUV);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_BGRA_TO_IYUV);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_RGB_TO_YV12);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_BGR_TO_YV12);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_RGBA_TO_YV12);
  run_unsupported(kleidicv_thread_rgb_to_yuv_u8, KLEIDICV_BGRA_TO_YV12);
}

INSTANTIATE_TEST_SUITE_P(, RgbToYuv420Thread,
                         testing::Values(P{2, 1, 1}, P{2, 2, 1}, P{2, 2, 2},
                                         P{2, 1, 2}, P{2, 2, 1}, P{2, 3, 2},
                                         P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2},
                                         P{2, 6, 3}, P{2, 7, 4}, P{12, 34, 5},
                                         P{12, 37, 5}, P{2, 1000, 2}));
