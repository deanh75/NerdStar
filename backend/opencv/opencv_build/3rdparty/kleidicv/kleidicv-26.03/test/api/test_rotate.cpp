// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"

class Rotate : public testing::TestWithParam<size_t> {
 public:
  void scalar_test(size_t padding) {
    size_t first_dim = test::Options::vector_lanes<uint8_t>() - 1;
    size_t second_dim = test::Options::vector_lanes<uint8_t>() + 1;
    // Exercise horizontal scalar path
    test(first_dim, second_dim, padding);
    test(second_dim, first_dim, padding);
  }

  void vector_test(size_t padding) {
    // Make at least two full vector passes
    size_t src_width = 2 * test::Options::vector_lanes<uint8_t>();
    // Set height to be different from width but still larger than vector_lanes
    size_t src_height = 3 * test::Options::vector_lanes<uint8_t>();
    test(src_width, src_height, padding);
  }

  void vector_plus_scalar_test(size_t padding) {
    size_t first_dim = 3 * test::Options::vector_lanes<uint8_t>() - 1;
    size_t second_dim = 3 * test::Options::vector_lanes<uint8_t>() - 1;
    test(first_dim, second_dim, padding);
    test(second_dim, first_dim, padding);
  }

 protected:
  void test(size_t src_width, size_t src_height, size_t padding) const {
    const size_t dst_width = src_height;
    const size_t dst_height = src_width;
    size_t element_size = GetParam();
    size_t src_stride = (src_width + padding) * element_size;
    size_t dst_stride = (dst_width + padding) * element_size;

    std::vector<uint8_t> source(src_stride * src_height, 0);
    std::mt19937 generator{
        static_cast<std::mt19937::result_type>(test::Options::seed())};
    std::generate(source.begin(), source.end(), generator);

    for (int angle : {90, -90}) {
      std::vector<uint8_t> expected(dst_stride * dst_height, 0);
      std::vector<uint8_t> actual_single(dst_stride * dst_height, 0);

      calculate_expected(source.data(), expected.data(), src_width, src_height,
                         src_stride, dst_stride, element_size, angle);

      ASSERT_EQ(KLEIDICV_OK,
                kleidicv_rotate(source.data(), src_stride, src_width,
                                src_height, actual_single.data(), dst_stride,
                                angle, element_size));

      expect_eq_vector2D(expected.data(), actual_single.data(), dst_width,
                         dst_height, dst_stride, element_size);
    }
  }

  void expect_eq_vector2D(const uint8_t *lhs, const uint8_t *rhs, size_t width,
                          size_t height, size_t stride,
                          size_t element_size) const {
    for (size_t i = 0; i < height; i++) {
      for (size_t j = 0; j < width * element_size; j++) {
        ASSERT_EQ(lhs[i * stride + j], rhs[i * stride + j]);
      }
    }
  }

  void calculate_expected(const uint8_t *source, uint8_t *expected,
                          size_t src_width, size_t src_height,
                          size_t src_stride, size_t dst_stride,
                          size_t element_size, int angle) const {
    for (size_t i = 0; i < src_height; i++) {
      for (size_t j = 0; j < src_width; j++) {
        if (angle == 90) {
          // dst[j][src_height - i - 1] = src[i][j]
          memcpy(
              expected + j * dst_stride + (src_height - i - 1) * element_size,
              source + i * src_stride + j * element_size, element_size);
        } else {
          // dst[src_width - j - 1][i] = src[i][j]
          memcpy(expected + (src_width - j - 1) * dst_stride + i * element_size,
                 source + i * src_stride + j * element_size, element_size);
        }
      }
    }
  }
};

TEST_P(Rotate, ScalarNoPadding) { scalar_test(0); }

TEST_P(Rotate, VectorNoPadding) { vector_test(0); }

TEST_P(Rotate, ScalarWithPadding) { scalar_test(1); }

TEST_P(Rotate, VectorWithPadding) { vector_test(1); }

TEST_P(Rotate, VectorPlusScalarNoPadding) { vector_plus_scalar_test(0); }

TEST_P(Rotate, VectorPlusScalarWithPadding) { vector_plus_scalar_test(1); }

TEST_P(Rotate, NullPointer) {
  std::vector<uint8_t> src(1, 0);
  std::vector<uint8_t> dst(1, 0);
  size_t element_size = GetParam();
  for (int angle : {90, -90}) {
    test::test_null_args(kleidicv_rotate, src.data(), element_size, 1, 1,
                         dst.data(), element_size, angle, element_size);
  }
}

TEST_P(Rotate, ZeroImageSize) {
  std::vector<uint8_t> src(1, 0);
  std::vector<uint8_t> dst(1, 0);
  size_t element_size = GetParam();
  for (int angle : {90, -90}) {
    EXPECT_EQ(KLEIDICV_OK,
              kleidicv_rotate(src.data(), element_size, 0, 1, dst.data(),
                              element_size, angle, element_size));
    EXPECT_EQ(KLEIDICV_OK,
              kleidicv_rotate(src.data(), element_size, 1, 0, dst.data(),
                              element_size, angle, element_size));
  }
}

TEST_P(Rotate, OversizeImage) {
  std::vector<uint8_t> src(1, 0);
  std::vector<uint8_t> dst(1, 0);
  size_t element_size = GetParam();
  for (int angle : {90, -90}) {
    EXPECT_EQ(
        KLEIDICV_ERROR_RANGE,
        kleidicv_rotate(src.data(), element_size, KLEIDICV_MAX_IMAGE_PIXELS + 1,
                        1, dst.data(), element_size, angle, element_size));
    EXPECT_EQ(
        KLEIDICV_ERROR_RANGE,
        kleidicv_rotate(src.data(), element_size, KLEIDICV_MAX_IMAGE_PIXELS,
                        KLEIDICV_MAX_IMAGE_PIXELS, dst.data(), element_size,
                        angle, element_size));
  }
}

TEST_P(Rotate, Misalignment) {
  size_t element_size = GetParam();
  if (element_size == 1) {
    // misalignment impossible
    return;
  }

  const size_t kBufSize = element_size * 10;
  std::vector<uint8_t> src(kBufSize, 0);
  std::vector<uint8_t> dst(kBufSize, 0);

  for (int angle : {90, -90}) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              kleidicv_rotate(src.data() + 1, element_size, 1, 1, dst.data(),
                              element_size, angle, element_size));
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              kleidicv_rotate(src.data(), element_size + 1, 1, 2, dst.data(),
                              element_size, angle, element_size));
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              kleidicv_rotate(src.data(), element_size, 1, 1, dst.data() + 1,
                              element_size, angle, element_size));
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              kleidicv_rotate(src.data(), element_size, 2, 1, dst.data(),
                              element_size + 1, angle, element_size));
    // Ignore stride if there's only one row
    EXPECT_EQ(KLEIDICV_OK,
              kleidicv_rotate(src.data(), element_size + 1, 1, 1, dst.data(),
                              element_size, angle, element_size));
    EXPECT_EQ(KLEIDICV_OK,
              kleidicv_rotate(src.data(), element_size, 1, 1, dst.data(),
                              element_size + 1, angle, element_size));
  }
}

INSTANTIATE_TEST_SUITE_P(, Rotate, testing::Values(1, 2, 4, 8),
                         testing::PrintToStringParamName());

TEST(RotateNotImplemented, InPlace) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 1;
  const size_t stride = width * element_size;

  uint8_t source[width * height] = {};
  for (int angle : {90, -90}) {
    ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_rotate(source, stride, width, height, source, stride,
                              angle, element_size));
  }
}

TEST(RotateNotImplemented, Angle) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 1;
  const size_t stride = width * element_size;
  const int angle = 180;

  uint8_t source[width * height] = {};
  uint8_t dst[width * height] = {};
  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_rotate(source, stride, width, height, dst, stride, angle,
                            element_size));
}

TEST(RotateNotImplemented, ElementSize) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 16;
  const size_t stride = width * element_size;
  const int angle = 90;

  std::vector<uint8_t> source(width * element_size * height, 0);
  std::vector<uint8_t> dst(width * element_size * height, 0);
  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_rotate(source.data(), stride, width, height, dst.data(),
                            stride, angle, element_size));
}
