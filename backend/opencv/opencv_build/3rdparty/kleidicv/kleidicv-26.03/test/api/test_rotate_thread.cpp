// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

class RotateThread : public testing::TestWithParam<size_t> {
 public:
  void scalar_test(size_t padding) {
    size_t first_dim = test::Options::vector_lanes<uint8_t>() - 1;
    size_t second_dim = test::Options::vector_lanes<uint8_t>() + 1;
    // Exercise horizontal scalar path
    test(first_dim, second_dim, padding);
    test(second_dim, first_dim, padding);
  }

  void vector_test(size_t padding) {
    // Make the size at least two batches 2 * 64 = 128 for all element_size
    size_t src_width = 8 * test::Options::vector_lanes<uint8_t>();
    // Set height to be different from width but still larger than vector_lanes
    size_t src_height = 12 * test::Options::vector_lanes<uint8_t>();
    test(src_width, src_height, padding);
  }

  void vector_plus_scalar_test(size_t padding) {
    size_t first_dim = 12 * test::Options::vector_lanes<uint8_t>() - 1;
    size_t second_dim = 12 * test::Options::vector_lanes<uint8_t>() - 1;
    test(first_dim, second_dim, padding);
    test(second_dim, first_dim, padding);
  }

 protected:
  void test(size_t src_width, size_t src_height, size_t padding) const {
    const size_t dst_width = src_height;
    const size_t dst_height = src_width;
    unsigned thread_count = 2;
    size_t element_size = GetParam();
    size_t src_stride = (src_width + padding) * element_size;
    size_t dst_stride = (dst_width + padding) * element_size;

    std::vector<uint8_t> source(src_stride * src_height, 0);
    std::mt19937 generator{
        static_cast<std::mt19937::result_type>(test::Options::seed())};
    std::generate(source.begin(), source.end(), generator);

    for (int angle : {90, -90}) {
      std::vector<uint8_t> actual_single(dst_stride * dst_height, 0);
      std::vector<uint8_t> actual_multi(dst_stride * dst_height, 0);

      ASSERT_EQ(KLEIDICV_OK,
                kleidicv_rotate(source.data(), src_stride, src_width,
                                src_height, actual_single.data(), dst_stride,
                                angle, element_size));

      ASSERT_EQ(KLEIDICV_OK,
                kleidicv_thread_rotate(source.data(), src_stride, src_width,
                                       src_height, actual_multi.data(),
                                       dst_stride, angle, element_size,
                                       get_multithreading_fake(thread_count)));

      expect_eq_vector2D(actual_multi.data(), actual_single.data(), dst_width,
                         dst_height, dst_stride, element_size);
    }
  }

  void expect_eq_vector2D(const uint8_t *lhs, const uint8_t *rhs, size_t width,
                          size_t height, size_t stride,
                          size_t element_size) const {
    for (size_t i = 0; i < height; i++) {
      ASSERT_EQ(
          std::memcmp(lhs + i * stride, rhs + i * stride, width * element_size),
          0);
    }
  }
};

TEST_P(RotateThread, ScalarNoPadding) { scalar_test(0); }

TEST_P(RotateThread, VectorNoPadding) { vector_test(0); }

TEST_P(RotateThread, ScalarWithPadding) { scalar_test(1); }

TEST_P(RotateThread, VectorWithPadding) { vector_test(1); }

TEST_P(RotateThread, VectorPlusScalarNoPadding) { vector_plus_scalar_test(0); }

TEST_P(RotateThread, VectorPlusScalarWithPadding) {
  vector_plus_scalar_test(1);
}

INSTANTIATE_TEST_SUITE_P(, RotateThread, testing::Values(1, 2, 4, 8),
                         testing::PrintToStringParamName());

TEST(RotateThreadNotImplemented, InPlace) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 1;
  const size_t stride = width * element_size;
  unsigned thread_count = 2;

  uint8_t source[width * height] = {};
  for (int angle : {90, -90}) {
    ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_rotate(source, stride, width, height, source,
                                     stride, angle, element_size,
                                     get_multithreading_fake(thread_count)));
  }
}

TEST(RotateThreadNotImplemented, Angle) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 1;
  const size_t stride = width * element_size;
  const int angle = 180;
  unsigned thread_count = 2;

  uint8_t source[width * height] = {};
  uint8_t dst[width * height] = {};
  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_rotate(source, stride, width, height, dst, stride,
                                   angle, element_size,
                                   get_multithreading_fake(thread_count)));
}

TEST(RotateThreadNotImplemented, ElementSize) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 16;
  const size_t stride = width * element_size;
  const int angle = 90;
  unsigned thread_count = 2;

  std::vector<uint8_t> source(width * element_size * height, 0);
  std::vector<uint8_t> dst(width * element_size * height, 0);
  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_rotate(source.data(), stride, width, height,
                                   dst.data(), stride, angle, element_size,
                                   get_multithreading_fake(thread_count)));
}
