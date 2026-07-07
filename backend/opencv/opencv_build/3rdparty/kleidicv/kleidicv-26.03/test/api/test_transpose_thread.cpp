// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "multithreading_fake.h"

class TransposeThread : public testing::TestWithParam<size_t> {
 public:
  void scalar_test(size_t padding) {
    size_t first_dim = test::Options::vector_lanes<uint8_t>() - 1;
    size_t second_dim = test::Options::vector_lanes<uint8_t>() + 1;
    test(first_dim, second_dim, padding);
    test(second_dim, first_dim, padding);
  }

  void vector_test(size_t padding) {
    size_t src_width = 2 * test::Options::vector_lanes<uint8_t>();
    size_t src_height = 3 * test::Options::vector_lanes<uint8_t>();
    test(src_width, src_height, padding);
  }

  void vector_plus_scalar_test(size_t padding) {
    size_t first_dim = 3 * test::Options::vector_lanes<uint8_t>() - 1;
    size_t second_dim = 3 * test::Options::vector_lanes<uint8_t>() + 1;
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
    std::vector<uint8_t> actual_single(dst_stride * dst_height, 0);
    std::vector<uint8_t> actual_multi(dst_stride * dst_height, 0);

    std::mt19937 generator{
        static_cast<std::mt19937::result_type>(test::Options::seed())};
    std::generate(source.begin(), source.end(), generator);

    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_transpose(source.data(), src_stride, actual_single.data(),
                           dst_stride, src_width, src_height, element_size));

    ASSERT_EQ(KLEIDICV_OK, kleidicv_thread_transpose(
                               source.data(), src_stride, actual_multi.data(),
                               dst_stride, src_width, src_height, element_size,
                               get_multithreading_fake(thread_count)));

    expect_eq_vector2D(actual_multi.data(), actual_single.data(), dst_width,
                       dst_height, dst_stride, element_size);
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

TEST_P(TransposeThread, ScalarNoPadding) { scalar_test(0); }

TEST_P(TransposeThread, VectorNoPadding) { vector_test(0); }

TEST_P(TransposeThread, ScalarWithPadding) { scalar_test(1); }

TEST_P(TransposeThread, VectorWithPadding) { vector_test(1); }

TEST_P(TransposeThread, VectorPlusScalarNoPadding) {
  vector_plus_scalar_test(0);
}

TEST_P(TransposeThread, VectorPlusScalarWithPadding) {
  vector_plus_scalar_test(1);
}

INSTANTIATE_TEST_SUITE_P(, TransposeThread, testing::Values(1, 2, 4, 8),
                         testing::PrintToStringParamName());

TEST(TransposeThreadNotImplemented, ElementSize) {
  const size_t width = 1;
  const size_t height = 1;
  const size_t element_size = 16;
  const size_t stride = width * element_size;
  unsigned thread_count = 2;

  std::vector<uint8_t> source(width * element_size * height, 0);
  std::vector<uint8_t> dst(width * element_size * height, 0);
  ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_transpose(source.data(), stride, dst.data(), stride,
                                      width, height, element_size,
                                      get_multithreading_fake(thread_count)));
}
