// SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_OPERATION_H_
#define KLEIDICV_TEST_FRAMEWORK_OPERATION_H_

#include <array>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include "framework/array.h"
#include "framework/utils.h"

// Abstract base class for operations with InputsSize number of inputs and
// OutputsSize number of outputs.
template <typename InputType, size_t InputsSize, size_t OutputsSize,
          typename OutputType = InputType>
class OperationTest {
 private:
  // Function declaration for the sole reason to use its trailing return type
  // with a decltype()
  template <size_t... IOindices>
  static auto tuple_type_creator(std::index_sequence<IOindices...>)
      -> std::tuple<std::conditional_t<(IOindices < InputsSize), InputType,
                                       OutputType>...>;

 public:
  // Elements is a tuple with `InputsSize` number of `InputType` and
  // `OutputsSize` number of `OutputType`
  //
  // For example if `InputsSize == 3` and `OutputsSize == 2` the resulted tuple
  // is: std::tuple<InputType, InputType, InputType, OutputType, OutputType>
  using Elements = decltype(tuple_type_creator(
      std::make_index_sequence<InputsSize + OutputsSize>{}));

  using InputArrayType = test::Array2D<InputType>;
  using OutputArrayType = test::Array2D<OutputType>;

  OperationTest() {
    inputs_padding_.fill(0);
    outputs_padding_.fill(0);
  }

  virtual ~OperationTest() = default;

  // Sets the number of padding bytes at the end of rows, same for all arrays
  OperationTest<InputType, InputsSize, OutputsSize, OutputType>& with_padding(
      size_t padding) {
    for (auto& in_padding : inputs_padding_) {
      in_padding = padding;
    }

    for (auto& out_padding : outputs_padding_) {
      out_padding = padding;
    }

    return *this;
  }

  // Sets the number of padding bytes at the end of rows.
  OperationTest<InputType, InputsSize, OutputsSize, OutputType>& with_paddings(
      std::initializer_list<size_t> inputs_padding,
      std::initializer_list<size_t> outputs_padding) {
    size_t i = 0;
    for (size_t p : inputs_padding) {
      inputs_padding_[i++] = p;
    }
    size_t j = 0;
    for (size_t q : outputs_padding) {
      outputs_padding_[j++] = q;
    }
    return *this;
  }

  // Sets the number of elements in a row.
  OperationTest<InputType, InputsSize, OutputsSize, OutputType>& with_width(
      size_t width) {
    width_ = width;
    return *this;
  }

  void test() {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      inputs_[i] = InputArrayType{width(), height(), inputs_padding_[i]};
      inputs_[i].fill(0);
      ASSERT_TRUE(inputs_[i].valid());
    }

    for (auto& expected : expected_) {
      expected = OutputArrayType{width(), height()};
      expected.fill(0);
      ASSERT_TRUE(expected.valid());
    }

    for (size_t i = 0; i < actual_.size(); ++i) {
      actual_[i] = OutputArrayType{width(), height(), outputs_padding_[i]};
      actual_[i].fill(42);  // fill with any value different than `expected`
      ASSERT_TRUE(actual_[i].valid());
    }

    setup();
    check(call_api());
  }

 protected:
  // Returns test data.
  virtual const std::vector<Elements>& test_elements() = 0;

  // Prepares inputs and expected outputs for the operation.
  virtual void setup() {
    auto elements_list = test_elements();
    // Check that the number of elements fit into the buffers.
    ASSERT_LE(elements_list.size(), height());

    size_t row_index = 0;
    for (auto elements : elements_list) {
      // Fill elements one vector length apart.
      for (size_t column_index = 0; column_index < width();
           column_index += test::Options::vector_lanes<InputType>()) {
        store_test_elements(
            row_index, column_index, elements,
            std::make_index_sequence<InputsSize + OutputsSize>{});
      }

      // Increment loop counter.
      ++row_index;
    }
  }

  template <size_t ElementIndex>
  void store_one_test_element(const size_t row_index, const size_t column_index,
                              const Elements& elements) {
    const auto& value = std::get<ElementIndex>(elements);
    if constexpr (ElementIndex < InputsSize) {
      inputs_[ElementIndex].set(row_index, column_index, {value});
    } else {
      expected_[ElementIndex - InputsSize].set(row_index, column_index,
                                               {value});
    }
  }

  template <size_t... ElementIndices>
  void store_test_elements(const size_t row_index, const size_t column_index,
                           const Elements& elements,
                           std::index_sequence<ElementIndices...>) {
    (store_one_test_element<ElementIndices>(row_index, column_index, elements),
     ...);
  }

  // Calls the API-under-test in the appropriate way.
  virtual kleidicv_error_t call_api() = 0;

  // Checks that the result meets the expectations.
  virtual void check(kleidicv_error_t err) {
    EXPECT_EQ(KLEIDICV_OK, err);
    for (size_t index = 0; index < expected_.size(); ++index) {
      EXPECT_EQ_ARRAY2D(expected_[index], actual_[index]);
    }
  }

  // Tested number of rows.
  virtual size_t height() { return test_elements().size(); }

  // Tested number of elements in a row.
  virtual size_t width() const { return width_; }

  // Returns the minimum value for integral types, or the minimum normalized
  // positive value for floating point types.
  static constexpr OutputType
  min_if_integral_else_smallest_positive_normalized() {
    return std::numeric_limits<OutputType>::min();
  }

  // Returns the lowest value for OutputType.
  static constexpr OutputType lowest() {
    return std::numeric_limits<OutputType>::lowest();
  }

  // Returns the maximum value for OutputType.
  static constexpr OutputType max() {
    return std::numeric_limits<OutputType>::max();
  }

  // Input operand(s) for the operation.
  std::array<InputArrayType, InputsSize> inputs_;
  // Expected result of the operation.
  std::array<OutputArrayType, OutputsSize> expected_;
  // Actual result of the operation.
  std::array<OutputArrayType, OutputsSize> actual_;
  // Number of padding bytes at the end of rows.
  std::array<size_t, InputsSize> inputs_padding_;
  std::array<size_t, OutputsSize> outputs_padding_;
  // Tested number of elements in a row.
  // Sufficient number of elements to exercise both vector and scalar paths.
  size_t width_{3 * test::Options::vector_lanes<InputType>() - 1};
};  // end of class OperationTest<InputType, InputsSize, OutputsSize,
    // OutputType>

template <typename ElementType>
class BinaryOperationTest : public OperationTest<ElementType, 2, 1> {
};  // end of class BinaryOperationTest<ElementType>

template <typename ElementType>
class UnaryOperationTest : public OperationTest<ElementType, 1, 1> {
};  // end of class UnaryOperationTest<ElementType>

#endif  // KLEIDICV_TEST_FRAMEWORK_OPERATION_H_
