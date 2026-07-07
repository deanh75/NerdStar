// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_GENERATOR_H_
#define KLEIDICV_TEST_FRAMEWORK_GENERATOR_H_

#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include "framework/abstract.h"
#include "framework/utils.h"

template <typename T, typename... Ts>
inline constexpr bool is_any_of_v = (std::is_same_v<T, Ts> || ...);

// uniform_int_distribution does not support 8-bit data types on some platforms
// int64_t is used for not supported types
template <typename T>
struct TypeForRandom {
  using type = std::conditional_t<
      is_any_of_v<T, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>,
      T, int64_t>;
};

template <typename T>
using IntTypeForRandom = typename TypeForRandom<T>::type;

namespace test {

template <typename ElementType>
class GenerateLinearSeries : public Generator<ElementType> {
 public:
  explicit GenerateLinearSeries(ElementType start_from)
      : counter_{start_from} {}

  std::optional<ElementType> next() override { return counter_++; }

 private:
  ElementType counter_;
};  // end of class GenerateLinearSeries

// Generates pseudo-random numbers of a given type.
template <typename ElementType>
class PseudoRandomNumberGenerator : public Generator<ElementType> {
 public:
  PseudoRandomNumberGenerator() : seed_{Options::seed()}, rng_{seed_} {}

  // Resets the generator to the initial state.
  void reset() override { rng_.seed(seed_); }

  // Yields the next value or std::nullopt.
  std::optional<ElementType> next() override {
    if constexpr (std::is_floating_point_v<ElementType>) {
      // Return a random floating point value.
      // The value is from a uniform distribution of the *representable*
      // finite floating point values, not the range. This means that the
      // likelihood of producing a tiny value is about the same as the
      // likelihood of producing a huge value.

      for (;;) {
        ElementType result;
        const auto r = rng_();
        static_assert(sizeof(result) <= sizeof(r));
        memcpy(&result, &r, sizeof(result));
        if (std::isfinite(result)) {
          return result;
        }
      }
    } else {
      return static_cast<ElementType>(
          std::uniform_int_distribution<IntTypeForRandom<ElementType>>(
              std::numeric_limits<ElementType>::lowest(),
              std::numeric_limits<ElementType>::max())(rng_));
    }
  }

 protected:
  uint64_t seed_;
  std::mt19937_64 rng_;
};  // end of class PseudoRandomNumberGenerator<ElementType>

// Generates pseudo-random integers of a given type within the range [min, max].
template <typename ElementType,
          std::enable_if_t<std::is_integral_v<ElementType>, bool> = true>
class PseudoRandomNumberGeneratorIntRange
    : public PseudoRandomNumberGenerator<ElementType> {
 public:
  PseudoRandomNumberGeneratorIntRange(ElementType min, ElementType max)
      : PseudoRandomNumberGenerator<ElementType>(), dist_(min, max) {}

  // Yields the next value or std::nullopt.
  std::optional<ElementType> next() override {
    return static_cast<ElementType>(dist_(this->rng_));
  }

 protected:
  std::uniform_int_distribution<IntTypeForRandom<ElementType>> dist_;
};  // end of class PseudoRandomNumberGeneratorIntRange<ElementType>

template <typename ElementType,
          std::enable_if_t<std::is_floating_point_v<ElementType>, bool> = true>
class PseudoRandomNumberGeneratorFloatRange
    : public PseudoRandomNumberGenerator<ElementType> {
 public:
  PseudoRandomNumberGeneratorFloatRange(ElementType min, ElementType max)
      : PseudoRandomNumberGenerator<ElementType>(), dist_(min, max) {}

  // Yields the next value or std::nullopt.
  std::optional<ElementType> next() override {
    return static_cast<ElementType>(dist_(this->rng_));
  }

 protected:
  std::uniform_real_distribution<ElementType> dist_;
};  // end of class PseudoRandomNumberGeneratorFloatRange<ElementType>

// Generator which yields values of an iterable container.
template <typename IterableType>
class SequenceGenerator : public Generator<typename IterableType::value_type> {
 public:
  explicit SequenceGenerator(const IterableType &container)
      : begin_{container.begin()},
        current_{container.begin()},
        end_{container.end()} {}

  // Resets the generator to its initial state.
  void reset() override { current_ = begin_; }

  // Yields the next value or std::nullopt.
  std::optional<typename IterableType::value_type> next() override {
    if (current_ == end_) {
      return std::nullopt;
    }

    return *current_++;
  }

 protected:
  typename IterableType::const_iterator begin_;
  typename IterableType::const_iterator current_;
  typename IterableType::const_iterator end_;
};  // end of class SequenceGenerator<IterableType>

}  // namespace test

#endif  // KLEIDICV_TEST_FRAMEWORK_GENERATOR_H_
