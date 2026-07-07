// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_BORDER_21X21_H
#define KLEIDICV_WORKSPACE_BORDER_21X21_H

#include "border_types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Border offsets for fixed-size filters.
template <typename T, const size_t S>
class FixedBorderInfo;

// Border offsets for 21x21 filters.
template <typename T>
class FixedBorderInfo<T, 21UL> final {
 public:
  // Simple object holding read-only constant offsets.
  class Offsets final {
   public:
    // NOLINTBEGIN(hicpp-member-init)
    Offsets() = default;
    // NOLINTEND(hicpp-member-init)

    Offsets(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t o3,
            ptrdiff_t o4, ptrdiff_t o5, ptrdiff_t o6, ptrdiff_t o7,
            ptrdiff_t o8, ptrdiff_t o9, ptrdiff_t o10, ptrdiff_t o11,
            ptrdiff_t o12, ptrdiff_t o13, ptrdiff_t o14, ptrdiff_t o15,
            ptrdiff_t o16, ptrdiff_t o17, ptrdiff_t o18, ptrdiff_t o19,
            ptrdiff_t o20)
        : offsets_{o0,  o1,  o2,  o3,  o4,  o5,  o6,  o7,  o8,  o9, o10,
                   o11, o12, o13, o14, o15, o16, o17, o18, o19, o20} {}

    ptrdiff_t c0() const { return offsets_[0]; }
    ptrdiff_t c1() const { return offsets_[1]; }
    ptrdiff_t c2() const { return offsets_[2]; }
    ptrdiff_t c3() const { return offsets_[3]; }
    ptrdiff_t c4() const { return offsets_[4]; }
    ptrdiff_t c5() const { return offsets_[5]; }
    ptrdiff_t c6() const { return offsets_[6]; }
    ptrdiff_t c7() const { return offsets_[7]; }
    ptrdiff_t c8() const { return offsets_[8]; }
    ptrdiff_t c9() const { return offsets_[9]; }
    ptrdiff_t c10() const { return offsets_[10]; }
    ptrdiff_t c11() const { return offsets_[11]; }
    ptrdiff_t c12() const { return offsets_[12]; }
    ptrdiff_t c13() const { return offsets_[13]; }
    ptrdiff_t c14() const { return offsets_[14]; }
    ptrdiff_t c15() const { return offsets_[15]; }
    ptrdiff_t c16() const { return offsets_[16]; }
    ptrdiff_t c17() const { return offsets_[17]; }
    ptrdiff_t c18() const { return offsets_[18]; }
    ptrdiff_t c19() const { return offsets_[19]; }
    ptrdiff_t c20() const { return offsets_[20]; }

   private:
    ptrdiff_t offsets_[21];
  };

  FixedBorderInfo(size_t width, FixedBorderType border_type)
      : width_(width), border_type_(border_type) {}

  // Returns offsets without the influence of any border.
  Offsets offsets_without_border() const KLEIDICV_STREAMING {
    return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
               8, 9, 10);
  }

  // NOLINTBEGIN(readability-function-cognitive-complexity)
  // Returns offsets for columns affected by left border.
  Offsets offsets_with_left_border(size_t column_index) const
      KLEIDICV_STREAMING {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
        if (column_index == 0) {
          return get(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10);
        } else if (column_index == 1) {
          return get(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 2) {
          return get(-2, -2, -2, -2, -2, -2, -2, -2, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 3) {
          return get(-3, -3, -3, -3, -3, -3, -3, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 4) {
          return get(-4, -4, -4, -4, -4, -4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 5) {
          return get(-5, -5, -5, -5, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 6) {
          return get(-6, -6, -6, -6, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 7) {
          return get(-7, -7, -7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 8) {
          return get(-8, -8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else {
          return get(-9, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        }
        break;

      case FixedBorderType::REFLECT:
        if (column_index == 0) {
          return get(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10);
        } else if (column_index == 1) {
          return get(7, 6, 5, 4, 3, 2, 1, 0, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                     9, 10);
        } else if (column_index == 2) {
          return get(5, 4, 3, 2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
                     8, 9, 10);
        } else if (column_index == 3) {
          return get(3, 2, 1, 0, -1, -2, -3, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
                     8, 9, 10);
        } else if (column_index == 4) {
          return get(1, 0, -1, -2, -3, -4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                     7, 8, 9, 10);
        } else if (column_index == 5) {
          return get(-1, -2, -3, -4, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 6) {
          return get(-3, -4, -5, -6, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 7) {
          return get(-5, -6, -7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 8) {
          return get(-7, -8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else {
          return get(-9, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        }
        break;

      case FixedBorderType::WRAP:
        if (column_index == 0) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7,
                     width_ - 6, width_ - 5, width_ - 4, width_ - 3, width_ - 2,
                     width_ - 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        } else if (column_index == 1) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7,
                     width_ - 6, width_ - 5, width_ - 4, width_ - 3, width_ - 2,
                     -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        } else if (column_index == 2) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7,
                     width_ - 6, width_ - 5, width_ - 4, width_ - 3, -2, -1, 0,
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        } else if (column_index == 3) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7,
                     width_ - 6, width_ - 5, width_ - 4, -3, -2, -1, 0, 1, 2, 3,
                     4, 5, 6, 7, 8, 9, 10);
        } else if (column_index == 4) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7,
                     width_ - 6, width_ - 5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 5) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7,
                     width_ - 6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                     9, 10);
        } else if (column_index == 6) {
          return get(width_ - 10, width_ - 9, width_ - 8, width_ - 7, -6, -5,
                     -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        } else if (column_index == 7) {
          return get(width_ - 10, width_ - 9, width_ - 8, -7, -6, -5, -4, -3,
                     -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        } else if (column_index == 8) {
          return get(width_ - 10, width_ - 9, -8, -7, -6, -5, -4, -3, -2, -1, 0,
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        } else {
          return get(width_ - 10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2,
                     3, 4, 5, 6, 7, 8, 9, 10);
        }
        break;

      case FixedBorderType::REVERSE:
        if (column_index == 0) {
          return get(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                     9, 10);
        } else if (column_index == 1) {
          return get(8, 7, 6, 5, 4, 3, 2, 1, 0, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                     9, 10);
        } else if (column_index == 2) {
          return get(6, 5, 4, 3, 2, 1, 0, -1, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                     9, 10);
        } else if (column_index == 3) {
          return get(4, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
                     8, 9, 10);
        } else if (column_index == 4) {
          return get(2, 1, 0, -1, -2, -3, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                     7, 8, 9, 10);
        } else if (column_index == 5) {
          return get(0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                     7, 8, 9, 10);
        } else if (column_index == 6) {
          return get(-2, -3, -4, -5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 7) {
          return get(-4, -5, -6, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else if (column_index == 8) {
          return get(-6, -7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        } else {
          return get(-8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10);
        }
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }

  // Returns offsets for columns affected by right border.
  Offsets offsets_with_right_border(size_t column_index) const
      KLEIDICV_STREAMING {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
        if (column_index == (width_ - 10)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 9);
        } else if (column_index == (width_ - 9)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 8, 8);
        } else if (column_index == (width_ - 8)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 7, 7, 7);
        } else if (column_index == (width_ - 7)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 6, 6, 6, 6);
        } else if (column_index == (width_ - 6)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     5, 5, 5, 5, 5);
        } else if (column_index == (width_ - 5)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4,
                     4, 4, 4, 4, 4);
        } else if (column_index == (width_ - 4)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3, 3,
                     3, 3, 3, 3, 3);
        } else if (column_index == (width_ - 3)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2, 2, 2,
                     2, 2, 2, 2, 2);
        } else if (column_index == (width_ - 2)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1);
        } else {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0);
        }
        break;

      case FixedBorderType::REFLECT:
        if (column_index == (width_ - 10)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 9);
        } else if (column_index == (width_ - 9)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 8, 7);
        } else if (column_index == (width_ - 8)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 7, 6, 5);
        } else if (column_index == (width_ - 7)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 6, 5, 4, 3);
        } else if (column_index == (width_ - 6)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     5, 4, 3, 2, 1);
        } else if (column_index == (width_ - 5)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4,
                     3, 2, 1, 0, -1);
        } else if (column_index == (width_ - 4)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3, 2,
                     1, 0, -1, -2, -3);
        } else if (column_index == (width_ - 3)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2, 1, 0,
                     -1, -2, -3, -4, -5);
        } else if (column_index == (width_ - 2)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1, 0, -1,
                     -2, -3, -4, -5, -6, -7);
        } else {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0, -1, -2, -3,
                     -4, -5, -6, -7, -8, -9);
        }
        break;

      case FixedBorderType::WRAP:
        if (column_index == (width_ - 10)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10 - width_);
        } else if (column_index == (width_ - 9)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9 - width_, 10 - width_);
        } else if (column_index == (width_ - 8)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8 - width_, 9 - width_, 10 - width_);
        } else if (column_index == (width_ - 7)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7 - width_, 8 - width_, 9 - width_, 10 - width_);
        } else if (column_index == (width_ - 6)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6 - width_, 7 - width_, 8 - width_, 9 - width_,
                     10 - width_);
        } else if (column_index == (width_ - 5)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     5 - width_, 6 - width_, 7 - width_, 8 - width_, 9 - width_,
                     10 - width_);
        } else if (column_index == (width_ - 4)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
                     4 - width_, 5 - width_, 6 - width_, 7 - width_, 8 - width_,
                     9 - width_, 10 - width_);
        } else if (column_index == (width_ - 3)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2,
                     3 - width_, 4 - width_, 5 - width_, 6 - width_, 7 - width_,
                     8 - width_, 9 - width_, 10 - width_);
        } else if (column_index == (width_ - 2)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2 - width_,
                     3 - width_, 4 - width_, 5 - width_, 6 - width_, 7 - width_,
                     8 - width_, 9 - width_, 10 - width_);
        } else {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1 - width_,
                     2 - width_, 3 - width_, 4 - width_, 5 - width_, 6 - width_,
                     7 - width_, 8 - width_, 9 - width_, 10 - width_);
        }
        break;

      case FixedBorderType::REVERSE:
        if (column_index == (width_ - 10)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 8);
        } else if (column_index == (width_ - 9)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 8, 7, 6);
        } else if (column_index == (width_ - 8)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 7, 6, 5, 4);
        } else if (column_index == (width_ - 7)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     6, 5, 4, 3, 2);
        } else if (column_index == (width_ - 6)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                     4, 3, 2, 1, 0);
        } else if (column_index == (width_ - 5)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 3,
                     2, 1, 0, -1, -2);
        } else if (column_index == (width_ - 4)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 2, 1,
                     0, -1, -2, -3, -4);
        } else if (column_index == (width_ - 3)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 1, 0, -1,
                     -2, -3, -4, -5, -6);
        } else if (column_index == (width_ - 2)) {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 0, -1, -2,
                     -3, -4, -5, -6, -7, -8);
        } else {
          return get(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, -1, -2, -3, -4,
                     -5, -6, -7, -8, -9, -10);
        }
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  // Returns offsets for rows or columns affected by any border.
  Offsets offsets_with_border(size_t row_or_column_index) const
      KLEIDICV_STREAMING {
    if (row_or_column_index < 10U) {
      // Rows and columns have the same offsets.
      return offsets_with_left_border(row_or_column_index);
    }
    if (row_or_column_index >= (width_ - 10U)) {
      // Rows and columns have the same offsets.
      return offsets_with_right_border(row_or_column_index);
    }
    return offsets_without_border();
  }

 private:
  // Takes care of static signed to unsigned casts.
  Offsets get(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t o3,
              ptrdiff_t o4, ptrdiff_t o5, ptrdiff_t o6, ptrdiff_t o7,
              ptrdiff_t o8, ptrdiff_t o9, ptrdiff_t o10, ptrdiff_t o11,
              ptrdiff_t o12, ptrdiff_t o13, ptrdiff_t o14, ptrdiff_t o15,
              ptrdiff_t o16, ptrdiff_t o17, ptrdiff_t o18, ptrdiff_t o19,
              ptrdiff_t o20) const KLEIDICV_STREAMING {
    return Offsets{o0,  o1,  o2,  o3,  o4,  o5,  o6,  o7,  o8,  o9, o10,
                   o11, o12, o13, o14, o15, o16, o17, o18, o19, o20};
  }

  size_t width_;
  FixedBorderType border_type_;
};  // end of class FixedBorderInfo<T, 21UL>

// Shorthand for 21x21 filter border type.
template <typename T>
using FixedBorderInfo21x21 = FixedBorderInfo<T, 21UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_BORDER_21X21_H
