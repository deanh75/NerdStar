// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTER_2D_WINDOW_LOADER_5X5_H
#define KLEIDICV_FILTER_2D_WINDOW_LOADER_5X5_H

#include "kleidicv/workspace/border_5x5.h"

namespace KLEIDICV_TARGET_NAMESPACE {
template <typename SourceType>
class Filter2dWindowLoader5x5 {
 public:
  using BorderInfoType =
      typename KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo5x5<SourceType>;
  using BorderOffsets = typename BorderInfoType::Offsets;

  template <typename LoadArrayElementFunctionType, typename KernelWindowFunctor>
  static void load_window(KernelWindowFunctor& KernelWindow,
                          LoadArrayElementFunctionType load_array_element,
                          Rows<const SourceType> src_rows,
                          BorderOffsets window_row_offsets,
                          BorderOffsets window_col_offsets,
                          size_t index) KLEIDICV_STREAMING {
    KernelWindow(0, 0) = load_array_element(
        src_rows.at(window_row_offsets.c0(), window_col_offsets.c0())[index]);
    KernelWindow(0, 1) = load_array_element(
        src_rows.at(window_row_offsets.c0(), window_col_offsets.c1())[index]);
    KernelWindow(0, 2) = load_array_element(
        src_rows.at(window_row_offsets.c0(), window_col_offsets.c2())[index]);
    KernelWindow(0, 3) = load_array_element(
        src_rows.at(window_row_offsets.c0(), window_col_offsets.c3())[index]);
    KernelWindow(0, 4) = load_array_element(
        src_rows.at(window_row_offsets.c0(), window_col_offsets.c4())[index]);
    KernelWindow(1, 0) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c0())[index]);
    KernelWindow(1, 1) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c1())[index]);
    KernelWindow(1, 2) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c2())[index]);
    KernelWindow(1, 3) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c3())[index]);
    KernelWindow(1, 4) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c4())[index]);
    KernelWindow(2, 0) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c0())[index]);
    KernelWindow(2, 1) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c1())[index]);
    KernelWindow(2, 2) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c2())[index]);
    KernelWindow(2, 3) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c3())[index]);
    KernelWindow(2, 4) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c4())[index]);
    KernelWindow(3, 0) = load_array_element(
        src_rows.at(window_row_offsets.c3(), window_col_offsets.c0())[index]);
    KernelWindow(3, 1) = load_array_element(
        src_rows.at(window_row_offsets.c3(), window_col_offsets.c1())[index]);
    KernelWindow(3, 2) = load_array_element(
        src_rows.at(window_row_offsets.c3(), window_col_offsets.c2())[index]);
    KernelWindow(3, 3) = load_array_element(
        src_rows.at(window_row_offsets.c3(), window_col_offsets.c3())[index]);
    KernelWindow(3, 4) = load_array_element(
        src_rows.at(window_row_offsets.c3(), window_col_offsets.c4())[index]);
    KernelWindow(4, 0) = load_array_element(
        src_rows.at(window_row_offsets.c4(), window_col_offsets.c0())[index]);
    KernelWindow(4, 1) = load_array_element(
        src_rows.at(window_row_offsets.c4(), window_col_offsets.c1())[index]);
    KernelWindow(4, 2) = load_array_element(
        src_rows.at(window_row_offsets.c4(), window_col_offsets.c2())[index]);
    KernelWindow(4, 3) = load_array_element(
        src_rows.at(window_row_offsets.c4(), window_col_offsets.c3())[index]);
    KernelWindow(4, 4) = load_array_element(
        src_rows.at(window_row_offsets.c4(), window_col_offsets.c4())[index]);
  }
};
}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_FILTER_2D_WINDOW_LOADER_5X5_H
