// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTER_2D_WINDOW_LOADER_3X3_H
#define KLEIDICV_FILTER_2D_WINDOW_LOADER_3X3_H

#include "kleidicv/types.h"
#include "kleidicv/workspace/border_3x3.h"

namespace KLEIDICV_TARGET_NAMESPACE {
template <typename SourceType>
class Filter2dWindowLoader3x3 {
 public:
  using BorderInfoType =
      typename KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo3x3<SourceType>;
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

    KernelWindow(1, 0) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c0())[index]);
    KernelWindow(1, 1) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c1())[index]);
    KernelWindow(1, 2) = load_array_element(
        src_rows.at(window_row_offsets.c1(), window_col_offsets.c2())[index]);

    KernelWindow(2, 0) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c0())[index]);
    KernelWindow(2, 1) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c1())[index]);
    KernelWindow(2, 2) = load_array_element(
        src_rows.at(window_row_offsets.c2(), window_col_offsets.c2())[index]);
  }

  template <typename LoadArrayElementFunctionType, typename KernelWindowFunctor>
  static void load_window_to_handle_dual_rows(
      KernelWindowFunctor& KernelWindow,
      LoadArrayElementFunctionType load_array_element,
      Rows<const SourceType> src_rows, BorderOffsets window_row_offsets_0,
      BorderOffsets window_row_offsets_1, BorderOffsets window_col_offsets,
      size_t index) KLEIDICV_STREAMING {
    load_window(KernelWindow, load_array_element, src_rows,
                window_row_offsets_0, window_col_offsets, index);

    KernelWindow(3, 0) = load_array_element(src_rows.at(
        window_row_offsets_1.c2() + 1, window_col_offsets.c0())[index]);
    KernelWindow(3, 1) = load_array_element(src_rows.at(
        window_row_offsets_1.c2() + 1, window_col_offsets.c1())[index]);
    KernelWindow(3, 2) = load_array_element(src_rows.at(
        window_row_offsets_1.c2() + 1, window_col_offsets.c2())[index]);
  }
};
}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_FILTER_2D_WINDOW_LOADER_3X3_BASE_H
