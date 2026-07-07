// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_7X7_H
#define KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_7X7_H

#include <algorithm>

#include "kleidicv/kleidicv.h"
#include "kleidicv/traits.h"

namespace KLEIDICV_TARGET_NAMESPACE {
// A. Adams, "Fast Median Filters using Separable Sorting Networks,"
// available at: https://andrew.adams.pub/fast_median_filters.pdf
template <class Comparator, typename KernelWindowFunctor, typename T,
          typename ContextType>
void sorting_network7x7(KernelWindowFunctor& KernelWindow, T& output_vec,
                        ContextType& context) KLEIDICV_STREAMING {
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(6, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(3, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(5, 0), context);
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(4, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(6, 0), context);
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(5, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(4, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(6, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(3, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(5, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(4, 0), context);
  Comparator::compare_and_swap(KernelWindow(5, 0), KernelWindow(6, 0), context);

  Comparator::compare_and_swap(KernelWindow(0, 1), KernelWindow(6, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(5, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 1), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(4, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(6, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 1), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(5, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(4, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(6, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(5, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(4, 1), context);
  Comparator::compare_and_swap(KernelWindow(5, 1), KernelWindow(6, 1), context);

  Comparator::compare_and_swap(KernelWindow(0, 2), KernelWindow(6, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(5, 2), context);
  Comparator::compare_and_swap(KernelWindow(0, 2), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(6, 2), context);
  Comparator::compare_and_swap(KernelWindow(0, 2), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(5, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(6, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(5, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(6, 2), context);

  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(6, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(5, 3), context);
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(4, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(6, 3), context);
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(5, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(4, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(6, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(5, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(4, 3), context);
  Comparator::compare_and_swap(KernelWindow(5, 3), KernelWindow(6, 3), context);

  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(6, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(5, 4), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(4, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(6, 4), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(5, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(4, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(6, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(5, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(4, 4), context);
  Comparator::compare_and_swap(KernelWindow(5, 4), KernelWindow(6, 4), context);

  Comparator::compare_and_swap(KernelWindow(0, 5), KernelWindow(6, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 5), KernelWindow(3, 5), context);
  Comparator::compare_and_swap(KernelWindow(4, 5), KernelWindow(5, 5), context);
  Comparator::compare_and_swap(KernelWindow(0, 5), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(1, 5), KernelWindow(4, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 5), KernelWindow(6, 5), context);
  Comparator::compare_and_swap(KernelWindow(0, 5), KernelWindow(1, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 5), KernelWindow(5, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 5), KernelWindow(4, 5), context);
  Comparator::compare_and_swap(KernelWindow(1, 5), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(4, 5), KernelWindow(6, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 5), KernelWindow(3, 5), context);
  Comparator::compare_and_swap(KernelWindow(4, 5), KernelWindow(5, 5), context);
  Comparator::compare_and_swap(KernelWindow(1, 5), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 5), KernelWindow(4, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 5), KernelWindow(6, 5), context);

  Comparator::compare_and_swap(KernelWindow(0, 6), KernelWindow(6, 6), context);
  Comparator::compare_and_swap(KernelWindow(2, 6), KernelWindow(3, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 6), KernelWindow(5, 6), context);
  Comparator::compare_and_swap(KernelWindow(0, 6), KernelWindow(2, 6), context);
  Comparator::compare_and_swap(KernelWindow(1, 6), KernelWindow(4, 6), context);
  Comparator::compare_and_swap(KernelWindow(3, 6), KernelWindow(6, 6), context);
  Comparator::compare_and_swap(KernelWindow(0, 6), KernelWindow(1, 6), context);
  Comparator::compare_and_swap(KernelWindow(2, 6), KernelWindow(5, 6), context);
  Comparator::compare_and_swap(KernelWindow(3, 6), KernelWindow(4, 6), context);
  Comparator::compare_and_swap(KernelWindow(1, 6), KernelWindow(2, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 6), KernelWindow(6, 6), context);
  Comparator::compare_and_swap(KernelWindow(2, 6), KernelWindow(3, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 6), KernelWindow(5, 6), context);
  Comparator::compare_and_swap(KernelWindow(1, 6), KernelWindow(2, 6), context);
  Comparator::compare_and_swap(KernelWindow(3, 6), KernelWindow(4, 6), context);
  Comparator::compare_and_swap(KernelWindow(5, 6), KernelWindow(6, 6), context);

  // sort row 0 for element 4, 5, 6
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(0, 6), context);
  Comparator::compare_and_swap(KernelWindow(0, 2), KernelWindow(0, 3), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(0, 5), context);
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(0, 1), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(0, 6), context);
  Comparator::max(KernelWindow(0, 0), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 2), KernelWindow(0, 5), context);
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(0, 4), context);
  Comparator::max(KernelWindow(0, 1), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(0, 6), context);
  Comparator::max(KernelWindow(0, 2), KernelWindow(0, 3), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(0, 5), context);
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(0, 5), KernelWindow(0, 6), context);
  // sort row 1 for element 3, 4, 5, 6
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(1, 6), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(1, 5), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(1, 6), context);
  Comparator::max(KernelWindow(1, 0), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(1, 5), context);
  Comparator::max(KernelWindow(1, 1), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(1, 6), context);
  Comparator::max(KernelWindow(1, 2), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(1, 5), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 5), KernelWindow(1, 6), context);
  // sort row 2 for element 2, 3, 4, 5, 6
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(2, 6), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(2, 6), context);
  Comparator::max(KernelWindow(2, 0), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(2, 6), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(2, 5), context);
  Comparator::max(KernelWindow(2, 1), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 5), KernelWindow(2, 6), context);
  // sort row 3 for element 1, 2, 3, 4, 5
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(3, 6), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(3, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(3, 6), context);
  Comparator::max(KernelWindow(3, 0), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(3, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(3, 6), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(3, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(3, 4), context);
  Comparator::min(KernelWindow(3, 5), KernelWindow(3, 6), context);
  // sort row 4 for element 0, 1, 2, 3, 4
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(4, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(4, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(4, 5), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(4, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(4, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(4, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(4, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(4, 5), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(4, 2), context);
  Comparator::min(KernelWindow(4, 4), KernelWindow(4, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(4, 3), context);
  Comparator::min(KernelWindow(4, 4), KernelWindow(4, 5), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(4, 4), context);
  // sort row 5 for element 0, 1, 2, 3
  Comparator::compare_and_swap(KernelWindow(5, 0), KernelWindow(5, 6), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(5, 3), context);
  Comparator::compare_and_swap(KernelWindow(5, 4), KernelWindow(5, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 1), KernelWindow(5, 4), context);
  Comparator::compare_and_swap(KernelWindow(5, 0), KernelWindow(5, 2), context);
  Comparator::compare_and_swap(KernelWindow(5, 3), KernelWindow(5, 6), context);
  Comparator::compare_and_swap(KernelWindow(5, 0), KernelWindow(5, 1), context);
  Comparator::compare_and_swap(KernelWindow(5, 3), KernelWindow(5, 4), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(5, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 1), KernelWindow(5, 2), context);
  Comparator::min(KernelWindow(5, 4), KernelWindow(5, 6), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(5, 3), context);
  Comparator::min(KernelWindow(5, 4), KernelWindow(5, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 1), KernelWindow(5, 2), context);
  Comparator::min(KernelWindow(5, 3), KernelWindow(5, 4), context);
  // sort row 6 for element 0, 1, 2
  Comparator::compare_and_swap(KernelWindow(6, 0), KernelWindow(6, 6), context);
  Comparator::compare_and_swap(KernelWindow(6, 2), KernelWindow(6, 3), context);
  Comparator::compare_and_swap(KernelWindow(6, 4), KernelWindow(6, 5), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(6, 4), context);
  Comparator::compare_and_swap(KernelWindow(6, 0), KernelWindow(6, 2), context);
  Comparator::min(KernelWindow(6, 3), KernelWindow(6, 6), context);
  Comparator::compare_and_swap(KernelWindow(6, 0), KernelWindow(6, 1), context);
  Comparator::min(KernelWindow(6, 3), KernelWindow(6, 4), context);
  Comparator::min(KernelWindow(6, 2), KernelWindow(6, 5), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(6, 2), context);
  Comparator::min(KernelWindow(6, 2), KernelWindow(6, 3), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(6, 2), context);

  // sort digonal 0
  Comparator::max(KernelWindow(1, 3), KernelWindow(2, 2), context);
  Comparator::max(KernelWindow(3, 1), KernelWindow(4, 0), context);
  Comparator::max(KernelWindow(2, 2), KernelWindow(4, 0), context);
  Comparator::max(KernelWindow(4, 0), KernelWindow(0, 4), context);
  // sort digonal 1
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(5, 0), KernelWindow(0, 5), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(1, 4), context);
  Comparator::max(KernelWindow(4, 1), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(1, 4), context);
  Comparator::max(KernelWindow(5, 0), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(0, 5), context);
  Comparator::max(KernelWindow(3, 2), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(0, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(1, 4), context);
  // sort digonal 2
  Comparator::compare_and_swap(KernelWindow(6, 0), KernelWindow(0, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(1, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 1), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(6, 0), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(0, 6), context);
  Comparator::max(KernelWindow(6, 0), KernelWindow(5, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(1, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 1), KernelWindow(4, 2), context);
  Comparator::min(KernelWindow(2, 4), KernelWindow(0, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(3, 3), context);
  Comparator::min(KernelWindow(2, 4), KernelWindow(1, 5), context);
  Comparator::max(KernelWindow(5, 1), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(2, 4), context);
  // sort digonal 3
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(1, 6), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(4, 3), context);
  Comparator::min(KernelWindow(3, 4), KernelWindow(2, 5), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(3, 4), context);
  Comparator::min(KernelWindow(4, 3), KernelWindow(1, 6), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(5, 2), context);
  Comparator::min(KernelWindow(4, 3), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(4, 3), context);
  // sort digonal 4
  Comparator::min(KernelWindow(5, 3), KernelWindow(4, 4), context);
  Comparator::min(KernelWindow(3, 5), KernelWindow(2, 6), context);
  Comparator::min(KernelWindow(5, 3), KernelWindow(3, 5), context);
  Comparator::min(KernelWindow(6, 2), KernelWindow(5, 3), context);

  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(6, 2), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(4, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(6, 1), KernelWindow(6, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(0, 5), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(6, 2), KernelWindow(4, 2), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(2, 4), context);
  Comparator::min(KernelWindow(1, 4), KernelWindow(0, 5), context);
  Comparator::max(KernelWindow(6, 1), KernelWindow(4, 3), context);
  Comparator::compare_and_swap(KernelWindow(6, 2), KernelWindow(5, 2), context);
  Comparator::max(KernelWindow(4, 2), KernelWindow(2, 4), context);
  Comparator::min(KernelWindow(3, 3), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(3, 3), context);
  Comparator::min(KernelWindow(2, 4), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(5, 2), KernelWindow(4, 3), context);
  Comparator::max(KernelWindow(4, 2), KernelWindow(2, 3), context);
  Comparator::min(KernelWindow(3, 3), KernelWindow(2, 4), context);
  Comparator::max(KernelWindow(4, 2), KernelWindow(4, 3), context);
  Comparator::min(KernelWindow(3, 3), KernelWindow(2, 3), context);
  Comparator::max(KernelWindow(4, 3), KernelWindow(3, 3), context);

  output_vec = KernelWindow(3, 3);
}
}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_7X7_H
