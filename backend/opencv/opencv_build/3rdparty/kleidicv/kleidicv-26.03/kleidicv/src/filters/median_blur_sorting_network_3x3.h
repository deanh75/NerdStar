// SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_3X3_H
#define KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_3X3_H

#include <algorithm>

#include "kleidicv/kleidicv.h"
#include "kleidicv/traits.h"

namespace KLEIDICV_TARGET_NAMESPACE {
template <class Comparator, typename KernelWindowFunctor, typename T,
          typename ContextType>
void sorting_network3x3_single_row(KernelWindowFunctor& KernelWindow,
                                   T& output_vec,
                                   ContextType& context) KLEIDICV_STREAMING {
  // full sort row
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 1), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(2, 2), context);
  // find max in col 0
  T max_0_1_2 =
      Comparator::get_max(KernelWindow(0, 0), KernelWindow(1, 0), context);
  max_0_1_2 = Comparator::get_max(max_0_1_2, KernelWindow(2, 0), context);
  // find mid in col 1
  T src_tmp_0 = KernelWindow(0, 1);
  T mid_0_1_2 = KernelWindow(1, 1);
  T src_tmp_2 = KernelWindow(2, 1);
  Comparator::compare_and_swap(src_tmp_0, src_tmp_2, context);
  Comparator::max(src_tmp_0, mid_0_1_2, context);
  Comparator::min(mid_0_1_2, src_tmp_2, context);
  // find min in col 2
  T min_0_1_2 =
      Comparator::get_min(KernelWindow(0, 2), KernelWindow(1, 2), context);
  min_0_1_2 = Comparator::get_min(min_0_1_2, KernelWindow(2, 2), context);
  // find median
  Comparator::compare_and_swap(min_0_1_2, max_0_1_2, context);
  Comparator::max(min_0_1_2, mid_0_1_2, context);
  Comparator::min(mid_0_1_2, max_0_1_2, context);
  output_vec = mid_0_1_2;
}

template <class Comparator, typename KernelWindowFunctor, typename T,
          typename ContextType>
void sorting_network3x3_dual_rows(KernelWindowFunctor& KernelWindow,
                                  T& output_vec_0, T& output_vec_1,
                                  ContextType& context) KLEIDICV_STREAMING {
  // full sort row
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(0, 0), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 1), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(3, 2), context);
  // sort common element
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(2, 1), context);
  // find first median
  // find max in col 0
  T max_0_1_2 =
      Comparator::get_max(KernelWindow(0, 0), KernelWindow(2, 0), context);
  // find mid in col 1
  T src_tmp_0 =
      Comparator::get_min(KernelWindow(0, 1), KernelWindow(2, 1), context);
  T mid_0_1_2 = Comparator::get_max(KernelWindow(1, 1), src_tmp_0, context);
  // find min in col 2
  T min_0_1_2 =
      Comparator::get_min(KernelWindow(0, 2), KernelWindow(1, 2), context);
  // find median
  Comparator::compare_and_swap(min_0_1_2, max_0_1_2, context);
  Comparator::max(min_0_1_2, mid_0_1_2, context);
  Comparator::min(mid_0_1_2, max_0_1_2, context);
  output_vec_0 = mid_0_1_2;
  // find second median
  // find max in col 0
  max_0_1_2 =
      Comparator::get_max(KernelWindow(2, 0), KernelWindow(3, 0), context);
  // find mid in col 1
  src_tmp_0 =
      Comparator::get_max(KernelWindow(1, 1), KernelWindow(3, 1), context);
  mid_0_1_2 = Comparator::get_min(KernelWindow(2, 1), src_tmp_0, context);
  // find min in col 2
  min_0_1_2 =
      Comparator::get_min(KernelWindow(1, 2), KernelWindow(3, 2), context);
  // find median
  Comparator::compare_and_swap(min_0_1_2, max_0_1_2, context);
  Comparator::max(min_0_1_2, mid_0_1_2, context);
  Comparator::min(mid_0_1_2, max_0_1_2, context);
  output_vec_1 = mid_0_1_2;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_3X3_H
