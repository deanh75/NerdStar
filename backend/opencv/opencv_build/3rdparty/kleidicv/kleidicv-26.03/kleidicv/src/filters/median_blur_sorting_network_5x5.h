// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_5X5_H
#define KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_5X5_H

#include <algorithm>

#include "kleidicv/kleidicv.h"
#include "kleidicv/traits.h"

namespace KLEIDICV_TARGET_NAMESPACE {
// R. B. Kent and M. S. Pattichis, ''Design of high-speed multiway merge
// sorting networks using fast single-stage N-sorters and N-filters,'' *IEEE
// Access*, vol. 10, pp. 79565-79581, Jul. 2022,
// doi: 10.1109/ACCESS.2022.3193370. The paper is currently available at:
// https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9837930
template <class Comparator, typename KernelWindowFunctor, typename T,
          typename ContextType>
void sorting_network5x5(KernelWindowFunctor& KernelWindow, T& output_vec,
                        ContextType& context) KLEIDICV_STREAMING {
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(0, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(0, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 0), KernelWindow(0, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 0), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(3, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 1), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 1), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(0, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(1, 2), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(0, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(0, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(0, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(3, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(2, 4), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(3, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(2, 4), context);
  // sort row zero for only element 3 and 4
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(0, 0), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 2), KernelWindow(0, 0), context);
  Comparator::compare_and_swap(KernelWindow(0, 3), KernelWindow(0, 1), context);
  Comparator::min(KernelWindow(0, 1), KernelWindow(0, 0), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(0, 2), context);
  Comparator::min(KernelWindow(0, 2), KernelWindow(0, 1), context);
  Comparator::compare_and_swap(KernelWindow(0, 4), KernelWindow(0, 3), context);
  Comparator::min(KernelWindow(0, 3), KernelWindow(0, 2), context);
  // sort row 1 for only element {2, 3, 4}
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 2), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(1, 1), context);
  Comparator::min(KernelWindow(1, 1), KernelWindow(1, 0), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(1, 2), context);
  Comparator::min(KernelWindow(1, 2), KernelWindow(1, 1), context);
  Comparator::compare_and_swap(KernelWindow(1, 4), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(1, 3), KernelWindow(1, 2), context);
  // sort row 2 {1, 2, 3}
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(2, 1), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(2, 1), context);
  Comparator::min(KernelWindow(2, 1), KernelWindow(2, 0), context);
  Comparator::compare_and_swap(KernelWindow(2, 4), KernelWindow(2, 2), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(2, 1), context);
  Comparator::max(KernelWindow(2, 4), KernelWindow(2, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 3), KernelWindow(2, 2), context);
  // sort row 3
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(3, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(3, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 3), KernelWindow(3, 1), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(3, 0), context);
  Comparator::compare_and_swap(KernelWindow(3, 4), KernelWindow(3, 2), context);
  Comparator::compare_and_swap(KernelWindow(3, 2), KernelWindow(3, 1), context);
  Comparator::max(KernelWindow(3, 4), KernelWindow(3, 3), context);
  Comparator::max(KernelWindow(3, 3), KernelWindow(3, 2), context);
  // sort row 4
  Comparator::compare_and_swap(KernelWindow(4, 3), KernelWindow(4, 0), context);
  Comparator::compare_and_swap(KernelWindow(4, 4), KernelWindow(4, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 2), KernelWindow(4, 0), context);
  Comparator::max(KernelWindow(4, 3), KernelWindow(4, 1), context);
  Comparator::compare_and_swap(KernelWindow(4, 1), KernelWindow(4, 0), context);
  Comparator::max(KernelWindow(4, 4), KernelWindow(4, 2), context);
  Comparator::max(KernelWindow(4, 2), KernelWindow(4, 1), context);
  // sort dig 0
  Comparator::min(KernelWindow(2, 1), KernelWindow(0, 3), context);
  Comparator::min(KernelWindow(3, 0), KernelWindow(1, 2), context);
  Comparator::min(KernelWindow(3, 0), KernelWindow(2, 1), context);
  // sort dig 1
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(1, 3), context);
  Comparator::compare_and_swap(KernelWindow(2, 2), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 1), KernelWindow(1, 3), context);
  Comparator::min(KernelWindow(1, 3), KernelWindow(0, 4), context);
  Comparator::compare_and_swap(KernelWindow(4, 0), KernelWindow(2, 2), context);
  Comparator::min(KernelWindow(2, 2), KernelWindow(1, 3), context);
  Comparator::max(KernelWindow(4, 0), KernelWindow(3, 1), context);
  Comparator::max(KernelWindow(3, 1), KernelWindow(2, 2), context);
  // sort dig 2
  Comparator::max(KernelWindow(3, 2), KernelWindow(1, 4), context);
  Comparator::max(KernelWindow(4, 1), KernelWindow(2, 3), context);
  Comparator::max(KernelWindow(2, 3), KernelWindow(1, 4), context);
  Comparator::compare_and_swap(KernelWindow(3, 0), KernelWindow(1, 4), context);
  Comparator::min(KernelWindow(2, 2), KernelWindow(1, 4), context);
  Comparator::max(KernelWindow(3, 0), KernelWindow(2, 2), context);
  output_vec = KernelWindow(2, 2);
}
}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MEDIAN_BLUR_SORTING_NETWORK_5X5_H
