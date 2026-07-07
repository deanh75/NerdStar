// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_BORDER_H_
#define KLEIDICV_TEST_FRAMEWORK_BORDER_H_

#include "framework/abstract.h"
#include "kleidicv/kleidicv.h"

namespace test {

// Prepares bordering elements given a border type and bordering requirements.
template <typename ElementType>
void prepare_borders(kleidicv_border_type_t border_type,
                     const ElementType *border_value, const Bordered *bordered,
                     TwoDimensional<ElementType> *elements);

}  // namespace test

#endif  // KLEIDICV_TEST_FRAMEWORK_BORDER_H_
