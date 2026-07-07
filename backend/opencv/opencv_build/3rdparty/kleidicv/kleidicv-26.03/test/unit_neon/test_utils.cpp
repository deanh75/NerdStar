// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "kleidicv/utils.h"

using kleidicv::neon::saturating_sub;

TEST(Utils, SaturatingSub1) { EXPECT_EQ(5UL, saturating_sub(10UL, 5UL)); }

TEST(Utils, SaturatingSub2) { EXPECT_EQ(0UL, saturating_sub(5UL, 10UL)); }
