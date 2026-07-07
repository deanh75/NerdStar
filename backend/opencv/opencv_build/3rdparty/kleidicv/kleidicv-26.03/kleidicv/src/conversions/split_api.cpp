// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/split.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(kleidicv_split, &kleidicv::neon::split,
                                        nullptr);
