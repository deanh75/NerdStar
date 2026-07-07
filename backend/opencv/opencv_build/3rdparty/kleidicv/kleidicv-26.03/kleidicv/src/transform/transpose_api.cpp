// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/transpose.h"

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(kleidicv_transpose,
                                        &kleidicv::neon::transpose, nullptr);
