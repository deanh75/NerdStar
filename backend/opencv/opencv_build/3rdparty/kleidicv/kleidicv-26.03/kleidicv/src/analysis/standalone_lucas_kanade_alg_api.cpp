// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/analysis/standalone_lucas_kanade_alg.h"
#include "kleidicv/dispatch.h"

KLEIDICV_MULTIVERSION_C_API_WITH_SME(
    kleidicv_standalone_lucas_kanade_alg_u8,
    &kleidicv::neon::standalone_lucas_kanade_alg_u8,
    &kleidicv::sve2::standalone_lucas_kanade_alg_u8,
    &kleidicv::sme::standalone_lucas_kanade_alg_u8,
    &kleidicv::sme2::standalone_lucas_kanade_alg_u8);
