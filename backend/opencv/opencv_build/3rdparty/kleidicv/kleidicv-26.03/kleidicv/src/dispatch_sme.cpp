// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>

#include "kleidicv/config.h"
#include "kleidicv/dispatch.h"

namespace kleidicv {
KLEIDICV_LOCALLY_STREAMING uint64_t svcntb_sme() { return svcntb(); }
}  // namespace kleidicv
