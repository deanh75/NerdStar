// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/remap.h"

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(kleidicv_remap_s16_u8,
                                        &kleidicv::neon::remap_s16<uint8_t>,
                                        &kleidicv::sve2::remap_s16<uint8_t>);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(kleidicv_remap_s16_u16,
                                        &kleidicv::neon::remap_s16<uint16_t>,
                                        &kleidicv::sve2::remap_s16<uint16_t>);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(
    kleidicv_remap_s16point5_u8, &kleidicv::neon::remap_s16point5<uint8_t>,
    &kleidicv::sve2::remap_s16point5<uint8_t>);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(
    kleidicv_remap_s16point5_u16, &kleidicv::neon::remap_s16point5<uint16_t>,
    &kleidicv::sve2::remap_s16point5<uint16_t>);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(kleidicv_remap_f32_u8,
                                        &kleidicv::neon::remap_f32<uint8_t>,
                                        &kleidicv::sve2::remap_f32<uint8_t>);

KLEIDICV_MULTIVERSION_C_API_WITHOUT_SME(kleidicv_remap_f32_u16,
                                        &kleidicv::neon::remap_f32<uint16_t>,
                                        &kleidicv::sve2::remap_f32<uint16_t>);
