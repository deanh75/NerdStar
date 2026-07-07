<!--
SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Overview {.chapter}

KleidiCV is Arm's high-performance image processing library for AArch64. It
exposes a C API and has only a standard C runtime dependency, making it easy to
integrate into existing pipelines. KleidiCV integrates with CV frameworks such
as OpenCV. It selects optimized implementations at runtime based on available
Arm CPU features.

KleidiCV uses Arm C Language Extensions (ACLE) intrinsics and provides
implementations targeting Neon, SVE2, SME, and SME2. Runtime dispatch selects
an appropriate backend for each API. It focuses on performance-critical,
low-level operations such as color conversions, filtering, morphology, resize,
and geometric transforms.

This documentation covers the public C API:

- Function groups (conversions, filters, morphology, resize, transform,
  analysis, arithmetics)
- Supported data types and constraints for each operation
- Required types, constants, and error handling
- Per-function reference pages

Platform support is organized into tiers (Tier 1 includes AArch64 Ubuntu on
Neoverse N1 and Android on Galaxy S22; Tier 3 includes Apple silicon and
Windows Arm). Integration examples are available, including OpenCV HAL
integration and extracting a single operation into a standalone library.

For build, test, benchmark, and integration workflows, refer to the main
KleidiCV documentation at <https://gitlab.arm.com/kleidi/kleidicv>.

## Backend Selection {.section}

Most APIs are exposed in a default form, for example
`kleidicv_resize_linear_u8(...)`. This form uses the default runtime dispatch
path, which prefers Neon or SVE2 backends.

Some APIs also provide a variant whose name ends in `_sme`, for example
`kleidicv_resize_linear_u8_sme(...)`. The `_sme` suffix does not change the
algorithm, parameter contract, or result format. It only changes backend
selection: the `_sme` form prefers SME or SME2 backends when they are
available, and otherwise falls back to the same non-SME paths as the default
form.

The default form can be made to use the SME-preferred dispatch path by setting
`KLEIDICV_PREFER_SME_BACKEND=ON` before the library is loaded. This setting only
affects APIs that provide an `_sme` variant.
