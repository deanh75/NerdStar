<!--
SPDX-FileCopyrightText: 2023 - 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Platform Support

Support for different platforms is organised in 3 tiers[^1].

## Tier 1

A released version of KleidiCV shall be tested for correctness and performance on a Tier 1 platform.

Tier 1 platforms:
* Android on Samsung Galaxy S22 with an up-to-date operating system, where applications are built with
  Android NDK r27c.
* AArch64 Ubuntu (as defined by `docker/Dockerfile` in the KleidiCV source tree) on Arm Neoverse N1 CPU.

## Tier 2

KleidiCV is tested for correctness on a Tier 2 platform, however performance is not checked.

There are no Tier 2 platforms currently.

## Tier 3

KleidiCV aims to support Tier 3 platforms but since no routine testing is done on these platforms
it may or may not work in practice.

Tier 3 platforms:
* Mac computers with Apple silicon
* Windows 11 Arm-based PCs

[^1] Inspired by [Rust's platform support](https://doc.rust-lang.org/nightly/rustc/platform-support.html).
