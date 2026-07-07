#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Local wrapper to run the full CI sequence in one go.

set -euo pipefail

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scripts/ci-lint-docs.sh
scripts/ci-clang.sh
scripts/ci-gcc.sh

# TODO: Cross-build OpenCV
if [[ $(dpkg --print-architecture) = arm64 ]]; then
  # Check OpenCV-KleidiCV integration
  CLEAN="ON" scripts/ci-opencv-tests.sh
  scripts/run_opencv_conformity_checks.sh
fi
