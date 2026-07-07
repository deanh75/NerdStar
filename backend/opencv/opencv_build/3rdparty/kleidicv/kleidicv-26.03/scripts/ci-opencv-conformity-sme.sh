#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $(dpkg --print-architecture) = arm64 ]]; then
  if ! command -v qemu-aarch64; then
    apt-get update
    apt-get -y --no-install-recommends install qemu-user
  fi

  CLEAN="ON" OPENCV_CONFORMITY_CPUS="max,sve128=on,sme512=on" \
    scripts/run_opencv_conformity_checks.sh
else
  echo "Skipping OpenCV conformity on non-arm64 runner."
fi
