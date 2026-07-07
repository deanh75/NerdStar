#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Commit message checks for merge requests.
if [[ -n "${CI_MERGE_REQUEST_DIFF_BASE_SHA:-}" ]]; then
  python3 scripts/check_commit_messages.py --range "${CI_MERGE_REQUEST_DIFF_BASE_SHA}..${CI_COMMIT_SHA}"
fi

# Ensure we're starting clean.
rm -rf build/ci
mkdir -p build/ci

# Check format of C++ files.
CHECK_ONLY=ON VERBOSE=ON scripts/format.sh

scripts/cpplint.sh

# Check format of shell scripts.
# shellcheck disable=SC2046
# Word splitting is essential here.
shellcheck $(find scripts -name '*.sh' | tr '\n' ' ')

# Check license headers.
reuse lint
