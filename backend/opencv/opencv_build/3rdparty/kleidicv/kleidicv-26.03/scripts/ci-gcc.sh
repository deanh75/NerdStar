#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Ensure we're doing a clean build.
rm -rf build/ci
mkdir -p build/ci

if ! command -v qemu-aarch64; then
  apt-get update
  apt-get -y --no-install-recommends install qemu-user
fi

# Force ccache for all CMake builds.
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Build with GCC.
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ cmake -S . -B build/ci/gcc -G Ninja \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static"
ninja -C build/ci/gcc

# Run tests on GCC build.
TESTRESULT=0
qemu-aarch64 -cpu cortex-a35 build/ci/gcc/test/api/kleidicv-api-test --gtest_output=xml:build/ci/test-results/gcc-neon/ || TESTRESULT=1

scripts/prefix_testsuite_names.py build/ci/test-results/gcc-neon/kleidicv-api-test.xml "gcc-neon."

exit $TESTRESULT
