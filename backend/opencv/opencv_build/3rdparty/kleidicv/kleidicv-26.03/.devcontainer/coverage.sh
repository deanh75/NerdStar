#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -eu

BUILD_ID="kleidicv-coverage"

BUILD_ID="${BUILD_ID}" \
COVERAGE="ON" \
CMAKE_CXX_FLAGS="--target=aarch64-linux-gnu" \
CMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -static -fuse-ld=lld" \
EXTRA_CMAKE_ARGS="-DKLEIDICV_ENABLE_SME2=ON -DKLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS=OFF -DKLEIDICV_ENABLE_SME=ON -DKLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS=OFF" \
./scripts/build.sh kleidicv-test

# Clean any coverage results from previous runs
find build/kleidicv-coverage/ -type f -name *.gcda -delete

LONG_VECTOR_TESTS="GRAY2.*:RGB*:Yuv*:Rgb*:Resize*"
SME_API_TESTS="SmeApi*"

qemu-aarch64 build/kleidicv-coverage/test/framework/kleidicv-framework-test
qemu-aarch64 -cpu cortex-a35 build/kleidicv-coverage/test/api/kleidicv-api-test
qemu-aarch64 -cpu cortex-a35 build/kleidicv-coverage/test/unit_neon/kleidicv-neon-unit-test
qemu-aarch64 -cpu max,sve128=on,sme=off build/kleidicv-coverage/test/api/kleidicv-api-test --vector-length=16
qemu-aarch64 -cpu max,sve2048=on,sve-default-vector-length=256,sme=off \
  build/kleidicv-coverage/test/api/kleidicv-api-test --gtest_filter="${LONG_VECTOR_TESTS}" --vector-length=256
KLEIDICV_PREFER_SME_BACKEND=ON qemu-aarch64 -cpu max,sve128=on,sme512=on build/kleidicv-coverage/test/api/kleidicv-api-test --vector-length=64
KLEIDICV_PREFER_SME_BACKEND=OFF qemu-aarch64 -cpu max,sve128=on,sme512=on build/kleidicv-coverage/test/api/kleidicv-api-test --gtest_filter="${SME_API_TESTS}" --vector-length=64
KLEIDICV_PREFER_SME_BACKEND=ON armie -mvl=16 -msvl=64 -mfeatures=scripts/armie_features.txt \
  build/kleidicv-coverage/test/api/kleidicv-api-test --vector-length=64

# Generate test coverage report
LLVM_COV=llvm-cov scripts/generate_coverage_report.py "build/${BUILD_ID}"
