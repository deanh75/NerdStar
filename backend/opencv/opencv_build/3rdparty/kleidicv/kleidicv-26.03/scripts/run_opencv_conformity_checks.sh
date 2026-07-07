#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Build and run OpenCV conformity checks under QEMU.

set -exu

SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

: "${CLEAN:=OFF}"
: "${OPENCV_VERSION:=4.13.0}"
: "${OPENCV_URL:=/opt/opencv-${OPENCV_VERSION}.tar.gz}"
: "${BUILD_PATH:=${SCRIPT_PATH}/../build/conformity}"

OPENCV_DEFAULT_PATH="${BUILD_PATH}/opencv_default"
OPENCV_KLEIDICV_PATH="${BUILD_PATH}/opencv_kleidicv"

OPENCV_CONFORMITY_CPUS="${OPENCV_CONFORMITY_CPUS:-cortex-a35 max,sve128=on,sme=off max,sve128=on,sme512=on}"

export OPENCV_VERSION OPENCV_URL

BUILD_PATH="${BUILD_PATH}" \
CLEAN="${CLEAN}" \
  "${SCRIPT_PATH}/opencv-conformity-build.sh"

TESTRESULT=0
for cpu in ${OPENCV_CONFORMITY_CPUS}; do
    KLEIDICV_PREFER_SME_BACKEND=ON qemu-aarch64 -cpu "${cpu}" "${OPENCV_KLEIDICV_PATH}/bin/manager" "${OPENCV_DEFAULT_PATH}/bin/subordinate" || TESTRESULT=1
done

exit $TESTRESULT
