#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Builds the perf benchmarks for OpenCV with Android target.
#
# The following builds will be created:
#   * opencv-vanilla:                          (without KleidiCV, but with VANILLA_EXTRA_CMAKE_OPTIONS)
#   * opencv-kleidicv:                         (with KleidiCV and KLEIDICV_EXTRA_CMAKE_OPTIONS)
#   * [opencv-kleidicv-<CUSTOM_BUILD_SUFFIX>]: (with KleidiCV and CUSTOM_CMAKE_OPTIONS,
#                                               but only if the CUSTOM_CMAKE_OPTIONS were provided)
#
# It uses the scripts/build-opencv.sh. See that script for detailed behavior and more options.
# Note that the OPENCV_PATH (for OpenCV) and CMAKE_TOOLCHAIN_FILE (for Android NDK)
# variables should be set as well.
#
# Options:
#   CMAKE_TARGETS:                Optional space separated list for CMake build targets.
#   COMMON_EXTRA_CMAKE_ARGS:      Optional commmon CMake options for all the opencv builds.
#   VANILLA_EXTRA_CMAKE_OPTIONS:  Optional extra CMake options for the opencv-vanilla build.
#   KLEIDICV_EXTRA_CMAKE_OPTIONS: Optional extra CMake options for the opencv-kleidicv build.
#   CUSTOM_CMAKE_OPTIONS:         If provided, a (first) extra build will be created with KleidiCV with these options.
#   CUSTOM2_CMAKE_OPTIONS:        If provided, a (second) extra build will be created with KleidiCV with these options.
#   CUSTOM3_CMAKE_OPTIONS:        If provided, a (third) extra build will be created with KleidiCV with these options.
#   CUSTOM_BUILD_SUFFIX:          Build name suffix for the first extra build. Defaults to 'custom'.
#   CUSTOM2_BUILD_SUFFIX:         Build name suffix for the second extra build. Defaults to 'custom2'.
#   CUSTOM3_BUILD_SUFFIX:         Build name suffix for the third extra build. Defaults to 'custom3'.
#
# ------------------------------------------------------------------------------

set -exu

if [[ ! -f "${CMAKE_TOOLCHAIN_FILE:-}" ]]; then
  echo "Please specify the path of the Android NDK toolchain file (e.g. android-ndk-r27c/build/cmake/android.toolchain.cmake) in the CMAKE_TOOLCHAIN_FILE env variable"
  exit 1
fi

# ------------------------------------------------------------------------------

BENCHMARK_SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
SCRIPT_PATH="${BENCHMARK_SCRIPT_PATH}/.."
KLEIDICV_SOURCE_PATH="$(realpath "${SCRIPT_PATH}/..")"

export COMMON_CMAKE_ARGS="\
  -DANDROID_ABI=arm64-v8a \
  -DBUILD_PERF_TESTS=ON \
  -DOPENCV_ALGO_HINT_DEFAULT=ALGO_HINT_APPROX \
"

# ------------------------------------------------------------------------------

export BUILD_ID="opencv-vanilla"
export EXTRA_CMAKE_ARGS="\
    ${COMMON_CMAKE_ARGS} \
    ${COMMON_EXTRA_CMAKE_ARGS:-} \
    -DWITH_KLEIDICV=OFF \
    ${VANILLA_EXTRA_CMAKE_OPTIONS:-} \
"

"${SCRIPT_PATH}"/build-opencv.sh "${CMAKE_TARGETS:-}"

# ------------------------------------------------------------------------------

export BUILD_ID="opencv-kleidicv"
export EXTRA_CMAKE_ARGS="\
    ${COMMON_CMAKE_ARGS} \
    ${COMMON_EXTRA_CMAKE_ARGS:-} \
    -DWITH_KLEIDICV=ON \
    -DKLEIDICV_SOURCE_PATH=${KLEIDICV_SOURCE_PATH} \
    ${KLEIDICV_EXTRA_CMAKE_OPTIONS:-} \
"

"${SCRIPT_PATH}"/build-opencv.sh "${CMAKE_TARGETS:-}"

# ------------------------------------------------------------------------------

if [[ -z "${CUSTOM_CMAKE_OPTIONS:-}" ]]; then
  exit 0;
fi

export BUILD_ID="opencv-kleidicv-${CUSTOM_BUILD_SUFFIX:-custom}"
export EXTRA_CMAKE_ARGS="\
    ${COMMON_CMAKE_ARGS} \
    ${COMMON_EXTRA_CMAKE_ARGS:-} \
    -DWITH_KLEIDICV=ON \
    -DKLEIDICV_SOURCE_PATH=${KLEIDICV_SOURCE_PATH} \
    ${CUSTOM_CMAKE_OPTIONS} \
"

"${SCRIPT_PATH}"/build-opencv.sh "${CMAKE_TARGETS:-}"

# ------------------------------------------------------------------------------

if [[ -z "${CUSTOM2_CMAKE_OPTIONS:-}" ]]; then
  exit 0;
fi

export BUILD_ID="opencv-kleidicv-${CUSTOM2_BUILD_SUFFIX:-custom2}"
export EXTRA_CMAKE_ARGS="\
    ${COMMON_CMAKE_ARGS} \
    ${COMMON_EXTRA_CMAKE_ARGS:-} \
    -DWITH_KLEIDICV=ON \
    -DKLEIDICV_SOURCE_PATH=${KLEIDICV_SOURCE_PATH} \
    ${CUSTOM2_CMAKE_OPTIONS} \
"

"${SCRIPT_PATH}"/build-opencv.sh "${CMAKE_TARGETS:-}"

# ------------------------------------------------------------------------------

if [[ -z "${CUSTOM3_CMAKE_OPTIONS:-}" ]]; then
  exit 0;
fi

export BUILD_ID="opencv-kleidicv-${CUSTOM3_BUILD_SUFFIX:-custom3}"
export EXTRA_CMAKE_ARGS="\
    ${COMMON_CMAKE_ARGS} \
    ${COMMON_EXTRA_CMAKE_ARGS:-} \
    -DWITH_KLEIDICV=ON \
    -DKLEIDICV_SOURCE_PATH=${KLEIDICV_SOURCE_PATH} \
    ${CUSTOM3_CMAKE_OPTIONS} \
"

"${SCRIPT_PATH}"/build-opencv.sh "${CMAKE_TARGETS:-}"

# ------------------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------------------
