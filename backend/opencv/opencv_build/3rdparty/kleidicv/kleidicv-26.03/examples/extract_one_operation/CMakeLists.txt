# SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

project("SME Gaussian Blur")

# Create example shared library containing only the SME backend of GaussainBlur.
add_library(
  sme_gaussian_blur
  SHARED
  sme_gaussian_blur.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kleidicv/src/filters/gaussian_blur_fixed_sme.cpp
)

set_target_properties(
  sme_gaussian_blur
  PROPERTIES
  CXX_STANDARD 17
)

target_include_directories(
  sme_gaussian_blur
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kleidicv/include
)

target_compile_options(sme_gaussian_blur
  PRIVATE
  "-march=armv9-a+sme"
  "-DKLEIDICV_TARGET_SME=1"
  "-DKLEIDICV_TARGET_NAMESPACE=kleidicv::sme"
  "-DKLEIDICV_TARGET_FN_ATTRS="
)

# Create example binary to use the library.
add_executable(
  example_usage
  example_usage.c
)

target_link_libraries(
  example_usage
  sme_gaussian_blur
)
