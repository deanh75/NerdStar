# SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

project("KleidiCV C example")

add_executable(
  kleidicv-c-example
  main.c
)

target_include_directories(
  kleidicv-c-example
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kleidicv/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../kleidicv/include
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Trying to link without the C++ runtime. Should work in case of gcc and clang.
  list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_LIBRARIES stdc++ c++)
  set_target_properties(
    kleidicv-c-example
    PROPERTIES
    LINKER_LANGUAGE "C"
  )

  # Linking the whole KleidiCV library to check whether any cpp runtime feature is used.
  target_link_libraries(
    kleidicv-c-example
    -Wl,--whole-archive
    kleidicv
    -Wl,--no-whole-archive
    m
  )
else()
  target_link_libraries(
    kleidicv-c-example
    kleidicv
  )
endif()
