# SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB kleidicv_neon_unit_test_sources CONFIGURE_DEPENDS "*.h" "test_*.cpp")

list(APPEND kleidicv_neon_unit_test_sources ${KLEIDICV_TEST_FRAMEWORK_SOURCES})

set_source_files_properties(
  ${kleidicv_neon_unit_test_sources}
  PROPERTIES COMPILE_OPTIONS "${KLEIDICV_TEST_CXX_FLAGS}"
)

add_executable(
  kleidicv-neon-unit-test
  ${kleidicv_neon_unit_test_sources}
)

set_target_properties(
  kleidicv-neon-unit-test
  PROPERTIES CXX_STANDARD 17
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    kleidicv-neon-unit-test
    PRIVATE -flax-vector-conversions
  )
endif()

target_include_directories(
  kleidicv-neon-unit-test
  PRIVATE ${KLEIDICV_INCLUDE_DIR}
  PRIVATE ${KLEIDICV_TEST_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_compile_definitions(
  kleidicv-neon-unit-test
  PRIVATE KLEIDICV_TARGET_NEON=1
)

if (KLEIDICV_ALLOCATION_TESTS)
  target_link_options(
    kleidicv-neon-unit-test
    PRIVATE -Wl,--wrap,malloc
  )
endif()

target_link_libraries(
  kleidicv-neon-unit-test
  gtest_main
)
