# SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/../../kleidicv/CMakeLists.txt")
include("${CMAKE_CURRENT_LIST_DIR}/../../kleidicv_thread/CMakeLists.txt")

project("KleidiCV OpenCV HAL")

set(KLEIDICV_HAL_VERSION "26.03" CACHE INTERNAL "")
set(KLEIDICV_HAL_LIBRARIES "kleidicv_hal" "kleidicv" "kleidicv_thread" CACHE INTERNAL "")
set(KLEIDICV_HAL_HEADERS "${CMAKE_CURRENT_LIST_DIR}/kleidicv_hal.h" CACHE INTERNAL "")
set(KLEIDICV_HAL_INCLUDE_DIRS "$<TARGET_PROPERTY:kleidicv_thread,INCLUDE_DIRECTORIES>" CACHE INTERNAL "")

add_library(kleidicv_hal STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} "${CMAKE_CURRENT_LIST_DIR}/kleidicv_hal.cpp")
target_link_libraries(kleidicv_hal PUBLIC kleidicv kleidicv_thread)
target_include_directories(kleidicv_hal PRIVATE
  ${KLEIDICV_HAL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${OpenCV_SOURCE_DIR}/modules/core/include
  ${OpenCV_SOURCE_DIR}/modules/imgproc/include
)
target_compile_options(
  kleidicv_hal
  PRIVATE
    $<TARGET_PROPERTY:kleidicv,COMPILE_OPTIONS>
    "-Wno-old-style-cast"
)

set_target_properties(kleidicv_hal
  PROPERTIES CXX_STANDARD 17
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

set_target_properties(kleidicv
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

set_target_properties(kleidicv_thread
  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(kleidicv_neon
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
  if(KLEIDICV_ENABLE_SVE2)
    ocv_install_target(kleidicv_sve2
      EXPORT OpenCVModules
      ARCHIVE
      DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
      COMPONENT dev
    )
  endif()
  if(KLEIDICV_ENABLE_SME)
    ocv_install_target(kleidicv_sme
      EXPORT OpenCVModules
      ARCHIVE
      DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
      COMPONENT dev
    )
  endif()
  if(KLEIDICV_ENABLE_SME2)
    ocv_install_target(kleidicv_sme2
      EXPORT OpenCVModules
      ARCHIVE
      DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
      COMPONENT dev
    )
  endif()
  ocv_install_target(kleidicv
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
  ocv_install_target(kleidicv_thread
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
  ocv_install_target(kleidicv_hal
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
endif()
