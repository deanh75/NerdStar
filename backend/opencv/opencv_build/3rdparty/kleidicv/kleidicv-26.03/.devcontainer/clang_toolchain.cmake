# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER clang-$ENV{LLVM_VERSION})
set(CMAKE_CXX_COMPILER clang++-$ENV{LLVM_VERSION})
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(GNU_MACHINE aarch64-linux-gnu)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_BUILD_RPATH "")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "--rtlib=compiler-rt -fuse-ld=lld -static ${CMAKE_EXE_LINKER_FLAGS}")
