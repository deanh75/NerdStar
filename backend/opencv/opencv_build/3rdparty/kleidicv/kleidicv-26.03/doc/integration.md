<!--
SPDX-FileCopyrightText: 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Integration guide

KleidiCV is designed to be easily reusable as it is a single C++ library with a
C API and its only runtime dependency is the C standard library. (Only
compile-time C++ constructs are used in the library.)

KleidiCV is already integrated into OpenCV, `adapters/opencv/kleidicv_hal.h` and
`adapters/opencv/kleidicv_hal.cpp` contains the implementation of it. It can be
 an example for integrating KleidiCV into other projects. OpenCV has a
`hal_replacement.hpp` file per OpenCV module which lists the HAL (Hardware
Acceleration Layer) functions for the given module.
`adapters/opencv/kleidicv_hal.h` sets macros (with some fallback mechanism to
other HAL implementations) to override the default HAL functions, and
`adapters/opencv/kleidicv_hal.cpp` contains implementations for these HAL
functions using KleidiCV as the backend.

## Extract one operation from KleidiCV

It can be desirable to extract only one function from KleidiCV, so the
repository contains an example about that at `examples/extract_one_operation`.
It is a CMake project which creates a shared library and an example application
to demonstrate the usage of the created shared library.

Generally, in KleidiCV one operation is implemented by one C++ source file, so
the example library uses one `.cpp` file,
`kleidicv/src/filters/gaussian_blur_fixed_sme.cpp` from the KleidiCV source
tree, and all the included header files.
`examples/extract_one_operation/sme_gaussian_blur.cpp` is needed to glue the
example library's public API with the implementation. The example library's
public API is defined by
`examples/extract_one_operation/sme_gaussian_blur_api.h`.

### Build the example

Building the example requires an SME capable toolchain and CMake. Example
command to build:

```
cmake -S kleidicv/examples/extract_one_operation \
      -B build/extract
```

If make was used as the generator for CMake
`build/extract/CMakeFiles/sme_gaussian_blur.dir/path/to/kleidicv/kleidicv/src/filters/gaussian_blur_fixed_sme.cpp.o.d`
contains which header files were used from KleidiCV. (But it also contains used system headers.)

To try the example just run:

```
./build/extract/example_usage
```

### Build the example for Android

In case of targeting Android the
[Android NDK](https://developer.android.com/ndk/) is also needed. (At least
version r29 RC 1.) Example build command is:

```
cmake -S kleidicv/examples/extract_one_operation \
      -B build/extract \
      -DANDROID_ABI=arm64-v8a \
      -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake
```
