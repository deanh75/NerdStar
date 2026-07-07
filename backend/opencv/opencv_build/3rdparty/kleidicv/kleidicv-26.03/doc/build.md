<!--
SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Building KleidiCV for Android

## Prerequisites
While the core functionality of the library does not rely on any third-party
libraries, there are build prerequisites that are essential for compiling the
source code and generating the executable. Please ensure that these tools are
installed on your system before proceeding with the build process.

To successfully build and compile this project for Android, you'll need the following tools:
- [Android NDK](https://developer.android.com/ndk/).
  See [the platform support page](platform-support.md) for supported versions.
- [CMake](https://cmake.org) 3.16 or higher.
- `make`
- `patch`

Running tests on Android devices requires [ADB](https://developer.android.com/tools/adb).
ADB is included in [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools).

## Building OpenCV & KleidiCV for Android

For details of which OpenCV function are accelerated by KleidiCV see
[KleidiCV's OpenCV documentation](opencv.md).

### Get and patch OpenCV source

This version of KleidiCV is compatible with [OpenCV](https://opencv.org) version 4.13 and later.
Earlier versions of KleidiCV are compatible with earlier versions of OpenCV.
OpenCV 5.x support is experimental.

Integration consists of the following steps:
1. Download OpenCV sources:
```
wget https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz
tar xf 4.13.0.tar.gz
cd opencv-4.13.0
```
2. Patch OpenCV:
```
patch -p1</path/to/kleidicv/adapters/opencv/opencv-4.13.patch
```
(It can happen that the current patch file is empty. In that case the patch command outputs a warning, but it is not an issue.)

### Build

OpenCV can be built with KleidiCV enabled via the following CMake variables:
- `WITH_KLEIDICV` - set this to `ON` to enable KleidiCV.
- `KLEIDICV_SOURCE_PATH` - the top-level `kleidicv` directory.

```
cmake \
  -S /path/to/opencv \
  -B build-opencv-android \
  -DANDROID_ABI=arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DBUILD_ANDROID_PROJECTS=OFF \
  -DWITH_KLEIDICV=ON \
  -DKLEIDICV_SOURCE_PATH=/path/to/kleidicv
cmake --build build-opencv-android --parallel
```

(`BUILD_ANDROID_PROJECTS=OFF` is specified just to simplify these build instructions.
See the OpenCV project's documentation to learn the requirements for enabling the option.)

## Build KleidiCV standalone for Android

From the top-level `kleidicv` directory run:
```
cmake -S . -B build-kleidicv-android \
-DANDROID_ABI=arm64-v8a \
-DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake
cmake --build build-kleidicv-android --parallel
```

# Building OpenCV & KleidiCV as an AAR package

> **Last tested with OpenCV 4.13**

The AAR package built from OpenCV can be conveniently used in an Android
application. This
[learning path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/android_opencv_kleidicv/)
can be checked as an example, but to use a locally built AAR package the project
dependencies need to be updated, changing the
`implementation("org.opencv:opencv:4.11.0")` line of `app/build.gradle.kts` to
`implementation(files("</full/path/of/opencv_java_shared_4.13.0.aar>"))`. (The
version number in the AAR package's filename might require an update if it was
built from a different OpenCV version.)

## Prerequisites
This how-to assumes a headless x86 based Ubuntu 24.04 machine to execute the
build on.

## Build steps

1. Install required packages:

   ```
   #!/bin/bash

   sudo apt install openjdk-17-jdk openjdk-17-jre cmake ninja-build
   ```

2. Install the Android SDK and Build Tools. These are part of Android Studio,
   but in a headless environment Command line tools can be downloaded from
   [developer.android.com/studio](https://developer.android.com/studio). Once
   downloaded follow these steps:

   ```
   #!/bin/bash

   mkdir android-sdk
   unzip commandlinetools-linux-*.zip -d android-sdk
   ./android-sdk/cmdline-tools/bin/sdkmanager --install "build-tools;36.0.0" --sdk_root=$(readlink -f ./android-sdk)
   <accept the Terms and Conditions>
   ```

3. Install and extract the Android NDK from
   [developer.android.com/ndk/downloads/](https://developer.android.com/ndk/downloads/).

4. Check out KleidiCV.

5. Follow the steps described at
   [Get and patch OpenCV source](#get-and-patch-opencv-source).

6. Build the OpenCV Android SDK with OpenCV's `build-sdk.sh` script.

   The script requires a configuration file, so create `kleidicv-config.py`
   with the following content to build OpenCV with the checked out KleidiCV
   source instead of the KleidiCV version shipped with OpenCV:

   ```
   ABIs = [
    ABI("3", "arm64-v8a", None, 21, cmake_vars=dict(KLEIDICV_SOURCE_PATH='</full/path/of/KleidiCV>')),
   ]
   ```

   The content is based on `opencv/platforms/android/ndk-18-api-level-21.config.py`
   from OpenCV 4.13 and some description of this configuration file format can
   be found
   [here](https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build#advanced-opencv-build-options).

   Then OpenCV's `build-sdk.sh` can be called:

   ```
   #!/bin/bash

   # 1st argument is the output folder
   # 2nd argument is the patched OpenCV source path
   python3 opencv/platforms/android/build_sdk.py \
           ./OpenCV_Android_SDK_build \
           ./opencv \
           --ndk_path=</full/path/of/android-ndk> \
           --sdk_path=</full/path/of/android-sdk> \
           --config=</full/path/of/kleidicv_oob_config.py> \
           --no_samples_build
   ```

8. Create the AAR package with OpenCV's `build_static_aar.py` script:

   ```
   #!/bin/bash

   mkdir OpenCV_AAR_build
   cd OpenCV_AAR_build

   ANDROID_HOME=</full/path/of/android-sdk> \
   python3 </full/path/of/patched/OpenCV>/platforms/android/build_java_shared_aar.py \
           </full/path/of/OpenCV_Android_SDK_build>/OpenCV-android-sdk \
           --ndk_location=</full/path/of/android-ndk>
   ```

At this point the AAR package should be available at
`OpenCV_AAR_build/outputs/opencv_java_shared_4.13.0.aar`. (The version number
in the AAR package's filename might be different based on the OpenCV version
used for the build.)

# Building KleidiCV for AArch64 Linux

If your build machine is AArch64 Linux then you can build KleidiCV with the system toolchain.

Cross-building from another architecture to AArch64 in the standard way
defined by your toolchain is also possible. See your toolchain
documentation for cross-building instructions.

## Prerequisites
To successfully build and compile this project for AArch64 Linux, you'll need the following tools:
- Either GCC 8.4 or higher, or Clang 12 or higher.
- Binutils
- [CMake](https://cmake.org) 3.16 or higher.
- `make`
- `patch`

## Building OpenCV & KleidiCV for AArch64 Linux

This is similar to building for Android, just with fewer settings required.
First [get and patch the OpenCV source](#get-and-patch-opencv-source).
Then build:
```
cmake \
  -S /path/to/opencv \
  -B build-opencv-linux \
  -DWITH_KLEIDICV=ON \
  -DKLEIDICV_SOURCE_PATH=/path/to/kleidicv
cmake --build build-opencv-linux --parallel
```

## Build KleidiCV standalone for AArch64 Linux

This is similar to building for Android, just with fewer settings required.
```
cmake -S /path/to/kleidicv -B build-kleidicv-linux
cmake --build build-kleidicv-linux --parallel
```

# KleidiCV Build Options

In addition to the standard CMake settings, KleidiCV behaviour can be
modified at build time via the following CMake options:
- `KLEIDICV_BENCHMARK` - Enable building KleidiCV benchmarks. The benchmarks use Google Benchmark which will be downloaded automatically. Off by default.
- `KLEIDICV_ENABLE_SME` - Enable Scalable Matrix Extension and Streaming Scalable Vector Extension code paths. Off by default while the [ACLE SME specification is in beta](https://github.com/ARM-software/acle/blob/main/main/acle.md#sme-language-extensions-and-intrinsics).
- `KLEIDICV_ENABLE_SME2` - Enable Scalable Matrix Extension 2 and Streaming Scalable Vector Extension code paths. Off by default while the [ACLE SME specification is in beta](https://github.com/ARM-software/acle/blob/main/main/acle.md#sme-language-extensions-and-intrinsics).
  - `KLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS` - Limit Scalable Matrix Extension 2 code paths to cases where it is expected to provide a benefit over SME code paths. On by default. Has no effect if `KLEIDICV_ENABLE_SME2` is off.
  - If enabled, enables `KLEIDICV_ENABLE_SME` as well.
- `KLEIDICV_ENABLE_SVE2` - Enable Scalable Vector Extension 2 code paths. This is on by default for some popular compilers known to support SVE2 but otherwise off by default.
  - `KLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS` - Limit Scalable Vector Extension 2 code paths to cases where it is expected to provide a benefit over the Neon code path. On by default. Has no effect if `KLEIDICV_ENABLE_SVE2` is off.
