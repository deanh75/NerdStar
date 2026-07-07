
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "ASM"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_ASM
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib/aarch64/SgemmKernelNeon.S" "/home/nerdstar/NerdStar/backend/opencv/opencv_build/3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/aarch64/SgemmKernelNeon.S.o"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib/aarch64/SgemvKernelNeon.S" "/home/nerdstar/NerdStar/backend/opencv/opencv_build/3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/aarch64/SgemvKernelNeon.S.o"
  )
set(CMAKE_ASM_COMPILER_ID "GNU")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_ASM
  "BUILD_MLAS_NO_ONNXRUNTIME=1"
  "MLAS_GEMM_ONLY=1"
  "MLAS_OPENCV_THREADING=1"
  "_USE_MATH_DEFINES"
  "__STDC_CONSTANT_MACROS"
  "__STDC_FORMAT_MACROS"
  "__STDC_LIMIT_MACROS"
  )

# The include file search paths:
set(CMAKE_ASM_TARGET_INCLUDE_PATH
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/dlpack/include"
  "3rdparty/kleidicv/kleidicv-26.03/kleidicv_thread/include"
  "3rdparty/kleidicv/kleidicv-26.03/kleidicv/include"
  "hal/kleidicv/include"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/inc"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/modules/core/include"
  "."
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/__/__/modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp.o" "gcc" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/__/__/modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp.o.d"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib/compute.cpp" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/compute.cpp.o" "gcc" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/compute.cpp.o.d"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib/flashattn.cpp" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/flashattn.cpp.o" "gcc" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/flashattn.cpp.o.d"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib/platform.cpp" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/platform.cpp.o" "gcc" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/platform.cpp.o.d"
  "/home/nerdstar/NerdStar/backend/opencv/opencv/3rdparty/mlas/lib/sgemm.cpp" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/sgemm.cpp.o" "gcc" "3rdparty/mlas/CMakeFiles/opencv_dnn_mlas.dir/lib/sgemm.cpp.o.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
