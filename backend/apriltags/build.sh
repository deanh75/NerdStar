#!/usr/bin/env bash
# build.sh — end-to-end build of the Python wrapper for the 971/Team766
# CUDA AprilTag detector, for a Jetson Orin Nano Super running Jetpack 6.x.
#
# What this does:
#   1. Installs apt deps (glgog, libapriltag dev, OpenCV dev, CUDA).
#   2. Clones cgpadwick/apriltag @ 3.3.0 (if not already present) and builds
#      libapriltag.so into a known prefix.
#   3. Uses your existing Team766/apriltags_cuda checkout (you already
#      cloned and built it) and installs libapriltag_cuda.so + headers into
#      the same prefix.
#   4. Builds the pybind11 Python extension in this directory.
#
# Assumptions:
#   * You're on the Orin Nano Super with Jetpack 6.1 or 6.2 installed
#     (sudo nvpmodel -m 0; jtop should show Jetpack 6.x and CUDA 12.x).
#   * Python 3.10+ with pip, numpy, opencv-python, pybind11 already
#     pip-installed.
#   * You have already cloned Team766/apriltags_cuda and built it at least
#     once following the README in that repo. The build/ directory should
#     contain libapriltag_cuda.so.
#
# Usage:
#   cd ~/apriltag_cuda_py
#   ./build.sh
#   pip install -e .
#   python -c "import apriltag_cuda_py; print('ok')"

set -euo pipefail

# --- Configuration ---------------------------------------------------------

# Where everything ends up. Change if you want a different prefix.
HOME="/home/nerdstar/NerdStar/backend/apriltags"
PREFIX="${HOME}/apriltag_cuda_install"
mkdir -p "${PREFIX}"

# Where the cgpadwick apriltag source goes.
APRILTAG_SRC_DIR="${HOME}/apriltag-src"

# Where your Team766 checkout is. If you cloned it elsewhere, change this.
TEAM766_SRC_DIR="${HOME}/apriltags_cuda"

# CUDA arch for the Orin Nano / Orin Nano Super: SM 8.7.
CUDA_ARCH="87"

# Number of build jobs.
JOBS="$(nproc)"

echo "=========================================="
echo "apriltag_cuda_py build"
echo "  PREFIX=${PREFIX}"
echo "  APRILTAG_SRC_DIR=${APRILTAG_SRC_DIR}"
echo "  TEAM766_SRC_DIR=${TEAM766_SRC_DIR}"
echo "  CUDA_ARCH=${CUDA_ARCH}"
echo "=========================================="

# --- 1. System dependencies -------------------------------------------------

echo ""
echo "[1/5] Installing apt dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libopencv-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-pip \
    zlib1g-dev

# Jetpack should already have CUDA. Verify.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found in PATH. Did you install Jetpack's CUDA?"
    echo "       Try: ls /usr/local/cuda/bin/nvcc"
    exit 1
fi
echo "    nvcc: $(nvcc --version | tail -1)"

# --- 2. Python deps --------------------------------------------------------

echo ""
# echo "[2/5] Installing Python dependencies..."
# pip install --user --upgrade pip
# pip install --user numpy opencv-python pybind11

# --- 3. Build cgpadwick apriltag 3.3.0 (libapriltag.so) --------------------

echo ""
echo "[3/5] Building cgpadwick/apriltag 3.3.0 -> libapriltag.so"
if [ ! -d "${APRILTAG_SRC_DIR}" ]; then
    git clone --branch 3.3.0 --depth 1 \
        https://github.com/cgpadwick/apriltag.git "${APRILTAG_SRC_DIR}"
fi

cd "${APRILTAG_SRC_DIR}"
cmake -B build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_PYTHON_WRAPPER=OFF \
    -DBUILD_SHARED_LIBS=ON
cmake --build build --parallel "${JOBS}"
cmake --install build
echo "    libapriltag.so installed to ${PREFIX}/lib/"

# --- 4. Build the 971/Team766 CUDA detector --------------------------------

echo ""
echo "[4/5] Building Team766/apriltags_cuda -> libapriltag_cuda.so"
if [ ! -d "${TEAM766_SRC_DIR}" ]; then
    echo "ERROR: ${TEAM766_SRC_DIR} not found."
    echo "       You said you already cloned and built it. Either point the"
    echo "       TEAM766_SRC_DIR env var to its location, or:"
    echo "         git clone https://github.com/Team766/apriltags_cuda.git ${TEAM766_SRC_DIR}"
    echo "       (The Team766 build is heavy: it pulls in OpenCV, WPILib, etc."
    echo "        Plan for 30-60 minutes on a fresh checkout.)"
    exit 1
fi

cd "${TEAM766_SRC_DIR}"

# Build only the library target to avoid pulling in the networktables/JSON
# test binaries that we don't need for the Python wrapper.
cmake -B build -GNinja \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release
    # -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    # -DCMAKE_PREFIX_PATH="${PREFIX}" \
    # -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4
cmake --build build --target apriltag_cuda --parallel "${JOBS}"
echo "    libapriltag_cuda.so built (in build/)."
echo "    Copying it to ${PREFIX}/lib/ ..."
mkdir -p "${PREFIX}/lib"
cp -f build/libapriltag_cuda.a "${PREFIX}/lib/"

# Also copy the headers we'll need to compile against. Team766 doesn't
# install them by default.
mkdir -p "${PREFIX}/include"
cp -f src/apriltag_gpu.h   "${PREFIX}/include/"
cp -f src/apriltag_utils.h "${PREFIX}/include/"
cp -f src/cuda_frc971.h    "${PREFIX}/include/"
cp -f src/gpu_image.h      "${PREFIX}/include/"
cp -f src/line_fit_filter.h "${PREFIX}/include/"
cp -f src/points.h         "${PREFIX}/include/"
cp -f src/threshold.h      "${PREFIX}/include/"
echo "    Headers installed to ${PREFIX}/include/"

# --- 5. Build the pybind11 Python extension -------------------------------

echo ""
echo "[5/5] Building apriltag_cuda_py Python extension..."
cd "${PREFIX}/.."
# We're somewhere in the user's home; let them know where to go next.
# The actual extension build happens below.
APRILTAG_CUDA_PY_DIR="${APRILTAG_CUDA_PY_DIR:-${HOME}/apriltag_cuda_py}"
if [ ! -d "${APRILTAG_CUDA_PY_DIR}" ]; then
    echo "ERROR: ${APRILTAG_CUDA_PY_DIR} not found."
    echo "       Clone or copy the apriltag_cuda_py source there, or set"
    echo "       APRILTAG_CUDA_PY_DIR to its location."
    exit 1
fi
cd "${APRILTAG_CUDA_PY_DIR}"

export APRILTAG_CUDA_LIB_DIR="${PREFIX}/lib"
export APRILTAG_INCLUDE_DIR="${PREFIX}/include"
export APRILTAG_SRC_DIR="${TEAM766_SRC_DIR}/src"
export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"

pip install --user -e .

echo ""
echo "=========================================="
echo "Build complete."
echo ""
echo "Quick test:"
echo "  python3 -c 'import apriltag_cuda_py; print(\"apriltag_cuda_py loaded:\", apriltag_cuda_py.GpuDetector.__doc__[:80])'"
echo ""
echo "If you ran into a 'libapriltag_cuda.so not found' error at import time,"
echo "run:  export LD_LIBRARY_PATH=${PREFIX}/lib:\$LD_LIBRARY_PATH"
echo "or add that line to your ~/.bashrc."
echo "=========================================="
