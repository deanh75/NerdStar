# setup.py for the apriltag_cuda_py Python extension.
#
# This builds a pybind11 C++ extension that links against:
#   * libapriltag_cuda.so  (from Team766/apriltags_cuda)
#   * libapriltag.so       (from cgpadwick/apriltag @ 3.3.0)
#   * libglog              (from apt: libgoogle-glog-dev)
#   * CUDA runtime         (from Jetpack)
#   * pybind11, numpy      (pip)
#
# The build.sh script at the package root handles building the C++ side
# (Team766's libapriltag_cuda.so + cgpadwick's libapriltag.so) into a
# predictable location, then this setup.py picks them up via env vars
# APRILTAG_CUDA_LIB_DIR and APRILTAG_INCLUDE_DIR.
#
# Typical usage:
#   ./build.sh          # builds the C++ libs and this extension
#   pip install -e .    # editable install into current env
# or:
#   pip install .       # build a wheel in place

import os
import sys
import sysconfig
import subprocess
from pathlib import Path
import pybind11

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class NVCCBuildExt(build_ext):
    def build_extensions(self):
        nvcc = os.environ.get("NVCC", str(CUDA_HOME / "bin" / "nvcc"))
        original_compile = self.compiler._compile

        def nvcc_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cmd = [nvcc, "-x", "cu"] + cc_args + [src, "-o", obj] + extra_postargs
            print("=== NVCC COMPILE ===", " ".join(cmd))
            self.compiler.spawn(cmd)

        def nvcc_link(target_desc, objects, output_filename, output_dir=None,
               libraries=None, library_dirs=None, runtime_library_dirs=None,
               export_symbols=None, debug=0, extra_preargs=None,
               extra_postargs=None, build_temp=None, target_lang=None):

            def to_nvcc_args(args):
                """Convert -Wl,opt1,opt2 style GNU linker args into nvcc-safe -Xlinker form."""
                out = []
                for arg in args or []:
                    if arg.startswith("-Wl,"):
                        for opt in arg[len("-Wl,"):].split(","):
                            out += ["-Xlinker", opt]
                    else:
                        out.append(arg)
                return out

            cmd = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-ccbin", "g++-11"]
            cmd += list(objects)
            for d in (library_dirs or []):
                cmd += [f"-L{d}"]
            for lib in (libraries or []):
                cmd += [f"-l{lib}"]
            cmd += ["-lcudadevrt"]
            cmd += to_nvcc_args(extra_preargs)
            cmd += to_nvcc_args(extra_postargs)
            cmd += ["-o", output_filename]
            print("=== NVCC LINK ===", " ".join(cmd))
            self.compiler.spawn(cmd)

        self.compiler._compile = nvcc_compile
        self.compiler.link = nvcc_link
        try:
            build_ext.build_extensions(self)
        finally:
            self.compiler._compile = original_compile

# ---- Find dependencies via environment variables set by build.sh ----------

APRILTAG_CUDA_LIB_DIR = Path(
    os.environ.get("APRILTAG_CUDA_LIB_DIR", str(Path(__file__).parent / "apriltag_cuda_install" / "lib"))
)
APRILTAG_INCLUDE_DIR = Path(
    os.environ.get("APRILTAG_INCLUDE_DIR", "/usr/local/include")
)
# The 971 detector's headers live in <apriltag_gpu.h>, <apriltag_utils.h>,
# etc. The cgpadwick apriltag lives in include/apriltag/. We add both.
APRILTAG_SRC_DIR = Path(
    os.environ.get("APRILTAG_SRC_DIR", str(Path.home() / "apriltags_cuda_src"))
)
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-11.8"))
CUB_INCLUDE = Path("/usr/local/cuda-11.8/targets/aarch64-linux/include")
assert (CUB_INCLUDE / "cub").exists(), f"cub headers not found under {CUB_INCLUDE}"

# ---- Compile the extension -------------------------------------------------

ext = Extension(
    "apriltag_cuda_py",
    sources=[str(Path(__file__).parent / "apriltag_cuda_py/binding.cpp")],
    include_dirs=[
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "apriltag_cuda_install", "include"),
        str(APRILTAG_INCLUDE_DIR / "apriltag"),  # for apriltag.h, apriltag_pose.h
        str(APRILTAG_INCLUDE_DIR),               # for apriltag_gpu.h etc.
        str(APRILTAG_SRC_DIR),                   # for apriltag_gpu.h etc.
        f"{sysconfig.get_path('include')}",      # for Python.h
        pybind11.get_include(),
        str(CUDA_HOME / "targets" / "aarch64-linux" / "include"),
        str(CUB_INCLUDE),
    ],
    library_dirs=[
        str(APRILTAG_CUDA_LIB_DIR),
        str(CUDA_HOME / "lib64"),
        # str(CUDA_HOME_11 / "targets" / "aarch64-linux" /"lib"),
    ],
    libraries=[
        "apriltag_cuda",   # 971/Team766 GPU detector
        "apriltag",        # cgpadwick apriltag 3.3.0
        "glog",
        "cudart",
        "cudadevrt",
        "cublas",
        "pthread",
    ],
    extra_compile_args=[
        "-O3",
        "-std=c++17",
        "-ccbin", "g++-11",
        "-rdc=true",
        "-Xcompiler", "-fPIC",
        "-Xcompiler", "-w",
        # "-Xcompiler", "-march=armv8-a",
        "-DNDEBUG",
        # "-allow-unsupported-compiler"
    ],
    extra_link_args=[
        # rpath so the .so finds libapriltag_cuda.so and libapriltag.so at
        # runtime even if the user hasn't run ldconfig.
        f"-Wl,-rpath,{APRILTAG_CUDA_LIB_DIR}",
    ],
    language="c++",
)

setup(
    name="apriltag_cuda_py",
    version="0.1.0",
    description="Python bindings for 971/Team766's CUDA AprilTag detector (Jetson).",
    long_description=Path(__file__).parent.joinpath("README.md").read_text() if (Path(__file__).parent / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="You",
    packages=find_packages(),
    # package_dir={"": "src"},
    ext_modules=[ext],
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy", "opencv-python"],
    cmdclass={"build_ext": NVCCBuildExt},
)
