# apriltag_cuda_py

Python bindings (pybind11) for the 971/Team766 CUDA AprilTag detector.
Drop-in replacement for the standard `apriltag` Python wrapper, but with
detection running on the Jetson GPU instead of the CPU.

Targeted at the **Jetson Orin Nano Super** running Jetpack 6.1 or 6.2.
Real-world performance on that hardware (verified by FRC teams):

* 1280x800, one camera, single tag in view: **~150-200 FPS**
* 1280x800, two cameras simultaneously: **~55 FPS per camera**
* 1600x1304: **240 FPS at 60% GPU utilization** (Austin Schuh / Team 971)

If you were getting 30-40 FPS with OpenCV's CPU detector, this is a 4-7x
improvement with no code changes beyond swapping the import.

## What you get

```python
import apriltag_cuda_py as g

detector = g.GpuDetector(
    width=1280, height=800,
    family="tag36h11", tagsize=0.165,
    fx=905.5, fy=907.9, cx=609.9, cy=352.7,   # your camera intrinsics
    k1=0, k2=0, p1=0, p2=0, k3=0,            # distortion (zeros if calibrated)
)
results = detector.detect(gray_image)   # gray_image is a uint8 HxW numpy array

for r in results:
    print(r['id'], r['center'], r['corners'])
    print(r['pose_R'], r['pose_t'], r['pose_err'])
```

Each `r` is a dict with:

* `id` (int) — the tag ID
* `hamming` (int) — bit errors corrected during decoding
* `margin` (float) — detection confidence
* `center` (np.ndarray, shape (2,)) — pixel center
* `corners` (np.ndarray, shape (4, 2)) — four corner pixels in [lb, rb, rt, lt] order
* `pose_R` (np.ndarray, shape (3, 3)) — tag-to-camera rotation matrix
* `pose_t` (np.ndarray, shape (3,)) — tag-to-camera translation in meters
* `pose_err` (float) — orthogonal-iteration object-space error (lower = better)

## Build

The build script does everything end-to-end:

```bash
cd ~/apriltag_cuda_py
./build.sh
```

What `build.sh` does:

1. Installs apt deps: `libgoogle-glog-dev`, `libopencv-dev`, etc.
2. Clones `cgpadwick/apriltag` @ 3.3.0 and builds `libapriltag.so`.
3. Builds `libapriltag_cuda.so` from your existing Team766 checkout.
4. Builds the pybind11 extension and `pip install -e .`s it.

It puts everything in `~/apriltag_cuda_install/` (configurable via `PREFIX=`).

### Prerequisites on the Jetson

* Jetpack 6.1 or 6.2 (CUDA 12.6+, Jetpack 6.2 is what the FRC teams verify against)
* `sudo nvpmodel -m 0` to put the Orin Nano in 25W MAXN_SUPER mode
* `pip install numpy opencv-python pybind11` (the build script does this too)
* You already cloned `Team766/apriltags_cuda` and built it once with its
  own README. The build script expects it at `~/apriltags_cuda/`.
  Override with `TEAM766_SRC_DIR=/path/to/apriltags_cuda ./build.sh`
  if it's elsewhere.

## Tag family note

The 971 GPU detector's CUDA code paths are written for `tag36h11` only.
The Python wrapper still accepts other family names (so the API matches
the standard apriltag wrapper), but if you pass anything other than
`tag36h11`, the detector will silently fall back to CPU decoding for the
tag-ID step. FRC 2026 REBUILT uses `tag36h11` so this is what you want
anyway.

## Camera intrinsics

`fx, fy, cx, cy` are the standard OpenCV pinhole camera matrix entries
in pixels. `k1, k2, p1, p2, k3` are the plumb-bob (a.k.a. OpenCV
`cv::distCoeffs`) distortion coefficients. If you calibrated with
OpenCV (`cv2.calibrateCamera`) or PhotonVision, the values from those
work directly. The 971 detector uses these for the GPU-side
undistortion that happens before tag decoding.

## Why the 971 detector and not Isaac ROS or PhotonVision

* **Isaac ROS Apriltag** (NVIDIA's official CUDA port) only supports
  `tag36h11` in the GPU backend, requires ROS 2, and does not publish
  Orin Nano benchmark numbers. Heavier.
* **PhotonVision** (FRC's standard vision tool) uses 971's detector
  internally on Jetson (via FRC Team 4143's fork), but it's a full
  Java+Gradle application with a web UI. If you want to run a vision
  pipeline on the same Python process that does the rest of your robot
  code, you don't want PhotonVision.
* **This wrapper** is what you'd write yourself if you wanted "the
  PhotonVision detector" without all the PhotonVision. Same 5-7x speedup,
  same hardware, ~200 lines of C++ instead of a 100MB Java app.

## Files in this package

```
apriltag_cuda_py/
├── setup.py                                  # builds the pybind11 extension
├── build.sh                                  # end-to-end build script
├── example.py                                # runnable camera+detection demo
├── src/apriltag_cuda_py/
│   └── binding.cpp                           # the pybind11 wrapper
└── README.md                                 # this file
```

## Verified against

* Team766/apriltags_cuda (libapriltag_cuda.so)
* cgpadwick/apriltag @ tag 3.3.0 (libapriltag.so)
* pybind11 >= 2.6
* glog (libgoogle-glog-dev on apt)
* OpenCV 4.x (libopencv-dev on apt)
* CUDA 12.6 (Jetpack 6.2)
* Python 3.10
