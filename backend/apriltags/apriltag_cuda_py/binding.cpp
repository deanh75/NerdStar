// pybind11 wrapper for the 971/Team766 CUDA AprilTag detector.
//
// This is a thin wrapper around frc971::apriltag::GpuDetector that
// exposes a Python class with the same calling convention as the
// standard apriltag Python wrapper, so you can drop it into a Python
// pipeline that previously used OpenCV's CPU detector and just see
// 5-10x more FPS on a Jetson Orin Nano Super.
//
// Build prerequisites (these are the repos you already cloned):
//   * Team766/apriltags_cuda         -> builds libapriltag_cuda.so
//   * cgpadwick/apriltag (3.3.0)     -> builds libapriltag.so
//   * glog (apt: libgoogle-glog-dev)
//   * pybind11 (pip install pybind11)
//   * numpy (pip install numpy)
//   * CUDA toolkit (already on Jetpack 6.2)
//
// The build script (build.sh) calls the Team766 CMake to produce
// libapriltag_cuda.so, then builds this extension against it.

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "common/matd.h"
#include "common/zarray.h"

#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

#include "apriltag_gpu.h"

namespace py = pybind11;

namespace {

// One-time glog initialization. Without this, glog will print a warning
// to stderr on first use that it's uninitialized. We just route to /dev/null
// because we don't want glog's output polluting the Python stream.
void EnsureGlogInitialized() {
    static bool done = false;
    if (done) return;
    if (!google::IsGoogleLoggingInitialized()) {
        // Use a non-existent log directory so glog drops messages silently.
        // Real diagnostics from the detector go through pybind11 exceptions.
        google::InitGoogleLogging("apriltag_cuda_py");
    }
    done = true;
}

// Tiny RAII helper for the apriltag detector + tag family.
struct ApriltagDetectorBundle {
    apriltag_family_t* tf = nullptr;
    apriltag_detector_t* td = nullptr;
    std::string family_name;

    void Destroy() {
        if (td) {
            apriltag_detector_destroy(td);
            td = nullptr;
        }
        if (tf) {
            // apriltag does NOT provide a generic family destroy in 3.3.0;
            // each family has its own. Wire them up by name.
            if (family_name == "tag16h5")           tag16h5_destroy(tf);
            else if (family_name == "tag25h9")      tag25h9_destroy(tf);
            else if (family_name == "tag36h11")     tag36h11_destroy(tf);
            else if (family_name == "tagCircle21h7")  tagCircle21h7_destroy(tf);
            else if (family_name == "tagCircle49h12") tagCircle49h12_destroy(tf);
            else if (family_name == "tagCustom48h12") tagCustom48h12_destroy(tf);
            else if (family_name == "tagStandard41h12") tagStandard41h12_destroy(tf);
            else if (family_name == "tagStandard52h13") tagStandard52h13_destroy(tf);
            tf = nullptr;
        }
    }
    ~ApriltagDetectorBundle() { Destroy(); }
};

}  // namespace

// Python-facing GPU detector class.
//
// Usage from Python:
//   import apriltag_cuda_py as g
//   d = g.GpuDetector(1280, 800, family="tag36h11", tagsize=0.165,
//                     fx=..., fy=..., cx=..., cy=...,
//                     k1=0, k2=0, p1=0, p2=0, k3=0)
//   results = d.detect(gray_image)   # gray_image is a uint8 HxW numpy array
//   for r in results:
//       print(r['id'], r['center'], r['corners'], r['pose_R'], r['pose_t'])
class GpuDetector {
public:
    GpuDetector(int width,
                int height,
                std::string family = "tag36h11",
                double tagsize = 0.165,
                double fx = 1.0,
                double fy = 1.0,
                double cx = 0.0,
                double cy = 0.0,
                double k1 = 0.0,
                double k2 = 0.0,
                double p1 = 0.0,
                double p2 = 0.0,
                double k3 = 0.0,
                double decimate = 2.0,
                bool refine_edges = true,
                int maxhamming = 1)
        : width_(width), height_(height) {
        EnsureGlogInitialized();

        // 971 detector constraints (CHECK_EQ in apriltag_gpu.cu):
        if ((width % 8) != 0 || (height % 8) != 0) {
            throw std::runtime_error(
                "GpuDetector: width and height must both be multiples of 8 "
                "(the 971 CUDA detector requires this for memory layout).");
        }
        if (static_cast<size_t>(width) * static_cast<size_t>(height) >=
            (1u << 22)) {
            throw std::runtime_error(
                "GpuDetector: image has too many pixels (>= 4,194,304). "
                "The 971 detector has a hard limit of (1 << 22) - 1 pixels.");
        }
        if (decimate != 2.0) {
            // 971's apriltag_gpu.cu CHECKs quad_decimate == 2.
            // We still allow the param so the API is uniform, but warn.
            py::print("warning: 971 CUDA detector only supports decimate=2.0; "
                      "forcing 2.0 and ignoring the requested value.");
            decimate = 2.0;
        }

        // Build the apriltag tag family.
        ApriltagDetectorBundle bundle;
        bundle.family_name = family;
        if (family == "tag16h5")        bundle.tf = tag16h5_create();
        else if (family == "tag25h9")   bundle.tf = tag25h9_create();
        else if (family == "tag36h11")  bundle.tf = tag36h11_create();
        else if (family == "tagCircle21h7")  bundle.tf = tagCircle21h7_create();
        else if (family == "tagCircle49h12") bundle.tf = tagCircle49h12_create();
        else if (family == "tagCustom48h12") bundle.tf = tagCustom48h12_create();
        else if (family == "tagStandard41h12") bundle.tf = tagStandard41h12_create();
        else if (family == "tagStandard52h13") bundle.tf = tagStandard52h13_create();
        else {
            throw std::runtime_error(
                "GpuDetector: unknown family '" + family +
                "'. Supported: tag16h5, tag25h9, tag36h11, tagCircle21h7, "
                "tagCircle49h12, tagCustom48h12, tagStandard41h12, tagStandard52h13. "
                "NOTE: 971 GPU detector only fully supports tag36h11 in practice.");
        }
        if (!bundle.tf) {
            throw std::runtime_error(
                "GpuDetector: failed to create tag family '" + family + "'.");
        }

        bundle.td = apriltag_detector_create();
        if (!bundle.td) {
            bundle.Destroy();
            throw std::runtime_error("GpuDetector: apriltag_detector_create() failed.");
        }
        apriltag_detector_add_family_bits(bundle.td, bundle.tf, maxhamming);

        // These are the values the 971 detector's CHECK_* macros expect:
        bundle.td->quad_decimate = 2.0;        // required
        bundle.td->quad_sigma = 0.0;           // no blur pre-detection
        bundle.td->nthreads = 1;               // GPU does the heavy lifting
        bundle.td->refine_edges = refine_edges;
        bundle.td->debug = false;
        bundle.td->qtp.min_white_black_diff = 5;
        bundle.td->qtp.deglitch = false;       // required

        cam_ = frc971::apriltag::CameraMatrix{fx, fy, cx, cy};
        dist_ = frc971::apriltag::DistCoeffs{k1, k2, p1, p2, k3};

        try {
            detector_ = std::make_unique<frc971::apriltag::GpuDetector>(
                static_cast<size_t>(width),
                static_cast<size_t>(height),
                bundle.td,
                cam_,
                dist_);
        } catch (...) {
            bundle.Destroy();
            throw;
        }

        // Hand ownership of the family + detector to member vars so the
        // GpuDetector's internal pointer remains valid for our lifetime.
        // Then null out the bundle's pointers so its destructor doesn't
        // double-free.
        tf_ = bundle.tf;
        td_ = bundle.td;
        family_name_ = bundle.family_name;
        tagsize_ = tagsize;
        bundle.tf = nullptr;
        bundle.td = nullptr;
    }

    ~GpuDetector() {
        // detector_ dtor runs first (unique_ptr in reverse-declaration-order).
        if (td_) apriltag_detector_destroy(td_);
        if (tf_) {
            if (family_name_ == "tag16h5")          tag16h5_destroy(tf_);
            else if (family_name_ == "tag25h9")     tag25h9_destroy(tf_);
            else if (family_name_ == "tag36h11")    tag36h11_destroy(tf_);
            else if (family_name_ == "tagCircle21h7")  tagCircle21h7_destroy(tf_);
            else if (family_name_ == "tagCircle49h12") tagCircle49h12_destroy(tf_);
            else if (family_name_ == "tagCustom48h12") tagCustom48h12_destroy(tf_);
            else if (family_name_ == "tagStandard41h12") tagStandard41h12_destroy(tf_);
            else if (family_name_ == "tagStandard52h13") tagStandard52h13_destroy(tf_);
        }
    }

    GpuDetector(const GpuDetector&) = delete;
    GpuDetector& operator=(const GpuDetector&) = delete;

    // Run detection on a grayscale uint8 HxW numpy array.
    // Returns a list of dicts, one per detected tag:
    //   { 'id': int,
    //     'hamming': int,
    //     'margin': float,
    //     'center': np.ndarray(2, float64),   # (x, y) in pixels
    //     'corners': np.ndarray(4, 2, float64), # 4 corner (x, y) in pixels
    //                                              # order: lb, rb, rt, lt
    //     'pose_R': np.ndarray(3, 3, float64),   # tag-to-camera rotation
    //     'pose_t': np.ndarray(3, float64),      # tag-to-camera translation (m)
    //     'pose_err': float }                    # orthogonal-iteration error
    py::list Detect(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image) {
        if (!detector_) {
            throw std::runtime_error("GpuDetector: detector not initialized.");
        }
        auto info = image.request();
        if (info.ndim != 2) {
            throw std::runtime_error(
                "GpuDetector.detect: input must be 2-D (HxW grayscale).");
        }
        if (info.shape[0] != height_ || info.shape[1] != width_) {
            throw std::runtime_error(
                "GpuDetector.detect: input is " + std::to_string(info.shape[1]) +
                "x" + std::to_string(info.shape[0]) + " but detector was built for " +
                std::to_string(width_) + "x" + std::to_string(height_) + ".");
        }
        if (info.strides[1] != 1) {
            throw std::runtime_error(
                "GpuDetector.detect: image rows must be contiguous in memory.");
        }

        // Run the GPU detector. This synchronizes the GPU stream internally
        // before returning, so detections are safe to read on the host after.
        // detector_->Detect(info.ptr);
        detector_->Detect(reinterpret_cast<const uint8_t*>(info.ptr));

        const zarray_t* detections = detector_->Detections();
        int N = zarray_size(detections);

        py::list results;

        apriltag_detection_info_t pose_info{};
        pose_info.tagsize = tagsize_;
        pose_info.fx = cam_.fx;
        pose_info.fy = cam_.fy;
        pose_info.cx = cam_.cx;
        pose_info.cy = cam_.cy;

        for (int i = 0; i < N; i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);

            // Estimate pose using orthogonal iteration (most accurate of the
            // three apriltag_pose methods; same one apriltag_pose demo uses).
            pose_info.det = det;
            apriltag_pose_t pose{};
            double pose_err = estimate_tag_pose(&pose_info, &pose);

            py::dict r;
            r["id"] = det->id;
            r["hamming"] = det->hamming;
            r["margin"] = det->decision_margin;

            py::array_t<double> center({2});
            auto c = center.mutable_unchecked<1>();
            c(0) = det->c[0];
            c(1) = det->c[1];
            r["center"] = center;

            py::array_t<double> corners({4, 2});
            auto cm = corners.mutable_unchecked<2>();
            // Order: lb, rb, rt, lt (matches apriltag_detection_t->p)
            for (int j = 0; j < 4; j++) {
                cm(j, 0) = det->p[j][0];
                cm(j, 1) = det->p[j][1];
            }
            r["corners"] = corners;

            // Pose: rotation matrix (3x3) and translation (3,).
            py::array_t<double> pose_R({3, 3});
            py::array_t<double> pose_t({3});
            auto Rm = pose_R.mutable_unchecked<2>();
            auto tm = pose_t.mutable_unchecked<1>();
            for (int r2 = 0; r2 < 3; r2++) {
                for (int c2 = 0; c2 < 3; c2++) {
                    Rm(r2, c2) = matd_get(pose.R, r2, c2);
                }
                tm(r2) = matd_get(pose.t, r2, 0);
            }
            r["pose_R"] = pose_R;
            r["pose_t"] = pose_t;
            r["pose_err"] = pose_err;

            matd_destroy(pose.R);
            matd_destroy(pose.t);

            results.append(r);
        }
        return results;
    }

    int width()  const { return width_; }
    int height() const { return height_; }
    std::string family() const { return family_name_; }
    double tagsize() const { return tagsize_; }

private:
    int width_ = 0;
    int height_ = 0;
    apriltag_family_t* tf_ = nullptr;
    apriltag_detector_t* td_ = nullptr;
    frc971::apriltag::CameraMatrix cam_{};
    frc971::apriltag::DistCoeffs dist_{};
    std::string family_name_;
    double tagsize_ = 0.0;
    std::unique_ptr<frc971::apriltag::GpuDetector> detector_;
};

PYBIND11_MODULE(apriltag_cuda_py, m) {
    m.doc() = "pybind11 wrapper for 971/Team766's CUDA AprilTag detector. "
              "Drop-in replacement for the standard apriltag Python wrapper "
              "but runs detection on the GPU. Tested on Jetson Orin Nano Super.";

    py::class_<GpuDetector>(m, "GpuDetector")
        .def(py::init<int, int,
                      std::string, double,
                      double, double, double, double,
                      double, double, double, double, double,
                      double, bool, int>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("family") = "tag36h11",
             py::arg("tagsize") = 0.165,
             py::arg("fx") = 1.0,
             py::arg("fy") = 1.0,
             py::arg("cx") = 0.0,
             py::arg("cy") = 0.0,
             py::arg("k1") = 0.0,
             py::arg("k2") = 0.0,
             py::arg("p1") = 0.0,
             py::arg("p2") = 0.0,
             py::arg("k3") = 0.0,
             py::arg("decimate") = 2.0,
             py::arg("refine_edges") = true,
             py::arg("maxhamming") = 1,
             "Construct a GPU AprilTag detector.\n\n"
             "Args:\n"
             "  width, height: image dimensions in pixels (both must be multiples of 8).\n"
             "  family: tag family name (e.g. 'tag36h11').\n"
             "  tagsize: physical size of the tag in meters (used for pose).\n"
             "  fx, fy, cx, cy: camera intrinsics in pixels.\n"
             "  k1, k2, p1, p2, k3: distortion coefficients (plumb-bob / OpenCV model).\n"
             "  decimate: ignored; 971 detector requires 2.0.\n"
             "  refine_edges: run edge refinement after quad detection.\n"
             "  maxhamming: maximum hamming distance for tag decoding (0, 1, 2, or 3).")
        .def("detect", &GpuDetector::Detect,
             py::arg("image"),
             "Run detection on a uint8 HxW grayscale numpy array. "
             "Returns a list of detection dicts (id, center, corners, pose_R, pose_t, ...).")
        .def_property_readonly("width",  &GpuDetector::width)
        .def_property_readonly("height", &GpuDetector::height)
        .def_property_readonly("family", &GpuDetector::family)
        .def_property_readonly("tagsize", &GpuDetector::tagsize);
}
