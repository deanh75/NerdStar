"""
example.py - End-to-end example for apriltag_cuda_py.

Reads frames from a V4L2 camera (or any OpenCV-supported source), runs
CUDA-accelerated AprilTag detection, and prints the detections.

You should see ~120-180 FPS on a single camera at 1280x800 on a Jetson
Orin Nano Super.

Run it with:
    python3 example.py [--camera 0] [--width 1280] [--height 800] \
                       [--tagsize 0.165] [--fx 905.5] [--fy 907.9] \
                       [--cx 609.9] [--cy 352.7]
"""
import argparse
import time

import cv2
import numpy as np

import apriltag_cuda_py as g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0,
                   help="OpenCV camera index (default 0).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--fps", type=int, default=120,
                   help="Requested FPS from V4L2 (best effort).")
    p.add_argument("--family", default="tag36h11",
                   help="AprilTag family (must be tag36h11 for FRC 2026 REBUILT).")
    p.add_argument("--tagsize", type=float, default=0.165,
                   help="Physical tag edge length in meters.")
    p.add_argument("--fx", type=float, default=905.5)
    p.add_argument("--fy", type=float, default=907.9)
    p.add_argument("--cx", type=float, default=609.9)
    p.add_argument("--cy", type=float, default=352.7)
    p.add_argument("--k1", type=float, default=0.0)
    p.add_argument("--k2", type=float, default=0.0)
    p.add_argument("--p1", type=float, default=0.0)
    p.add_argument("--p2", type=float, default=0.0)
    p.add_argument("--k3", type=float, default=0.0)
    p.add_argument("--show", action="store_true",
                   help="Show the live video with detected tag overlays.")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print FPS every N frames.")
    args = p.parse_args()

    # ---- Build the GPU detector (one-time setup) --------------------------
    print(f"Building GPU detector for {args.width}x{args.height} {args.family}...")
    t0 = time.perf_counter()
    detector = g.GpuDetector(
        width=args.width,
        height=args.height,
        family=args.family,
        tagsize=args.tagsize,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        k1=args.k1, k2=args.k2, p1=args.p1, p2=args.p2, k3=args.k3,
    )
    print(f"Detector ready in {time.perf_counter() - t0:.2f}s")

    # ---- Open the camera --------------------------------------------------
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open camera {args.camera}.")

    # Set MJPG so we can actually hit 120fps. The Arducam 120fps cams need
    # MJPG; with YUYV they top out at ~30fps no matter what you ask for.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Some Arducam drivers ignore the FPS request. Verify what we got:
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.1f} fps (requested {args.fps})")
    if (actual_w, actual_h) != (args.width, args.height):
        print(f"WARNING: camera returned {actual_w}x{actual_h}, detector was built "
              f"for {args.width}x{args.height}. Detection will fail.")
        cap.release()
        return

    # ---- Main loop --------------------------------------------------------
    frame_count = 0
    t_start = time.perf_counter()
    t_log = t_start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # Arducam MJPG frames come in as BGR. Convert to grayscale.
            # If you have a monochrome camera, frame may already be 1-channel
            # and cvtColor will be a no-op.
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # THE GPU DETECTOR CALL.
            results = detector.detect(gray)

            # ---- Display ----------------------------------------------------
            if args.show and frame.ndim == 3:
                for r in results:
                    # r['corners'] is (4, 2) float64 in [lb, rb, rt, lt] order.
                    pts = r["corners"].astype(int).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], isClosed=True,
                                  color=(0, 255, 0), thickness=2)
                    cx, cy = r["center"].astype(int)
                    cv2.putText(frame, f"id {r['id']}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("apriltag_cuda_py", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ---- FPS log ---------------------------------------------------
            frame_count += 1
            if frame_count % args.log_every == 0:
                now = time.perf_counter()
                fps = args.log_every / (now - t_log)
                t_log = now
                avg = frame_count / (now - t_start)
                print(f"  {fps:6.1f} FPS  (avg {avg:6.1f})  "
                      f"{len(results):2d} tags last frame")

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - t_start
        print(f"\nTotal: {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count / elapsed:.1f} FPS overall)")
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
