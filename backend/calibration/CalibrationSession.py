# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import datetime
import os
from typing import List, Union

import cv2
import numpy


class CalibrationSession:
    _all_charuco_corners: List[numpy.ndarray] = []
    _all_charuco_ids: List[numpy.ndarray] = []
    _imsize = None

    def __init__(self) -> None:
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._charuco_board = cv2.aruco.CharucoBoard((12, 9), 0.030, 0.023, self._aruco_dict)
        self._calib_flags = (
            cv2.CALIB_USE_QR |
            cv2.CALIB_FIX_ASPECT_RATIO |
            cv2.CALIB_FIX_PRINCIPAL_POINT |
            cv2.CALIB_ZERO_TANGENT_DIST |
            cv2.CALIB_FIX_K1 |
            cv2.CALIB_FIX_K2 |
            cv2.CALIB_FIX_K3 |
            cv2.CALIB_FIX_K4 |
            cv2.CALIB_FIX_K5 |
            cv2.CALIB_FIX_K6
        )

    def process_frame(self, image: cv2.Mat) -> None:
        # Get image size
        if self._imsize == None:
            self._imsize = (image.shape[1], image.shape[0])

        # Detect tags
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, self._aruco_dict, parameters=self._aruco_params)
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners)

            # Find Charuco corners
            (retval, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self._charuco_board
            )
            if self._is_good_frame(charuco_corners, charuco_ids):
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

                self._all_charuco_corners.append(charuco_corners)
                self._all_charuco_ids.append(charuco_ids)
            else: 
                print("Frame rejected: Not enough corners detected")

    def _is_good_frame(self, charuco_corners, charuco_ids) -> bool:
        if charuco_corners is None or charuco_ids is None:
            return False
        if len(charuco_corners) < 12:
            return False
        return True
    
    def select_best_frames(self, max_frames=350):
        """
        Robust Charuco frame selector:
        - Keeps ALL frames if dataset <= max_frames
        - Otherwise selects best + diverse subset
        """

        scored = []

        for c, i in zip(self._all_charuco_corners, self._all_charuco_ids):
            if c is None or i is None:
                continue

            pts = numpy.asarray(c, dtype=numpy.float32).reshape(-1, 2)

            if len(pts) < 6:
                continue

            # -----------------------------
            # QUALITY SCORE (2D only, stable)
            # -----------------------------
            num_points = len(pts)

            x_std = numpy.std(pts[:, 0])
            y_std = numpy.std(pts[:, 1])
            spread = x_std + y_std

            min_xy = numpy.min(pts, axis=0)
            max_xy = numpy.max(pts, axis=0)
            area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])

            score = (
                num_points * 1.0 +
                spread * 0.01 +
                area * 0.0001
            )

            # pseudo pose proxy (no solvePnP)
            center = numpy.mean(pts, axis=0)

            scored.append((score, c, i, center))

        if len(scored) == 0:
            return [], []

        scored.sort(key=lambda x: x[0], reverse=True)

        # -----------------------------
        # CASE 1: small dataset → use ALL
        # -----------------------------
        if len(scored) <= max_frames:
            corners = [
                numpy.asarray(c, dtype=numpy.float32).reshape(-1, 1, 2)
                for _, c, _, _ in scored
            ]
            ids = [
                numpy.asarray(i, dtype=numpy.int32).reshape(-1, 1)
                for _, _, i, _ in scored
            ]
            return corners, ids

        # -----------------------------
        # CASE 2: large dataset → select diverse best
        # -----------------------------
        selected_corners = []
        selected_ids = []
        selected_centers = []

        for score, c, i, center in scored:
            if len(selected_corners) >= max_frames:
                break

            # diversity check (image-space only, stable)
            too_similar = False
            for sc in selected_centers:
                center_norm = center / numpy.array([self._imsize[1], self._imsize[0]])
                sc_norm = sc / numpy.array([self._imsize[1], self._imsize[0]])
                if numpy.linalg.norm(center_norm - sc_norm) < 0.008: # lower number loosens the constraint
                    too_similar = True
                    break

            if too_similar:
                continue

            selected_corners.append(
                numpy.asarray(c, dtype=numpy.float32).reshape(-1, 1, 2)
            )
            selected_ids.append(
                numpy.asarray(i, dtype=numpy.int32).reshape(-1, 1)
            )
            selected_centers.append(center)

        return selected_corners, selected_ids

    def finish(self, cam_id: str) -> Union[bool, str]:
        if len(self._all_charuco_corners) == 0 or len(self._all_charuco_ids) == 0:
            return "ERROR: No calibration data"
        
        if len(self._all_charuco_corners) < 5:
            return "ERROR: Not enough data for calibration"
        
        if self._imsize is None:
            return "ERROR: Image size not set"

        if len(self._all_charuco_corners) != len(self._all_charuco_ids):
            return "ERROR: Corners/IDs mismatch"
        
        # Filter out bad frames
        valid_corners = []
        valid_ids = []

        valid_corners, valid_ids = self.select_best_frames(max_frames=500)

        if len(valid_corners) < 250:
            return "ERROR: Not enough valid frames after filtering"
        
        path = f"backend/data/{cam_id}_calibration.yml"

        if os.path.exists(path):
            os.remove(path)

        try:
            retval, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                valid_corners, valid_ids, self._charuco_board, self._imsize, None, None, flags=self._calib_flags
            )
        except cv2.error as e:
            return "OpenCV calibration error:" + e

        # Validate result
        if retval is None or retval <= 0:
            return "ERROR: Calibration returned invalid retval"

        if camera_matrix is None or distortion_coefficients is None:
            return "ERROR: Calibration returned empty matrices"

        calibration_store = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        calibration_store.write("calibration_date", str(datetime.datetime.now()))
        calibration_store.write("camera_resolution", self._imsize)
        calibration_store.write("camera_matrix", camera_matrix)
        calibration_store.write("distortion_coefficients", distortion_coefficients)
        calibration_store.release()

        self._all_charuco_corners.clear()
        self._all_charuco_ids.clear()
        self._imsize = None

        return True