# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from typing import List

import cv2
# import apriltag_cuda_py
from backend.config.config import ConfigStore
from backend.vision_types import FiducialImageObservation


class FiducialDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> List[FiducialImageObservation]:
        raise NotImplementedError


class ArucoFiducialDetector(FiducialDetector):
    def __init__(self, dictionary_id) -> None:
        cv2.setNumThreads(6)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_params.aprilTagQuadDecimate = 2.0   # same meaning as pupil-apriltags' quad_decimate
        self._aruco_params.aprilTagQuadSigma = 0.0
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> List[FiducialImageObservation]:
        corners, ids, _ = self._detector.detectMarkers(image)
        if len(corners) == 0:
            return []
        return [FiducialImageObservation(id[0], corner) for id, corner in zip(ids, corners)]

# class cudaFiducialDetector(FiducialDetector):
#     def __init__(self, dictionary):
        