# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue
from typing import List, Tuple, Union

import cv2
from backend.config.config import ConfigStore, LocalConfig
from backend.pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from backend.pipeline.FiducialDetector import ArucoFiducialDetector
from backend.vision_types import CameraPoseObservation, FiducialImageObservation


def apriltag_worker(
    q_in: queue.Queue[Tuple[float, cv2.Mat, ConfigStore, LocalConfig]],
    q_out: queue.Queue[
        Tuple[
            float,
            List[FiducialImageObservation],
            Union[CameraPoseObservation, None],
        ]
    ],
):
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
    camera_pose_estimator = MultiTargetCameraPoseEstimator()

    while True:
        sample = q_in.get()
        timestamp: float = sample[0]
        image: cv2.Mat = sample[1]
        config: ConfigStore = sample[2]
        local_config: LocalConfig = sample[3]

        image_observations = fiducial_detector.detect_fiducials(image, config)
        camera_pose_observation = camera_pose_estimator.solve_camera_pose(
            image_observations, config, local_config
        )

        q_out.put(
            (timestamp, image_observations, camera_pose_observation)
        )
