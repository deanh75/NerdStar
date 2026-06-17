# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import math
from typing import List

import ntcore
from backend.config.config import LocalConfig
from backend.vision_types import ObjDetectObservation, RobotPoseEstimation


class OutputPublisher:
    def send_pose_estimation(self, local_config: LocalConfig, pose: RobotPoseEstimation) -> None:
        raise NotImplementedError

    def send_objdetect_fps(self, local_config: LocalConfig, timestamp: float, fps: int) -> None:
        raise NotImplementedError

    def send_objdetect_observation(
        self, local_config: LocalConfig, timestamp: float, observations: List[ObjDetectObservation]
    ) -> None:
        raise NotImplementedError


class NTOutputPublisher(OutputPublisher):
    _nt: ntcore.NetworkTableInstance
    _init_complete: bool = False
    _pose_pub: ntcore.StructPublisher
    _observations_pub: ntcore.DoubleArrayPublisher
    _objdetect_fps_pub: ntcore.IntegerPublisher
    _objdetect_observations_pub: ntcore.DoubleArrayPublisher

    def _check_init(self, local_config: LocalConfig):
        # Initialize publishers on first call
        if not self._init_complete:
            self._init_complete = True
            self._nt = ntcore.NetworkTableInstance.getDefault()
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + local_config.device_id + "/output"
            )
            self._pose_pub = nt_table.getStructTopic("pose_estimation", RobotPoseEstimation).publish(
                ntcore.PubSubOptions(periodic=0.01, sendAll=True, keepDuplicates=True)
            )
            self._objdetect_fps_pub = nt_table.getIntegerTopic("fps_objdetect").publish()
            self._objdetect_observations_pub = nt_table.getDoubleArrayTopic("objdetect_observations").publish(
                ntcore.PubSubOptions(periodic=0.01, sendAll=True, keepDuplicates=True)
            )

    def send_pose_estimation(self, local_config: LocalConfig, pose: RobotPoseEstimation) -> None:
        self._check_init(local_config)
        offset = self._nt.getServerTimeOffset()
        if offset is None:
            return
        pose.timestamp = (pose.timestamp + offset) / 1e6
        self._pose_pub.set(pose)

    def send_objdetect_fps(self, local_config: LocalConfig, timestamp: float, fps: int) -> None:
        self._check_init(local_config)
        self._objdetect_fps_pub.set(fps)

    def send_objdetect_observation(
        self, local_config: LocalConfig, timestamp: float, observations: List[ObjDetectObservation]
    ) -> None:
        self._check_init(local_config)

        observation_data: List[float] = []
        for observation in observations:
            observation_data.append(observation.obj_class)
            observation_data.append(observation.confidence)
            for angle in observation.corner_angles.ravel():
                observation_data.append(angle)

        self._objdetect_observations_pub.set(observation_data, math.floor(timestamp * 1000000))
