# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from dataclasses import dataclass
from typing import List, Union

import numpy
import numpy.typing
from wpimath.geometry import *
from wpiutil.wpistruct import make_wpistruct
import struct as _struct


@dataclass(frozen=True)
class FiducialImageObservation:
    tag_id: int
    corners: numpy.typing.NDArray[numpy.float64]


@dataclass(frozen=True)
class FiducialPoseObservation:
    tag_id: int
    pose_0: Pose3d
    error_0: float
    pose_1: Pose3d
    error_1: float


@dataclass(frozen=True)
class CameraPoseObservation:
    tag_ids: List[int]
    pose_0: Pose3d
    error_0: float
    pose_1: Union[Pose3d, None]
    error_1: Union[float, None]

    @staticmethod
    def _pose_to_dict(pose: Pose3d):
        return {
            "x": float(pose.translation().X()),
            "y": float(pose.translation().Y()),
            "z": float(pose.translation().Z()),
            "qw": float(pose.rotation().getQuaternion().W()),
            "qx": float(pose.rotation().getQuaternion().X()),
            "qy": float(pose.rotation().getQuaternion().Y()),
            "qz": float(pose.rotation().getQuaternion().Z()),
        }

    def to_dict(self):
        return {
            "tag_ids": [int(t) for t in self.tag_ids],
            "pose_0": self._pose_to_dict(self.pose_0) if self.pose_0 else None,
            "pose_1": self._pose_to_dict(self.pose_1) if self.pose_1 is not None else None,
        }


@dataclass(frozen=True)
class TagAngleObservation:
    tag_id: int
    corners: numpy.typing.NDArray[numpy.float64]
    distance: float


@dataclass(frozen=True)
class ObjDetectObservation:
    obj_class: int
    confidence: float
    corner_angles: numpy.typing.NDArray[numpy.float64]
    corner_pixels: numpy.typing.NDArray[numpy.float64]

@dataclass
class TimestampedObservation:
    observation: CameraPoseObservation
    timestamp: int
    index: int

@dataclass
class PoseEstimate:
    pose: Pose3d
    xy_std_dev: float
    theta_std_dev: float

@make_wpistruct(name="RobotPoseEstimation")
@dataclass(frozen=True)
class RobotPoseEstimation:
    pose: Pose2d
    timestamp: int
    xy_std_dev: float
    theta_std_dev: float
