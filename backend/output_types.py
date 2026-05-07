# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from dataclasses import dataclass
from typing import List
from backend.vision_types import CameraPoseObservation, ObjDetectObservation

@dataclass
class ApriltagOutput:
    fps: int
    pose_observation: CameraPoseObservation

    def to_dict(self):
        return {
            "fps": self.fps,
            "pose_observation": self.pose_observation.to_dict() if self.pose_observation else None
        }

@dataclass
class ObjDetectionOutput:
    fps: int
    observations: List[ObjDetectObservation]

    def to_dict(self):
        return {
            "fps": self.fps,
            "observation_num": len(self.observations),
        }