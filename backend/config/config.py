# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from dataclasses import dataclass, field

import numpy
import numpy.typing
import json

from wpimath.geometry import Pose3d, Rotation3d, Translation3d

from backend.config import ConfigSource


@dataclass
class LocalConfig:
    device_id: str = ""
    team_number: int = 0
    obj_detect_model: str = ""
    obj_detect_max_fps: int = -1
    video_folder: str = ""
    video_framerate: int = 25
    fiducial_size_m: float = 0 
    tag_layout_name: str = ""
    tag_layout: any = None
    should_record: bool = False
    robot_size_x: float = 0.86
    robot_size_y: float = 0.86
    robot_size_z: float = 0.25

    def load_tag_layout(self):
        if self.tag_layout_name:
            with open(f"backend/data/layouts/{self.tag_layout_name}", "r") as tag_layout_file:
                self.tag_layout = json.load(tag_layout_file)
                print("Tag layout loaded successfully")

@dataclass
class CameraConfig:
    camera_id: str = ""
    camera_name: str = ""
    camera_max_fps: int = 120
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_auto_white_balance: bool = False
    camera_white_balance: int = 0
    camera_auto_exposure: bool = False
    camera_exposure: int = 0
    camera_gain: int = 0
    apriltags_enable: bool = False
    objdetect_enable: bool = False
    driverCam_enable: bool = False
    process_frames_enable: bool = False
    camera_transform: Pose3d = field(default_factory=lambda: Pose3d(0.3, 0.0, 0.05, Rotation3d(0, 0, 0)))
    camera_horiz_fov: float = 70
    is_calibrating: bool = False
    has_calibration: bool = False
    camera_matrix: numpy.typing.NDArray[numpy.float64] = None
    distortion_coefficients: numpy.typing.NDArray[numpy.float64] = None

@dataclass
class RemoteConfig:
    event_name: str = ""
    match_type: int = 0
    match_number: int = 0
    is_recording: bool = False
    timestamp: int = 0


@dataclass
class ConfigStore:
    camera_config: CameraConfig
    remote_config: RemoteConfig
    camera_config_source: "ConfigSource.FileConfigSource"
    remote_config_source: "ConfigSource.NTConfigSource"
