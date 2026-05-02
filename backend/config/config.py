# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from dataclasses import dataclass

import numpy
import numpy.typing


@dataclass
class LocalConfig:
    device_id: str = ""
    server_ip: str = ""
    device_ip: str = ""
    obj_detect_model: str = ""
    obj_detect_max_fps: int = -1
    video_folder: str = ""
    video_framerate: int = 25
    fiducial_size_m: float = 0 
    tag_layout: any = None
    should_record: bool = False

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
    apriltags_stream_port: int = 8000 # TODO: Need?
    objdetect_stream_port: int = 8001 # TODO: Need?
    apriltags_enable: bool = False
    objdetect_enable: bool = False
    driverCam_enable: bool = False
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
    local_config: LocalConfig
    camera_config: CameraConfig
    remote_config: RemoteConfig
    camera_config_source: any
