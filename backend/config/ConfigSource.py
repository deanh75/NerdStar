# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import json
import shutil

import cv2
import ntcore
import numpy
from wpimath.geometry import Rotation3d, Translation3d, Pose3d
from backend.config.config import ConfigStore, LocalConfig


class ConfigSource:
    def update(self, config_store: ConfigStore) -> None:
        raise NotImplementedError
    
    def save(self, obj: str, value) -> None:
        raise NotImplementedError

class LocalConfigSource(ConfigSource):
    def __init__(self) -> None:
        self._local_config_filename = "backend/data/local.json"
        pass

    def update(self, local_config: LocalConfig) -> None:
        with open(self._local_config_filename, "r") as local_config_file:
            local_config_data = json.loads(local_config_file.read())
            local_config.device_id = local_config_data["device_id"]
            local_config.team_number = local_config_data["team_number"]
            local_config.obj_detect_model = local_config_data["obj_detect_model"]
            local_config.obj_detect_max_fps = local_config_data["obj_detect_max_fps"]
            local_config.video_folder = local_config_data["video_folder"]
            local_config.video_framerate = local_config_data["video_framerate"]
            local_config.fiducial_size_m = local_config_data["fiducial_size_m"]
            local_config.should_record = local_config_data["should_record"]
            local_config.tag_layout_name = local_config_data["tag_layout_name"]
            local_config.load_tag_layout()
            local_config.robot_size_x = local_config_data["robot_size_x"]
            local_config.robot_size_y = local_config_data["robot_size_y"]
            local_config.robot_size_z = local_config_data["robot_size_z"]

    def save(self, obj: str, value) -> None:
        with open(self._local_config_filename, "r") as local_config_file:
            local_config_data = json.loads(local_config_file.read())
            local_config_data[obj] = value

        with open(self._local_config_filename, "w") as local_config_file:
            json.dump(local_config_data, local_config_file, indent=4)

class FileConfigSource(ConfigSource):
    def __init__(self, cam_id: str) -> None:
        self._cam_id = cam_id
        self._cam_config_filename = f"backend/data/{cam_id}_cam.json"
        self._calibration_filename = f"backend/data/{cam_id}_calibration.yml"
        pass

    def update(self, config_store: ConfigStore) -> None:
        # Get config
        try:
            with open(self._cam_config_filename, "r") as cam_config_file:
                cam_config_data = json.loads(cam_config_file.read())
                config_store.camera_config.camera_id = self._cam_id
                cam_config_data["camera_id"] = self._cam_id
                config_store.camera_config.camera_name = cam_config_data["camera_name"]
                config_store.camera_config.camera_max_fps = cam_config_data["camera_max_fps"]
                config_store.camera_config.camera_resolution_width = cam_config_data["camera_resolution_width"]
                config_store.camera_config.camera_resolution_height = cam_config_data["camera_resolution_height"]
                config_store.camera_config.camera_auto_white_balance = cam_config_data["camera_auto_white_balance"]
                config_store.camera_config.camera_white_balance = cam_config_data["camera_white_balance"]
                config_store.camera_config.camera_auto_exposure = cam_config_data["camera_auto_exposure"]
                config_store.camera_config.camera_exposure = cam_config_data["camera_exposure"]
                config_store.camera_config.camera_gain = cam_config_data["camera_gain"]
                config_store.camera_config.apriltags_enable = cam_config_data["apriltags_enable"]
                config_store.camera_config.objdetect_enable = cam_config_data["objdetect_enable"]
                config_store.camera_config.driverCam_enable = cam_config_data["driverCam_enable"]
                config_store.camera_config.process_frames_enable = cam_config_data["process_frames_enable"]
                if cam_config_data["camera_transform"] is not None:
                    config_store.camera_config.camera_transform = self.dict_to_pose3d(
                        cam_config_data["camera_transform"])
                config_store.camera_config.camera_horiz_fov = cam_config_data["camera_horiz_fov"]

            with open(self._cam_config_filename, "w") as cam_config_file:
                json.dump(cam_config_data, cam_config_file, indent=4)

            # Get calibration
            calibration_store = cv2.FileStorage(self._calibration_filename, cv2.FILE_STORAGE_READ)
            camera_matrix = calibration_store.getNode("camera_matrix").mat()
            distortion_coefficients = calibration_store.getNode("distortion_coefficients").mat()
            calibration_store.release()
            if type(camera_matrix) == numpy.ndarray and type(distortion_coefficients) == numpy.ndarray:
                config_store.camera_config.camera_matrix = camera_matrix
                config_store.camera_config.distortion_coefficients = distortion_coefficients
                config_store.camera_config.has_calibration = True
        except: 
            shutil.copy("backend/data/default_cam.json", self._cam_config_filename)
            self.update(config_store)

    def save(self, obj: str, value) -> None:
        with open(self._cam_config_filename, "r") as cam_config_file:
            cam_config_data = json.loads(cam_config_file.read())
            cam_config_data[obj] = value

        with open(self._cam_config_filename, "w") as cam_config_file:
            json.dump(cam_config_data, cam_config_file, indent=4)

    def dict_to_pose3d(self, d: dict) -> Pose3d:
        t = d["translation"]
        r = d["rotation"]
        return Pose3d(
            Translation3d(t["x"], t["y"], t["z"]),
            Rotation3d(r["roll"], r["pitch"], r["yaw"]),
        )


class NTConfigSource(ConfigSource):
    _init_complete: bool = False
    _is_recording_sub: ntcore.BooleanSubscriber
    _timestamp_sub: ntcore.IntegerSubscriber
    _event_name_sub: ntcore.StringSubscriber
    _match_type_sub: ntcore.IntegerSubscriber
    _match_number_sub: ntcore.IntegerSubscriber

    def update(self, config_store: ConfigStore, local_config: LocalConfig) -> None:
        # Initialize subscribers on first call
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + local_config.device_id + "/config"
            )
            self._is_recording_sub = nt_table.getBooleanTopic("is_recording").subscribe(False)
            self._timestamp_sub = nt_table.getIntegerTopic("timestamp").subscribe(0)
            self._event_name_sub = nt_table.getStringTopic("event_name").subscribe("")
            self._match_type_sub = nt_table.getIntegerTopic("match_type").subscribe(0)
            self._match_number_sub = nt_table.getIntegerTopic("match_number").subscribe(0)
            self._init_complete = True

        # Read config data fron NetworkTables
        config_store.remote_config.is_recording = self._is_recording_sub.get()
        config_store.remote_config.timestamp = self._timestamp_sub.get()
        config_store.remote_config.event_name = self._event_name_sub.get()
        config_store.remote_config.match_type = self._match_type_sub.get()
        config_store.remote_config.match_number = self._match_number_sub.get()
