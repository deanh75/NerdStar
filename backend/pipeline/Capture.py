# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import dataclasses
import gc
import subprocess
from typing import Dict, List, Tuple

import cv2
from backend.config.config import ConfigStore
from backend.NerdAVF import NerdAVF


class Capture:
    """Interface for receiving camera frames."""

    def __init__(self) -> None:
        raise NotImplementedError

    # def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
    #     """Return the next frame from the camera."""
    #     raise NotImplementedError
    
    def get_frame(self, cam_id: str, configs: List[ConfigStore]) -> Tuple[bool, cv2.Mat]:
        """Return the next frame from the camera."""
        raise NotImplementedError
    
    def getCameras() -> list[AVFoundation.AVCaptureDevice]: 
        """Return a list of available cameras."""
        raise NotImplementedError
    
    def stop(self) -> None: 
        raise NotImplementedError

    @classmethod
    def _config_changed(cls, config_a: ConfigStore, config_b: ConfigStore) -> bool:
        if config_a == None and config_b == None:
            return False
        if config_a == None or config_b == None:
            return True

        camera_a = config_a.camera_config
        camera_b = config_b.camera_config

        return (
            camera_a.camera_id != camera_b.camera_id
            or camera_a.camera_resolution_width != camera_b.camera_resolution_width
            or camera_a.camera_resolution_height != camera_b.camera_resolution_height
            or camera_a.camera_auto_white_balance != camera_b.camera_auto_white_balance
            or camera_a.camera_white_balance != camera_b.camera_white_balance
            or camera_a.camera_auto_exposure != camera_b.camera_auto_exposure
            or camera_a.camera_exposure != camera_b.camera_exposure
            or camera_a.camera_gain != camera_b.camera_gain
            or camera_a.camera_max_fps != camera_b.camera_max_fps
        )
        
class AVFoundationMjpegCapture(Capture):
    """ "Read from camera with OpenCV and AVFoundation through an Mjeg server."""

    def __init__(self) -> None:
        self._videos: Dict[str, cv2.VideoCapture] = {}
        self._last_configs: Dict[str, ConfigStore] = {}
        pass

    def get_frame(self, cam_name: str, configs: List[ConfigStore]) -> Tuple[bool, cv2.Mat]:
        config_store: ConfigStore = next((c for c in configs if c.camera_config.camera_name == cam_name), None)
        last_config: ConfigStore = self._last_configs[cam_name] if cam_name in self._last_configs else None
        video: cv2.VideoCapture = self._videos[cam_name] if self._videos != None and cam_name in self._videos else None

        if config_store == None:
            print(f"No config found for camera {cam_name}")
            return False, None

        if video != None and self._config_changed(last_config, config_store):
            print("Restarting capture session")
            video.release()
            video = None
            self._videos[cam_name] = video

        if video == None:
            if config_store.camera_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                devices = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
                devices.sort(key=lambda x: x.uniqueID())
                for index, device in enumerate(devices):
                    if device.uniqueID() == config_store.camera_config.camera_id:
                        s: str = device.uniqueID()
                        s = s.replace("0x", "")
                        result = subprocess.run([
                            "./backend/camera-ctrl/build/camera-ctrl",
                            "0x" + s[8:12],
                            "0x" + s[12:16],
                            "0x" + s[0:8],
                            str(int(config_store.camera_config.camera_auto_white_balance)),
                            str(config_store.camera_config.camera_white_balance),
                            str(int(config_store.camera_config.camera_auto_exposure)),
                            str(config_store.camera_config.camera_exposure),
                            str(config_store.camera_config.camera_gain)
                        ], capture_output=True, check=False, text=True)
                        print("RETURN CODE:", result.returncode)
                        print("STDOUT:\n", result.stdout)
                        print("STDERR:\n", result.stderr)

                        video = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
                        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        video.set(cv2.CAP_PROP_FRAME_WIDTH, config_store.camera_config.camera_resolution_width)
                        video.set(cv2.CAP_PROP_FRAME_HEIGHT, config_store.camera_config.camera_resolution_height)
                        video.set(cv2.CAP_PROP_FPS, config_store.camera_config.camera_max_fps)

                        self._videos[cam_name] = video
                        break

        last_config = ConfigStore(
            dataclasses.replace(config_store.local_config), dataclasses.replace(config_store.camera_config), 
            dataclasses.replace(config_store.remote_config), config_store.camera_config_source
        )
        self._last_configs[cam_name] = last_config

        if video == None:
            if config_store.camera_config.camera_id != "":
                print("Camera not found, restarting")
            return False, None
        else:
            video.grab()
            retval, image = video.retrieve()
            if not retval:
                print("Capture session failed, restarting")
                video.release()
                video = None  # Force reconnect
                self._videos[cam_name] = video
            return retval, image
        
    def getCameras(self) -> list[AVFoundation.AVCaptureDevice]: 
        cameras = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
        cameras.sort(key=lambda x: x.uniqueID())
        return cameras
        
    def stop(self) -> None:
        try:
            if self._videos != None:
                for video in self._videos.values():
                    video.release()
                self._videos.clear()
                self._last_configs.clear()
                gc.collect()
        except Exception as e:
            print("Stop error:", e)

# CAPTURE_IMPLS = {
#     "": Capture,
#     "avfoundation": AVFoundationCapture,
#     "avfoundationmjpeg": AVFoundationMjpegCapture,
# }