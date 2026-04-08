# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import dataclasses
import subprocess
import os
from typing import Dict, List, Tuple

import AVFoundation
import cv2
from backend.config.config import ConfigStore


class Capture:
    """Interface for receiving camera frames."""

    def __init__(self) -> None:
        raise NotImplementedError

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Return the next frame from the camera."""
        raise NotImplementedError
    
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
            or camera_a.camera_auto_exposure != camera_b.camera_auto_exposure
            or camera_a.camera_exposure != camera_b.camera_exposure
            or camera_a.camera_iso != camera_b.camera_iso
        )
        
class AVFoundationMjpegCapture(Capture):
    """ "Read from camera with OpenCV and AVFoundation through an Mjeg server."""

    def __init__(self) -> None:
        self._videos: Dict[str, cv2.VideoCapture] = {}
        self._last_configs: List[ConfigStore] = []
        pass

    def get_frame(self, cam_name: str, configs: List[ConfigStore]) -> Tuple[bool, cv2.Mat]:
        config_store: ConfigStore = next((c for c in configs if c.camera_config.camera_name == cam_name), None)
        last_config: ConfigStore = next((c for c in self._last_configs if c.camera_config.camera_name == cam_name), None)
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
                        subprocess.run(
                            [
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../camera-ctrl/CameraControl"),
                                device.uniqueID(),
                                str(config_store.camera_config.camera_exposure),
                                str(config_store.camera_config.camera_iso),
                                str(config_store.camera_config.camera_auto_exposure),
                                str(config_store.camera_config.camera_auto_white_balance),
                            ],
                            check=True,
                        )

                        video = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
                        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        video.set(cv2.CAP_PROP_FRAME_WIDTH, config_store.camera_config.camera_resolution_width)
                        video.set(cv2.CAP_PROP_FRAME_HEIGHT, config_store.camera_config.camera_resolution_height)
                        self._videos[cam_name] = video
                        break

        last_config = ConfigStore(
            dataclasses.replace(config_store.local_config), dataclasses.replace(config_store.camera_config), 
            dataclasses.replace(config_store.remote_config), config_store.camera_config_source
        )
        self._last_configs.append(last_config)

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
        except:
            pass

# CAPTURE_IMPLS = {
#     "": Capture,
#     "avfoundation": AVFoundationCapture,
#     "avfoundationmjpeg": AVFoundationMjpegCapture,
# }