from typing import List

import cv2
import ntcore

from backend.config.ConfigSource import ConfigSource, FileConfigSource, LocalConfigSource, NTConfigSource
from backend.config.config import ConfigStore, LocalConfig, CameraConfig, RemoteConfig
from backend.pipeline.Capture import AVFoundationMjpegCapture

class Wrapper:
    def __init__(self):
        self._nt: ntcore.NetworkTableInstance = ntcore.NetworkTableInstance.getDefault()

        loc_config: LocalConfig = LocalConfig()
        loc_config_source: ConfigSource = LocalConfigSource()
        loc_config_source.update(loc_config)
        self._configs: List[ConfigStore] = []

        self._nt.setServer(loc_config.server_ip)
        self._nt.startClient4(loc_config.device_id)

        self._capture: AVFoundationMjpegCapture = AVFoundationMjpegCapture()
        for cam in self._capture.getCameras():
            camera_config_source: ConfigSource = FileConfigSource(cam.uniqueID())
            remote_config_source: ConfigSource = NTConfigSource()
            config = ConfigStore(loc_config, CameraConfig(), RemoteConfig(), camera_config_source)
            camera_config_source.update(config)
            remote_config_source.update(config)
            self._configs.append(config)

    def get_raw_frame(self, cam_name_supplier: callable):
        try:
            while True: 
                ret, img = self._capture.get_frame(cam_name_supplier(), self._configs)
                if not ret:
                    with open("static/Camera_Lost.jpg", "rb") as f:
                        frame_bytes = f.read()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    _, frame_buf = cv2.imencode('.jpg', img)
                    frame_bytes = frame_buf.tobytes()
                    
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            print(f"Stopping stream for {cam_name_supplier()}")
    
    def get_cameras(self) -> List[str]:
        cams: List[str] = []
        for config in self._configs:
            cams.append(config.camera_config.camera_name)
        return cams
    
    def update_config(self, index: int, obj: str, value) -> bool:
        if 0 <= index < len(self._configs):
            self._configs[index].camera_config_source.save(obj, value)
            setattr(self._configs[index].camera_config, obj, value)
            return True
        return False