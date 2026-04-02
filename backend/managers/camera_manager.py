import cv2
import time
from threading import Thread

class Camera:
    def __init__(self, cam_id, name):
        self.cap = cv2.VideoCapture(cam_id)
        self.name = name

        self.frame = None
        self.running = True
        self.fps = 0

        # Your default config
        self.config = {
            "camera_id": str(cam_id),
            "camera_name": name,
            "apriltags_enable": "False",
            "objdetect_enable": "False",
            "driverCam_enable": "False",
            "apriltags_stream_port": "8000",
            "objdetect_stream_port": "8001"
        }

        Thread(target=self.update, daemon=True).start()

    def update(self):
        prev = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

                now = time.time()
                self.fps = 1 / (now - prev)
                prev = now

    def get_jpeg(self):
        if self.frame is None:
            return None
        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()


class CameraManager:
    def __init__(self):
        self.cameras = {
            "Cam1": Camera(0, "Cam1")
        }
        self.selected = "Cam1"

    def get_all(self):
        return {
            "cameras": [
                cam.config for cam in self.cameras.values()
            ],
            "selected": self.selected
        }

    def get_camera(self, name):
        return self.cameras.get(name)

    def select(self, name):
        self.selected = name

    def rename(self, old, new):
        if old in self.cameras:
            cam = self.cameras.pop(old)
            cam.name = new
            cam.config["camera_name"] = new
            self.cameras[new] = cam

    def update_config(self, data):
        cam = self.cameras[self.selected]
        cam.config.update(data)

    def get_fps(self, name):
        cam = self.cameras.get(name)
        return cam.fps if cam else 0