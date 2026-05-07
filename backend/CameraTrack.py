import asyncio

import cv2
import time
import av
from aiortc import VideoStreamTrack

from backend.wrapper import Wrapper

class CameraTrack(VideoStreamTrack):
    def __init__(self, wrapper: Wrapper, camera_supplier: callable):
        super().__init__()
        self.wrapper = wrapper
        self.camera_supplier = camera_supplier

        self.last_cam_name = None
        self.index = -1

        self.frame_interval = 1 / 120
        self.last_time = 0

        self.fallback_frame = cv2.imread("static/Cam_Lost.png")

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        now = time.time()

        if now - self.last_time < self.frame_interval:
            return await self.recv()

        self.last_time = time.time()

        cam_name = self.camera_supplier()
        if cam_name != self.last_cam_name:
            self.index = next(
                (i for i, cfg in enumerate(self.wrapper._configs) 
                if cfg.camera_config.camera_name == cam_name), 
                -1
            )
            self.last_cam_name = cam_name

        try:
            if self.index < 0 or self.index >= len(self.wrapper._configs):
                raise ValueError("Invalid camera index")
                
            frame = self.wrapper.latest_frames[self.index]

            if frame is None:
                raise ValueError("No frame available for this camera")
        
        except Exception:
            frame = self.fallback_frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame