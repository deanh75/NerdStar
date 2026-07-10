import asyncio

import cv2
import time
import av
from aiortc import VideoStreamTrack
from aiortc.codecs import h264

from backend.wrapper import Wrapper

h264.MAX_BITRATE = 5_000_000       # raise ceiling above our target
h264.DEFAULT_BITRATE = 3_500_000   # 5 Mbps target
h264.MIN_BITRATE = 500_000         # floor, unchanged from aiortc default
h264.MAX_FRAME_RATE = 120

class CameraTrack(VideoStreamTrack):
    def __init__(self, wrapper: Wrapper, camera_supplier: callable):
        super().__init__()
        self.wrapper = wrapper
        self.camera_supplier = camera_supplier

        self.last_cam_name = None
        self.index = -1

        self.frame_interval = 1 / 70
        self.last_time = 0

        self.fallback_frame = cv2.imread("static/Cam_Lost.png")

    async def recv(self):
        now = time.time()
        elapsed = now - self.last_time
        wait = self.frame_interval - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_time = time.time()

        pts, time_base = await self.next_timestamp()

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

        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame