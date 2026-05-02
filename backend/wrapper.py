import enum
import queue
import threading
import time
from typing import List

from cscore import CameraServer
import cv2
import ntcore

from backend.calibration.CalibrationSession import CalibrationSession
from backend.config.ConfigSource import ConfigSource, FileConfigSource, LocalConfigSource, NTConfigSource
from backend.config.config import ConfigStore, LocalConfig, CameraConfig, RemoteConfig
from backend.output.OutputPublisher import NTOutputPublisher, OutputPublisher
from backend.output.VideoWriter import FFmpegVideoWriter, VideoWriter
from backend.output.overlay_util import overlay_obj_detect_observation
from backend.pipeline.Capture import AVFoundationMjpegCapture
from backend.vision_types import FiducialImageObservation, ObjDetectObservation

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
            camera_config_source: ConfigSource = FileConfigSource(cam.uniqueID)
            remote_config_source: ConfigSource = NTConfigSource()
            config = ConfigStore(loc_config, CameraConfig(), RemoteConfig(), camera_config_source, remote_config_source)
            camera_config_source.update(config)
            remote_config_source.update(config)
            self._configs.append(config)

        self.latest_frames: list[cv2.Mat] = [None] * len(self._configs)
        self.frame_lock = threading.Lock()

    def get_frame(self, cam_name_supplier: callable):
        while True:
            index = next((i for i, cfg in enumerate(self._configs) if cfg.camera_config.camera_name == cam_name_supplier()), -1)
            if index < 0 or index >= len(self._configs):
                with open("static/Cam_Lost.png", "rb") as f:
                    frame_bytes = f.read()
                yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')
                continue
            elif self.latest_frames[index] is None:
                with open("static/Cam_Lost.png", "rb") as f:
                    frame_bytes = f.read()
                yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')
                continue
                
            _, frame_buf = cv2.imencode('.jpg', self.latest_frames[index].copy())
            frame_bytes = frame_buf.tobytes()
                            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
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
    
    def get_camera_settings(self, index: int):
        if index < 0:
            return {
                "apriltags_enable": False,
                "objdetect_enable": False,
                "driverCam_enable": False,
                "process_frames_enable": False,
                "camera_resolution_width": 640,
                "camera_resolution_height": 480,
                "camera_auto_white_balance": False,
                "camera_white_balance": 0,
                "camera_auto_exposure": False,
                "camera_exposure": 0,
                "camera_gain": 0
            }
        elif 0 <= index < len(self._configs):
            config = self._configs[index].camera_config
            return {
                "apriltags_enable": config.apriltags_enable,
                "objdetect_enable": config.objdetect_enable,
                "driverCam_enable": config.driverCam_enable,
                "process_frames_enable": config.process_frames_enable,
                "camera_resolution_width": config.camera_resolution_width,
                "camera_resolution_height": config.camera_resolution_height,
                "camera_auto_white_balance": config.camera_auto_white_balance, 
                "camera_white_balance": config.camera_white_balance,
                "camera_auto_exposure": config.camera_auto_exposure,
                "camera_exposure": config.camera_exposure,
                "camera_gain": config.camera_gain
            }
        return None
    
    def get_config_value(self, index: int, obj: str):
        if 0 <= index < len(self._configs):
            return getattr(self._configs[index].camera_config, obj, None)
        return None
    
    def start_backend(self, index: int):
        if self._configs[index].camera_config.apriltags_enable:
            apriltag_worker_in = queue.Queue(maxsize=1)
            apriltag_worker_out = queue.Queue(maxsize=1)
            apriltag_worker = threading.Thread(
                target=apriltag_worker,
                args=(apriltag_worker_in, apriltag_worker_out),
                daemon=True,
            )
            apriltag_worker.start()

        if self._configs[index].camera_config.objdetect_enable:
            objdetect_worker_in = queue.Queue(maxsize=1)
            objdetect_worker_out = queue.Queue(maxsize=1)
            objdetect_worker = threading.Thread(
                target=objdetect_worker,
                args=(objdetect_worker_in, objdetect_worker_out),
                daemon=True,
            )
            objdetect_worker.start()

        output_publisher: OutputPublisher = NTOutputPublisher()
        calib_session = CalibrationSession()
        video_writer: VideoWriter = FFmpegVideoWriter()
        objdetect_next_frame = -1
        objdetect_frame_count = 0
        objdetect_last_print = 0
        was_calibrating = False
        was_recording = False
        was_streaming = False
        last_image_observations: List[FiducialImageObservation] = []
        last_objdetect_observations: List[ObjDetectObservation] = []
        video_frame_cache: List[cv2.Mat] = []
        while True:
            self._configs[index].remote_config_source.update(self._configs[index])
            success, image = self._capture.get_frame(self._configs[index])
            timestamp = time.time()

            # Start and stop recording
            should_record = (
                success
                and self._configs[index].remote_config.is_recording
                and self._configs[index].camera_config.camera_resolution_width > 0
                and self._configs[index].camera_config.camera_resolution_height > 0
                and self._configs[index].remote_config.timestamp > 0
            )
            if should_record and not was_recording:
                print("Starting recording")
                video_writer.start(self._configs[index], len(image.shape) == 2)
            elif not should_record and was_recording:
                print("Stopping recording")
                video_writer.stop()
            was_recording = should_record

            # Exit if no frame
            if not success:
                print("No frame available")
                # time.sleep(0.5)
                continue

            if self._configs[index].camera_config.driverCam_enable:
                if not was_streaming:
                    cs = CameraServer.putVideo(
                        "Driver Cam" + self._configs[index].camera_config.camera_name, 
                        self._configs[index].camera_config.camera_resolution_width, 
                        self._configs[index].camera_config.camera_resolution_height
                    )
                
                processed_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                cs.putFrame(processed_frame)

            if self._configs[index].camera_config.is_calibrating:
                # Calibration mode
                was_calibrating = True
                calib_session.process_frame(image)
                with self.frame_lock:
                    self.latest_frames[index] = image.copy()

            elif was_calibrating:
                # Just finished calibration, save results
                print("Saving calibration")
                calib_session.finish(self._configs[index].camera_config.camera_id)
                self._configs[index].camera_config_source.update(self._configs[index])
                was_calibrating = False

            elif self._configs[index].camera_config.has_calibration:
                # AprilTag pipeline
                if self._configs[index].camera_config.apriltags_enable:
                    try:
                        apriltag_worker_in.put((timestamp, image, self._configs[index]), block=False)
                    except:  # No space in queue
                        pass
                    try:
                        (
                            timestamp_out,
                            image_observations,
                            pose_observation,
                            tag_angle_observations,
                            demo_pose_observation,
                        ) = apriltag_worker_out.get(block=False)
                    except:  # No new frames
                        pass
                    else:
                        # Publish observation
                        # output_publisher.send_apriltag_observation(
                        #     config, timestamp_out, pose_observation, tag_angle_observations, demo_pose_observation
                        # )

                        # Store last observations
                        last_image_observations = image_observations

                # Object detection pipeline
                if self._configs[index].camera_config.objdetect_enable:
                    # Apply FPS limit for object detection
                    if objdetect_next_frame == -1:
                        objdetect_next_frame = timestamp
                    if self._configs[index].local_config.obj_detect_max_fps < 0 or timestamp > objdetect_next_frame:
                        objdetect_next_frame += 1 / self._configs[index].local_config.obj_detect_max_fps
                        try:
                            objdetect_worker_in.put((timestamp, image, self._configs[index]), block=False)
                        except:  # No space in queue
                            pass
                    try:
                        timestamp_out, observations = objdetect_worker_out.get(block=False)
                    except:  # No new frames
                        pass
                    else:
                        # Publish observation
                        output_publisher.send_objdetect_observation(self._configs[index], timestamp_out, observations)

                        # Store last observations
                        last_objdetect_observations = observations

                        # Measure FPS
                        fps = None
                        objdetect_frame_count += 1
                        if time.time() - objdetect_last_print > 1:
                            objdetect_last_print = time.time()
                            print("Running object detection pipeline at", objdetect_frame_count, "fps")
                            output_publisher.send_objdetect_fps(self._configs[index], timestamp, objdetect_frame_count)
                            objdetect_frame_count = 0

                # Save frame to video
                if self._configs[index].remote_config.is_recording:
                    if len(video_frame_cache) >= 2:
                        # Delay output by two frames to improve alignment with overlays
                        video_writer.write_frame(
                            timestamp, video_frame_cache.pop(0), last_image_observations, last_objdetect_observations
                        )
                    video_frame_cache.append(image)
                else:
                    video_frame_cache = []

            # else:
                # No calibration
                # print("No calibration found")
                # time.sleep(0.5)

            if (self._configs[index].camera_config.process_frames_enable 
                and (self._configs[index].camera_config.apriltags_enable or self._configs[index].camera_config.objdetect_enable)
                and self._configs[index].camera_config.has_calibration):
                img: cv2.Mat = image.copy()
                if self._configs[index].camera_config.apriltags_enable:
                    [overlay_obj_detect_observation(img, x) for x in last_image_observations]
                if self._configs[index].camera_config.objdetect_enable:
                    [overlay_obj_detect_observation(img, x) for x in last_objdetect_observations]
                    
                with self.frame_lock:
                    self.latest_frames[index] = img.copy()
            else:
                with self.frame_lock:
                    self.latest_frames[index] = image.copy()