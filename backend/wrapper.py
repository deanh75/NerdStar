import queue
import threading
import time
from typing import List

from cscore import CameraServer
import cv2
import ntcore
from starlette.responses import ContentStream

from backend.calibration.CalibrationSession import CalibrationSession
from backend.config.ConfigSource import ConfigSource, FileConfigSource, LocalConfigSource, NTConfigSource
from backend.config.config import ConfigStore, LocalConfig, CameraConfig, RemoteConfig
from backend.output.OutputPublisher import NTOutputPublisher, OutputPublisher
from backend.output.VideoWriter import FFmpegVideoWriter, VideoWriter
from backend.output.overlay_util import overlay_obj_detect_observation
from backend.output_types import ApriltagOutput, ObjDetectionOutput
from backend.pipeline.Capture import AVFoundationMjpegCapture
from backend.vision_types import FiducialImageObservation, ObjDetectObservation
from backend.workers.apriltag_worker import apriltag_worker
from backend.workers.objdetect_worker import objdetect_worker

class Wrapper:
    def __init__(self):
        self._nt: ntcore.NetworkTableInstance = ntcore.NetworkTableInstance.getDefault()

        self.local_config: LocalConfig = LocalConfig()
        self.local_config_source: ConfigSource = LocalConfigSource()
        self.local_config_source.update(self.local_config)

        self._configs: List[ConfigStore] = []

        self.setup_nt()

        self._capture: AVFoundationMjpegCapture = AVFoundationMjpegCapture()
        for cam in self._capture.getCameras():
            camera_config_source: ConfigSource = FileConfigSource(cam.uniqueID)
            remote_config_source: ConfigSource = NTConfigSource()
            config = ConfigStore(CameraConfig(), RemoteConfig(), camera_config_source, remote_config_source)
            camera_config_source.update(config)
            remote_config_source.update(config, self.local_config)
            self._configs.append(config)

        self.apriltag_lock = threading.Lock()
        self.obj_lock = threading.Lock()
        self.calib_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        self.latest_frames: list[cv2.Mat] = [None] * len(self._configs)

        self.calib_done = None

        self.output_apriltag: list[ApriltagOutput] = [None] * len(self._configs)
        self.output_objdetect: list[ObjDetectionOutput] = [None] * len(self._configs)

    def setup_nt(self):
        team_number = self.local_config.team_number
        am = team_number % 100
        te = team_number // 100
        ip = f"10.{te}.{am}.2"
        self._nt.setServer(ip)
        self._nt.startClient4(self.local_config.device_id)

    def restart_nt(self):
        self._nt.stopClient()
        self.setup_nt()

    def get_name(self):
        return self.local_config.device_id
    
    def update_local_settings(self, obj: str, value) -> bool:
        self.local_config_source.save(obj, value)
        setattr(self.local_config, obj, value)
        if obj == "tag_layout_name":
            self.local_config.load_tag_layout()
        return True

    def get_selected_model(self):
        return self.local_config.obj_detect_model
    
    def get_selected_layout(self):
        return self.local_config.tag_layout_name
    
    def get_local_settings(self):
        return {
            "team_number": self.local_config.team_number,
            "device_id": self.local_config.device_id,
            "obj_detect_max_fps": self.local_config.obj_detect_max_fps,
            "video_framerate": self.local_config.video_framerate,
            "fiducial_size_m": self.local_config.fiducial_size_m,
            "should_record": self.local_config.should_record
        }    
               
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
        if 0 <= index < len(self._configs):
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
    
    def get_config_value(self, index: int, obj: str):
        if 0 <= index < len(self._configs):
            return getattr(self._configs[index].camera_config, obj, None)
        return None
    
    def get_done(self):
        with self.calib_lock:
            return self.calib_done
        
    def get_apriltag_data(self, cam_supplier: callable) -> dict[str, any]:
        index = next((i for i, cam in enumerate(self._configs) if cam.camera_config.camera_name == cam_supplier()), -1)
    
        if 0 <= index < len(self.output_apriltag):
            if self.output_apriltag[index] is not None:
                with self.apriltag_lock:
                    return self.output_apriltag[index].to_dict()
        return {}
    
    def get_obj_data(self, cam_supplier: callable) -> dict[str, any]:
        index = next((i for i, cam in enumerate(self._configs) if cam.camera_config.camera_name == cam_supplier()), -1)

        if 0 <= index < len(self.output_objdetect):
            if self.output_objdetect[index] is not None:
                with self.obj_lock:
                    return self.output_objdetect[index].to_dict()
        return {}

    def start_backend(self, index: int):
        if self._configs[index].camera_config.apriltags_enable:
            apriltag_worker_in = queue.Queue(maxsize=1)
            apriltag_worker_out = queue.Queue(maxsize=1)
            apriltag_work = threading.Thread(
                target=apriltag_worker,
                args=(apriltag_worker_in, apriltag_worker_out),
                daemon=True,
            )
            apriltag_work.start()

        if self._configs[index].camera_config.objdetect_enable:
            objdetect_worker_in = queue.Queue(maxsize=1)
            objdetect_worker_out = queue.Queue(maxsize=1)
            objdetect_work = threading.Thread(
                target=objdetect_worker,
                args=(objdetect_worker_in, objdetect_worker_out),
                daemon=True,
            )
            objdetect_work.start()

        output_publisher: OutputPublisher = NTOutputPublisher()
        calib_session = CalibrationSession()
        video_writer: VideoWriter = FFmpegVideoWriter()
        apriltags_frame_count = 0
        apriltags_last_print = 0
        objdetect_last_frame_time = 0
        objdetect_frame_count = 0
        objdetect_last_print = 0
        was_calibrating = False
        was_recording = False
        was_streaming = False
        last_image_observations: List[FiducialImageObservation] = []
        last_objdetect_observations: List[ObjDetectObservation] = []
        video_frame_cache: List[cv2.Mat] = []
        while True:
            self._configs[index].remote_config_source.update(self._configs[index], self.local_config)
            success, image = self._capture.get_frame(self._configs[index])
            timestamp = time.time()

            # Start and stop recording
            should_record = (
                success
                and self.local_config.should_record
                and self._configs[index].remote_config.is_recording
                and self._configs[index].camera_config.camera_resolution_width > 0
                and self._configs[index].camera_config.camera_resolution_height > 0
                and self._configs[index].remote_config.timestamp > 0
            )
            if should_record and not was_recording:
                print("Starting recording")
                video_writer.start(self._configs[index], self.local_config, len(image.shape) == 2)
            elif not should_record and was_recording:
                print("Stopping recording")
                video_writer.stop()
            was_recording = should_record

            # Exit if no frame
            if not success:
                print("No frame available")
                continue

            if self._configs[index].camera_config.driverCam_enable:
                if not was_streaming:
                    cs = CameraServer.putVideo(
                        "Driver Cam " + self._configs[index].camera_config.camera_name, 
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
                with self.calib_lock:
                    self.calib_done = calib_session.finish(self._configs[index].camera_config.camera_id)
                if self.calib_done:
                    self._configs[index].camera_config_source.update(self._configs[index])
                    was_calibrating = False
                else:
                    was_calibrating = False
                    with self.calib_lock:
                        self.calib_done = None

            elif self._configs[index].camera_config.has_calibration:
                # AprilTag pipeline
                if self._configs[index].camera_config.apriltags_enable:
                    try:
                        apriltag_worker_in.put((timestamp, image, self._configs[index], self.local_config), block=False)
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

                        # Measure FPS
                        apriltags_frame_count += 1
                        if time.time() - apriltags_last_print > 1:
                            apriltags_last_print = time.time()
                            print("Running AprilTag pipeline at", apriltags_frame_count, "fps")
                            with self.apriltag_lock:
                                self.output_apriltag[index] = ApriltagOutput(
                                    fps=apriltags_frame_count,
                                    pose_observation=pose_observation
                                )
                            apriltags_frame_count = 0

                # Object detection pipeline
                if self._configs[index].camera_config.objdetect_enable:
                    # Apply FPS limit for object detection
                    if self.local_config.obj_detect_max_fps < 0 or (timestamp - objdetect_last_frame_time) >= (1.0 / self.local_config.obj_detect_max_fps):
                        objdetect_last_frame_time = timestamp
                        try:
                            objdetect_worker_in.put((timestamp, image, self._configs[index], self.local_config), block=False)
                        except:  # No space in queue
                            pass
                    try:
                        timestamp_out, observations = objdetect_worker_out.get(block=False)
                    except:  # No new frames
                        pass
                    else:
                        # Publish observation
                        output_publisher.send_objdetect_observation(self.local_config, timestamp_out, observations)

                        # Store last observations
                        last_objdetect_observations = observations

                        # Measure FPS
                        objdetect_frame_count += 1
                        if time.time() - objdetect_last_print > 1:
                            objdetect_last_print = time.time()
                            print("Running object detection pipeline at", objdetect_frame_count, "fps")
                            with self.obj_lock:
                                self.output_objdetect[index] = ObjDetectionOutput(
                                    fps=objdetect_frame_count,
                                    observations=observations
                                )
                            output_publisher.send_objdetect_fps(self.local_config, timestamp, objdetect_frame_count)
                            objdetect_frame_count = 0

                # Save frame to video
                if should_record:
                    if len(video_frame_cache) >= 2:
                        # Delay output by two frames to improve alignment with overlays
                        video_writer.write_frame(
                            timestamp, video_frame_cache.pop(0), last_image_observations, last_objdetect_observations
                        )
                    video_frame_cache.append(image)
                else:
                    video_frame_cache = []

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