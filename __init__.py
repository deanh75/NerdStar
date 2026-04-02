# Copyright (c) 2022-2026 Littleton Robotics
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import argparse
import atexit
import queue
import signal
import sys
import threading
import time
from typing import List, Union

import ntcore
from apriltag_worker import apriltag_worker
from calibration.CalibrationCommandSource import CalibrationCommandSource, NTCalibrationCommandSource
from calibration.CalibrationSession import CalibrationSession
from config.config import ConfigStore, LocalConfig, CameraConfig, RemoteConfig
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from objdetect_worker import objdetect_worker
from output.OutputPublisher import NTOutputPublisher, OutputPublisher
from output.StreamServer import MjpegStreamServer, StreamServer
from output.overlay_util import *
from output.VideoWriter import FFmpegVideoWriter, VideoWriter
from pipeline.Capture import CAPTURE_IMPLS, Capture

from cscore import CameraServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac_config", default="config_default.json")
    parser.add_argument("--camera_config", default="cam_config_default.json")
    parser.add_argument("--calibration", default="calibration_new.yml")
    args = parser.parse_args()

    config = ConfigStore(LocalConfig(), CameraConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource(args.mac_config, args.camera_config, args.calibration)
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()
    local_config_source.update(config)

    capture: Capture = CAPTURE_IMPLS[config.local_config.capture_impl]()
    output_publisher: OutputPublisher = NTOutputPublisher()
    video_writer: VideoWriter = FFmpegVideoWriter()
    calibration_session = CalibrationSession()
    calibration_session_server: Union[StreamServer, None] = None

    if config.camera_config.apriltags_enable:
        apriltag_worker_in = queue.Queue(maxsize=1)
        apriltag_worker_out = queue.Queue(maxsize=1)
        apriltag_worker = threading.Thread(
            target=apriltag_worker,
            args=(apriltag_worker_in, apriltag_worker_out, config.camera_config.apriltags_stream_port),
            daemon=True,
        )
        apriltag_worker.start()

    if config.camera_config.objdetect_enable:
        objdetect_worker_in = queue.Queue(maxsize=1)
        objdetect_worker_out = queue.Queue(maxsize=1)
        objdetect_worker = threading.Thread(
            target=objdetect_worker,
            args=(objdetect_worker_in, objdetect_worker_out, config.camera_config.objdetect_stream_port),
            daemon=True,
        )
        objdetect_worker.start()

    nt: ntcore.NetworkTableInstance = ntcore.NetworkTableInstance.getDefault()
    nt.setServer(config.local_config.server_ip)
    nt.startClient4(config.local_config.device_id + config.camera_config.camera_name)

    def cleanup() -> None:
        print("Cleaning Up And Saving")
        if was_calibrating:
            calibration_session.finish()

        if was_recording:
            video_writer.stop()
        
        capture.stop()
        nt.disconnect()
    
    def _signal_handler(signum, frame) -> None:
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    atexit.register(cleanup)

    apriltags_frame_count = 0
    apriltags_last_print = 0
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
        remote_config_source.update(config)
        success, image = capture.get_frame(config)
        timestamp = time.time()

        # Start and stop recording
        should_record = (
            success
            and config.remote_config.is_recording
            and config.remote_config.camera_resolution_width > 0
            and config.remote_config.camera_resolution_height > 0
            and config.remote_config.timestamp > 0
        )
        if should_record and not was_recording:
            print("Starting recording")
            video_writer.start(config, len(image.shape) == 2)
        elif not should_record and was_recording:
            print("Stopping recording")
            video_writer.stop()
        was_recording = should_record

        # Exit if no frame
        if not success:
            time.sleep(0.5)
            continue

        if config.camera_config.driverCam_enable:
            if not was_streaming:
                cs = CameraServer.putVideo(
                    "Driver Cam", 
                    config.remote_config.camera_resolution_width, 
                    config.remote_config.camera_resolution_height
                )
            
            processed_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            cs.putFrame(processed_frame)

        if calibration_command_source.get_calibrating(config):
            # Calibration mode
            if not was_calibrating:
                calibration_session_server = MjpegStreamServer()
                calibration_session_server.start(9999)
                was_calibrating = True
            calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))
            calibration_session_server.set_frame(image)

        elif was_calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        elif config.camera_config.has_calibration:
            # AprilTag pipeline
            if config.camera_config.apriltags_enable:
                try:
                    apriltag_worker_in.put((timestamp, image, config), block=False)
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
                    output_publisher.send_apriltag_observation(
                        config, timestamp_out, pose_observation, tag_angle_observations, demo_pose_observation
                    )

                    # Store last observations
                    last_image_observations = image_observations

                    # Measure FPS
                    fps = None
                    apriltags_frame_count += 1
                    if time.time() - apriltags_last_print > 1:
                        apriltags_last_print = time.time()
                        print("Running AprilTag pipeline at", apriltags_frame_count, "fps")
                        output_publisher.send_apriltag_fps(config, timestamp_out, apriltags_frame_count)
                        apriltags_frame_count = 0

            # Object detection pipeline
            if config.camera_config.objdetect_enable:
                # Apply FPS limit for object detection
                if objdetect_next_frame == -1:
                    objdetect_next_frame = timestamp
                if config.local_config.obj_detect_max_fps < 0 or timestamp > objdetect_next_frame:
                    objdetect_next_frame += 1 / config.local_config.obj_detect_max_fps
                    try:
                        objdetect_worker_in.put((timestamp, image, config), block=False)
                    except:  # No space in queue
                        pass
                try:
                    timestamp_out, observations = objdetect_worker_out.get(block=False)
                except:  # No new frames
                    pass
                else:
                    # Publish observation
                    output_publisher.send_objdetect_observation(config, timestamp_out, observations)

                    # Store last observations
                    last_objdetect_observations = observations

                    # Measure FPS
                    fps = None
                    objdetect_frame_count += 1
                    if time.time() - objdetect_last_print > 1:
                        objdetect_last_print = time.time()
                        print("Running object detection pipeline at", objdetect_frame_count, "fps")
                        output_publisher.send_objdetect_fps(config, timestamp, objdetect_frame_count)
                        objdetect_frame_count = 0

            # Save frame to video
            if config.remote_config.is_recording:
                if len(video_frame_cache) >= 2:
                    # Delay output by two frames to improve alignment with overlays
                    video_writer.write_frame(
                        timestamp, video_frame_cache.pop(0), last_image_observations, last_objdetect_observations
                    )
                video_frame_cache.append(image)
            else:
                video_frame_cache = []

        else:
            # No calibration
            print("No calibration found")
            time.sleep(0.5)
