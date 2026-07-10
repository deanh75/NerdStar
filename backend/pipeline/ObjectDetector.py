# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import math
from typing import List, Union

from ultralytics import YOLO
import cv2
import numpy as np
from backend.config.config import ConfigStore, LocalConfig
from PIL import Image
from backend.vision_types import ObjDetectObservation


class ObjectDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect(self, image: cv2.Mat, config: ConfigStore) -> List[ObjDetectObservation]:
        raise NotImplementedError


class YOLOObjectDetector(ObjectDetector):
    _model: Union[YOLO, None] = None

    def __init__(self) -> None:
        self.last_model = None
        pass

    def detect(self, image: cv2.cuda.GpuMat, config: ConfigStore, local_config: LocalConfig) -> List[ObjDetectObservation]:
        if local_config.obj_detect_model == "":
            return []

        if self._model != None and self.last_model != local_config.obj_detect_model:
            print("New model detected, reloading...")
            self._model = None

        # Load CoreML model
        if self._model == None:
            print("Loading object detection model")
            try:
                self._model = YOLO("backend/data/models/" + local_config.obj_detect_model)
                print("Finished loading object detection model")
            except Exception as e:
                self._model = None
                print(f"Error loading object detection model: {e}")
                return []

        self.last_model = local_config.obj_detect_model

        # Create scaled frame for model
        image_scaled = np.zeros((640, 640, 3), dtype=np.uint8)
        scaled_height = int(640 / (image.size()[0] / image.size()[1]))
        bar_height = int((640 - scaled_height) / 2)
        image_scaled[bar_height: bar_height + scaled_height, 0:640] = cv2.cuda.resize(image, (640, scaled_height))
        

        # Run CoreML model
        image_coreml = Image.fromarray(image_scaled)
        prediction = self._model.predict({"image": image_coreml})

        observations: List[ObjDetectObservation] = []
        for coordinates, confidence in zip(prediction["coordinates"], prediction["confidence"]):
            obj_class = max(range(len(confidence)), key=confidence.__getitem__)
            confidence = float(confidence[obj_class])
            x = coordinates[0] * image.shape[1]
            y = ((coordinates[1] * 640 - bar_height) / scaled_height) * image.shape[0]
            width = coordinates[2] * image.shape[1]
            height = coordinates[3] / (scaled_height / 640) * image.shape[0]

            corners = np.array(
                [
                    [x - width / 2, y - height / 2],
                    [x + width / 2, y - height / 2],
                    [x - width / 2, y + height / 2],
                    [x + width / 2, y + height / 2],
                ]
            )
            corners_undistorted = cv2.undistortPoints(
                corners,
                config.camera_config.camera_matrix,
                config.camera_config.distortion_coefficients,
                None,
                config.camera_config.camera_matrix,
            )

            corner_angles = np.zeros((4, 2))
            for index, corner in enumerate(corners_undistorted):
                vec = np.linalg.inv(config.camera_config.camera_matrix).dot(np.array([corner[0][0], corner[0][1], 1]).T)
                corner_angles[index][0] = math.atan(vec[0])
                corner_angles[index][1] = math.atan(vec[1])

            observations.append(ObjDetectObservation(obj_class, confidence, corner_angles, corners))

        return observations
