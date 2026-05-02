# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import queue
from typing import List, Tuple

import cv2
from backend.config.config import ConfigStore
from backend.pipeline.ObjectDetector import CoreMLObjectDetector
from backend.vision_types import ObjDetectObservation


def objdetect_worker(
    q_in: queue.Queue[Tuple[float, cv2.Mat, ConfigStore]],
    q_out: queue.Queue[Tuple[float, List[ObjDetectObservation]]],
):
    object_detector = CoreMLObjectDetector()

    while True:
        sample = q_in.get()
        timestamp: float = sample[0]
        image: cv2.Mat = sample[1]
        config: ConfigStore = sample[2]

        observations = object_detector.detect(image, config)

        q_out.put((timestamp, observations))
