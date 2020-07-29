from enum import IntEnum, auto
import os

import numpy as np
import torch
from yolov4.tool.class_names import COCO_NAMES
from yolov4.tool.config import YOLO_V4
from yolov4.tool.darknet2pytorch import Darknet
from yolov4.tool.torch_utils import do_detect, time
from yolov4.tool.utils import load_class_names
from yolov4.tool.weights import download_weights


class DetectorType(IntEnum):
    PYTORCH_YOLOV4 = auto()


class Detector(object):
    def __init__(self, detector_type=DetectorType.PYTORCH_YOLOV4, conf_threshold=0.5, nms_threshold=0.6, use_cuda=False):
        self.detector_type = detector_type
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_cuda = use_cuda

        self._init_detector()

    def _init_detector(self):
        weight_file = download_weights()

        detector = Darknet(YOLO_V4)
        detector.load_weights(weight_file)

        if self.use_cuda:
            detector.cuda()

        self.detector = detector

    def _yolov4_detect(self, imgs):
        if type(imgs) == torch.Tensor:
            imgs = imgs.numpy()

        detections = do_detect(self.detector, imgs, self.conf_threshold, self.nms_threshold, self.use_cuda)

        class_names = load_class_names(COCO_NAMES)
        person_class_id = class_names.index('person')

        person_detections = []
        for det in detections:
            filtered_class_ids = np.array(det)[:, 6] == person_class_id
            filtered_detections = np.array(det)[filtered_class_ids, :]
            person_detections.append(filtered_detections)

        #filtered_class_ids = np.array(detections)[0, :, 6] == person_class_id
        #filtered_detections = np.array(detections)[:, filtered_class_ids, :]

        width = imgs.shape[2]
        height = imgs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        for det in person_detections:
            det_boxes = det[:, 0:4]

            # Convert from 0.0-1.0 float value to specific pixel location
            # xmin = (det_boxes[:, 0] - (det_boxes[:, 2] / 2.0)) * width
            # ymin = (det_boxes[:, 1] - (det_boxes[:, 3] / 2.0)) * height
            # xmax = (det_boxes[:, 0] + (det_boxes[:, 2] / 2.0)) * width
            # ymax = (det_boxes[:, 1] + (det_boxes[:, 3] / 2.0)) * height
            xmin = det_boxes[:, 0] * width
            ymin = det_boxes[:, 1] * height
            xmax = det_boxes[:, 2] * width
            ymax = det_boxes[:, 3] * height

            det_boxes[:, 0] = xmin
            det_boxes[:, 1] = ymin
            det_boxes[:, 2] = xmax
            det_boxes[:, 3] = ymax

            boxes.append(det_boxes)
            scores.append(det[:, 5:6])
            class_ids.append(det[:, 6:7])

        return imgs, class_ids, scores, boxes

    def detect(self, imgs):
        # transpose from [<<batch size>>, 3, 608, 608] to [<<batch size>>, 608, 608, 3]
        timgs = np.array(imgs).transpose((0, 2, 3, 1)).copy()
        imgs, class_ids, scores, bounding_boxes = self._yolov4_detect(timgs)

        # # Filter bounding box detections by threshold
        # if isinstance(scores, mx.nd.NDArray):
        #     scores = scores.asnumpy()

        filtered_scores = []
        filtered_boxes = []
        filtered_class_ids = []
        for idx, score in enumerate(scores):
            threshold_indices = np.where(score[:, 0] > self.conf_threshold)

            if len(threshold_indices[0]) == 0:
                filtered_scores.append([])
                filtered_boxes.append([])
                filtered_class_ids.append([])

            filtered_scores.append(score[threshold_indices])
            filtered_boxes.append(bounding_boxes[idx][threshold_indices])
            filtered_class_ids.append(class_ids[idx][threshold_indices])

        # Filter and retain the original array size
        # filtered_scores = scores[threshold_indices][np.newaxis, ...]
        # filtered_boxes = bounding_boxes[threshold_indices][np.newaxis, ...]
        # filtered_class_ids = class_ids[threshold_indices][np.newaxis, ...]
        return imgs, filtered_class_ids, filtered_scores, filtered_boxes
