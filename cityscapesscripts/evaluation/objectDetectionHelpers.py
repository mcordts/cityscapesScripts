#!/usr/bin/python3
#
# helper functions for 3D object detection evaluation
#

import os
import fnmatch
import numpy as np

from typing import List

class Box3DObject:
    """Helper class storing information about a 3D-Box-instance.

    Attributes:
        box_2d_modal: modal 2d box
        box_2d_amodal: amodal 2d box
        center: center in 3D space
        dims: dimensions in 3D
        rotation: rotation of object in quaternion
        class_name: class name in cityscapes name format
        score: predicted score
    """
    def __init__(
        self,
        annotation: dict,
    ) -> None:

        self.box_2d_modal = annotation["2d"]["modal"]
        self.box_2d_amodal = annotation["2d"]["amodal"]
        self.center = annotation["3d"]["center"]
        self.dims = annotation["3d"]["dimensions"]
        self.rotation = annotation["3d"]["rotation"]
        self.class_name = annotation["class_name"]
        self.score = annotation["score"]

    def getDepth(self):
        return np.sqrt(self.center[0]**2 + self.center[2]**2).astype(int)

class IgnoreObject:
    """Helper class storing information about an ignore region.

    Attributes:
        box_2d: Coordinates of 2d-bounding box,
    """
    def __init__(
        self,
        annotation: dict,
    ) -> None:

        self.box_2d = annotation["2d"]


class EvaluationParameters:
    """Helper class managing the evaluation parameters

    Attributes:
        labels_to_evaluate: list of labels to evaluate
        min_iou_to_match_mapping: min iou required to accept as TP
        max_depth: max depth for evaluation
        step_size: step/bin size for DDTP metrics
    """

    def __init__(
        self,
        labels_to_evaluate: List[str],
        min_iou_to_match_mapping: float=0.7,
        max_depth: int=100,
        step_size: int=5
    ) -> None:

        self.labels_to_evaluate = labels_to_evaluate
        self.min_iou_to_match_mapping = min_iou_to_match_mapping
        self.max_depth = max_depth
        self.step_size = step_size


def calcIouMatrix(gts, preds):
    xmin_1, ymin_1, xmax_1, ymax_1 = np.split(gts, 4, axis=1)
    xmin_2, ymin_2, xmax_2, ymax_2 = np.split(preds, 4, axis=1)

    inter_xmin = np.maximum(xmin_1, np.transpose(xmin_2))
    inter_ymin = np.maximum(ymin_1, np.transpose(ymin_2))
    inter_xmax = np.minimum(xmax_1, np.transpose(xmax_2))
    inter_ymax = np.minimum(ymax_1, np.transpose(ymax_2))

    interArea = np.maximum((inter_xmax - inter_xmin + 1), 0) * np.maximum((inter_ymax - inter_ymin + 1), 0)

    area_1 = (xmax_1 - xmin_1 + 1) * (ymax_1 - ymin_1 + 1)
    area_2 = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)
    iou = interArea / (area_1 + np.transpose(area_2) - interArea + 1e-10)

    return iou

def calcOverlapMatrix(gt_ignores, preds):
    xmin_1, ymin_1, xmax_1, ymax_1 = np.split(gt_ignores, 4, axis=1)
    xmin_2, ymin_2, xmax_2, ymax_2 = np.split(preds, 4, axis=1)

    inter_xmin = np.maximum(xmin_1, np.transpose(xmin_2))
    inter_ymin = np.maximum(ymin_1, np.transpose(ymin_2))
    inter_xmax = np.minimum(xmax_1, np.transpose(xmax_2))
    inter_ymax = np.minimum(ymax_1, np.transpose(ymax_2))

    interArea = np.maximum((inter_xmax - inter_xmin + 1), 0) * np.maximum((inter_ymax - inter_ymin + 1), 0)

    area_2 = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)
    overlap = interArea / (np.transpose(area_2) + 1e-10)

    return overlap

def getFiles(folder):
    file_list = []
    for root, dirnames, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(".json"):
                file_list.append(os.path.join(root, f))
    file_list.sort()

    return file_list