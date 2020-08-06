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

        self._box_2d_modal = annotation["2d"]["modal"]
        self._box_2d_amodal = annotation["2d"]["amodal"]
        self._center = annotation["3d"]["center"]
        self._dims = annotation["3d"]["dimensions"]
        self._rotation = annotation["3d"]["rotation"]
        self._class_name = annotation["class_name"]
        self._score = annotation["score"]

    @property
    def box_2d_modal(self):
        return self._box_2d_modal

    @property
    def box_2d_amodal(self):
        return self._box_2d_amodal

    @property
    def center(self):
        return self._center

    @property
    def dims(self):
        return self._dims

    @property
    def rotation(self):
        return self._rotation

    @property
    def class_name(self):
        return self._class_name

    @property
    def score(self):
        return self._score

    @property
    def depth(self):
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

        self._box_2d = annotation["2d"]

    @property
    def box_2d(self):
        return self._box_2d

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

        self._labels_to_evaluate = labels_to_evaluate
        self._min_iou_to_match_mapping = min_iou_to_match_mapping
        self._max_depth = max_depth
        self._step_size = step_size

    @property
    def labels_to_evaluate(self):
        return self._labels_to_evaluate

    @property
    def min_iou_to_match_mapping(self):
        return self._min_iou_to_match_mapping

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def step_size(self):
        return self._step_size


def calcIouMatrix(gts, preds):
    """[summary]

    Args:
        gts ([type]): [description]
        preds ([type]): [description]

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Args:
        gt_ignores ([type]): [description]
        preds ([type]): [description]

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Args:
        folder ([type]): [description]

    Returns:
        [type]: [description]
    """
    file_list = []
    for root, dirnames, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(".json"):
                file_list.append(os.path.join(root, f))
    file_list.sort()

    return file_list