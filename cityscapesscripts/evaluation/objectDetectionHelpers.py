#!/usr/bin/python3
#
# helper functions for 3D object detection evaluation
#

import os
import fnmatch
import numpy as np

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