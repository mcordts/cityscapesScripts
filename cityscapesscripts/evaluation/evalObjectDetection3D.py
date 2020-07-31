#!/usr/bin/python3
#
# The evaluation script for Cityscapes 3D object detection (https://arxiv.org/abs/2006.07864)
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#

# python imports
import coloredlogs, logging
import numpy as np
import json
import os
import argparse

from pyquaternion import Quaternion
import concurrent.futures

from typing import (
    Dict, 
    List, 
    Tuple, 
    Union
)

from cityscapesscripts.helpers.labels import labels
from cityscapesscripts.evaluation.plot_3d_results import (
    prepare_data, 
    plot_data
) 

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


"""
File format:
Each json is a list of detections or GTs
{
    "annotation": [
        {
            "2d": {
                "modal": [xmin, ymin, xmax, ymax],
                "amodal": [xmin, ymin, xmax, ymax]
            },
            "3d": {
                "center": [x, y, z],
                "dimensions": [length, width, height],
                "rotation": [q1, q2, q3, q4],
                "format": "CRS_ISO8855"
            },
            "class_name": str,
            "score": 1.0
        }
    ]
}

"""

def printErrorAndExit(msg):
    logging.error(msg)
    logging.info("========================")
    logging.info("=== Stop evaluation ====")
    logging.info("========================")
    
    exit(1)

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

class Box3DEvaluator:
    def __init__(
        self,
        evaluation_params: EvaluationParameters
    ) -> None:

        self.evaluation_params = evaluation_params

        self._num_steps = 25
        self._score_thresholds = np.arange(0.0, 1.01, 1.0 / self.num_steps)

        self.reset()

    def calcIouMatrix(self, gts, preds):
        x11, y11, x12, y12 = np.split(gts, 4, axis=1)
        x21, y21, x22, y22 = np.split(preds, 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / \
            (boxAArea + np.transpose(boxBArea) - interArea + 1e-10)

        return iou

    def calcIntersectionMatrix(self, gt_ignores, preds):
        x11, y11, x12, y12 = np.split(gt_ignores, 4, axis=1)
        x21, y21, x22, y22 = np.split(preds, 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (np.transpose(boxBArea) + 1e-10)

        return iou

    def getMatches(self, iou_matrix):
        matched_gts = []
        matched_preds = []
        matched_ious = []

        # we either have gt and no predictions or no predictions but gt
        if iou_matrix.shape[0] == 0 or iou_matrix.shape[1] == 0:
            return [], [], []

        # iteratively select the max of the iou_matrix and set the corresponding
        # rows and cols to 0.
        while np.max(iou_matrix) > self.min_iou_to_match_mapping:
            tmp_row, tmp_col = np.where(iou_matrix == np.max(iou_matrix))

            used_row = tmp_row[0]
            used_col = tmp_col[0]

            matched_gts.append(used_row)
            matched_preds.append(used_col)
            matched_ious.append(np.max(iou_matrix))

            iou_matrix[used_row, ...] = 0.0
            iou_matrix[..., used_col] = 0.0

        return matched_gts, matched_preds, matched_ious

    def calcCenterDistances(self, class_name, gt_boxes, pred_boxes):
        """
        Calculates the BEV distance for a TP box
        d = sqrt(dx*dx + dz*dz)

        Args:
            gt_boxes:   GT boxes
            pred_boxes: Predicted boxes
        """

        gt_boxes = np.asarray([x.center for x in gt_boxes])
        pred_boxes = np.asarray([x.center for x in pred_boxes])

        gt_dists = np.sqrt(gt_boxes[..., 0]**2 +
                           gt_boxes[..., 2]**2).astype(int)
        center_dists = gt_boxes - pred_boxes
        center_dists = np.sqrt(
            center_dists[..., 0]**2 + center_dists[..., 2]**2)

        for gt_dist, center_dist in zip(gt_dists, center_dists):
            if gt_dist >= self.evaluation_params.max_depth:
                continue

            # instead of unbound distances in m we want to transform this in a score between 0 and 1
            # e.g. if the max_depth == 100
            # score = 1. - (dist / 100)

            gt_dist = int(gt_dist / self.evaluation_params.step_size) * self.evaluation_params.step_size

            self._stats["working_data"][class_name]["Center_Dist"][gt_dist].append(
                1. - min(center_dist / float(self.evaluation_params.max_depth), 1.))  # norm it to 1.

        return gt_dists

    def calcSizeSimilarities(self, class_name, gt_boxes, pred_boxes, gt_dists):
        """
        Calculates the size similarity for a TP box
        s = min(w/w', w'/w) * min(h/h', h'/h) * min(l/l', l'/l)

        Args:
            gt_boxes:   GT boxes
            pred_boxes: Predicted boxes
        """

        gt_boxes = np.asarray([x.dims for x in gt_boxes])
        pred_boxes = np.asarray([x.dims for x in pred_boxes])

        size_similarities = np.prod(np.minimum(
            gt_boxes / pred_boxes, pred_boxes / gt_boxes), axis=1)

        for gt_dist, size_simi in zip(gt_dists, size_similarities):
            if gt_dist >= self.evaluation_params.max_depth:
                continue

            gt_dist = int(gt_dist / self.evaluation_params.step_size) * self.evaluation_params.step_size

            self._stats["working_data"][class_name]["Size_Similarity"][gt_dist].append(
                size_simi)

    def calcOrientationSimilarities(self, class_name, gt_boxes, pred_boxes, gt_dists):
        """
        Calculates the orientation similarity for a TP box.
        os_yaw = (1 + cos(delta)) / 2.
        os_pitch/roll = 0.5 + (cos(delta_pitch) + cos(delta_roll)) / 4.

        Args:
            class_name (str): name of the class
            gt_boxes:   GT boxes
            pred_boxes: Predicted boxes
        """

        gt_vals = np.asarray(
            [Quaternion(x.rotation).yaw_pitch_roll for x in gt_boxes])
        pred_vals = np.asarray(
            [Quaternion(x.rotation).yaw_pitch_roll for x in pred_boxes])

        os_yaws = (1. + np.cos(gt_vals[..., 0] - pred_vals[..., 0])) / 2.
        os_pitch_rolls = 0.5 + \
            (np.cos(gt_vals[..., 1] - pred_vals[..., 1]) +
             np.cos(gt_vals[..., 2] - pred_vals[..., 2])) / 4.

        for gt_dist, os_yaw, os_pitch_roll in zip(gt_dists, os_yaws, os_pitch_rolls):
            if gt_dist >= self.evaluation_params.max_depth:
                continue

            gt_dist = int(gt_dist / self.evaluation_params.step_size) * self.evaluation_params.step_size

            self._stats["working_data"][class_name]["OS_Yaw"][gt_dist].append(
                os_yaw)
            self._stats["working_data"][class_name]["OS_Pitch_Roll"][gt_dist].append(
                os_pitch_roll)

    def calculateAUC(self, class_name):
        parameter_depth_data = self._stats["working_data"][class_name]

        for parameter_name, value_dict in parameter_depth_data.items():
            curr_mean = -1.
            result_dict = {}
            result_items = {}
            result_auc = 0.
            accum_values = 0.
            num_items = 0

            depths = []
            vals = []
            num_items_list = []
            all_items = []

            for depth, values in value_dict.items():
                if len(values) > 0:
                    accum_values += sum(values)
                    num_items += len(values)
                    all_items += values

                    # accum_values / float(num_items)
                    curr_mean = sum(values) / float(len(values))

                    depths.append(depth)
                    vals.append(curr_mean)
                    num_items_list.append(len(values))

            depths = np.asarray([0.] + depths + [self.evaluation_params.max_depth])
            if len(vals) > 0:
                vals = np.asarray(vals[0:1] + vals + vals[-1:])
            else:
                vals = np.asarray([0., 0.])

            idx = np.arange(1, len(depths), 1)
            result_auc = np.sum((depths[idx] - depths[idx - 1]) * vals[idx])

            for d, v, n in zip(depths, vals, num_items_list):
                result_dict[d] = v
                result_items[d] = n

            # norm it over depth to 1
            result_auc /= self.evaluation_params.max_depth

            self.results[parameter_name][class_name]["data"] = result_dict
            self.results[parameter_name][class_name]["auc"] = result_auc
            self.results[parameter_name][class_name]["items"] = result_items

    def calcTpStats(self):
        """Retrieves working point for each class and calculate TP stats.
        
        Calculated stats are:
          - BEV mean center distance
          - size similarity
          - orientation score for yaw and pitch/roll
        """

        parameters = ["AP", "Center_Dist",
                      "Size_Similarity", "OS_Yaw", "OS_Pitch_Roll"]

        for parameter in parameters:
            if parameter == "AP":
                continue
            self.results[parameter] = {
                x: {
                    "data": {},
                    "items": {},
                    "auc": 0.
                } for x in self.mapping_information.values()
            }

        for class_name in self.mapping_information.values():
            working_point = self._stats["working_point"][class_name]
            working_data = self._stats[working_point]["data"]

            self._stats["working_data"] = {}
            self._stats["working_data"][class_name] = {
                "Center_Dist": {x: [] for x in range(0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)},
                "Size_Similarity": {x: [] for x in range(0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)},
                "OS_Yaw": {x: [] for x in range(0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)},
                "OS_Pitch_Roll": {x: [] for x in range(0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)}
            }

            for base_img, tp_fp_fn_data in working_data.items():
                gt_boxes = self.gts[base_img]
                pred_boxes = self.preds[base_img]

                tp_idx_gt = tp_fp_fn_data["tp_idx_gt"]
                tp_idx_pred = tp_fp_fn_data["tp_idx_pred"]

                # only select the GT boxes
                gt_boxes = [gt_boxes[x] for x in tp_idx_gt[class_name]]
                pred_boxes = [pred_boxes[x] for x in tp_idx_pred[class_name]]

                # there is no prediction or GT -> no TP statistics
                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                # calculate center_dists for image
                gt_dists = self.calcCenterDistances(
                    class_name, gt_boxes, pred_boxes)

                # calculate size similarities
                self.calcSizeSimilarities(
                    class_name, gt_boxes, pred_boxes, gt_dists)

                # calculate orientation similarities
                self.calcOrientationSimilarities(
                    class_name, gt_boxes, pred_boxes, gt_dists)

            # calc AUC and detection score
            self.calculateAUC(class_name)

        # determine which categories have GT data and can be used for mean calculation
        accept_cats = []
        for cat, count in self._stats["GT_stats"].items():
            if count == 0:
                logger.warn("Category " + cat + " has no GT!")
            else:
                accept_cats.append(cat)

        # calculate mean over each entry
        for parameter_name in parameters:
            self.results["m" + parameter_name] = np.mean(
                [x["auc"] for cat, x in self.results[parameter_name].items() if cat in accept_cats])

        # calculate detection scores
        self.results["Detection_Score"] = {}
        logger.info(" === DETECTION SCORES ===")
        for class_name in self.mapping_information.values():
            vals = {p: self.results[p][class_name]["auc"] for p in parameters}
            det_score = vals["AP"] * (vals["Center_Dist"] + vals["Size_Similarity"] +
                                      vals["OS_Yaw"] + vals["OS_Pitch_Roll"]) / 4.
            self.results["Detection_Score"][class_name] = det_score
            logger.info(class_name + ": %.2f" % det_score)

        self.results["mDetection_Score"] = np.mean(
            [x for cat, x in self.results["Detection_Score"].items() if cat in accept_cats])
        logger.info("Mean Detection Score: %.2f" %
                    self.results["mDetection_Score"])
        self.results["GT_stats"] = self._stats["GT_stats"]

    def saveResults(self):
        """Saves ``self.results`` results to ``"results.json"`` 
        and ``self._stats`` to ``"stats.json"``.
        """
        with open('results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        with open('stats.json', 'w') as f:
            json.dump(self._stats, f, indent=4)

    def reset(self):
        """Resets state of this instance to a newly initialised one."""
        self.gts = {}
        self.preds = {}
        self._stats = {}
        self.ap = {}
        self.results = {}

    def loadPredictions(self, pred_folder: str):
        """Loads all predictions from the given folder

        Args:
            pred_folder (str): Prediction folder
        """

        logger.info("Loading predictions...")
        predictions = []
        for root, dirnames, filenames in os.walk(pred_folder):
            for f in filenames:
                if f.endswith(".json"):
                    predictions.append(os.path.join(root, f))

        predictions.sort()
        logger.info("Found " + str(len(predictions)) + " prediction files")

        for p in predictions[:]:
            preds_for_image = []
            base = os.path.basename(p)
            with open(p) as f:
                data = json.load(f)

            for d in data:
                box_data = Box3D(d["2d"], d["3d"],
                                 d["class_name"], d["score"], d["ignore"])
                preds_for_image.append(box_data)

            self.preds[base] = preds_for_image

    def loadGT(self, gt_folder: str):
        """
        Loads ground truth from the given folder

        Args:
            gt_folder (str): Ground truth folder
        """

        logger.info("Loading GT...")
        gts = []
        for root, dirnames, filenames in os.walk(gt_folder):
            for f in filenames:
                if f.endswith(".json"):
                    gts.append(os.path.join(root, f))

        gts.sort()
        logger.info("Found " + str(len(gts)) + " GT files")

        self._stats["GT_stats"] = {
            x: 0 for x in self.mapping_information.values()}

        for p in gts[:]:
            gts_for_image = []
            base = os.path.basename(p)
            with open(p) as f:
                data = json.load(f)

            for d in data:
                self._stats["GT_stats"][d["class_name"]] += 1
                box_data = Box3D(d["2d"], d["3d"],
                                 d["class_name"], d["score"], d["ignore"])
                gts_for_image.append(box_data)

            self.gts[base] = gts_for_image

    def evaluate(self):
        # fill up with empty detections if prediction file not found
        for base in self.gts.keys():
            if base not in self.preds.keys():
                logging.critical(
                    "Could not find any prediction for image " + base)
                self.preds[base] = []

        # initialize empty data
        for s in self.score_thresholds:
            self._stats[s] = {
                "data": {}
            }

        logger.info("Evaluating images...")
        # calculate stats for each image
        self.calcImageStats()

        logger.info("Calculate AP...")
        # calculate 2D ap
        self.calculateAp()

        logger.info("Calculate TP stats...")
        # calculate FP stats (center dist, size similarity, orientation score)
        self.calcTpStats()

        self.results["min_iou"] = self.min_iou_to_match_mapping

    def _worker(self, base):

        tmp_stats = {}

        gt_boxes = self.gts[base]
        pred_boxes = []

        pred_boxes = self.preds[base]

        for s in self.score_thresholds:
            tmp_stats[s] = {
                "data": {}
            }
            (tp_idx_gt, tp_idx_pred, fp_idx_pred,
             fn_idx_gt) = self._addImageEvaluation(gt_boxes, pred_boxes, s)

            assert len(tp_idx_gt) == len(tp_idx_pred)

            tmp_stats[s]["data"][base] = {
                "tp_idx_gt": tp_idx_gt,
                "tp_idx_pred": tp_idx_pred,
                "fp_idx_pred": fp_idx_pred,
                "fn_idx_gt": fn_idx_gt
            }

        return tmp_stats

    def calcImageStats(self):
        """Calculate Precision and Recall values for whole dataset."""
        # for x in self.gts.keys():
        #    self._worker(x)
        with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
            results = list(executor.map(self._worker, self.gts.keys()))

        # update internal result dict with the curresponding results
        for thread_result in results:
            for score, eval_data in thread_result.items():
                data = eval_data["data"]
                for img_base, match_data in data.items():
                    self._stats[score]["data"][img_base] = match_data

    def calculateAp(self):
        """Calculate Average Precision (AP) values for the whole dataset."""

        for s in self.score_thresholds:
            score_data = self._stats[s]["data"]
            tp_per_depth = {x: {d: [] for d in range(
                0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)} for x in self.mapping_information.values()}
            fp_per_depth = {x: {d: [] for d in range(
                0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)} for x in self.mapping_information.values()}
            fn_per_depth = {x: {d: [] for d in range(
                0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size)} for x in self.mapping_information.values()}

            precision_per_depth = {x: {}
                                   for x in self.mapping_information.values()}
            recall_per_depth = {x: {}
                                for x in self.mapping_information.values()}
            auc_per_depth = {x: {} for x in self.mapping_information.values()}

            tp = {x: 0 for x in self.mapping_information.values()}
            fp = {x: 0 for x in self.mapping_information.values()}
            fn = {x: 0 for x in self.mapping_information.values()}

            precision = {x: 0 for x in self.mapping_information.values()}
            recall = {x: 0 for x in self.mapping_information.values()}
            auc = {x: 0 for x in self.mapping_information.values()}

            for img_base, img_base_stats in score_data.items():
                gt_depths = [x.getDepth() for x in self.gts[img_base]]
                pred_depths = [x.getDepth() for x in self.preds[img_base]]

                for class_name, idxs in img_base_stats["tp_idx_gt"].items():
                    tp[class_name] += len(idxs)

                    for idx in idxs:
                        tp_depth = gt_depths[idx]
                        if tp_depth >= self.evaluation_params.max_depth:
                            continue

                        tp_depth = int(tp_depth / self.evaluation_params.step_size) * self.evaluation_params.step_size

                        tp_per_depth[class_name][tp_depth].append(idx)

                for class_name, idxs in img_base_stats["fp_idx_pred"].items():
                    fp[class_name] += len(idxs)

                    for idx in idxs:
                        fp_depth = pred_depths[idx]
                        if fp_depth >= self.evaluation_params.max_depth:
                            continue

                        fp_depth = int(fp_depth / self.evaluation_params.step_size) * self.evaluation_params.step_size

                        fp_per_depth[class_name][fp_depth].append(idx)

                for class_name, idxs in img_base_stats["fn_idx_gt"].items():
                    fn[class_name] += len(idxs)

                    for idx in idxs:
                        fn_depth = gt_depths[idx]
                        if fn_depth >= self.evaluation_params.max_depth:
                            continue

                        fn_depth = int(fn_depth / self.evaluation_params.step_size) * self.evaluation_params.step_size

                        fn_per_depth[class_name][fn_depth].append(idx)

            for class_name in self.mapping_information.values():
                accum_tp = 0
                accum_fp = 0
                accum_fn = 0

                for i in range(0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size):
                    accum_tp += len(tp_per_depth[class_name][i])
                    accum_fp += len(fp_per_depth[class_name][i])
                    accum_fn += len(fn_per_depth[class_name][i])

                    if accum_tp == 0 and accum_fn == 0:
                        precision_per_depth[class_name][i] = -1
                        recall_per_depth[class_name][i] = -1
                    elif accum_tp == 0:
                        precision_per_depth[class_name][i] = 0
                        recall_per_depth[class_name][i] = 0
                    else:
                        precision_per_depth[class_name][i] = accum_tp / \
                            float(accum_tp + accum_fp)
                        recall_per_depth[class_name][i] = accum_tp / \
                            float(accum_tp + accum_fn)

                    # TODO: shall we calculate AP up to depth? Or AP within depth range as for TP metrics?
                    auc_per_depth[class_name][i] = precision_per_depth[class_name][i] * \
                        recall_per_depth[class_name][i]

                if tp[class_name] == 0:
                    precision[class_name] = 0
                    recall[class_name] = 0
                else:
                    precision[class_name] = tp[class_name] / \
                        float(tp[class_name] + fp[class_name])
                    recall[class_name] = tp[class_name] / \
                        float(tp[class_name] + fn[class_name])

                auc[class_name] = precision[class_name] * recall[class_name]

            # write to stats
            self._stats[s]["pr_data"] = {
                "tp": tp,
                "fp": tp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "tp_per_depth": tp_per_depth,
                "fp_per_depth": fp_per_depth,
                "fn_per_depth": fn_per_depth,
                "precision_per_depth": precision_per_depth,
                "recall_per_depth": recall_per_depth,
                "auc_per_depth": auc_per_depth,
            }

        # calculate AP and mAP and working point
        ap = {
            x: {
                "data": {},
                "auc": 0.
            } for x in self.mapping_information.values()
        }

        ap_per_depth = {
            x: {} for x in self.mapping_information.values()
        }

        working_point = {x: 0 for x in self.mapping_information.values()}

        # calculate standard mAP
        for class_name in self.mapping_information.values():
            best_auc = 0.
            best_score = 0.

            recalls_ = []
            precisions_ = []
            for s in self.score_thresholds:
                if self._stats[s]["pr_data"]["auc"][class_name] > best_auc:
                    best_auc = self._stats[s]["pr_data"]["auc"][class_name]
                    best_score = s

                recalls_.append(
                    self._stats[s]["pr_data"]["recall"][class_name])
                precisions_.append(
                    self._stats[s]["pr_data"]["precision"][class_name])

            # sort for an ascending recalls list
            sorted_pairs = sorted(
                zip(recalls_, precisions_), key=lambda pair: pair[0])
            recalls = [r for r, _ in sorted_pairs]
            precisions = [p for _, p in sorted_pairs]

            # convert the data to numpy tensor for easier processing
            precisions = np.asarray([0] + precisions + [0])
            recalls = np.asarray([0] + recalls + [1])

            # make precision values to be decreasing only
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = np.maximum(precisions[i], precisions[i + 1])

            # gather indices of distinct recall values
            recall_idx = np.where(recalls[1:] != recalls[:-1])[0] + 1

            # calculate ap
            class_ap = np.sum(
                (recalls[recall_idx] - recalls[recall_idx - 1]) * precisions[recall_idx])

            ap[class_name]["auc"] = float(class_ap)
            ap[class_name]["data"]["recall"] = [float(x) for x in recalls_]
            ap[class_name]["data"]["precision"] = [
                float(x) for x in precisions_]
            working_point[class_name] = best_score

        # calculate depth dependent mAP
        for class_name in self.mapping_information.values():
            for d in range(0, self.evaluation_params.max_depth + 1, self.evaluation_params.step_size):
                tmp_dict = {
                    "data": {},
                    "auc": 0.
                }

                recalls_ = []
                precisions_ = []

                valid_depth = True
                for s in self.score_thresholds:
                    if d not in self._stats[s]["pr_data"]["recall_per_depth"][class_name].keys():
                        valid_depth = False
                        break

                    tmp_recall = self._stats[s]["pr_data"]["recall_per_depth"][class_name][d]
                    tmp_precision = self._stats[s]["pr_data"]["precision_per_depth"][class_name][d]

                    if tmp_recall >= 0 and tmp_precision >= 0:
                        recalls_.append(tmp_recall)
                        precisions_.append(tmp_precision)

                if len(precisions_) > 0 and len(recalls_) > 0:
                    if not valid_depth:
                        continue

                    # sort for an ascending recalls list
                    sorted_pairs = sorted(
                        zip(recalls_, precisions_), key=lambda pair: pair[0])
                    recalls = [r for r, _ in sorted_pairs]
                    precisions = [p for _, p in sorted_pairs]

                    # convert the data to numpy tensor for easier processing
                    precisions = np.asarray([0] + precisions + [0])
                    recalls = np.asarray([0] + recalls + [1])

                    # make precision values to be decreasing only
                    for i in range(len(precisions) - 2, -1, -1):
                        precisions[i] = np.maximum(
                            precisions[i], precisions[i + 1])

                    # gather indices of distinct recall values
                    recall_idx = np.where(recalls[1:] != recalls[:-1])[0] + 1

                    # calculate ap
                    class_ap = np.sum(
                        (recalls[recall_idx] - recalls[recall_idx - 1]) * precisions[recall_idx])

                    tmp_dict["auc"] = float(class_ap)
                    tmp_dict["data"]["recall"] = [float(x) for x in recalls_]
                    tmp_dict["data"]["precision"] = [
                        float(x) for x in precisions_]

                    ap_per_depth[class_name][d] = tmp_dict
                else:  # no valid detection until this depth
                    tmp_dict["auc"] = -1.
                    tmp_dict["data"]["recall"] = []
                    tmp_dict["data"]["precision"] = []

        self._stats["min_iou"] = self.min_iou_to_match_mapping
        self._stats["working_point"] = working_point

        self.results["AP"] = ap
        self.results["AP_per_depth"] = ap_per_depth

    def _addImageEvaluation(self, gt_boxes, pred_boxes, min_score):
        tp_idx_gt = {}
        tp_idx_pred = {}
        fp_idx_pred = {}
        fn_idx_gt = {}

        # calculate stats per class
        for i in self.mapping_information.values():
            # get idx for pred boxes for current class with ignore == False
            pred_idx = [idx for idx, box in enumerate(
                pred_boxes) if box.ignore == False and box.class_name == i and box.score >= min_score]

            # get idx for gt boxes for current class and ignores
            gt_idx = [idx for idx, box in enumerate(
                gt_boxes) if box.ignore == False and box.class_name == i]
            gt_idx_ignores = [idx for idx, box in enumerate(
                gt_boxes) if box.ignore == False and box.class_name == i]

            # create 2D box matrix for predictions and gts
            boxes_2d_pred = np.zeros((0, 4))
            if len(pred_idx) > 0:
                boxes_2d_pred = np.asarray(
                    [pred_boxes[x].box_2d for x in pred_idx])

            boxes_2d_gt = np.zeros((0, 4))
            if len(gt_idx) > 0:
                boxes_2d_gt = np.asarray([gt_boxes[x].box_2d for x in gt_idx])

            boxes_2d_gt_ignores = np.zeros((0, 4))
            if len(gt_idx_ignores) > 0:
                boxes_2d_gt_ignores = np.asarray(
                    [gt_boxes[x].box_2d for x in gt_idx_ignores])

            # calculate IoU matrix between GTs and Preds
            iou_matrix = self.calcIouMatrix(boxes_2d_gt, boxes_2d_pred)

            # get matches
            gt_tp_row_idx, pred_tp_col_idx, _ = self.getMatches(iou_matrix)

            # convert it to box idx
            gt_tp_idx = [gt_idx[x] for x in gt_tp_row_idx]
            pred_tp_idx = [pred_idx[x] for x in pred_tp_col_idx]
            gt_fn_idx = [x for x in gt_idx if x not in gt_tp_idx]
            pred_fp_idx_check_for_ignores = [
                x for x in pred_idx if x not in pred_tp_idx]

            # check if remaining FP idx match with ignored GT
            boxes_2d_pred_fp = np.zeros((0, 4))
            if len(pred_fp_idx_check_for_ignores) > 0:
                boxes_2d_pred_fp = np.asarray(
                    [pred_boxes[x].box_2d for x in pred_fp_idx_check_for_ignores])

            intersection_matrix = self.calcIntersectionMatrix(
                boxes_2d_gt_ignores, boxes_2d_pred_fp)

            # get matches and convert to actual box idx
            _, pred_tp_col_idx, _ = self.getMatches(iou_matrix)
            pred_tp_ignores_idx = [
                pred_fp_idx_check_for_ignores[x] for x in pred_tp_col_idx]
            pred_fp_idx = [
                x for x in pred_fp_idx_check_for_ignores if x not in pred_tp_ignores_idx]

            # dump data to result dicts
            tp_idx_gt[i] = gt_tp_idx
            tp_idx_pred[i] = pred_tp_idx
            fp_idx_pred[i] = pred_fp_idx
            fn_idx_gt[i] = gt_fn_idx

        return (tp_idx_gt, tp_idx_pred, fp_idx_pred, fn_idx_gt)

# perform the evaluation on given GT and predction folder
def evaluate3DObjectDetection(gt_folder, pred_folder, result_file, eval_params):
    # initialize the evaluator
    extended_evaluator = Box3DEvaluator(eval_params)

    # load GT and predictions
    extended_evaluator.loadGT(gt_folder)
    extended_evaluator.loadPredictions(pred_folder)

    # perform evaluation
    extended_evaluator.evaluate()

    # save results and plot them
    extended_evaluator.saveResults(result_file)
    data_to_plot = prepare_data(result_file)
    plot_data(data_to_plot, max_depth=eval_params.max_depth)

    return

# main method
def main():
    logging.info("========================")
    logging.info("=== Start evaluation ===")
    logging.info("========================")

    # get cityscapes paths
    cityscapesPath = os.environ.get(
        'CITYSCAPES_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    )
    gtFolder = os.path.join(cityscapesPath, "box3Dgt")

    predictionPath = os.environ.get(
        'CITYSCAPES_RESULTS',
        os.path.join(cityscapesPath, "results")
    )
    predictionFolder = os.path.join(predictionPath, "box3Dpred")

    parser = argparse.ArgumentParser()
    # setup location
    parser.add_argument("--gt-folder",
                        dest="gtFolder",
                        help= '''path to folder that contains ground truth *.json files. If the
                            argument is not provided this script will look for the *.json files in
                            the 'box3dgt' folder in CITYSCAPES_DATASET.
                        ''',
                        default=gtFolder,
                        type=str)
    parser.add_argument("--prediction-folder",
                        dest="predictionFolder",
                        help='''path to folder that contains ground truth *.json files. If the
                            argument is not provided this script will look for the *.json files in
                            the 'box3dpred' folder in CITYSCAPES_RESULTS.
                        ''',
                        default=predictionFolder,
                        type=str)
    resultFile = "result3DObjectDetection.json"
    parser.add_argument("--results-file",
                        dest="resultsFile",
                        help="File to store evaluation results. Default: {}".format(resultFile),
                        default=resultFile,
                        type=str)

    # setup evaluation parameters
    evalLabels = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]
    parser.add_argument("--eval-labels",
                        dest="evalLabels",
                        help="Labels to be evaluated separated with a space. Default: {}".format(" ".join(evalLabels)),
                        default=evalLabels,
                        nargs="+",
                        type=str)
    minIou = 0.7
    parser.add_argument("--min-iou",
                        dest="minIou",
                        help="Minimum IoU required to accept a detection as TP. Default: {}".format(minIou),
                        default=minIou,
                        type=float)
    maxDepth = 100
    parser.add_argument("--max-depth",
                        dest="maxDepth",
                        help="Maximum depth for DDTP metrics. Default: {}".format(maxDepth),
                        default=maxDepth,
                        type=int)
    stepSize = 5
    parser.add_argument("--step-size",
                        dest="stepSize",
                        help="Step size for DDTP metrics. Default: {}".format(stepSize),
                        default=stepSize,
                        type=int)
    args = parser.parse_args()

    if not os.path.exists(args.gtFolder):
        printErrorAndExit("Could not find gt folder {}. Please run the script with '--help'".format(args.gtFolder))
    
    if not os.path.exists(args.predictionFolder):
        printErrorAndExit("Could not find prediction folder {}. Please run the script with '--help'".format(args.predictionFolder))

    # setup the evaluation parameters
    eval_params = EvaluationParameters(
        evalLabels,
        min_iou_to_match_mapping=minIou,
        max_depth=maxDepth,
        step_size=stepSize
    )

    evaluate3DObjectDetection(args.gtFolder, args.predictionFolder, args.resultsFile, eval_params)

    return

if __name__ == "__main__":
    # call the main method
    main()

    