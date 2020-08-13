#!/usr/bin/python3
#
# The evaluation script for Cityscapes 3D object detection (https://arxiv.org/abs/2006.07864)
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#

# python imports
import coloredlogs
import logging
import numpy as np
import json
import os
import argparse

from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy
import concurrent.futures

from cityscapesscripts.helpers.labels import labels
from cityscapesscripts.evaluation.objectDetectionHelpers import (
    annotation_valid,
    Box3DObject,
    IgnoreObject,
    EvaluationParameters,
    getFiles,
    calcIouMatrix,
    calcOverlapMatrix
)
from cityscapesscripts.evaluation.objectDetectionHelpers import (
    MATCHING_MODAL,
    MATCHING_AMODAL
)
from cityscapesscripts.evaluation.plot3DResults import (
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
    logger.error(msg)
    logger.info("========================")
    logger.info("=== Stop evaluation ====")
    logger.info("========================")

    exit(1)


class Box3DEvaluator:
    """The Box3DEvaluator object contains the data as well as the parameters
    for the evluation of the dataset.
    :param eval_params: evaluation params including max depth, min iou etc.
    :type eval_params: EvaluationParameters
    :param _num_steps: number of confidence threshold steps during evaluation
    :type _num_steps: int
    :

    """

    def __init__(
        self,
        evaluation_params: EvaluationParameters
    ) -> None:

        self.eval_params = evaluation_params
        self._num_steps = 50

        # dict containing the GTs per image
        self.gts = {}

        # dict containing the predictions per image
        self.preds = {}

        # dict containing information for AP per class
        self.ap = {}

        # dict containing all required results
        self.results = {}

        # internal dict keeping addtional statistics
        self._stats = {}

        # the actual confidence thresholds
        self._conf_thresholds = np.arange(0.0, 1.01, 1.0 / self._num_steps)

        # the actual depth bins
        self._depth_bins = np.arange(0, self.eval_params.max_depth + 1, self.eval_params.step_size)

    def reset(self):
        """Resets state of this instance to a newly initialised one."""

        self.gts = {}
        self.preds = {}
        self._stats = {}
        self.ap = {}
        self.results = {}

    def getMatches(self, iou_matrix):
        """Gets the TP matches between the predictions and the GT data

        Args:
            iou_matrix (2x2 Array): The matrix containing the pairwise overlap or IoU

        Returns:
            Tuple(List[int],List[int],List[float]): A tuple containing the TP indices 
            for GT and predicions and the corresponding iou
        """
        matched_gts = []
        matched_preds = []
        matched_ious = []

        # we either have gt and no predictions or no predictions but gt
        if iou_matrix.shape[0] == 0 or iou_matrix.shape[1] == 0:
            return [], [], []

        # iteratively select the max of the iou_matrix and set the corresponding
        # rows and cols to 0.
        tmp_iou_max = np.max(iou_matrix)

        while tmp_iou_max > self.eval_params.min_iou_to_match:
            tmp_row, tmp_col = np.where(iou_matrix == tmp_iou_max)

            used_row = tmp_row[0]
            used_col = tmp_col[0]

            matched_gts.append(used_row)
            matched_preds.append(used_col)
            matched_ious.append(np.max(iou_matrix))

            iou_matrix[used_row, ...] = 0.0
            iou_matrix[..., used_col] = 0.0

            tmp_iou_max = np.max(iou_matrix)

        return (matched_gts, matched_preds, matched_ious)

    def calcCenterDistances(self, class_name, gt_boxes, pred_boxes):
        """Calculates the BEV distance for a TP box
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
        center_dists = np.sqrt(center_dists[..., 0]**2 + 
                               center_dists[..., 2]**2)

        for gt_dist, center_dist in zip(gt_dists, center_dists):
            if gt_dist >= self.eval_params.max_depth:
                continue

            # instead of unbound distances in m we want to transform this in a score between 0 and 1
            # e.g. if the max_depth == 100
            # score = 1. - (dist / 100)

            gt_dist = int(gt_dist / self.eval_params.step_size) * \
                self.eval_params.step_size

            self._stats["working_data"][class_name]["Center_Dist"][gt_dist].append(
                1. - min(center_dist / float(self.eval_params.max_depth), 1.))  # norm it to 1.

        return gt_dists

    def calcSizeSimilarities(self, class_name, gt_boxes, pred_boxes, gt_dists):
        """Calculates the size similarity for a TP box
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
            if gt_dist >= self.eval_params.max_depth:
                continue

            gt_dist = int(gt_dist / self.eval_params.step_size) * \
                self.eval_params.step_size

            self._stats["working_data"][class_name]["Size_Similarity"][gt_dist].append(
                size_simi)

    def calcOrientationSimilarities(self, class_name, gt_boxes, pred_boxes, gt_dists):
        """Calculates the orientation similarity for a TP box.
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
            if gt_dist >= self.eval_params.max_depth:
                continue

            gt_dist = int(gt_dist / self.eval_params.step_size) * \
                self.eval_params.step_size

            self._stats["working_data"][class_name]["OS_Yaw"][gt_dist].append(
                os_yaw)
            self._stats["working_data"][class_name]["OS_Pitch_Roll"][gt_dist].append(
                os_pitch_roll)

    def calculateAUC(self, class_name):
        """[summary]

        Args:
            class_name ([type]): [description]
        """
        parameter_depth_data = self._stats["working_data"][class_name]

        for parameter_name, value_dict in parameter_depth_data.items():
            curr_mean = -1.
            result_dict = {}
            result_items = {}
            result_auc = 0.
            num_items = 0

            depths = []
            vals = []
            num_items_list = []
            all_items = []

            for depth, values in value_dict.items():
                if len(values) > 0:
                    num_items += len(values)
                    all_items += values

                    curr_mean = sum(values) / float(len(values))

                    depths.append(depth)
                    vals.append(curr_mean)
                    num_items_list.append(len(values))

            # AUC is calculated as the mean of all values for available depths
            if len(vals) > 1:
                result_auc = np.mean(vals)
            else:
                result_auc = 0.

            # remove the expanded entries
            for d, v, n in list(zip(depths, vals, num_items_list)):
                result_dict[d] = v
                result_items[d] = n

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
                } for x in self.eval_params.labels_to_evaluate
            }

        for class_name in self.eval_params.labels_to_evaluate:
            working_point = self._stats["working_point"][class_name]
            working_data = self._stats[working_point]["data"]

            self._stats["working_data"] = {}
            self._stats["working_data"][class_name] = {
                "Center_Dist": {x: [] for x in self._depth_bins},
                "Size_Similarity": {x: [] for x in self._depth_bins},
                "OS_Yaw": {x: [] for x in self._depth_bins},
                "OS_Pitch_Roll": {x: [] for x in self._depth_bins}
            }

            for base_img, tp_fp_fn_data in working_data.items():
                gt_boxes = self.gts[base_img]["objects"]
                pred_boxes = self.preds[base_img]["objects"]

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
                logger.warn("Category %s has no GT!" % cat)
            else:
                accept_cats.append(cat)

        # add GT statistics to reults
        self.results["GT_stats"] = self._stats["GT_stats"]

        # add evaluation parameters to results
        modal_amodal_modifier = "Amodal" 
        if self.eval_params.matching_method == MATCHING_MODAL:
            modal_amodal_modifier = "Modal"

        self.results["eval_params"] = {
            "labels": self.eval_params.labels_to_evaluate,
            "min_iou_to_match": self.eval_params.min_iou_to_match,
            "max_depth": self.eval_params.max_depth,
            "step_size": self.eval_params.step_size,
            "matching_method": modal_amodal_modifier
        }

        # calculate detection scores and them to results
        self.results["Detection_Score"] = {}
        logger.info("========================")
        logger.info("======= Results ========")
        logger.info("========================")

        for class_name in self.eval_params.labels_to_evaluate:

            vals = {p: self.results[p][class_name]["auc"] for p in parameters}
            det_score = vals["AP"] * (vals["Center_Dist"] + vals["Size_Similarity"] +
                                      vals["OS_Yaw"] + vals["OS_Pitch_Roll"]) / 4.
            self.results["Detection_Score"][class_name] = det_score

            logger.info(class_name)
            logger.info(" -> 2D AP %-6s                : %.2f" % (modal_amodal_modifier, vals["AP"]))
            logger.info(" -> BEV Center Distance (DDTP)  : %.2f" % vals["Center_Dist"])
            logger.info(" -> Yaw Similarity (DDTP)       : %.2f" % vals["OS_Yaw"])
            logger.info(" -> Pitch/Roll Similarity (DDTP): %.2f" % vals["OS_Pitch_Roll"])
            logger.info(" -> Size Similarity (DDTP)      : %.2f" % vals["Size_Similarity"])
            logger.info(" -> Detection Score             : %.2f" % det_score)

        self.results["mDetection_Score"] = np.mean(
            [x for cat, x in self.results["Detection_Score"].items() if cat in accept_cats])
        logger.info("Mean Detection Score: %.2f" %
                    self.results["mDetection_Score"])

        # add mean evaluation results
        for parameter_name in parameters:
            self.results["m" + parameter_name] = np.mean(
                [x["auc"] for cat, x in self.results[parameter_name].items() if cat in accept_cats])

    def saveResults(self, result_folder):
        """Saves ``self.results`` results to ``"results.json"`` 
        and ``self._stats`` to ``"stats.json"``.
        """

        result_file = os.path.join(result_folder, "results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=4)

        # dump internal stats for debugging        
        # stats_file = os.path.join(result_folder, "stats.json")
        # with open(stats_file, 'w') as f:
        #    json.dump(self._stats, f, indent=4)

        return result_file

    def loadPredictions(self, pred_folder: str):
        """Loads all predictions from the given folder

        Args:
            pred_folder (str): Prediction folder
        """

        logger.info("Loading predictions...")
        predictions = getFiles(pred_folder)

        predictions.sort()
        logger.info("Found %d prediction files." % len(predictions))

        for p in predictions:
            preds_for_image = []

            # extract city_record_image from filepath
            base = os.path.basename(p)

            base = base[:base.rfind("_")]

            with open(p) as f:
                data = json.load(f)

            for d in data["annotation"]:
                if not annotation_valid(d):
                    logger.critical("Found incorrect annotation in %s." % p)
                    continue

                if d["class_name"] in self.eval_params.labels_to_evaluate:
                    box_data = Box3DObject(d)
                    preds_for_image.append(box_data)

            self.preds[base] = {
                "objects": preds_for_image
            }

    def loadGT(self, gt_folder: str) -> None:
        """Loads ground truth from the given folder

        Args:
            gt_folder (str): Ground truth folder
        """

        logger.info("Loading GT...")
        gts = getFiles(gt_folder)

        logger.info("Found %d GT files." % len(gts))

        self._stats["GT_stats"] = {x: 0 for x in self.eval_params.labels_to_evaluate}

        for p in gts:
            gts_for_image = []
            ignores_for_image = []

            # extract city_record_image from filepath
            base = os.path.basename(p)
            base = base[:base.rfind("_")]

            with open(p) as f:
                data = json.load(f)

            # load 3D boxes
            for d in data["annotation"]:
                if d["class_name"] in self.eval_params.labels_to_evaluate:
                    self._stats["GT_stats"][d["class_name"]] += 1
                    box_data = Box3DObject(d)
                    gts_for_image.append(box_data)

            # load ignore regions
            for d in data["ignore"]:
                box_data = IgnoreObject(d)
                ignores_for_image.append(box_data)

            self.gts[base] = {
                "objects": gts_for_image,
                "ignores": ignores_for_image
            }

    def evaluate(self):
        """[summary]
        """
        # fill up with empty detections if prediction file not found
        for base in self.gts.keys():
            if base not in self.preds.keys():
                logger.critical(
                    "Could not find any prediction for image %s." % base)
                self.preds[base] = {"objects": []}

        # initialize empty data
        for s in self._conf_thresholds:
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

    def _worker(self, base):
        """[summary]

        Args:
            base ([type]): [description]

        Returns:
            [type]: [description]
        """
        tmp_stats = {}

        gt_boxes = self.gts[base]
        pred_boxes = self.preds[base]

        for s in self._conf_thresholds:
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

    def calcImageStats(self) -> None:
        """Calculates Precision and Recall values for whole dataset."""

        # single threaded
        results = []
        for x in tqdm(self.gts.keys()):
            results.append(self._worker(x))

        # multi threaded
        # keep in mind that this will not work out of the box due the global interpreter lock
        # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        #    results = list(tqdm(executor.map(self._worker, self.gts.keys()), total=len(self.gts.keys())))

        # update internal result dict with the curresponding results
        for thread_result in results:
            for score, eval_data in thread_result.items():
                data = eval_data["data"]
                for img_base, match_data in data.items():
                    self._stats[score]["data"][img_base] = match_data

    def calculateAp(self) -> None:
        """Calculates Average Precision (AP) values for the whole dataset."""

        for s in self._conf_thresholds:
            score_data = self._stats[s]["data"]

            # dicts containing TP, FP and FN per depth per class
            tp_per_depth = {x: {d: [] for d in self._depth_bins} for x in self.eval_params.labels_to_evaluate}
            fp_per_depth = {x: {d: [] for d in self._depth_bins} for x in self.eval_params.labels_to_evaluate}
            fn_per_depth = {x: {d: [] for d in self._depth_bins} for x in self.eval_params.labels_to_evaluate}

            # dicts containing precision and recall and AP per depth per class
            precision_per_depth = {x: {} for x in self.eval_params.labels_to_evaluate}
            recall_per_depth = {x: {} for x in self.eval_params.labels_to_evaluate}
            auc_per_depth = {x: {} for x in self.eval_params.labels_to_evaluate}

            # dicts containing overall TP, FP and FN per class
            tp = {x: 0 for x in self.eval_params.labels_to_evaluate}
            fp = {x: 0 for x in self.eval_params.labels_to_evaluate}
            fn = {x: 0 for x in self.eval_params.labels_to_evaluate}

            # dicts containing overall precision, recall and AP per class
            precision = {x: 0 for x in self.eval_params.labels_to_evaluate}
            recall = {x: 0 for x in self.eval_params.labels_to_evaluate}
            auc = {x: 0 for x in self.eval_params.labels_to_evaluate}

            # get the statistics for each image
            for img_base, img_base_stats in score_data.items():
                gt_depths = [x.depth for x in self.gts[img_base]["objects"]]
                pred_depths = [x.depth for x in self.preds[img_base]["objects"]]

                for class_name, idxs in img_base_stats["tp_idx_gt"].items():
                    tp[class_name] += len(idxs)

                    for idx in idxs:
                        tp_depth = gt_depths[idx]
                        if tp_depth >= self.eval_params.max_depth:
                            continue

                        tp_depth = int(tp_depth / self.eval_params.step_size) * self.eval_params.step_size

                        tp_per_depth[class_name][tp_depth].append(idx)

                for class_name, idxs in img_base_stats["fp_idx_pred"].items():
                    fp[class_name] += len(idxs)

                    for idx in idxs:
                        fp_depth = pred_depths[idx]
                        if fp_depth >= self.eval_params.max_depth:
                            continue

                        fp_depth = int(fp_depth / self.eval_params.step_size) * self.eval_params.step_size

                        fp_per_depth[class_name][fp_depth].append(idx)

                for class_name, idxs in img_base_stats["fn_idx_gt"].items():
                    fn[class_name] += len(idxs)

                    for idx in idxs:
                        fn_depth = gt_depths[idx]
                        if fn_depth >= self.eval_params.max_depth:
                            continue

                        fn_depth = int(fn_depth / self.eval_params.step_size) * self.eval_params.step_size

                        fn_per_depth[class_name][fn_depth].append(idx)

            # calculate per depth precision and recall per class
            for class_name in self.eval_params.labels_to_evaluate:
                for i in self._depth_bins:
                    tp_at_depth = len(tp_per_depth[class_name][i])
                    fp_at_depth = len(fp_per_depth[class_name][i])
                    accum_fn = len(fn_per_depth[class_name][i])

                    if tp_at_depth == 0 and accum_fn == 0:
                        precision_per_depth[class_name][i] = -1
                        recall_per_depth[class_name][i] = -1
                    elif tp_at_depth == 0:
                        precision_per_depth[class_name][i] = 0
                        recall_per_depth[class_name][i] = 0
                    else:
                        precision_per_depth[class_name][i] = tp_at_depth / \
                            float(tp_at_depth + fp_at_depth)
                        recall_per_depth[class_name][i] = tp_at_depth / \
                            float(tp_at_depth + accum_fn)

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

        # dict containing data for AP and mAP
        ap = {
            x: {
                "data": {},
                "auc": 0.
            } for x in self.eval_params.labels_to_evaluate
        }

        ap_per_depth = {
            x: {} for x in self.eval_params.labels_to_evaluate
        }

        # dict containing the working point for DDTP metrics
        working_point = {x: 0 for x in self.eval_params.labels_to_evaluate}

        # calculate standard AP per class
        for class_name in self.eval_params.labels_to_evaluate:
            # best_auc and best_score are used for determining working point
            best_auc = 0.
            best_score = 0.

            recalls_ = []
            precisions_ = []
            for s in self._conf_thresholds:
                current_auc_for_score = self._stats[s]["pr_data"]["auc"][class_name]
                if current_auc_for_score > best_auc:
                    best_auc = current_auc_for_score
                    best_score = s

                recalls_.append(self._stats[s]["pr_data"]["recall"][class_name])
                precisions_.append(self._stats[s]["pr_data"]["precision"][class_name])

            # sort for an ascending recalls list
            sorted_pairs = sorted(zip(recalls_, precisions_), key=lambda pair: pair[0])
            recalls, precisions = map(list, zip(*sorted_pairs))

            # convert the data to numpy tensor for easier processing and add leading and trailing zeros/ones
            precisions = np.asarray([0] + precisions + [0])
            recalls = np.asarray([0] + recalls + [1])

            # precision values should be decreasing only
            # p(r) = max{r' > r} p(r')
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = np.maximum(precisions[i], precisions[i + 1])

            # gather indices of distinct recall values
            recall_idx = np.where(recalls[1:] != recalls[:-1])[0] + 1

            # calculate ap
            class_ap = np.sum(
                (recalls[recall_idx] - recalls[recall_idx - 1]) * precisions[recall_idx])

            ap[class_name]["auc"] = float(class_ap)
            ap[class_name]["data"]["recall"] = [float(x) for x in recalls_]
            ap[class_name]["data"]["precision"] = [float(x) for x in precisions_]
            working_point[class_name] = best_score

        # calculate depth dependent mAP
        for class_name in self.eval_params.labels_to_evaluate:
            for d in self._depth_bins:
                tmp_dict = {
                    "data": {},
                    "auc": 0.
                }

                recalls_ = []
                precisions_ = []

                valid_depth = True
                for s in self._conf_thresholds:
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
                    recalls, precisions = map(list, zip(*sorted_pairs))

                    # convert the data to numpy tensor for easier processing and add leading and trailing zeros/ones
                    precisions = np.asarray([0] + precisions + [0])
                    recalls = np.asarray([0] + recalls + [1])

                    # precision values should be decreasing only
                    # p(r) = max{r' > r} p(r')
                    for i in range(len(precisions) - 2, -1, -1):
                        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

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

        # dump mAP and working points to internal stats
        self._stats["min_iou"] = self.eval_params.min_iou_to_match
        self._stats["working_point"] = working_point
        self.results["AP"] = ap
        self.results["AP_per_depth"] = ap_per_depth

    def _addImageEvaluation(self, gt_boxes, pred_boxes, min_score):
        """[summary]

        Args:
            gt_boxes ([type]): [description]
            pred_boxes ([type]): [description]
            min_score ([type]): [description]

        Returns:
            [type]: [description]
        """
        tp_idx_gt = {}
        tp_idx_pred = {}
        fp_idx_pred = {}
        fn_idx_gt = {}

        # pre-load all ignore regions as they are the same for all classes
        gt_idx_ignores = [idx for idx,
            box in enumerate(gt_boxes["ignores"])]

        # calculate stats per class
        for i in self.eval_params.labels_to_evaluate:
            # get idx for pred boxes for current class
            pred_idx = [idx for idx, box in enumerate(
                pred_boxes["objects"]) if box.class_name == i and box.score >= min_score]

            # get idx for gt boxes for current class
            gt_idx = [idx for idx, box in enumerate(
                gt_boxes["objects"]) if box.class_name == i]

            # if there is no prediction at all, just return an empty result
            if len(pred_idx) == 0:
                # dump data to result dicts
                tp_idx_gt[i] = []
                tp_idx_pred[i] = []
                fp_idx_pred[i] = pred_idx
                fn_idx_gt[i] = gt_idx
                continue

            # create 2D box matrix for predictions and gts
            boxes_2d_pred = np.zeros((0, 4))
            if len(pred_idx) > 0:
                # get modal or amodal boxes depending on matching strategy
                if self.eval_params.matching_method == MATCHING_AMODAL:
                    boxes_2d_pred = np.asarray(
                        [pred_boxes["objects"][x].box_2d_amodal for x in pred_idx])
                elif self.eval_params.matching_method == MATCHING_MODAL:
                    boxes_2d_pred = np.asarray(
                        [pred_boxes["objects"][x].box_2d_modal for x in pred_idx])
                else:
                    raise ValueError("Matching method %d not known!" % self.eval_params.matching_method)

            boxes_2d_gt = np.zeros((0, 4))
            if len(gt_idx) > 0:
                # get modal or amodal boxes depending on matching strategy
                if self.eval_params.matching_method == MATCHING_AMODAL:
                    boxes_2d_gt = np.asarray(
                        [gt_boxes["objects"][x].box_2d_amodal for x in gt_idx])
                elif self.eval_params.matching_method == MATCHING_MODAL:
                    boxes_2d_gt = np.asarray(
                        [gt_boxes["objects"][x].box_2d_modal for x in gt_idx])
                else:
                    raise ValueError("Matching method %d not known!" % self.eval_params.matching_method)

            boxes_2d_gt_ignores = np.zeros((0, 4))
            if len(gt_idx_ignores) > 0:
                boxes_2d_gt_ignores = np.asarray(
                    [gt_boxes["ignores"][x].box_2d for x in gt_idx_ignores])

            # calculate IoU matrix between GTs and Preds
            iou_matrix = calcIouMatrix(boxes_2d_gt, boxes_2d_pred)

            # get matches
            (gt_tp_row_idx, pred_tp_col_idx, _) = self.getMatches(iou_matrix)

            # convert it to box idx
            gt_tp_idx = [gt_idx[x] for x in gt_tp_row_idx]
            pred_tp_idx = [pred_idx[x] for x in pred_tp_col_idx]
            gt_fn_idx = [x for x in gt_idx if x not in gt_tp_idx]
            pred_fp_idx_check_for_ignores = [
                x for x in pred_idx if x not in pred_tp_idx]

            # check if remaining FP idx match with ignored GT
            boxes_2d_pred_fp = np.zeros((0, 4))
            if len(pred_fp_idx_check_for_ignores) > 0:
                # as there are no amodal boxes for ignore regions
                # matching with ignore regions should only be performed on 
                # modal predictions.
                boxes_2d_pred_fp = np.asarray(
                    [pred_boxes["objects"][x].box_2d_modal for x in pred_fp_idx_check_for_ignores])
                
            overlap_matrix = calcOverlapMatrix(
                boxes_2d_gt_ignores, boxes_2d_pred_fp)

            # get matches and convert to actual box idx
            (_, pred_tp_col_idx, _) = self.getMatches(overlap_matrix)
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


def evaluate3DObjectDetection(gt_folder, pred_folder, result_folder, eval_params):
    """[summary]

    Args:
        gt_folder ([type]): [description]
        pred_folder ([type]): [description]
        result_folder ([type]): [description]
        eval_params ([type]): [description]
    """
    logger.info("Use the following options")
    logger.info(" -> GT folder    : %s"   % gt_folder)
    logger.info(" -> Pred folder  : %s"   % pred_folder)
    logger.info(" -> Classes      : %s"   % ", ".join(eval_params.labels_to_evaluate))
    logger.info(" -> Min IoU:     : %.2f" % eval_params.min_iou_to_match)
    logger.info(" -> Max depth [m]: %d"   % eval_params.max_depth)
    logger.info(" -> Step size [m]: %.2f" % eval_params.step_size)

    # initialize the evaluator
    boxEvaluator = Box3DEvaluator(eval_params)

    # load GT and predictions
    boxEvaluator.loadGT(gt_folder)
    boxEvaluator.loadPredictions(pred_folder)

    # perform evaluation
    boxEvaluator.evaluate()

    # save results and plot them
    result_file = boxEvaluator.saveResults(result_folder)
    data_to_plot = prepare_data(result_file)
    plot_data(data_to_plot, max_depth=eval_params.max_depth)

    return

# main method


def main():
    logger.info("========================")
    logger.info("=== Start evaluation ===")
    logger.info("========================")

    # get cityscapes paths
    cityscapesPath = os.environ.get(
        'CITYSCAPES_DATASET', os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', '..')
    )
    gtFolder = os.path.join(cityscapesPath, "box3Dgt")

    predictionPath = os.environ.get(
        'CITYSCAPES_RESULTS',
        os.path.join(cityscapesPath, "results")
    )
    predictionFolder = os.path.join(predictionPath, "box3Dpred")

    ###
    # TMP
    ###
    gtFolder = predictionFolder = "/lhome/ngaehle/Desktop/cs_QC_final_export_TEST/export/val/"
    ###
    ###
    ###

    parser = argparse.ArgumentParser()
    # setup location
    parser.add_argument("--gt-folder",
                        dest="gtFolder",
                        help='''path to folder that contains ground truth *.json files. If the
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
    resultFolder = ""
    parser.add_argument("--results-file",
                        dest="resultsFolder",
                        help="File to store evaluation results. Default: prediction folder",
                        default=resultFolder,
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
    stepSize = 5.
    parser.add_argument("--step-size",
                        dest="stepSize",
                        help="Step size for DDTP metrics. Default: {}".format(stepSize),
                        default=stepSize,
                        type=float)

    parser.add_argument("--modal",
                        action="store_true",
                        help="Use modal 2D boxes for matching",)
    args = parser.parse_args()

    if not os.path.exists(args.gtFolder):
        printErrorAndExit(
            "Could not find gt folder {}. Please run the script with '--help'".format(args.gtFolder))

    if not os.path.exists(args.predictionFolder):
        printErrorAndExit(
            "Could not find prediction folder {}. Please run the script with '--help'".format(args.predictionFolder))

    if resultFolder == "":
        resultFolder = args.predictionFolder
    os.makedirs(resultFolder, exist_ok=True)

    # setup the evaluation parameters
    eval_params = EvaluationParameters(
        args.evalLabels,
        min_iou_to_match=args.minIou,
        max_depth=args.maxDepth,
        step_size=args.stepSize,
        matching_method=int(args.modal)
    )

    evaluate3DObjectDetection(args.gtFolder, args.predictionFolder, args.resultsFolder, eval_params)

    logger.info("========================")
    logger.info("=== Stop evaluation ====")
    logger.info("========================")

    return


if __name__ == "__main__":
    # call the main method
    main()
