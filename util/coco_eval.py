import copy
import io
import logging
import os
from contextlib import redirect_stdout

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 添加这一行导入
import util.utils as utils


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            
            # suppress pycocotools prints
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            self.logger.info(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

    def _compute_tp_fp_fn(self, coco_eval, class_idx, iou_idx, area_idx, max_dets_idx):
        """
        从evalImgs中统计TP, FP, FN
        
        Args:
            coco_eval: COCO评估对象
            class_idx: 类别索引
            iou_idx: IoU阈值索引
            area_idx: 面积范围索引
            max_dets_idx: 最大检测数索引
        
        Returns:
            tp, fp, fn: 真阳性、假阳性、假阴性的数量
        """
        tp = 0
        fp = 0
        fn = 0
        
        # 获取目标类别ID
        if class_idx >= len(coco_eval.params.catIds):
            return tp, fp, fn
        target_cat_id = coco_eval.params.catIds[class_idx]
        target_area_rng = coco_eval.params.areaRng[area_idx]
        max_det = coco_eval.params.maxDets[max_dets_idx]
        
        # evalImgs是一个列表,每个元素对应一个(图像,类别,面积范围)的评估结果
        if not hasattr(coco_eval, 'evalImgs') or coco_eval.evalImgs is None:
            return tp, fp, fn
        
        for eval_img in coco_eval.evalImgs:
            if eval_img is None:
                continue
            
            # 检查是否是目标类别
            if eval_img['category_id'] != target_cat_id:
                continue
            
            # 检查area范围
            if eval_img['aRng'] != target_area_rng:
                continue
            
            # 获取该图像的dtMatches和gtMatches
            # dtMatches: 检测框的匹配情况 [IoU阈值 x 检测框数]
            # gtMatches: 真值框的匹配情况 [IoU阈值 x 真值框数]
            dt_matches = eval_img['dtMatches']  # shape: [num_iou_thresholds, num_detections]
            gt_matches = eval_img['gtMatches']  # shape: [num_iou_thresholds, num_groundtruths]
            dt_scores = eval_img['dtScores']
            
            # 获取当前IoU阈值下的匹配结果
            if iou_idx >= len(dt_matches):
                continue
            
            dt_match = dt_matches[iou_idx]  # 当前IoU阈值下的检测框匹配
            gt_match = gt_matches[iou_idx]  # 当前IoU阈值下的真值框匹配
            
            # 限制最大检测数
            if len(dt_match) > max_det:
                # 按照score排序,取top max_det个
                if len(dt_scores) > max_det:
                    dt_match = dt_match[:max_det]
            
            # 计算TP: 被成功匹配的检测框 (dtMatches > 0表示匹配到了GT)
            tp += np.sum(dt_match > 0)
            
            # 计算FP: 未被匹配的检测框 (dtMatches == 0表示没有匹配到GT)
            fp += np.sum(dt_match == 0)
            
            # 计算FN: 未被匹配的真值框 (gtMatches == 0表示GT没有被任何检测框匹配)
            fn += np.sum(gt_match == 0)
        
        return tp, fp, fn

    def calculate_precision_recall_metrics(self):
        """
        计算详细的精确率和召回率指标
        正确处理检测框的score排序和top-k筛选
        """
        results = {}
        
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            
            if not hasattr(coco_eval, 'evalImgs') or coco_eval.evalImgs is None:
                self.logger.warning(f"No evaluation results for {iou_type}")
                continue
            
            # 获取参数
            num_classes = len(coco_eval.params.catIds)
            
            # IoU@0.5对应索引0, IoU@0.75对应索引5
            iou_50_idx = 0
            iou_75_idx = 5
            
            # 存储每个类别的结果
            precision_50_per_class = []
            precision_75_per_class = []
            recall_50_per_class = []
            recall_75_per_class = []
            ap_50_per_class = []
            ap_75_per_class = []
            ap_avg_per_class = []
            
            # 遍历每个类别
            for class_idx in range(num_classes):
                cat_id = coco_eval.params.catIds[class_idx]
                
                # 收集所有检测框和GT框的信息
                all_dt_scores = []
                all_dt_matches_50 = []
                all_dt_matches_75 = []
                all_dt_ignore = []
                total_gt_50 = 0  # 非ignore的GT总数 @IoU=0.5
                total_gt_75 = 0  # 非ignore的GT总数 @IoU=0.75
                
                # 遍历所有图像的评估结果
                for eval_img in coco_eval.evalImgs:
                    if eval_img is None:
                        continue
                    
                    # 只处理当前类别且area='all'的结果
                    if eval_img['category_id'] != cat_id:
                        continue
                    if eval_img['aRng'] != coco_eval.params.areaRng[0]:  # 'all' area
                        continue
                    
                    # 获取检测框信息
                    dt_scores = eval_img.get('dtScores', [])
                    dt_matches = eval_img.get('dtMatches', [])
                    dt_ignore = eval_img.get('dtIgnore', [])
                    gt_ignore = eval_img.get('gtIgnore', [])
                    gt_matches = eval_img.get('gtMatches', [])
                    
                    if len(dt_scores) == 0:
                        # 没有检测框，但可能有GT
                        if len(gt_matches) > 0 and iou_50_idx < len(gt_matches):
                            gt_ig = gt_ignore if len(gt_ignore) > 0 else np.zeros(len(gt_matches[0]), dtype=bool)
                            total_gt_50 += np.sum(~gt_ig)
                        if len(gt_matches) > 0 and iou_75_idx < len(gt_matches):
                            gt_ig = gt_ignore if len(gt_ignore) > 0 else np.zeros(len(gt_matches[0]), dtype=bool)
                            total_gt_75 += np.sum(~gt_ig)
                        continue
                    
                    # 收集检测框信息
                    all_dt_scores.extend(dt_scores)
                    
                    # IoU@0.5的匹配结果
                    if iou_50_idx < len(dt_matches):
                        all_dt_matches_50.extend(dt_matches[iou_50_idx])
                        if iou_50_idx < len(dt_ignore):
                            all_dt_ignore.extend(dt_ignore[iou_50_idx])
                        else:
                            all_dt_ignore.extend([False] * len(dt_scores))
                    else:
                        all_dt_matches_50.extend([0] * len(dt_scores))
                        all_dt_ignore.extend([False] * len(dt_scores))
                    
                    # 统计GT数量（IoU@0.5）
                    if iou_50_idx < len(gt_matches):
                        gt_ig = gt_ignore if len(gt_ignore) > 0 else np.zeros(len(gt_matches[0]), dtype=bool)
                        total_gt_50 += np.sum(~gt_ig)
                
                # IoU@0.75需要重新遍历（因为匹配结果不同）
                all_dt_matches_75_list = []
                for eval_img in coco_eval.evalImgs:
                    if eval_img is None:
                        continue
                    if eval_img['category_id'] != cat_id:
                        continue
                    if eval_img['aRng'] != coco_eval.params.areaRng[0]:
                        continue
                    
                    dt_scores = eval_img.get('dtScores', [])
                    dt_matches = eval_img.get('dtMatches', [])
                    gt_matches = eval_img.get('gtMatches', [])
                    gt_ignore = eval_img.get('gtIgnore', [])
                    
                    if len(dt_scores) == 0:
                        if len(gt_matches) > 0 and iou_75_idx < len(gt_matches):
                            gt_ig = gt_ignore if len(gt_ignore) > 0 else np.zeros(len(gt_matches[0]), dtype=bool)
                            total_gt_75 += np.sum(~gt_ig)
                        continue
                    
                    # IoU@0.75的匹配结果
                    if iou_75_idx < len(dt_matches):
                        all_dt_matches_75_list.extend(dt_matches[iou_75_idx])
                    else:
                        all_dt_matches_75_list.extend([0] * len(dt_scores))
                    
                    # 统计GT数量（IoU@0.75）
                    if iou_75_idx < len(gt_matches):
                        gt_ig = gt_ignore if len(gt_ignore) > 0 else np.zeros(len(gt_matches[0]), dtype=bool)
                        total_gt_75 += np.sum(~gt_ig)
                
                all_dt_matches_75 = all_dt_matches_75_list
                
                # 如果没有检测框，设置为0
                if len(all_dt_scores) == 0:
                    precision_50_per_class.append(0.0)
                    precision_75_per_class.append(0.0)
                    recall_50_per_class.append(0.0)
                    recall_75_per_class.append(0.0)
                    ap_50_per_class.append(0.0)
                    ap_75_per_class.append(0.0)
                    ap_avg_per_class.append(0.0)
                    continue
                
                # 转换为numpy数组
                all_dt_scores = np.array(all_dt_scores)
                all_dt_matches_50 = np.array(all_dt_matches_50)
                all_dt_matches_75 = np.array(all_dt_matches_75)
                all_dt_ignore = np.array(all_dt_ignore, dtype=bool)
                
                # 按score降序排序
                sorted_indices = np.argsort(-all_dt_scores)
                all_dt_scores = all_dt_scores[sorted_indices]
                all_dt_matches_50 = all_dt_matches_50[sorted_indices]
                all_dt_matches_75 = all_dt_matches_75[sorted_indices]
                all_dt_ignore = all_dt_ignore[sorted_indices]
                
                # 限制最大检测数为100（COCO标准）
                max_dets = 100
                if len(all_dt_scores) > max_dets:
                    all_dt_scores = all_dt_scores[:max_dets]
                    all_dt_matches_50 = all_dt_matches_50[:max_dets]
                    all_dt_matches_75 = all_dt_matches_75[:max_dets]
                    all_dt_ignore = all_dt_ignore[:max_dets]
                
                # 计算IoU@0.5的指标
                # 只统计非ignore的检测框
                valid_dt_50 = ~all_dt_ignore
                tp_50 = np.sum((all_dt_matches_50 > 0) & valid_dt_50)
                fp_50 = np.sum((all_dt_matches_50 == 0) & valid_dt_50)
                
                precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
                recall_50 = tp_50 / total_gt_50 if total_gt_50 > 0 else 0.0
                
                precision_50_per_class.append(precision_50)
                recall_50_per_class.append(recall_50)
                
                # 计算IoU@0.75的指标
                valid_dt_75 = ~all_dt_ignore
                tp_75 = np.sum((all_dt_matches_75 > 0) & valid_dt_75)
                fp_75 = np.sum((all_dt_matches_75 == 0) & valid_dt_75)
                
                precision_75 = tp_75 / (tp_75 + fp_75) if (tp_75 + fp_75) > 0 else 0.0
                recall_75 = tp_75 / total_gt_75 if total_gt_75 > 0 else 0.0
                
                precision_75_per_class.append(precision_75)
                recall_75_per_class.append(recall_75)
                
                # 从eval中获取AP（这个是正确的）
                if hasattr(coco_eval, 'eval') and coco_eval.eval is not None:
                    precision = coco_eval.eval['precision']
                    
                    # AP@0.5
                    p50 = precision[iou_50_idx, :, class_idx, 0, 2]  # area=all, maxDets=100
                    valid_p50 = p50[p50 > -1]
                    ap_50 = valid_p50.mean() if len(valid_p50) > 0 else 0.0
                    ap_50_per_class.append(ap_50)
                    
                    # AP@0.75
                    p75 = precision[iou_75_idx, :, class_idx, 0, 2]
                    valid_p75 = p75[p75 > -1]
                    ap_75 = valid_p75.mean() if len(valid_p75) > 0 else 0.0
                    ap_75_per_class.append(ap_75)
                    
                    # AP@0.5:0.95
                    p_avg = precision[:, :, class_idx, 0, 2]
                    valid_p_avg = p_avg[p_avg > -1]
                    ap_avg = valid_p_avg.mean() if len(valid_p_avg) > 0 else 0.0
                    ap_avg_per_class.append(ap_avg)
            
            # 计算平均值
            mean_precision_50 = np.mean(precision_50_per_class) if precision_50_per_class else 0.0
            mean_precision_75 = np.mean(precision_75_per_class) if precision_75_per_class else 0.0
            mean_recall_50 = np.mean(recall_50_per_class) if recall_50_per_class else 0.0
            mean_recall_75 = np.mean(recall_75_per_class) if recall_75_per_class else 0.0
            mean_ap_50 = np.mean(ap_50_per_class) if ap_50_per_class else 0.0
            mean_ap_75 = np.mean(ap_75_per_class) if ap_75_per_class else 0.0
            mean_ap_avg = np.mean(ap_avg_per_class) if ap_avg_per_class else 0.0
            
            results[iou_type] = {
                # AP值
                'mean_ap_50': float(mean_ap_50),
                'mean_ap_75': float(mean_ap_75),
                'mean_ap_avg': float(mean_ap_avg),
                
                # Precision值
                'mean_precision_50': float(mean_precision_50),
                'mean_precision_75': float(mean_precision_75),
                
                # Recall值
                'mean_recall_50': float(mean_recall_50),
                'mean_recall_75': float(mean_recall_75),
            }
        
        return results

    def print_precision_recall_stats(self):
        """打印精确率、召回率和AP的详细统计信息"""
        results = self.calculate_precision_recall_metrics()
    
        for iou_type, metrics in results.items():
            self.logger.info(f"\n{'='*100}")
            self.logger.info(f"详细评估指标统计 ({iou_type})")
            self.logger.info(f"{'='*100}")
            
            self.logger.info(f"\n{'指标类型':<25} | {'@IoU=0.50':<12} | {'@IoU=0.75':<12} | {'@IoU=0.50:0.95':<15}")
            self.logger.info(f"{'-'*100}")
            
            # mAP (mean Average Precision)
            self.logger.info(f"{'mAP':<25} | {metrics['mean_ap_50']:>11.3f} | {metrics['mean_ap_75']:>11.3f} | {metrics['mean_ap_avg']:>14.3f}")
            
            # Precision 
            self.logger.info(f"{'Precision':<25} | {metrics['mean_precision_50']:>11.3f} | {metrics['mean_precision_75']:>11.3f} | {'N/A':>14}")
            
            # Recall
            self.logger.info(f"{'Recall':<25} | {metrics['mean_recall_50']:>11.3f} | {metrics['mean_recall_75']:>11.3f} | {'N/A':>14}")
            
            self.logger.info(f"{'='*100}\n")


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))