"""
Evaluation Metrics for Object Detection.
"""
import torch
import numpy as np
from typing import List, Dict


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between boxes in [x_center, y_center, w, h] format.
    """
    # Convert to corners [x1, y1, x2, y2]
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2
    
    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2
    
    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union area
    b1_area = box1[..., 2] * box1[..., 3]
    b2_area = box2[..., 2] * box2[..., 3]
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def detection_accuracy(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, 
                       iou_threshold: float = 0.5) -> Dict:
    """Calculate detection metrics at a given IoU threshold."""
    iou_scores = calculate_iou(pred_boxes, target_boxes)
    
    detections = (iou_scores >= iou_threshold).float()
    detection_rate = detections.mean().item()
    mean_iou = iou_scores.mean().item()
    
    return {
        'detection_rate': detection_rate,
        'mean_iou': mean_iou,
        'iou_scores': iou_scores
    }


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap


def evaluate_detections(all_predictions: List[Dict], all_targets: List[Dict],
                        iou_thresholds: List[float] = [0.5, 0.75]) -> Dict:
    """
    Evaluate detection results across multiple IoU thresholds.
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        total_correct = 0
        total_samples = 0
        total_iou = 0
        
        for pred, target in zip(all_predictions, all_targets):
            pred_box = pred['box']
            target_box = target['box']
            
            iou = calculate_iou(pred_box, target_box)
            total_iou += iou.item()
            total_samples += 1
            
            if iou >= iou_thresh:
                total_correct += 1
        
        results[f'accuracy@{iou_thresh}'] = total_correct / total_samples if total_samples > 0 else 0
        results[f'mean_iou'] = total_iou / total_samples if total_samples > 0 else 0
    
    return results
