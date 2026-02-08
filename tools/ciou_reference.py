"""
CIoU Loss Implementation - Upgrade from DIoU
Add to utils/loss.py if you want to experiment
"""

import torch
import math

def calculate_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple:
    """
    Calculate CIoU (Complete IoU) - adds aspect ratio penalty to DIoU.
    
    CIoU = IoU - (distance² / diagonal²) - α·v
    
    Where:
    - v measures aspect ratio consistency
    - α is a balance parameter
    """
    # Convert to corners (same as DIoU)
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    b1_area = boxes1[:, 2] * boxes1[:, 3]
    b2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = b1_area + b2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    # Diagonal of enclosing box
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # Center distance (same as DIoU)
    center_distance = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1]) ** 2
    
    # NEW: Aspect ratio penalty
    w_gt = boxes2[:, 2]
    h_gt = boxes2[:, 3]
    w_pred = boxes1[:, 2]
    h_pred = boxes1[:, 3]
    
    # v measures aspect ratio consistency
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(w_gt / (h_gt + 1e-6)) - torch.atan(w_pred / (h_pred + 1e-6)), 2
    )
    
    # α is the trade-off parameter
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    
    # CIoU = IoU - (center_distance / diagonal) - α·v
    ciou = iou - center_distance / (enclose_diagonal + 1e-6) - alpha * v
    
    return ciou, iou


# To use CIoU instead of DIoU, modify SingleObjectLoss:
# In the forward method, replace calculate_diou with calculate_ciou
