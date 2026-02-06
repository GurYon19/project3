"""
Loss Functions for Object Detection.
"""
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between boxes in [x_center, y_center, w, h] format."""
    # Convert to corners
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
    
    return inter_area / (union_area + 1e-6)


def calculate_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate DIoU (Distance IoU) between boxes in [x_center, y_center, w, h] format.
    
    DIoU = IoU - (distance² / diagonal²)
    
    Benefits over standard IoU:
    - Provides gradient even when boxes don't overlap
    - Penalizes center distance, encouraging boxes to move towards target
    - Converges faster than GIoU
    """
    # Convert to corners
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
    
    # Enclosing box (smallest box that contains both boxes)
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    # Diagonal of enclosing box
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # Center distance
    center_distance = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1]) ** 2
    
    # DIoU = IoU - (center_distance / enclosing_diagonal)
    diou = iou - center_distance / (enclose_diagonal + 1e-6)
    
    return diou, iou


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
    
    # CIoU = IoU - (Center Distance Penalty) - (Aspect Ratio Penalty)
    ciou = iou - (center_distance / (enclose_diagonal + 1e-6)) - (alpha * v)
    
    return ciou, iou


def calculate_eiou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple:
    """
    Calculate EIoU (Efficient IoU).
    
    EIoU = IoU - (distance² / diagonal²) - (w_diff² / C_w²) - (h_diff² / C_h²)
    
    Directly penalizes width and height differences, stricter on "false positive" pixels.
    """
    # 1. IoU and Enclosing Box (Standard)
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    b1_area = boxes1[:, 2] * boxes1[:, 3]
    b2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = b1_area + b2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)

    # Enclosing box (C)
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    cw = enclose_x2 - enclose_x1
    ch = enclose_y2 - enclose_y1
    c2 = cw ** 2 + ch ** 2 + 1e-6 # Diagonal squared
    
    # 2. Distance term (Center distance / Diagonal^2)
    rho2 = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1]) ** 2
    
    # 3. Aspect Ratio terms (Width diff / Width_C^2) + (Height diff / Height_C^2)
    w_diff = (boxes1[:, 2] - boxes2[:, 2]) ** 2
    h_diff = (boxes1[:, 3] - boxes2[:, 3]) ** 2
    
    eiou = iou - (rho2 / c2) - (w_diff / (cw ** 2 + 1e-6)) - (h_diff / (ch ** 2 + 1e-6))
    
    return eiou, iou


class SingleObjectLoss(nn.Module):
    """
    Loss for Part 2: EIoU (Efficient IoU) Loss.
    
    EIoU improves on CIoU by explicitly penalizing width and height errors
    separately, rather than just aspect ratio. This is more robust against
    'good center, bad size' errors.
    """
    
    def __init__(self, coord_weight: float = 0.0, iou_weight: float = 1.0, use_eiou: bool = True):
        super().__init__()
        self.coord_weight = coord_weight
        self.iou_weight = iou_weight
        self.use_eiou = use_eiou
        self.smooth_l1 = nn.SmoothL1Loss() if coord_weight > 0 else None
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> dict:
        # Calculate coordinate loss only if weight > 0
        if self.coord_weight > 0 and self.smooth_l1:
            coord_loss = self.smooth_l1(pred_boxes, target_boxes)
        else:
            coord_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        if self.use_eiou:
            eiou, iou = calculate_eiou(pred_boxes, target_boxes)
            iou_loss = 1 - eiou.mean()  # EIoU loss
            mean_iou = iou.mean()
        else:
            # Fallback (Legacy)
            diou, iou = calculate_diou(pred_boxes, target_boxes)
            iou_loss = 1 - diou.mean()
            mean_iou = iou.mean()
        
        total_loss = self.coord_weight * coord_loss + self.iou_weight * iou_loss
        
        return {
            'loss': total_loss,
            'coord_loss': coord_loss,
            'iou_loss': iou_loss,
            'mean_iou': mean_iou  # Standard IoU for evaluation
        }


class MultiObjectLoss(nn.Module):
    """Loss for Part 3: Box + Classification + Objectness."""
    
    def __init__(self, lambda_box=5.0, lambda_cls=1.0, lambda_obj=1.0, num_classes=20):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        pred_boxes = predictions['boxes']
        pred_classes = predictions['classes']
        pred_obj = predictions['objectness']
        
        target_boxes = targets['boxes']
        target_classes = targets['classes']
        target_valid = targets['valid_mask']
        
        B = pred_boxes.shape[0]
        
        total_box_loss = 0
        total_cls_loss = 0
        total_obj_loss = 0
        total_iou = 0
        num_valid = 0
        
        for b in range(B):
            valid_mask = target_valid[b].bool()
            num_valid_objs = valid_mask.sum().item()
            
            if num_valid_objs > 0:
                box_loss = self.smooth_l1(pred_boxes[b][valid_mask], target_boxes[b][valid_mask]).mean()
                total_box_loss += box_loss
                
                iou = calculate_iou(pred_boxes[b][valid_mask], target_boxes[b][valid_mask])
                total_iou += iou.sum()
                num_valid += num_valid_objs
                
                cls_loss = self.ce_loss(pred_classes[b][valid_mask], target_classes[b][valid_mask].long()).mean()
                total_cls_loss += cls_loss
            
            obj_loss = self.bce_loss(pred_obj[b], target_valid[b].float()).mean()
            total_obj_loss += obj_loss
        
        total_box_loss /= B
        total_cls_loss /= B
        total_obj_loss /= B
        mean_iou = total_iou / max(num_valid, 1)
        
        total_loss = self.lambda_box * total_box_loss + self.lambda_cls * total_cls_loss + self.lambda_obj * total_obj_loss
        
        return {
            'loss': total_loss,
            'box_loss': total_box_loss,
            'cls_loss': total_cls_loss,
            'obj_loss': total_obj_loss,
            'mean_iou': mean_iou
        }


def get_loss_function(phase: int, **kwargs):
    if phase == 2:
        return SingleObjectLoss(**kwargs)
    elif phase == 3:
        return MultiObjectLoss(**kwargs)
    else:
        raise ValueError(f"Invalid phase: {phase}")
