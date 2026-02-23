# part3/loss.py
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_ce(
    logits: torch.Tensor,          # [K, C+1]
    targets: torch.Tensor,         # [K]
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,  # [C+1] optional
) -> torch.Tensor:
    """
    Focal loss built on top of cross-entropy:
      FL = (1 - pt)^gamma * CE
    Returns mean loss over K.
    """
    logp = F.log_softmax(logits, dim=-1)           # [K, C+1]
    ce = F.nll_loss(logp, targets, weight=weight, reduction="none")  # [K]
    pt = torch.exp(-ce)                            # [K]
    fl = ((1.0 - pt) ** gamma) * ce
    return fl.mean()


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU between sets of boxes in xyxy.
    a: [N,4], b: [M,4] => [N,M]
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)

    union = area_a + area_b - inter + 1e-7
    return inter / union


def ciou_loss_xyxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CIoU loss for matched pairs of boxes.
    pred:   [N,4]
    target: [N,4]
    returns: [N]
    """
    # IoU
    iou = torch.diag(box_iou_xyxy(pred, target))  # [N]

    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # centers
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0

    # squared center distance
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # smallest enclosing box diagonal squared
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-7

    # aspect ratio term
    pw = (px2 - px1).clamp(min=1e-7)
    ph = (py2 - py1).clamp(min=1e-7)
    tw = (tx2 - tx1).clamp(min=1e-7)
    th = (ty2 - ty1).clamp(min=1e-7)

    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2) - alpha * v
    return 1.0 - ciou.clamp(min=-1.0, max=1.0)


def match_small_k_bruteforce(cost: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact matching for small K using brute-force permutations.
    Works well for K up to ~6.

    cost: [K,K] cost matrix (pred_i vs gt_j)
    valid_mask: [K] bool, True for real GT objects, False for padded GT slots.
               We'll only match to the first N valid GT slots.

    Returns:
      pred_idx: [N] indices in [0..K-1]
      gt_idx:   [N] indices in [0..N-1]
    """
    K = cost.shape[0]
    assert cost.shape == (K, K)

    N = int(valid_mask.long().sum().item())
    if N == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=cost.device),
            torch.empty((0,), dtype=torch.long, device=cost.device),
        )

    gt_inds = list(range(N))
    best_cost = float("inf")
    best_perm = None

    for pred_subset in itertools.permutations(range(K), N):
        total = 0.0
        for pi, gi in zip(pred_subset, gt_inds):
            total = total + cost[pi, gi]
        total_val = float(total.detach())
        if total_val < best_cost:
            best_cost = total_val
            best_perm = pred_subset

    pred_idx = torch.tensor(best_perm, dtype=torch.long, device=cost.device)
    gt_idx = torch.tensor(gt_inds, dtype=torch.long, device=cost.device)
    return pred_idx, gt_idx


def hungarian_match_k3(cost: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Very small matching for fixed K=3 using brute-force permutations.

    cost: [K,K] cost matrix (pred_i vs gt_j)
    valid_mask: [K] bool, True for real GT objects, False for padded GT slots.
               We'll only match to the first N valid GT slots.

    Returns:
      pred_idx: [N] indices in [0..K-1]
      gt_idx:   [N] indices in [0..N-1]
    """
    K = cost.shape[0]
    assert cost.shape == (K, K)

    # Determine N valid GT
    N = int(valid_mask.long().sum().item())
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=cost.device), torch.empty((0,), dtype=torch.long, device=cost.device)

    gt_inds = list(range(N))
    best_cost = None
    best_perm = None

    # Choose N predictions out of K, and assign them to N GT in some order.
    for pred_subset in itertools.permutations(range(K), N):
        total = 0.0
        for pi, gi in zip(pred_subset, gt_inds):
            total = total + cost[pi, gi]
        if best_cost is None or total < best_cost:
            best_cost = total
            best_perm = pred_subset

    pred_idx = torch.tensor(best_perm, dtype=torch.long, device=cost.device)
    gt_idx = torch.tensor(gt_inds, dtype=torch.long, device=cost.device)
    return pred_idx, gt_idx


@dataclass
class LossWeights:
    cls: float = 1.0
    box: float = 5.0


class FixedSlotLoss(nn.Module):
    """
    Loss for fixed-slot detection (K=3):
      - Hungarian matching (brute-force for K=3)
      - Classification loss (CE) over C+1 classes (incl background), optionally class-weighted
      - CIoU loss for matched boxes
    """

    def __init__(
        self,
        num_classes: int,
        max_objects: int = 3,
        weights: LossWeights = LossWeights(),
        class_weights: torch.Tensor | None = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg = num_classes
        self.K = max_objects
        self.w = weights
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            assert cw.ndim == 1 and cw.numel() == (num_classes + 1)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["boxes"]      # [B,K,4]
        pred_logits = outputs["logits"]    # [B,K,C+1]

        tgt_boxes = targets["boxes"]       # [B,K,4]
        tgt_labels = targets["labels"]     # [B,K]
        tgt_mask = targets["mask"]         # [B,K]

        B = pred_boxes.shape[0]
        device = pred_boxes.device

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        total_matched = 0

        for b in range(B):
            pb = pred_boxes[b]    # [K,4]
            pl = pred_logits[b]   # [K,C+1]

            tb = tgt_boxes[b]     # [K,4]
            tl = tgt_labels[b]    # [K]
            m = tgt_mask[b]       # [K]

            N = int(m.long().sum().item())

            # Default label for all K predictions is background
            assigned_labels = torch.full((self.K,), self.bg, dtype=torch.long, device=device)

            if N > 0:
                tb_valid = tb[:N]
                tl_valid = tl[:N]

                # Cost = (1 - IoU) + class cost
                iou = box_iou_xyxy(pb, tb_valid)  # [K,N]
                logp = F.log_softmax(pl, dim=-1)  # [K,C+1]
                cls_cost = -logp[:, tl_valid]     # [K,N]
                cost = (1.0 - iou) + 0.5 * cls_cost

                # Pad cost to [K,K] for brute-force matcher
                cost_full = torch.full((self.K, self.K), fill_value=10.0, device=device, dtype=cost.dtype)
                cost_full[:, :N] = cost

                pred_idx, gt_idx = match_small_k_bruteforce(cost_full, valid_mask=m)

                # Assign GT labels to matched prediction slots
                assigned_labels[pred_idx] = tl_valid[gt_idx]

                # Box loss for matched pairs
                matched_pb = pb[pred_idx]
                matched_tb = tb_valid[gt_idx]
                box_l = ciou_loss_xyxy(matched_pb, matched_tb).mean()
                total_box = total_box + box_l
                total_matched += len(pred_idx)

            # Classification loss over all K slots (weighted)
            if self.use_focal:
                cls_l = focal_loss_ce(pl, assigned_labels, gamma=self.focal_gamma, weight=self.class_weights)
            else:
                cls_l = F.cross_entropy(pl, assigned_labels, weight=self.class_weights, reduction="mean")

            total_cls = total_cls + cls_l

        total_cls = total_cls / B
        total_box = total_box / B if total_matched > 0 else torch.tensor(0.0, device=device)

        loss = self.w.cls * total_cls + self.w.box * total_box
        return {
            "loss": loss,
            "loss_cls": total_cls.detach(),
            "loss_box": total_box.detach(),
        }
