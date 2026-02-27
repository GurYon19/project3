from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_ce(
    logits: torch.Tensor,                # [K, C]
    targets: torch.Tensor,               # [K]
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,  # [C]
) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(logp, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    fl = ((1.0 - pt) ** gamma) * ce
    return fl.mean()


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    # IoU diagonal
    iou = torch.diag(box_iou_xyxy(pred, target))

    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0

    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-7

    pw = (px2 - px1).clamp(min=1e-7)
    ph = (py2 - py1).clamp(min=1e-7)
    tw = (tx2 - tx1).clamp(min=1e-7)
    th = (ty2 - ty1).clamp(min=1e-7)

    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2) - alpha * v
    return 1.0 - ciou.clamp(min=-1.0, max=1.0)


def match_small_k_bruteforce(cost: torch.Tensor, n_valid_gt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact matching for small K by brute-force permutations.
    cost: [K,K] but only first n_valid_gt columns are real GT.
    returns pred_idx, gt_idx (both length n_valid_gt)
    """
    K = cost.shape[0]
    N = int(n_valid_gt)
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
        total_val = float(total)
        if total_val < best_cost:
            best_cost = total_val
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
    Fixed-slot detection loss (K=3):

    Inputs:
      outputs:
        boxes:  [B,K,4]  xyxy (pixels in resized image)
        logits: [B,K,C]  C includes background class
      targets:
        boxes:  [B,K,4]
        labels: [B,K]
        mask:   [B,K]  True for real objects (not padded)

    Steps:
      1) For each image, match predictions to GT (only mask=True) by brute-force Hungarian
      2) Box CIoU loss for matched pairs
      3) Classification loss on all K slots using assigned labels (unmatched -> background)
    """

    def __init__(
        self,
        num_classes_total: int,
        bg_id: int,
        max_objects: int = 3,
        weights: LossWeights = LossWeights(),
        class_weights: torch.Tensor | None = None,  # [C]
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        match_cls_cost_weight: float = 0.5,
    ):
        super().__init__()
        self.C = int(num_classes_total)
        self.bg_id = int(bg_id)
        self.K = int(max_objects)
        self.w = weights
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)
        self.match_cls_cost_weight = float(match_cls_cost_weight)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            assert cw.ndim == 1 and cw.numel() == self.C
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["boxes"]      # [B,K,4]
        pred_logits = outputs["logits"]    # [B,K,C]

        tgt_boxes = targets["boxes"]       # [B,K,4]
        tgt_labels = targets["labels"]     # [B,K]
        tgt_mask = targets["mask"]         # [B,K]

        B = pred_boxes.shape[0]
        device = pred_boxes.device

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        matched_images = 0

        for b in range(B):
            pb = pred_boxes[b]    # [K,4]
            pl = pred_logits[b]   # [K,C]

            tb = tgt_boxes[b]     # [K,4]
            tl = tgt_labels[b]    # [K]
            m = tgt_mask[b]       # [K]

            n_gt = int(m.long().sum().item())

            # default: all predictions are background
            assigned_labels = torch.full((self.K,), self.bg_id, dtype=torch.long, device=device)

            if n_gt > 0:
                tb_valid = tb[m]     # [N,4]
                tl_valid = tl[m]     # [N]

                # Cost = (1 - IoU) + w * class cost
                iou = box_iou_xyxy(pb, tb_valid)  # [K,N]
                logp = F.log_softmax(pl, dim=-1)  # [K,C]
                cls_cost = -logp[:, tl_valid]     # [K,N]
                cost_kn = (1.0 - iou) + self.match_cls_cost_weight * cls_cost

                # Expand to [K,K] for matcher convenience
                cost_full = torch.full((self.K, self.K), fill_value=10.0, device=device, dtype=cost_kn.dtype)
                cost_full[:, :n_gt] = cost_kn

                pred_idx, gt_idx = match_small_k_bruteforce(cost_full, n_valid_gt=n_gt)

                assigned_labels[pred_idx] = tl_valid[gt_idx]

                # Box loss for matched pairs
                box_l = ciou_loss_xyxy(pb[pred_idx], tb_valid[gt_idx]).mean()
                total_box = total_box + box_l
                matched_images += 1

            # Classification loss (all K slots)
            if self.use_focal:
                cls_l = focal_loss_ce(pl, assigned_labels, gamma=self.focal_gamma, weight=self.class_weights)
            else:
                cls_l = F.cross_entropy(pl, assigned_labels, weight=self.class_weights, reduction="mean")

            total_cls = total_cls + cls_l

        total_cls = total_cls / B
        total_box = (total_box / matched_images) if matched_images > 0 else torch.tensor(0.0, device=device)

        loss = self.w.cls * total_cls + self.w.box * total_box
        return {
            "loss": loss,
            "loss_cls": total_cls.detach(),
            "loss_box": total_box.detach(),
        }