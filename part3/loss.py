from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
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


def match_small_k_bruteforce(cost_full: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    K = cost_full.shape[0]
    if N <= 0:
        return (
            torch.empty((0,), dtype=torch.long, device=cost_full.device),
            torch.empty((0,), dtype=torch.long, device=cost_full.device),
        )

    gt_inds = list(range(N))
    best_cost = float("inf")
    best_perm = None

    for pred_subset in itertools.permutations(range(K), N):
        total = 0.0
        for pi, gi in zip(pred_subset, gt_inds):
            total = total + cost_full[pi, gi]
        total_val = float(total.detach())
        if total_val < best_cost:
            best_cost = total_val
            best_perm = pred_subset

    pred_idx = torch.tensor(best_perm, dtype=torch.long, device=cost_full.device)
    gt_idx = torch.tensor(gt_inds, dtype=torch.long, device=cost_full.device)
    return pred_idx, gt_idx


@dataclass
class LossWeights:
    cls: float = 1.0
    box: float = 5.0


class FixedSlotLoss(nn.Module):
    """
    Consistent fixed-slot loss:
      - matching uses IoU + class cost (requires boxes be valid/ordered)
      - box loss uses SmoothL1 directly on pixel xyxy (same representation as eval)
    """

    def __init__(
        self,
        num_classes: int,
        max_objects: int = 3,
        weights: LossWeights = LossWeights(),
        class_weights: torch.Tensor | None = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        cls_cost_weight: float = 0.5,
        iou_cost_weight: float = 1.0,
        image_size: int = 448,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.bg = self.num_classes
        self.K = int(max_objects)
        self.w = weights
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)

        self.cls_cost_weight = float(cls_cost_weight)
        self.iou_cost_weight = float(iou_cost_weight)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            assert cw.ndim == 1 and cw.numel() == (self.num_classes + 1)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

        self.box_reg = nn.SmoothL1Loss(reduction="mean")
        self.image_size = int(image_size)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["boxes"]      # [B,K,4] pixel xyxy (MUST be valid)
        pred_logits = outputs["logits"]    # [B,K,C+1]

        tgt_boxes = targets["boxes"]       # [B,K,4] pixel xyxy
        tgt_labels = targets["labels"]     # [B,K]
        tgt_mask = targets["mask"]         # [B,K]

        B = pred_boxes.shape[0]
        device = pred_boxes.device

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        total_matched = 0

        for b in range(B):
            pb = pred_boxes[b]
            pl = pred_logits[b]

            tb = tgt_boxes[b]
            tl = tgt_labels[b]
            m = tgt_mask[b]
            N = int(m.long().sum().item())

            assigned_labels = torch.full((self.K,), self.bg, dtype=torch.long, device=device)

            if N > 0:
                tb_valid = tb[:N]
                tl_valid = tl[:N]

                iou = box_iou_xyxy(pb, tb_valid)      # [K,N]
                logp = F.log_softmax(pl, dim=-1)      # [K,C+1]
                cls_cost = -logp[:, tl_valid]         # [K,N]

                cost = (self.iou_cost_weight * (1.0 - iou)) + (self.cls_cost_weight * cls_cost)

                cost_full = torch.full((self.K, self.K), 10.0, device=device, dtype=cost.dtype)
                cost_full[:, :N] = cost

                pred_idx, gt_idx = match_small_k_bruteforce(cost_full, N=N)

                assigned_labels[pred_idx] = tl_valid[gt_idx]

                matched_pb = pb[pred_idx]
                matched_tb = tb_valid[gt_idx]
                s = max(float(self.image_size - 1), 1.0)
                pred_n = matched_pb / s
                tgt_n  = matched_tb / s

                # IMPORTANT: restore gradient scale so pixel-space learning isn't 447× slower
                total_box = total_box + (self.box_reg(pred_n, tgt_n) * s)
                total_matched += int(pred_idx.numel())

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