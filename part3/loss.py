from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Classification losses
# -------------------------

def focal_loss_ce(
    logits: torch.Tensor,                # [K, C+1]
    targets: torch.Tensor,               # [K]
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,  # [C+1] optional
) -> torch.Tensor:
    """
    Focal loss on top of cross-entropy:
      FL = (1 - pt)^gamma * CE
    Returns mean loss over K.
    """
    logp = F.log_softmax(logits, dim=-1)                           # [K, C+1]
    ce = F.nll_loss(logp, targets, weight=weight, reduction="none") # [K]
    pt = torch.exp(-ce)                                            # [K]
    fl = ((1.0 - pt) ** gamma) * ce
    return fl.mean()


# -------------------------
# Box helpers (SAFE for raw preds)
# -------------------------

def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1)
    h = (y2 - y1)
    return torch.stack([cx, cy, w, h], dim=-1)


def _safe_pred_to_norm_xyxy(pred_boxes_px: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Convert raw predicted pixel xyxy -> valid normalized xyxy in [0,1],
    using a tanh-based parameterization to avoid sigmoid saturation.

    Input:  pred_boxes_px [..., 4] raw xyxy in pixels (can be any real)
    Output: norm_xyxy   [..., 4] valid ordered xyxy normalized to [0,1]
    """
    # Convert raw xyxy into raw center/size (in pixel units)
    # (raw may produce negative w/h; that's fine)
    cxcywh = _xyxy_to_cxcywh(pred_boxes_px)
    cx_raw, cy_raw, w_raw, h_raw = cxcywh.unbind(dim=-1)

    # Map centers into [0,1] using tanh, then clamp slightly for stability
    cx_n = 0.5 * (torch.tanh(cx_raw / float(image_size)) + 1.0)
    cy_n = 0.5 * (torch.tanh(cy_raw / float(image_size)) + 1.0)

    # Map sizes into (0,1] using tanh as well (positive, smooth)
    # Start with abs to avoid sign flips, then tanh to bound
    w_n = torch.tanh(torch.abs(w_raw) / float(image_size)).clamp(min=1e-4)
    h_n = torch.tanh(torch.abs(h_raw) / float(image_size)).clamp(min=1e-4)

    # Convert back to xyxy normalized
    x1 = (cx_n - 0.5 * w_n).clamp(0.0, 1.0)
    y1 = (cy_n - 0.5 * h_n).clamp(0.0, 1.0)
    x2 = (cx_n + 0.5 * w_n).clamp(0.0, 1.0)
    y2 = (cy_n + 0.5 * h_n).clamp(0.0, 1.0)

    # Ensure ordering + non-degenerate
    x1o = torch.minimum(x1, x2)
    y1o = torch.minimum(y1, y2)
    x2o = torch.maximum(x1, x2)
    y2o = torch.maximum(y1, y2)

    # Guarantee at least 1 pixel in normalized units
    eps = 1.0 / max(float(image_size), 1.0)
    x2o = torch.maximum(x2o, x1o + eps)
    y2o = torch.maximum(y2o, y1o + eps)

    return torch.stack([x1o, y1o, x2o, y2o], dim=-1)


def _gt_px_to_norm_xyxy(gt_boxes_px: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    GT is already valid; just normalize + clamp.
    """
    s = max(float(image_size - 1), 1.0)
    b = gt_boxes_px / s
    x1, y1, x2, y2 = b.unbind(dim=-1)
    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = x2.clamp(0.0, 1.0)
    y2 = y2.clamp(0.0, 1.0)
    # ensure ordering
    x1o = torch.minimum(x1, x2)
    y1o = torch.minimum(y1, y2)
    x2o = torch.maximum(x1, x2)
    y2o = torch.maximum(y1, y2)
    eps = 1.0 / max(float(image_size), 1.0)
    x2o = torch.maximum(x2o, x1o + eps)
    y2o = torch.maximum(y2o, y1o + eps)
    return torch.stack([x1o, y1o, x2o, y2o], dim=-1)


def _pairwise_l1_cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [K,4], b: [N,4] -> cost [K,N] using L1 on cxcywh (normalized or px)
    """
    ac = _xyxy_to_cxcywh(a)  # [K,4]
    bc = _xyxy_to_cxcywh(b)  # [N,4]
    # broadcast
    diff = (ac[:, None, :] - bc[None, :, :]).abs().sum(dim=-1)
    return diff


# -------------------------
# Matching (brute force, small K)
# -------------------------

def match_small_k_bruteforce(cost_full: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact matching for small K using brute-force permutations.
    cost_full: [K,K] where columns 0..N-1 are real GTs and remaining are dummy.
    Returns pred_idx, gt_idx (gt_idx in [0..N-1])
    """
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


# -------------------------
# Main loss
# -------------------------

@dataclass
class LossWeights:
    cls: float = 1.0
    box: float = 5.0


class FixedSlotLoss(nn.Module):
    """
    Loss for fixed-slot detection:
      - stable matching (no IoU in cost; works with raw box predictions)
      - classification loss on C+1 classes (incl background)
      - SmoothL1 on normalized xyxy after safe pred transform

    This is designed to be robust when the model outputs raw/unconstrained boxes during training.
    """

    def __init__(
        self,
        num_classes: int,
        max_objects: int = 3,
        weights: LossWeights = LossWeights(),
        class_weights: torch.Tensor | None = None,   # [C+1]
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        image_size: int = 448,
        cls_cost_weight: float = 0.5,
        box_cost_weight: float = 1.0,
        dummy_cost: float = 10.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.bg = self.num_classes
        self.K = int(max_objects)
        self.w = weights
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)

        self.image_size = int(image_size)
        self.cls_cost_weight = float(cls_cost_weight)
        self.box_cost_weight = float(box_cost_weight)
        self.dummy_cost = float(dummy_cost)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            assert cw.ndim == 1 and cw.numel() == (self.num_classes + 1)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

        self.box_reg = nn.SmoothL1Loss(reduction="mean")

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["boxes"]      # [B,K,4] RAW (pixels)
        pred_logits = outputs["logits"]    # [B,K,C+1]

        tgt_boxes = targets["boxes"]       # [B,K,4] pixels
        tgt_labels = targets["labels"]     # [B,K]
        tgt_mask = targets["mask"]         # [B,K] bool

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

            # background default for all K slots
            assigned_labels = torch.full((self.K,), self.bg, dtype=torch.long, device=device)

            if N > 0:
                tb_valid = tb[:N]      # [N,4] px
                tl_valid = tl[:N]      # [N]

                # ----- Stable cost matrix -----
                # Box cost: L1 on cxcywh in pixel space (robust even if pb is raw)
                box_cost = _pairwise_l1_cost(pb, tb_valid)  # [K,N]

                # Class cost: negative log prob of the GT class
                logp = F.log_softmax(pl, dim=-1)            # [K,C+1]
                cls_cost = -logp[:, tl_valid]               # [K,N]

                cost = (self.box_cost_weight * box_cost) + (self.cls_cost_weight * cls_cost)  # [K,N]

                # pad to [K,K] for brute-force (dummy columns)
                cost_full = torch.full((self.K, self.K), fill_value=self.dummy_cost, device=device, dtype=cost.dtype)
                cost_full[:, :N] = cost

                pred_idx, gt_idx = match_small_k_bruteforce(cost_full, N=N)

                # assign labels for matched preds
                assigned_labels[pred_idx] = tl_valid[gt_idx]

                # ----- Box regression loss on safe-normalized boxes -----
                matched_pb = pb[pred_idx]           # [N,4] raw px
                matched_tb = tb_valid[gt_idx]       # [N,4] gt px

                pred_norm = _safe_pred_to_norm_xyxy(matched_pb, image_size=self.image_size)   # [N,4]
                tgt_norm = _gt_px_to_norm_xyxy(matched_tb, image_size=self.image_size)        # [N,4]

                box_l = self.box_reg(pred_norm, tgt_norm)
                total_box = total_box + box_l
                total_matched += int(pred_idx.numel())

            # ----- Classification loss over all K slots -----
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