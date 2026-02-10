# part3/loss.py
"""
Part 3 loss for fixed-capacity K=3 detection.

We have:
  pred_boxes:      (B, K, 4)   normalized xyxy in [0,1]
  pred_obj_logits: (B, K)      logits
  pred_cls_logits: (B, K, C)   logits

Targets:
  boxes:  (B, K, 4)  normalized xyxy
  labels: (B, K)     class ids
  mask:   (B, K)     bool (True for real gt objects, False for padding)

Key idea:
- Because K is small (3), we do a simple greedy matching per image:
    match predicted slots to GT boxes by highest IoU (one-to-one).
- Then compute:
    box loss:  (1 - CIoU) on matched pairs
    cls loss:  CrossEntropy on matched pairs
    obj loss:  BCEWithLogits on all K slots (matched=1, unmatched=0)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ----------------------------
# Geometry: IoU + CIoU
# ----------------------------

def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """boxes (...,4) xyxy in [0,1]"""
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1]


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (K,4), b: (N,4) -> iou: (K,N)
    """
    K = a.shape[0]
    N = b.shape[0]
    if K == 0 or N == 0:
        return a.new_zeros((K, N))

    a = a.unsqueeze(1).expand(K, N, 4)
    b = b.unsqueeze(0).expand(K, N, 4)

    inter_tl = torch.maximum(a[..., 0:2], b[..., 0:2])
    inter_br = torch.minimum(a[..., 2:4], b[..., 2:4])
    inter_wh = (inter_br - inter_tl).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)
    union = (area_a + area_b - inter).clamp(min=1e-9)
    return inter / union


def ciou_loss_xyxy(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    pred,tgt: (M,4) xyxy in [0,1]
    returns: (M,) CIoU loss = 1 - CIoU
    """
    # IoU
    iou = pairwise_iou_xyxy(pred, tgt).diagonal() if pred.shape[0] == tgt.shape[0] else None
    if iou is None:
        # fallback: compute elementwise IoU
        iou = pairwise_iou_xyxy(pred, tgt).diagonal()

    # centers
    px = (pred[:, 0] + pred[:, 2]) / 2.0
    py = (pred[:, 1] + pred[:, 3]) / 2.0
    tx = (tgt[:, 0] + tgt[:, 2]) / 2.0
    ty = (tgt[:, 1] + tgt[:, 3]) / 2.0

    # center distance
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    # enclosing box diagonal
    enc_tl = torch.minimum(pred[:, 0:2], tgt[:, 0:2])
    enc_br = torch.maximum(pred[:, 2:4], tgt[:, 2:4])
    c2 = ((enc_br[:, 0] - enc_tl[:, 0]) ** 2 + (enc_br[:, 1] - enc_tl[:, 1]) ** 2).clamp(min=eps)

    # aspect ratio penalty
    pw = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    ph = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    tw = (tgt[:, 2] - tgt[:, 0]).clamp(min=eps)
    th = (tgt[:, 3] - tgt[:, 1]).clamp(min=eps)

    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return 1.0 - ciou


# ----------------------------
# Matching (greedy by IoU)
# ----------------------------

def greedy_match_by_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> List[Tuple[int, int]]:
    """
    pred_boxes: (K,4)
    gt_boxes:   (N,4)
    Returns list of (pred_idx, gt_idx) pairs (one-to-one), length = min(K,N)
    """
    K = pred_boxes.shape[0]
    N = gt_boxes.shape[0]
    if K == 0 or N == 0:
        return []

    iou = pairwise_iou_xyxy(pred_boxes, gt_boxes)  # (K,N)
    # Greedy: repeatedly pick best remaining iou pair
    pairs: List[Tuple[int, int]] = []
    used_p = set()
    used_g = set()

    for _ in range(min(K, N)):
        # mask out used rows/cols
        iou_masked = iou.clone()
        if used_p:
            iou_masked[list(used_p), :] = -1.0
        if used_g:
            iou_masked[:, list(used_g)] = -1.0

        flat_idx = torch.argmax(iou_masked).item()
        p = flat_idx // N
        g = flat_idx % N
        if iou_masked[p, g].item() < 0:
            break
        pairs.append((p, g))
        used_p.add(p)
        used_g.add(g)

    return pairs


# ----------------------------
# Loss module
# ----------------------------

@dataclass
class LossOutput:
    total: torch.Tensor
    loss_box: torch.Tensor
    loss_cls: torch.Tensor
    loss_obj: torch.Tensor


class Part3Loss(nn.Module):
    def __init__(
        self,
        lambda_box: float | None = None,
        lambda_cls: float | None = None,
        lambda_obj: float | None = None,
    ):
        super().__init__()
        cfg = config.PART3_CONFIG
        self.lambda_box = float(lambda_box) if lambda_box is not None else float(cfg["lambda_box"])
        self.lambda_cls = float(lambda_cls) if lambda_cls is not None else float(cfg["lambda_cls"])
        self.lambda_obj = float(lambda_obj) if lambda_obj is not None else float(cfg["lambda_obj"])

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        pred_boxes: torch.Tensor,       # (B,K,4)
        pred_obj_logits: torch.Tensor,  # (B,K)
        pred_cls_logits: torch.Tensor,  # (B,K,C)
        targets: Dict[str, torch.Tensor],
    ) -> LossOutput:
        """
        targets keys: boxes(B,K,4), labels(B,K), mask(B,K)
        """
        gt_boxes_all = targets["boxes"]
        gt_labels_all = targets["labels"]
        gt_mask_all = targets["mask"]

        B, K, _ = pred_boxes.shape
        device = pred_boxes.device

        # Objectness targets for all slots
        obj_tgt = torch.zeros((B, K), device=device, dtype=torch.float32)

        box_losses = []
        cls_losses = []

        for b in range(B):
            gt_mask = gt_mask_all[b]  # (K,)
            n_gt = int(gt_mask.sum().item())
            if n_gt == 0:
                # all slots should be empty
                continue

            gt_boxes = gt_boxes_all[b][gt_mask]    # (N,4)
            gt_labels = gt_labels_all[b][gt_mask]  # (N,)

            pairs = greedy_match_by_iou(pred_boxes[b], gt_boxes)
            if len(pairs) == 0:
                continue

            p_idx = torch.tensor([p for p, _ in pairs], device=device, dtype=torch.long)
            g_idx = torch.tensor([g for _, g in pairs], device=device, dtype=torch.long)

            # Mark matched preds as object-present
            obj_tgt[b, p_idx] = 1.0

            # Box loss: CIoU on matched pairs
            pb = pred_boxes[b, p_idx]  # (M,4)
            gb = gt_boxes[g_idx]       # (M,4)
            l_box = ciou_loss_xyxy(pb, gb).mean()
            box_losses.append(l_box)

            # Class loss on matched pairs
            pc = pred_cls_logits[b, p_idx]  # (M,C)
            gl = gt_labels[g_idx]           # (M,)
            l_cls = self.ce(pc, gl)
            cls_losses.append(l_cls)

        # If no matched pairs in the batch (should be rare), set losses to 0
        if box_losses:
            loss_box = torch.stack(box_losses).mean()
        else:
            loss_box = pred_boxes.sum() * 0.0

        if cls_losses:
            loss_cls = torch.stack(cls_losses).mean()
        else:
            loss_cls = pred_cls_logits.sum() * 0.0

        # Objectness loss across all slots
        loss_obj = self.bce(pred_obj_logits, obj_tgt)

        total = self.lambda_box * loss_box + self.lambda_cls * loss_cls + self.lambda_obj * loss_obj
        return LossOutput(total=total, loss_box=loss_box, loss_cls=loss_cls, loss_obj=loss_obj)


# ----------------------------
# Smoke test: forward + loss + backward
# ----------------------------

def _smoke_test(batch_size: int = 2, device: torch.device | None = None) -> None:
    from torch.utils.data import DataLoader
    from part3.dataset import Part3CocoDataset, collate_fn
    from part3.model import Part3Detector

    device = device or config.DEVICE

    ds = Part3CocoDataset(split="train")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    imgs, targets, metas = next(iter(dl))

    model = Part3Detector().to(device)
    crit = Part3Loss().to(device)

    imgs = imgs.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    model.train()
    out = model(imgs)

    losses = crit(
        pred_boxes=out.pred_boxes,
        pred_obj_logits=out.pred_obj_logits,
        pred_cls_logits=out.pred_cls_logits,
        targets=targets,
    )

    print("[LOSS SMOKE TEST]")
    print("  total:", float(losses.total.item()))
    print("  box  :", float(losses.loss_box.item()))
    print("  cls  :", float(losses.loss_cls.item()))
    print("  obj  :", float(losses.loss_obj.item()))

    # Backward check
    losses.total.backward()
    # Check one gradient exists
    head_grad = None
    for n, p in model.named_parameters():
        if p.grad is not None:
            head_grad = (n, float(p.grad.abs().mean().item()))
            break
    print("  grad sample:", head_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = config.DEVICE if args.device is None else torch.device(args.device)
    _smoke_test(batch_size=args.batch_size, device=device)


if __name__ == "__main__":
    main()
