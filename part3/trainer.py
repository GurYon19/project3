from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import random
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def _mean_iou_matched(pred_boxes, pred_logits, tgt_boxes, tgt_labels, tgt_mask, bg_id: int) -> torch.Tensor:
    """
    Greedy class-consistent matching (for monitoring only).
    Computes mean IoU over GT objects by matching predicted boxes of same class.
    """
    B, K, _ = pred_boxes.shape
    total_iou = 0.0
    total_gt = 0

    pred_cls = pred_logits.argmax(dim=-1)  # [B,K]

    for b in range(B):
        N = int(tgt_mask[b].long().sum().item())
        if N == 0:
            continue

        gt_boxes = tgt_boxes[b, :N]
        gt_labels = tgt_labels[b, :N]

        keep = pred_cls[b] != bg_id
        if keep.sum().item() == 0:
            total_gt += N
            continue

        pb = pred_boxes[b][keep]
        pc = pred_cls[b][keep]
        if pb.numel() == 0:
            total_gt += N
            continue

        ious = _box_iou_xyxy(pb, gt_boxes)

        used_p = set()
        used_g = set()
        cand = []
        for pi in range(ious.shape[0]):
            for gi in range(ious.shape[1]):
                if int(pc[pi].item()) != int(gt_labels[gi].item()):
                    continue
                cand.append((float(ious[pi, gi].item()), pi, gi))
        cand.sort(reverse=True, key=lambda x: x[0])

        matched_iou = 0.0
        for v, pi, gi in cand:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            matched_iou += v

        total_iou += matched_iou
        total_gt += N

    if total_gt == 0:
        return torch.tensor(0.0, device=pred_boxes.device)
    return torch.tensor(total_iou / total_gt, device=pred_boxes.device)


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_metric: float = -1e9


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str | Path = "logs/part3",
        ckpt_dir: str | Path = "checkpoints/part3",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        background_id: int = 3,
        grad_clip: float = 1.0,
        amp: bool = False,
        multi_scale: bool = False,
        ms_sizes: Optional[list[int]] = None,
        base_image_size: int = 448,
        backbone_lr_mult: float = 0.2,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.background_id = int(background_id)
        self.grad_clip = float(grad_clip) if grad_clip is not None else 0.0

        # AMP only on CUDA (safe on MPS/CPU)
        self.amp = bool(amp) and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.state = TrainState()

        self.multi_scale = bool(multi_scale)
        self.ms_sizes = ms_sizes or []
        self.base_image_size = int(base_image_size)
        self.backbone_lr_mult = float(backbone_lr_mult)

        if self.multi_scale and len(self.ms_sizes) == 0:
            self.multi_scale = False
        if self.multi_scale:
            print(f"[TRAIN] Multi-scale enabled sizes={self.ms_sizes}")

    def save(self, name: str):
        path = self.ckpt_dir / name
        obj = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "state": self.state.__dict__,
        }
        torch.save(obj, path)

    def load(self, path: str | Path):
        obj = torch.load(path, map_location=self.device)
        self.model.load_state_dict(obj["model"])
        self.optimizer.load_state_dict(obj["optimizer"])
        if self.scheduler and obj.get("scheduler") is not None:
            self.scheduler.load_state_dict(obj["scheduler"])
        self.state = TrainState(**obj.get("state", {}))

    def _set_backbone_lr_groups(self):
        """
        After unfreezing, optionally reduce backbone LR relative to head LR.
        We detect backbone parameters by name if possible; otherwise fall back to all params.
        """
        # If optimizer already has param groups > 1, do nothing
        if len(self.optimizer.param_groups) > 1:
            return

        # Try to separate backbone and non-backbone params
        backbone_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                head_params.append(p)

        # If we couldn't split, don't change anything
        if len(backbone_params) == 0 or len(head_params) == 0:
            return

        base_lr = self.optimizer.param_groups[0]["lr"]
        wd = self.optimizer.param_groups[0].get("weight_decay", 0.0)

        self.optimizer.param_groups.clear()
        self.optimizer.add_param_group({"params": head_params, "lr": base_lr, "weight_decay": wd})
        self.optimizer.add_param_group(
            {"params": backbone_params, "lr": base_lr * self.backbone_lr_mult, "weight_decay": wd}
        )

        print(f"[OPT] Split param groups: head_lr={base_lr:.2e} backbone_lr={base_lr*self.backbone_lr_mult:.2e}")

    def _apply_multiscale(
        self, images: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        """
        Per-batch multi-scale resizing.
        Returns (images, targets, new_size). If no resize, new_size=old_size.
        """
        if (not self.multi_scale) or (len(self.ms_sizes) == 0):
            return images, targets, images.shape[-1]

        B, C, H, W = images.shape
        old_h, old_w = H, W
        new_size = random.choice(self.ms_sizes)

        if hasattr(self.criterion, "image_size"):
            self.criterion.image_size = int(new_size)

        if new_size == old_h and new_size == old_w:
            return images, targets, new_size

        images = F.interpolate(images, size=(new_size, new_size), mode="bilinear", align_corners=False)

        boxes = targets["boxes"]
        scale_x = new_size / float(old_w)
        scale_y = new_size / float(old_h)

        boxes = boxes.clone()
        boxes[..., 0] *= scale_x
        boxes[..., 2] *= scale_x
        boxes[..., 1] *= scale_y
        boxes[..., 3] *= scale_y

        boxes[..., 0::2] = boxes[..., 0::2].clamp(0, new_size - 1)
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0, new_size - 1)

        targets = dict(targets)
        targets["boxes"] = boxes

        return images, targets, new_size

    def train_one_epoch(self, loader, epoch: int):
        self.model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        n_batches = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = {
                "boxes": targets["boxes"].to(self.device),
                "labels": targets["labels"].to(self.device),
                "mask": targets["mask"].to(self.device),
            }

            # Multi-scale (train only)
            prev_size = getattr(self.model, "image_size", self.base_image_size)
            images, targets, new_size = self._apply_multiscale(images, targets)
            if hasattr(self.model, "image_size"):
                try:
                    self.model.image_size = int(new_size)
                except Exception:
                    pass

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets)
                    loss = losses["loss"]
                self.scaler.scale(loss).backward()
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                loss = losses["loss"]
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # restore base image size so validation is stable even if train used multiscale
            if hasattr(self.model, "image_size"):
                try:
                    self.model.image_size = int(prev_size)
                except Exception:
                    pass

            running["loss"] += float(loss.item())
            running["loss_cls"] += float(losses["loss_cls"].item())
            running["loss_box"] += float(losses["loss_box"].item())
            n_batches += 1

            lr0 = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/step_lr", lr0, self.state.global_step)
            self.writer.add_scalar("train/step_loss", float(loss.item()), self.state.global_step)
            self.writer.add_scalar("train/step_loss_cls", float(losses["loss_cls"].item()), self.state.global_step)
            self.writer.add_scalar("train/step_loss_box", float(losses["loss_box"].item()), self.state.global_step)

            self.state.global_step += 1

        for k in running:
            running[k] /= max(1, n_batches)

        self.writer.add_scalar("train/epoch_loss", running["loss"], epoch)
        self.writer.add_scalar("train/epoch_loss_cls", running["loss_cls"], epoch)
        self.writer.add_scalar("train/epoch_loss_box", running["loss_box"], epoch)

        if self.scheduler:
            self.scheduler.step()

        return running

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> Dict[str, float]:
        self.model.eval()

        # Ensure model is on base image size in val
        if hasattr(self.model, "image_size"):
            try:
                self.model.image_size = int(self.base_image_size)
            except Exception:
                pass

        running = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "mean_iou": 0.0}
        n_batches = 0

        for images, targets in loader:
            images = images.to(self.device)
            tgt = {
                "boxes": targets["boxes"].to(self.device),
                "labels": targets["labels"].to(self.device),
                "mask": targets["mask"].to(self.device),
            }

            outputs = self.model(images)
            losses = self.criterion(outputs, tgt)

            miou = _mean_iou_matched(
                outputs["boxes"],
                outputs["logits"],
                tgt["boxes"],
                tgt["labels"],
                tgt["mask"],
                bg_id=self.background_id,
            )

            running["loss"] += float(losses["loss"].item())
            running["loss_cls"] += float(losses["loss_cls"].item())
            running["loss_box"] += float(losses["loss_box"].item())
            running["mean_iou"] += float(miou.item())
            n_batches += 1

        for k in running:
            running[k] /= max(1, n_batches)

        self.writer.add_scalar("val/loss", running["loss"], epoch)
        self.writer.add_scalar("val/loss_cls", running["loss_cls"], epoch)
        self.writer.add_scalar("val/loss_box", running["loss_box"], epoch)
        self.writer.add_scalar("val/mean_iou", running["mean_iou"], epoch)

        return running

    def fit(self, train_loader, val_loader, epochs: int, unfreeze_epoch: int = 10, save_best: bool = True):
        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch

            # Unfreeze backbone at epoch threshold
            if epoch == unfreeze_epoch and hasattr(self.model, "unfreeze_backbone"):
                print(f"[TRAIN] Unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()
                self._set_backbone_lr_groups()

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            self.save("last.pth")

            if save_best:
                score = float(val_metrics["mean_iou"])
                if score > self.state.best_val_metric:
                    self.state.best_val_metric = score
                    self.save("best.pth")

            print(
                f"[E{epoch:03d}] "
                f"train loss={train_metrics['loss']:.4f} (cls={train_metrics['loss_cls']:.4f}, box={train_metrics['loss_box']:.4f}) | "
                f"val loss={val_metrics['loss']:.4f} miou={val_metrics['mean_iou']:.4f}"
            )