# part3/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import random


def _mean_iou_matched(pred_boxes, pred_logits, tgt_boxes, tgt_labels, tgt_mask, bg_id: int) -> torch.Tensor:
    from part3.loss import box_iou_xyxy

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

        ious = box_iou_xyxy(pb, gt_boxes)

        used_p = set()
        used_g = set()
        cand = []
        for pi in range(ious.shape[0]):
            for gi in range(ious.shape[1]):
                if pc[pi].item() != gt_labels[gi].item():
                    continue
                cand.append((ious[pi, gi].item(), pi, gi))
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
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.background_id = background_id

        self.amp = bool(amp) and (device.type == "cuda")

        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Updated AMP API (only used on CUDA)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.state = TrainState()

        self.multi_scale = bool(multi_scale)
        self.ms_sizes = ms_sizes or []
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

    def _apply_multiscale(self, images: torch.Tensor, targets: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Per-batch multi-scale resizing.
        images: [B,3,H,W]  (usually square)
        targets['boxes']: [B,K,4] in pixel coords for the current H,W
        """
        if (not self.multi_scale) or (len(self.ms_sizes) == 0):
            return images, targets

        B, C, H, W = images.shape
        # Assume square inputs in this project; keep it robust anyway:
        old_h, old_w = H, W

        new_size = random.choice(self.ms_sizes)
        if new_size == old_h and new_size == old_w:
            return images, targets

        # Resize images
        images = F.interpolate(images, size=(new_size, new_size), mode="bilinear", align_corners=False)

        # Scale boxes
        boxes = targets["boxes"]
        scale_x = new_size / float(old_w)
        scale_y = new_size / float(old_h)

        boxes = boxes.clone()
        boxes[..., 0] *= scale_x
        boxes[..., 2] *= scale_x
        boxes[..., 1] *= scale_y
        boxes[..., 3] *= scale_y

        # Clamp to new image bounds
        boxes[..., 0::2] = boxes[..., 0::2].clamp(0, new_size - 1)
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0, new_size - 1)

        targets = dict(targets)
        targets["boxes"] = boxes

        # If model uses a stored image_size for scaling outputs, update it
        if hasattr(self.model, "image_size"):
            try:
                self.model.image_size = new_size
            except Exception:
                pass

        return images, targets

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
            images, targets = self._apply_multiscale(images, targets)

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

            running["loss"] += float(loss.item())
            running["loss_cls"] += float(losses["loss_cls"].item())
            running["loss_box"] += float(losses["loss_box"].item())
            n_batches += 1

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/lr", lr, self.state.global_step)
            self.writer.add_scalar("train/loss", float(loss.item()), self.state.global_step)
            self.writer.add_scalar("train/loss_cls", float(losses["loss_cls"].item()), self.state.global_step)
            self.writer.add_scalar("train/loss_box", float(losses["loss_box"].item()), self.state.global_step)
            self.state.global_step += 1

        for k in running:
            running[k] /= max(1, n_batches)

        self.writer.add_scalar("train_epoch/loss", running["loss"], epoch)
        self.writer.add_scalar("train_epoch/loss_cls", running["loss_cls"], epoch)
        self.writer.add_scalar("train_epoch/loss_box", running["loss_box"], epoch)

        if self.scheduler:
            self.scheduler.step()

        return running

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
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

            if epoch == unfreeze_epoch and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            self.save("last.pth")

            if save_best:
                score = val_metrics["mean_iou"]
                if score > self.state.best_val_metric:
                    self.state.best_val_metric = score
                    self.save("best.pth")

            print(
                f"[E{epoch:03d}] "
                f"train loss={train_metrics['loss']:.4f} (cls={train_metrics['loss_cls']:.4f}, box={train_metrics['loss_box']:.4f}) | "
                f"val loss={val_metrics['loss']:.4f} miou={val_metrics['mean_iou']:.4f}"
            )
