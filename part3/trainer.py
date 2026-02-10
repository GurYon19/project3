# part3/trainer.py
"""
Trainer for Part 3 fixed-capacity multi-object detection.

Responsibilities:
- Train loop (forward -> loss -> backward -> step)
- Validation loop (compute val loss)
- Progressive unfreezing of backbone
- Checkpointing (best model by val loss)
- Basic TensorBoard logging

Note: We'll add mAP evaluation later in evaluate.py; for now we validate by loss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

import config
from part3.loss import Part3Loss


@dataclass
class TrainStats:
    epoch: int
    train_total: float
    train_box: float
    train_cls: float
    train_obj: float
    val_total: float
    val_box: float
    val_cls: float
    val_obj: float
    lr: float
    seconds: float


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader,
        val_loader,
        device: torch.device,
        out_dir: Path,
        log_dir: Path,
        loss_fn: Optional[Part3Loss] = None,
        unfreeze_epoch: int = 30,
        grad_clip: Optional[float] = 1.0,
        log_every: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = Path(out_dir)
        self.log_dir = Path(log_dir)
        self.loss_fn = loss_fn or Part3Loss()
        self.unfreeze_epoch = int(unfreeze_epoch)
        self.grad_clip = grad_clip
        self.log_every = int(log_every)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.best_val = float("inf")

    def close(self):
        self.writer.close()

    def _step_batch(self, imgs, targets, train: bool) -> Dict[str, torch.Tensor]:
        imgs = imgs.to(self.device, non_blocking=True)
        targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            out = self.model(imgs)
            losses = self.loss_fn(
                pred_boxes=out.pred_boxes,
                pred_obj_logits=out.pred_obj_logits,
                pred_cls_logits=out.pred_cls_logits,
                targets=targets,
            )

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

        return {
            "total": losses.total.detach(),
            "box": losses.loss_box.detach(),
            "cls": losses.loss_cls.detach(),
            "obj": losses.loss_obj.detach(),
        }

    @torch.no_grad()
    def _run_val(self) -> Dict[str, float]:
        self.model.eval()
        sums = {"total": 0.0, "box": 0.0, "cls": 0.0, "obj": 0.0}
        n = 0

        for imgs, targets, _ in self.val_loader:
            batch_losses = self._step_batch(imgs, targets, train=False)
            bs = imgs.shape[0]
            for k in sums:
                sums[k] += float(batch_losses[k].item()) * bs
            n += bs

        for k in sums:
            sums[k] /= max(1, n)
        return sums

    def fit(self, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Progressive unfreeze
            if epoch == self.unfreeze_epoch:
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                else:
                    for p in self.model.backbone.parameters():
                        p.requires_grad = True

            # Train
            sums = {"total": 0.0, "box": 0.0, "cls": 0.0, "obj": 0.0}
            n = 0

            for step, (imgs, targets, _) in enumerate(self.train_loader, start=1):
                batch_losses = self._step_batch(imgs, targets, train=True)
                bs = imgs.shape[0]
                for k in sums:
                    sums[k] += float(batch_losses[k].item()) * bs
                n += bs

                if step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss_total_step", float(batch_losses["total"].item()),
                                           (epoch - 1) * len(self.train_loader) + step)
                    self.writer.add_scalar("train/lr_step", lr,
                                           (epoch - 1) * len(self.train_loader) + step)

            # Epoch averages
            train_avg = {k: sums[k] / max(1, n) for k in sums}

            # Scheduler step (epoch-based)
            if self.scheduler is not None:
                self.scheduler.step()

            # Validate
            val_avg = self._run_val()
            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            # TensorBoard epoch logs
            self.writer.add_scalar("train/loss_total", train_avg["total"], epoch)
            self.writer.add_scalar("train/loss_box", train_avg["box"], epoch)
            self.writer.add_scalar("train/loss_cls", train_avg["cls"], epoch)
            self.writer.add_scalar("train/loss_obj", train_avg["obj"], epoch)

            self.writer.add_scalar("val/loss_total", val_avg["total"], epoch)
            self.writer.add_scalar("val/loss_box", val_avg["box"], epoch)
            self.writer.add_scalar("val/loss_cls", val_avg["cls"], epoch)
            self.writer.add_scalar("val/loss_obj", val_avg["obj"], epoch)
            self.writer.add_scalar("train/lr", lr, epoch)

            stats = TrainStats(
                epoch=epoch,
                train_total=train_avg["total"],
                train_box=train_avg["box"],
                train_cls=train_avg["cls"],
                train_obj=train_avg["obj"],
                val_total=val_avg["total"],
                val_box=val_avg["box"],
                val_cls=val_avg["cls"],
                val_obj=val_avg["obj"],
                lr=lr,
                seconds=dt,
            )

            print(
                f"[Epoch {epoch:03d}] "
                f"train total={stats.train_total:.4f} (box={stats.train_box:.4f}, cls={stats.train_cls:.4f}, obj={stats.train_obj:.4f}) | "
                f"val total={stats.val_total:.4f} (box={stats.val_box:.4f}, cls={stats.val_cls:.4f}, obj={stats.val_obj:.4f}) | "
                f"lr={stats.lr:.2e} | {stats.seconds:.1f}s"
            )

            # Save checkpoints
            self._save_checkpoint("last_model.pth", epoch, val_avg["total"])

            if val_avg["total"] < self.best_val:
                self.best_val = val_avg["total"]
                self._save_checkpoint("best_model.pth", epoch, val_avg["total"])

        self.close()

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": {
                "PART3_CONFIG": config.PART3_CONFIG,
                "IMAGE_SIZE": config.IMAGE_SIZE,
                "BACKBONE": config.BACKBONE,
            },
        }
        if self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()

        path = self.out_dir / filename
        torch.save(ckpt, path)
