# part3/trainer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

# We reuse matching + IoU from the loss module so metrics match training behavior.
from part3.loss import box_iou_xyxy, match_small_k_bruteforce
from utils import loss


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -1.0  # we’ll track best val mIoU here


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str | Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        ckpt_dir: Optional[str | Path] = None,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = float(grad_clip)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.log_dir))

        # checkpoints live next to tensorboard by default (one folder up)
        if ckpt_dir is None:
            self.ckpt_dir = self.log_dir.parent
        else:
            self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.state = TrainState()

    # -------------------------
    # Checkpointing
    # -------------------------
    def _ckpt_payload(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._ckpt_payload(), str(path))

    # -------------------------
    # Train / Val
    # -------------------------
    def train_one_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total = 0.0
        total_cls = 0.0
        total_box = 0.0
        n = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            
            #with torch.autograd.set_detect_anomaly(True):
            out = self.model(images)
            loss_dict = self.loss_fn(out, targets)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            #torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total += float(loss.item())
            total_cls += float(loss_dict["loss_cls"].item())
            total_box += float(loss_dict["loss_box"].item())
            n += 1

            # TensorBoard (per step)
            self.tb.add_scalar("train/loss", float(loss.item()), self.state.global_step)
            self.tb.add_scalar("train/loss_cls", float(loss_dict["loss_cls"].item()), self.state.global_step)
            self.tb.add_scalar("train/loss_box", float(loss_dict["loss_box"].item()), self.state.global_step)

            lr = self.optimizer.param_groups[0]["lr"]
            self.tb.add_scalar("train/lr", float(lr), self.state.global_step)

            self.state.global_step += 1

        return {
            "loss": total / max(1, n),
            "loss_cls": total_cls / max(1, n),
            "loss_box": total_box / max(1, n),
        }

    @torch.no_grad()
    def validate_one_epoch(self, loader, iou_thresh: float = 0.5) -> Dict[str, float]:
        """
        Validation metrics:
        - val loss (same as training loss_fn)
        - matched mIoU over GT objects (using same brute-force matching style)
        - matched recall@IoU (fraction of GT matched with IoU>=thr)
        """
        self.model.eval()
        total = 0.0
        total_cls = 0.0
        total_box = 0.0
        n_batches = 0

        sum_iou = 0.0
        n_gt_total = 0
        n_gt_hit = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

            out = self.model(images)
            loss_dict = self.loss_fn(out, targets)

            total += float(loss_dict["loss"].item())
            total_cls += float(loss_dict["loss_cls"].item())
            total_box += float(loss_dict["loss_box"].item())
            n_batches += 1

            # --- metric: matched IoU / recall ---
            pb = out["boxes"]          # [B,K,4]
            tb = targets["boxes"]      # [B,K,4]
            m = targets["mask"]        # [B,K]  True for real GT

            B, K, _ = pb.shape
            for b in range(B):
                n_gt = int(m[b].long().sum().item())
                if n_gt == 0:
                    continue

                gt_boxes = tb[b][m[b]]  # [N,4]
                pred_boxes = pb[b]      # [K,4]

                iou_kn = box_iou_xyxy(pred_boxes, gt_boxes)  # [K,N]
                cost_full = torch.full((K, K), 10.0, device=iou_kn.device, dtype=iou_kn.dtype)
                cost_full[:, :n_gt] = (1.0 - iou_kn)

                pred_idx, gt_idx = match_small_k_bruteforce(cost_full, n_valid_gt=n_gt)
                matched_ious = iou_kn[pred_idx, gt_idx]  # [N]

                sum_iou += float(matched_ious.sum().item())
                n_gt_total += n_gt
                n_gt_hit += int((matched_ious >= iou_thresh).sum().item())

        val_loss = total / max(1, n_batches)
        val_loss_cls = total_cls / max(1, n_batches)
        val_loss_box = total_box / max(1, n_batches)

        miou = (sum_iou / max(1, n_gt_total))
        recall = (n_gt_hit / max(1, n_gt_total))

        return {
            "loss": val_loss,
            "loss_cls": val_loss_cls,
            "loss_box": val_loss_box,
            "miou": miou,
            "recall@0.5": recall,
        }

    # -------------------------
    # Fit loop
    # -------------------------
    def fit(self, dl_train, dl_val, epochs: int) -> None:
        """
        Runs training for `epochs` epochs.
        Saves:
          - last.pth every epoch
          - best.pth when val miou improves
          - summary.json updated each epoch
        """
        epochs = int(epochs)
        summary_path = self.ckpt_dir / "summary.json"
        history = []

        for e in range(epochs):
            self.state.epoch = e

            tr = self.train_one_epoch(dl_train)
            va = self.validate_one_epoch(dl_val)

            # TB (per epoch)
            self.tb.add_scalar("val/loss", va["loss"], e)
            self.tb.add_scalar("val/loss_cls", va["loss_cls"], e)
            self.tb.add_scalar("val/loss_box", va["loss_box"], e)
            self.tb.add_scalar("val/miou", va["miou"], e)
            self.tb.add_scalar("val/recall@0.5", va["recall@0.5"], e)

            # Console
            print(
                f"[E{e:03d}] "
                f"train loss={tr['loss']:.4f} (cls={tr['loss_cls']:.4f}, box={tr['loss_box']:.4f}) | "
                f"val loss={va['loss']:.4f} miou={va['miou']:.4f} rec@0.5={va['recall@0.5']:.4f}"
            )

            # Save last
            self.save_checkpoint(self.ckpt_dir / "last.pth")

            # Save best by val miou
            if va["miou"] > self.state.best_metric:
                self.state.best_metric = float(va["miou"])
                self.save_checkpoint(self.ckpt_dir / "best.pth")
                print(f"[CKPT] new best miou={self.state.best_metric:.4f} -> best.pth")

            # Update summary.json
            row = {"epoch": e, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
            history.append(row)
            summary_path.write_text(json.dumps({"history": history, "best_metric": self.state.best_metric}, indent=2), encoding="utf-8")

        self.tb.flush()
        self.tb.close()