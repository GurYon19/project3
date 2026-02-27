from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_map: float = -1.0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str | Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.log_dir))

        self.state = TrainState()

    def train_one_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total = 0.0
        total_cls = 0.0
        total_box = 0.0
        n = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

            out = self.model(images)
            loss_dict = self.loss_fn(out, targets)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total += float(loss.item())
            total_cls += float(loss_dict["loss_cls"].item())
            total_box += float(loss_dict["loss_box"].item())
            n += 1

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