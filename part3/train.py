# part3/train.py
"""
Entry point to train Part 3 detector.

Usage:
  python -m part3.train
  python -m part3.train --epochs 10 --batch-size 16 --device cuda
  python -m part3.train --limit-train 500 --limit-val 200   (quick smoke run)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from part3.dataset import Part3CocoDataset, collate_fn
from part3.model import Part3Detector
from part3.loss import Part3Loss
from part3.trainer import Trainer


def _limit_loader(ds, limit: int):
    """
    Wrap dataset to only use first 'limit' samples (for quick debugging).
    """
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    from torch.utils.data import Subset
    return Subset(ds, list(range(limit)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.PART3_CONFIG["epochs"])
    ap.add_argument("--batch-size", type=int, default=config.PART3_CONFIG["batch_size"])
    ap.add_argument("--lr", type=float, default=config.PART3_CONFIG["learning_rate"])
    ap.add_argument("--weight-decay", type=float, default=config.PART3_CONFIG["weight_decay"])
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)

    # Debug / quick run
    ap.add_argument("--limit-train", type=int, default=0, help="Use only first N train samples (0 = full).")
    ap.add_argument("--limit-val", type=int, default=0, help="Use only first N val samples (0 = full).")

    args = ap.parse_args()

    device = config.DEVICE if args.device is None else torch.device(args.device)

    # Datasets
    train_ds = Part3CocoDataset(split="train")
    val_ds = Part3CocoDataset(split="valid")

    train_ds = _limit_loader(train_ds, args.limit_train)
    val_ds = _limit_loader(val_ds, args.limit_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Model
    model = Part3Detector().to(device)

    # Freeze backbone initially if requested
    if config.PART3_CONFIG.get("freeze_backbone", True):
        if hasattr(model, "freeze_backbone"):
            model.freeze_backbone()

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine LR (epoch-based)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    loss_fn = Part3Loss()

    # Output dirs
    ckpt_dir = Path(config.CHECKPOINTS_DIR) / "part3"
    log_dir = Path(config.LOGS_DIR) / "part3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=ckpt_dir,
        log_dir=log_dir,
        loss_fn=loss_fn,
        unfreeze_epoch=int(config.PART3_CONFIG.get("unfreeze_epoch", 30)),
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )

    print(f"[TRAIN] device={device} epochs={args.epochs} batch_size={args.batch_size}")
    print(f"[TRAIN] checkpoints -> {ckpt_dir}")
    print(f"[TRAIN] tensorboard  -> {log_dir}")
    if args.limit_train:
        print(f"[TRAIN] DEBUG: limit_train={args.limit_train}")
    if args.limit_val:
        print(f"[TRAIN] DEBUG: limit_val={args.limit_val}")

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
