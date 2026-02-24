from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from part3.dataset import Part3VOCDataset, collate_part3
from part3.model import FixedSlotDetector
from part3.loss import FixedSlotLoss, LossWeights
from part3.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser("Part 3 training (fixed-slot detector)")
    p.add_argument("--data-dir", type=str, default="datasets/part3")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--unfreeze-epoch", type=int, default=10)
    p.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.2,
        help="When unfreezing backbone, set backbone lr = lr * backbone_lr_mult",
    )

    p.add_argument("--use-focal", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)

    p.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training (per-batch resize)")
    p.add_argument("--ms-sizes", type=str, default="384,448,512", help="Comma-separated sizes, e.g. 384,448,512")

    p.add_argument("--aug", action="store_true", help="Enable train-time augmentations")
    p.add_argument("--aug-scale-min", type=float, default=0.5)
    p.add_argument("--aug-scale-max", type=float, default=1.0)
    p.add_argument("--aug-flip-p", type=float, default=0.5)
    p.add_argument("--aug-jitter-p", type=float, default=0.8)
    p.add_argument("--aug-blur-p", type=float, default=0.15)

    p.add_argument("--log-dir", type=str, default="logs/part3")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/part3")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    # Prefer MPS if available (your setup), then CUDA, then CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    classes_json = data_dir / "classes.json"

    train_ds = Part3VOCDataset(
        train_json,
        classes_json,
        image_size=args.image_size,
        max_objects=args.max_objects,
        augment=args.aug,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_flip_p=args.aug_flip_p,
        aug_jitter_p=args.aug_jitter_p,
        aug_blur_p=args.aug_blur_p,
    )

    # IMPORTANT: no augmentations in val
    val_ds = Part3VOCDataset(
        val_json,
        classes_json,
        image_size=args.image_size,
        max_objects=args.max_objects,
        augment=False,
    )

    num_classes = len(train_ds.classes)
    bg_id = num_classes

    device = pick_device()
    pin_memory = device.type == "cuda"  # only useful on CUDA

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_part3,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_part3,
        drop_last=False,
    )

    # Model (MobileNetV3-Large is already inside FixedSlotDetector now)
    model = FixedSlotDetector(
        num_classes=num_classes,
        max_objects=args.max_objects,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        # If your loss uses IoU during training and you see NaNs, flip this True:
        constrain_boxes_in_train=False,
    ).to(device)

    print(f"[DEVICE] {device}")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} classes={train_ds.classes} bg_id={bg_id}")
    print(f"[LOSS] focal={args.use_focal} gamma={args.focal_gamma} class_weights=None")
    print(f"[MS] enabled={args.multi_scale} sizes={args.ms_sizes} (train only), val_size={args.image_size}")
    print(f"[AUG] enabled={args.aug}")

    criterion = FixedSlotLoss(
        num_classes=num_classes,
        max_objects=args.max_objects,
        weights=LossWeights(cls=1.0, box=5.0),
        class_weights=None,  # keep None to isolate focal effect
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
    ).to(device)

    # Optimizer: start with all params (backbone may be frozen so it's fine)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ms_sizes = [int(s) for s in args.ms_sizes.split(",") if s.strip()]
    if not ms_sizes:
        ms_sizes = [args.image_size]

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        background_id=bg_id,
        amp=args.amp,
        multi_scale=args.multi_scale,
        ms_sizes=ms_sizes,
        base_image_size=args.image_size,
        backbone_lr_mult=args.backbone_lr_mult,
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"[RESUME] loaded: {args.resume}")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        unfreeze_epoch=args.unfreeze_epoch,
    )


if __name__ == "__main__":
    main()