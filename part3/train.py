# part3/train.py
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
    p.add_argument("--use-focal", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training (per-batch resize)")
    p.add_argument("--ms-sizes", type=str, default="384,448,512", help="Comma-separated sizes, e.g. 384,448,512")
    p.add_argument("--aug", action="store_true", help="Enable train-time augmentations (Run5)")
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_class_weights(counts, bg_weight=0.25):
    import numpy as np
    counts = np.array(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    C = len(counts)

    w = total / (C * counts)      # inverse frequency normalized
    w = w / w.mean()              # mean=1 normalization

    w_full = np.zeros(C + 1, dtype=np.float64)
    w_full[:C] = w
    w_full[C] = bg_weight
    return torch.tensor(w_full, dtype=torch.float32)


import json
from collections import Counter

def count_objects_per_class_from_index(index_json_path: Path, num_classes: int) -> list[int]:
    """
    Count GT objects per class index (0..num_classes-1) from the dataset index JSON.
    Expects each sample dict to contain 'labels' as int indices (no background).
    """
    data = json.loads(Path(index_json_path).read_text(encoding="utf-8"))
    c = Counter()
    for item in data:
        for y in item.get("labels", []):
            c[int(y)] += 1
    return [c.get(i, 0) for i in range(num_classes)]


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    classes_json = data_dir / "classes.json"

    train_ds = Part3VOCDataset(
        train_json, classes_json,
        image_size=args.image_size, max_objects=args.max_objects,
        augment=args.aug,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_flip_p=args.aug_flip_p,
        aug_jitter_p=args.aug_jitter_p,
        aug_blur_p=args.aug_blur_p,
    )

    # IMPORTANT: no augmentations in val
    val_ds = Part3VOCDataset(
        val_json, classes_json,
        image_size=args.image_size, max_objects=args.max_objects,
        augment=False,
    )

    num_classes = len(train_ds.classes)
    bg_id = num_classes

    device = pick_device()

    # pin_memory is only useful on CUDA
    pin_memory = device.type == "cuda"

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

    model = FixedSlotDetector(
        num_classes=num_classes,
        max_objects=args.max_objects,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
    ).to(device)

    # Using counts you measured:
    counts = count_objects_per_class_from_index(train_json, num_classes)
    class_w = compute_class_weights(counts, bg_weight=0.25)
    print(f"[WEIGHTS] counts={counts} classes={train_ds.classes} bg_weight=0.25")
    print(f"[WEIGHTS] class_w={class_w.tolist()} (classes + bg)")
    
    print(f"[LOSS] focal={args.use_focal} gamma={args.focal_gamma} class_weights=None")

    criterion = FixedSlotLoss(
        num_classes=num_classes,
        max_objects=args.max_objects,
        weights=LossWeights(cls=1.0, box=5.0),
        class_weights=None,                 # <-- IMPORTANT: Run A isolates focal
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
    ).to(device)

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
    )
    print(f"[MS] enabled={args.multi_scale} sizes={ms_sizes} (train only), val_size={args.image_size}")


    if args.resume:
        trainer.load(args.resume)
        print(f"[RESUME] loaded: {args.resume}")

    print(f"[DEVICE] {device}")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} classes={train_ds.classes} bg_id={bg_id}")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        unfreeze_epoch=args.unfreeze_epoch,
    )


if __name__ == "__main__":
    main()
