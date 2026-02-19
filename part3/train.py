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


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    classes_json = data_dir / "classes.json"

    train_ds = Part3VOCDataset(train_json, classes_json, image_size=args.image_size, max_objects=args.max_objects)
    val_ds = Part3VOCDataset(val_json, classes_json, image_size=args.image_size, max_objects=args.max_objects)

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
    # person=8005, car=867, dog=1002
    class_w = compute_class_weights([8005, 867, 1002], bg_weight=0.25)
    print(f"[WEIGHTS] class_w={class_w.tolist()} (person,car,dog,bg)")

    criterion = FixedSlotLoss(
        num_classes=num_classes,
        max_objects=args.max_objects,
        weights=LossWeights(cls=1.0, box=5.0),
        class_weights=class_w.to(device),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
    )

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
