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
    # person=8005, car=867, dog=1002
    class_w = compute_class_weights([8005, 867, 1002], bg_weight=0.25)
    print(f"[WEIGHTS] class_w={class_w.tolist()} (person,car,dog,bg)")

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
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from part3.dataset import Part3VOCDataset, collate_part3
from part3.loss import FixedSlotLoss, LossWeights
from part3.model import FixedSlotDetector, ModelConfig
from part3.trainer import Trainer
from part3.evaluate import evaluate_map50, load_classes, pick_device


def build_sampler(dataset: Part3VOCDataset, boost_classes: set[int], boost: float = 3.0):
    # Weight images higher if they contain car/dog (or any boosted class) in real slots.
    weights = []
    for i in range(len(dataset)):
        _, t = dataset[i]
        labels = t["labels"]
        mask = t["mask"]
        real = labels[mask]
        w = 1.0
        if any(int(c.item()) in boost_classes for c in real):
            w = boost
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="checkpoints/part3")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--use-focal", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--sampler-boost", type=float, default=3.0)
    p.add_argument("--no-sampler", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    print(f"[DEVICE] {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    classes, bg_id = load_classes(data_dir / "classes.json")
    print(f"[CLASSES] {classes} bg_id={bg_id}")

    train_ds = Part3VOCDataset(data_dir / "train.json", data_dir / "classes.json", image_size=args.image_size, max_objects=args.max_objects, augment=True)
    val_ds = Part3VOCDataset(data_dir / "val.json", data_dir / "classes.json", image_size=args.image_size, max_objects=args.max_objects, augment=False)

    sampler = None
    if not args.no_sampler:
        # boost minority classes: car and dog (if present)
        boost_ids = set()
        for name in ["car", "dog"]:
            if name in classes:
                boost_ids.add(classes.index(name))
        sampler = build_sampler(train_ds, boost_ids, boost=args.sampler_boost)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_part3,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_part3)

    cfg = ModelConfig(
        image_size=args.image_size,
        max_objects=args.max_objects,
        num_classes_total=len(classes),
        bg_id=bg_id,
        backbone="mobilenet_v3_small",
        pretrained=True,
    )
    model = FixedSlotDetector(cfg).to(device)

    loss_fn = FixedSlotLoss(
        num_classes_total=len(classes),
        bg_id=bg_id,
        max_objects=args.max_objects,
        weights=LossWeights(cls=1.0, box=5.0),
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Step each iteration in trainer (simple). No scheduler needed to start.

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        log_dir=out_dir / "tb",
        scheduler=None,
    )

    best_map = -1.0
    best_path = out_dir / "best.pth"

    for epoch in range(args.epochs):
        trainer.state.epoch = epoch
        metrics = trainer.train_one_epoch(train_loader)
        print(f"[EPOCH {epoch+1:03d}] train loss={metrics['loss']:.4f} cls={metrics['loss_cls']:.4f} box={metrics['loss_box']:.4f}")

        # Evaluate mAP@0.5
        ap = evaluate_map50(model, val_loader, classes, bg_id, conf_thresh=0.25, topk=args.max_objects, device=device)
        val_map = float(ap.get("mAP@0.5", 0.0))
        print(f"[EPOCH {epoch+1:03d}] val mAP@0.5={val_map:.4f}  " + "  ".join([f"{k}:{v:.3f}" for k,v in ap.items() if k not in ["mAP@0.5"]]))

        trainer.tb.add_scalar("val/mAP@0.5", val_map, epoch)
        for k, v in ap.items():
            if k != "mAP@0.5":
                trainer.tb.add_scalar(f"val/AP@0.5/{k}", float(v), epoch)

        # Save best
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "classes": classes,
            "bg_id": bg_id,
            "image_size": args.image_size,
            "max_objects": args.max_objects,
        }
        torch.save(ckpt, out_dir / "last.pth")

        if val_map > best_map:
            best_map = val_map
            torch.save(ckpt, best_path)
            print(f"[OK] new best: {best_path} (mAP@0.5={best_map:.4f})")

    # save training summary
    (out_dir / "summary.json").write_text(json.dumps({"best_map@0.5": best_map}, indent=2), encoding="utf-8")
    print(f"[DONE] best mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()