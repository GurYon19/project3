# part3/train.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from part3.dataset import Part3VOCDataset, collate_part3
from part3.model import FixedSlotDetector, ModelConfig
from part3.loss import FixedSlotLoss, LossWeights
from part3.trainer import Trainer


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classes(classes_json: str | Path) -> Dict:
    classes_json = Path(classes_json)
    obj = json.loads(classes_json.read_text(encoding="utf-8"))
    # support both formats:
    # 1) {"classes":[...], "bg_id":3}
    # 2) {"classes":[...]} with bg implied as last
    classes = obj["classes"]
    if "bg_id" in obj:
        bg_id = int(obj["bg_id"])
    else:
        bg_id = len(classes) - 1
    return {"classes": classes, "bg_id": bg_id}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Part 3 training (Fixed-slot multi-object detector)")

    p.add_argument("--data-dir", type=str, required=True, help="Dir with train.json/val.json/test.json/classes.json")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    p.add_argument("--pretrained", action="store_true")

    p.add_argument("--use-focal", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)

    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--w-box", type=float, default=5.0)
    p.add_argument("--match-cls-cost-weight", type=float, default=0.5)

    p.add_argument("--tag", type=str, default="run1")
    p.add_argument("--out-dir", type=str, default="checkpoints/part3_relaxed")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = pick_device()
    data_dir = Path(args.data_dir)

    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    classes_json = data_dir / "classes.json"

    classes_info = load_classes(classes_json)
    classes: List[str] = classes_info["classes"]
    bg_id: int = classes_info["bg_id"]
    num_classes_total = len(classes)

    print(f"[DEVICE] {device.type}")
    print(f"[DATA] train={train_json} val={val_json}")
    print(f"[CLASSES] {classes} bg_id={bg_id}")
    print(f"[CFG] image_size={args.image_size} K={args.max_objects} backbone={args.backbone}")

    # Datasets
    ds_train = Part3VOCDataset(
        index_json=train_json,
        classes_json=classes_json,
        image_size=args.image_size,
        max_objects=args.max_objects,
        augment=True,
    )
    ds_val = Part3VOCDataset(
        index_json=val_json,
        classes_json=classes_json,
        image_size=args.image_size,
        max_objects=args.max_objects,
        augment=False,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_part3,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_part3,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    cfg = ModelConfig(
        image_size=args.image_size,
        max_objects=args.max_objects,
        num_classes_total=num_classes_total,
        bg_id=bg_id,
        backbone=args.backbone,
        pretrained=args.pretrained,
    )
    model = FixedSlotDetector(cfg).to(device)

    # Loss
    loss_fn = FixedSlotLoss(
        num_classes_total=num_classes_total,
        bg_id=bg_id,
        max_objects=args.max_objects,
        weights=LossWeights(cls=args.w_cls, box=args.w_box),
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        match_cls_cost_weight=args.match_cls_cost_weight,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Output dirs
    run_dir = Path(args.out_dir) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        log_dir=run_dir / "tb",
        ckpt_dir=run_dir,   # so best.pth/last.pth go into the run folder
    )

    trainer.fit(dl_train=dl_train, dl_val=dl_val, epochs=args.epochs)


if __name__ == "__main__":
    main()