# part3/dataset.py
"""
Part 3 COCO-style dataset loader (fixed capacity K=3).

Reads:
  data/part3/{split}/annotations.json
  data/part3/{split}/images/<file_name>

Returns per sample:
  img:    FloatTensor (3, IMAGE_SIZE, IMAGE_SIZE)  (ImageNet normalized)
  target: dict with:
    boxes: FloatTensor (K=3, 4)  normalized xyxy in [0,1]
    labels: LongTensor (K=3,)    class ids in [0..C-1]
    mask:  BoolTensor (K=3,)     True for real objects, False for padding

Includes a small test you can run:
  python -m part3.dataset --split train --n 3
It will print shapes and save a few GT visualization images to outputs/part3/gt_sanity/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


@dataclass
class Sample:
    img: torch.Tensor
    target: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    # xywh: (..., 4) with x,y,w,h
    x = xywh[..., 0]
    y = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


def _clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    xyxy = xyxy.copy()
    xyxy[..., 0] = np.clip(xyxy[..., 0], 0, w - 1)
    xyxy[..., 2] = np.clip(xyxy[..., 2], 0, w - 1)
    xyxy[..., 1] = np.clip(xyxy[..., 1], 0, h - 1)
    xyxy[..., 3] = np.clip(xyxy[..., 3], 0, h - 1)
    return xyxy


def _normalize_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    # Normalize to [0,1] in xyxy
    out = xyxy.astype(np.float32).copy()
    out[..., 0] /= float(w)
    out[..., 2] /= float(w)
    out[..., 1] /= float(h)
    out[..., 3] /= float(h)
    return np.clip(out, 0.0, 1.0)


def _resize_image_and_boxes(
    img_bgr: np.ndarray, boxes_xyxy: np.ndarray, out_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes image to (out_size, out_size) with stretch (no aspect preservation),
    and scales boxes accordingly.
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension.")
    sx = out_size / float(w)
    sy = out_size / float(h)

    img_resized = cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    boxes = _clip_xyxy(boxes, out_size, out_size)
    return img_resized, boxes


class Part3CocoDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_root: Path | None = None,
        image_size: int | None = None,
        max_objects: int | None = None,
        classes_json: Path | None = None,
    ):
        """
        split: 'train' | 'valid' | 'test'
        """
        assert split in {"train", "valid", "test"}, f"Bad split: {split}"

        self.split = split
        self.data_root = Path(data_root) if data_root is not None else Path(config.PART3_CONFIG["data_root"])
        self.split_root = self.data_root / split
        self.image_dir = self.split_root / "images"
        self.ann_path = self.split_root / "annotations.json"

        self.image_size = int(image_size) if image_size is not None else int(config.IMAGE_SIZE)
        self.max_objects = int(max_objects) if max_objects is not None else int(config.PART3_CONFIG["max_objects"])

        # Load class mapping as source of truth
        cls_path = Path(classes_json) if classes_json is not None else (self.data_root / "classes.json")
        cls = _load_json(cls_path)
        self.num_classes = int(cls["num_classes"])
        self.id_to_name = {int(k): v for k, v in cls["id_to_name"].items()}

        coco = _load_json(self.ann_path)

        # Images
        self.images: List[dict] = coco["images"]
        self.img_by_id: Dict[int, dict] = {im["id"]: im for im in self.images}

        # Annotations grouped by image_id
        self.anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Keep deterministic ordering for reproducibility
        self.images.sort(key=lambda x: x["id"])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        im = self.images[idx]
        img_id = im["id"]
        file_name = im["file_name"]

        img_path = self.image_dir / file_name
        if not img_path.exists():
            # some exports place images at split root; fallback
            alt = self.split_root / file_name
            if alt.exists():
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path} (or {alt})")

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        anns = self.anns_by_img.get(img_id, [])
        # COCO bbox is [x, y, w, h] in pixels
        boxes_xywh = np.array([a["bbox"] for a in anns], dtype=np.float32)
        labels = np.array([a["category_id"] for a in anns], dtype=np.int64)

        # Safety: dataset should already be filtered to 1..K objects
        if boxes_xywh.shape[0] > self.max_objects:
            boxes_xywh = boxes_xywh[: self.max_objects]
            labels = labels[: self.max_objects]

        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

        # Resize to model input size and scale boxes accordingly
        img_resized, boxes_resized = _resize_image_and_boxes(img_bgr, boxes_xyxy, self.image_size)

        # Normalize boxes to [0,1]
        boxes_norm = _normalize_xyxy(boxes_resized, self.image_size, self.image_size)

        # Pad to K
        K = self.max_objects
        n = boxes_norm.shape[0]

        boxes_out = np.zeros((K, 4), dtype=np.float32)
        labels_out = np.zeros((K,), dtype=np.int64)
        mask_out = np.zeros((K,), dtype=bool)

        if n > 0:
            boxes_out[:n] = boxes_norm
            labels_out[:n] = labels
            mask_out[:n] = True

        # Convert image to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(config.IMAGENET_STD, dtype=np.float32)
        img_rgb = (img_rgb - mean) / std

        # To tensor CHW
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()

        target = {
            "boxes": torch.from_numpy(boxes_out),         # (K,4) float
            "labels": torch.from_numpy(labels_out),       # (K,) long
            "mask": torch.from_numpy(mask_out),           # (K,) bool
        }

        meta = {
            "image_id": img_id,
            "file_name": file_name,
            "path": str(img_path),
        }

        return Sample(img=img_t, target=target, meta=meta)


def collate_fn(batch: List[Sample]):
    # Fixed shapes, so stacking is simple
    imgs = torch.stack([b.img for b in batch], dim=0)
    boxes = torch.stack([b.target["boxes"] for b in batch], dim=0)
    labels = torch.stack([b.target["labels"] for b in batch], dim=0)
    mask = torch.stack([b.target["mask"] for b in batch], dim=0)
    metas = [b.meta for b in batch]
    targets = {"boxes": boxes, "labels": labels, "mask": mask}
    return imgs, targets, metas


def _denormalize_img(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (3,H,W) ImageNet normalized RGB tensor
    returns uint8 RGB image
    """
    img = img_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)
    img = (img * std) + mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _draw_gt(sample: Sample, out_path: Path, id_to_name: Dict[int, str]) -> None:
    img_rgb = _denormalize_img(sample.img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    boxes = sample.target["boxes"].numpy()  # normalized xyxy
    labels = sample.target["labels"].numpy()
    mask = sample.target["mask"].numpy()

    H, W = img_bgr.shape[:2]
    for i in range(boxes.shape[0]):
        if not mask[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * W)
        x2 = int(x2 * W)
        y1 = int(y1 * H)
        y2 = int(y2 * H)
        cls = int(labels[i])
        name = id_to_name.get(cls, str(cls))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            name,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=3, help="How many samples to visualize.")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    ds = Part3CocoDataset(split=args.split)
    print(f"[DATASET] split={args.split}  len={len(ds)}  image_size={ds.image_size}  K={ds.max_objects}")
    print(f"[CLASSES] {ds.id_to_name}")

    # Single-sample check
    s0 = ds[0]
    print("[SAMPLE 0]")
    print("  img:", tuple(s0.img.shape), s0.img.dtype)
    print("  boxes:", tuple(s0.target["boxes"].shape), s0.target["boxes"].dtype)
    print("  labels:", tuple(s0.target["labels"].shape), s0.target["labels"].dtype)
    print("  mask:", tuple(s0.target["mask"].shape), s0.target["mask"].dtype)
    print("  meta:", s0.meta)

    # Dataloader check
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    imgs, targets, metas = next(iter(dl))
    print("[BATCH]")
    print("  imgs:", tuple(imgs.shape))
    print("  boxes:", tuple(targets["boxes"].shape))
    print("  labels:", tuple(targets["labels"].shape))
    print("  mask:", tuple(targets["mask"].shape))
    print("  metas[0]:", metas[0])

    # Save a few GT visualizations
    out_dir = Path(config.OUTPUTS_DIR) / "part3" / "gt_sanity" / args.split
    for i in range(min(args.n, len(ds))):
        sample = ds[i]
        out_path = out_dir / f"{args.split}_{i:04d}.jpg"
        _draw_gt(sample, out_path, ds.id_to_name)
    print(f"[DONE] Wrote {min(args.n, len(ds))} GT sanity images to: {out_dir}")


if __name__ == "__main__":
    main()
