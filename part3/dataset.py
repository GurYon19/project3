# part3/dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF

import random
from torchvision.transforms import ColorJitter, GaussianBlur


@dataclass
class Part3Sample:
    image_id: str
    image_path: str
    width: int
    height: int
    boxes: List[List[float]]   # xyxy in original image pixels
    labels: List[int]          # 0..C-1


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resize_boxes_xyxy(
    boxes_xyxy: torch.Tensor,
    orig_w: int,
    orig_h: int,
    new_w: int,
    new_h: int,
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    sx = new_w / float(orig_w)
    sy = new_h / float(orig_h)
    x1 = boxes_xyxy[:, 0] * sx
    y1 = boxes_xyxy[:, 1] * sy
    x2 = boxes_xyxy[:, 2] * sx
    y2 = boxes_xyxy[:, 3] * sy
    return torch.stack([x1, y1, x2, y2], dim=1)


def _clamp_boxes_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    return boxes


def _filter_valid_boxes_xyxy(boxes: torch.Tensor, labels: torch.Tensor, min_wh: float = 2.0):
    if boxes.numel() == 0:
        return boxes, labels
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    keep = (w >= min_wh) & (h >= min_wh)
    return boxes[keep], labels[keep]


def _random_square_crop_params(w: int, h: int, scale_min: float, scale_max: float):
    """
    Returns (left, top, crop_size) for a square crop inside WxH.
    scale is relative to min(w,h).
    """
    base = min(w, h)
    s = random.uniform(scale_min, scale_max)
    crop = int(round(base * s))
    crop = max(16, min(crop, base))  # avoid tiny / invalid

    max_left = w - crop
    max_top = h - crop
    left = 0 if max_left <= 0 else random.randint(0, max_left)
    top = 0 if max_top <= 0 else random.randint(0, max_top)
    return left, top, crop


def _crop_boxes_xyxy(boxes: torch.Tensor, left: int, top: int, crop: int) -> torch.Tensor:
    """
    Crop boxes by (left, top, crop_size). Output boxes in cropped coords (still xyxy).
    Boxes are clipped to crop window.
    """
    if boxes.numel() == 0:
        return boxes

    x1 = boxes[:, 0] - left
    y1 = boxes[:, 1] - top
    x2 = boxes[:, 2] - left
    y2 = boxes[:, 3] - top

    boxes2 = torch.stack([x1, y1, x2, y2], dim=1)
    boxes2 = _clamp_boxes_xyxy(boxes2, crop, crop)
    return boxes2


class Part3VOCDataset(Dataset):
    """
    Reads datasets/part3/{train,val,test}.json produced by tools/filter_voc_for_part3.py

    Returns:
      image:  FloatTensor [3,H,W] in [0,1]
      target:
        boxes: [K,4] padded
        labels:[K] padded with background_id=C
        mask:  [K] bool
        meta:  dict
    """

    def __init__(
        self,
        index_json: str | Path,
        classes_json: str | Path,
        image_size: int = 448,
        max_objects: int = 3,
        augment: bool = False,
        aug_scale_min: float = 0.5,
        aug_scale_max: float = 1.0,
        aug_flip_p: float = 0.5,
        aug_jitter_p: float = 0.8,
        aug_blur_p: float = 0.15,
    ):
        self.index_json = Path(index_json)
        self.classes_json = Path(classes_json)
        self.image_size = int(image_size)
        self.max_objects = int(max_objects)

        self.augment = bool(augment)
        self.aug_scale_min = float(aug_scale_min)
        self.aug_scale_max = float(aug_scale_max)
        self.aug_flip_p = float(aug_flip_p)
        self.aug_jitter_p = float(aug_jitter_p)
        self.aug_blur_p = float(aug_blur_p)

        # Define transforms once
        self._jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)
        self._blur = GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))


        classes_obj = _load_json(self.classes_json)
        self.classes: List[str] = classes_obj["classes"]
        self.num_classes = len(self.classes)
        self.background_id = self.num_classes

        items = _load_json(self.index_json)
        self.samples: List[Part3Sample] = []
        for it in items:
            self.samples.append(
                Part3Sample(
                    image_id=it["image_id"],
                    image_path=it["image_path"],
                    width=int(it["width"]),
                    height=int(it["height"]),
                    boxes=it["boxes"],
                    labels=it["labels"],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        s = self.samples[idx]

        img = Image.open(s.image_path).convert("RGB")
        orig_w, orig_h = img.size
        # prefer JSON metadata if present
        if s.width > 0 and s.height > 0:
            orig_w, orig_h = s.width, s.height

        boxes = torch.tensor(s.boxes, dtype=torch.float32)   # [N,4] in orig pixels
        labels = torch.tensor(s.labels, dtype=torch.long)    # [N]

        # --- Train-time augmentation (operates in original image coords) ---
        if self.augment:
            # 1) Random square crop (scale jitter), retry a few times so we don't crop away all objects
            for _ in range(5):
                left, top, crop = _random_square_crop_params(orig_w, orig_h, self.aug_scale_min, self.aug_scale_max)
                boxes_c = _crop_boxes_xyxy(boxes, left, top, crop)
                boxes_c, labels_c = _filter_valid_boxes_xyxy(boxes_c, labels, min_wh=2.0)
                if labels_c.numel() > 0:
                    # apply crop
                    img = TF.crop(img, top, left, crop, crop)
                    boxes, labels = boxes_c, labels_c
                    orig_w, orig_h = crop, crop
                    break
            # if all retries fail, keep original image/boxes

            # 2) Horizontal flip
            if random.random() < self.aug_flip_p:
                img = TF.hflip(img)
                if boxes.numel() > 0:
                    # x coords flip inside [0, orig_w)
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = (orig_w - 1) - x2
                    boxes[:, 2] = (orig_w - 1) - x1

            # 3) Color jitter
            if random.random() < self.aug_jitter_p:
                img = self._jitter(img)

            # 4) Mild blur (helps with video motion blur)
            if random.random() < self.aug_blur_p:
                img = self._blur(img)

        # --- Resize to network input ---
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        x = TF.to_tensor(img)  # [3,H,W] float in [0,1]

        # Resize boxes to match final image_size
        boxes = _resize_boxes_xyxy(boxes, orig_w, orig_h, self.image_size, self.image_size)
        boxes, labels = _filter_valid_boxes_xyxy(boxes, labels, min_wh=2.0)


        K = self.max_objects
        N = min(labels.numel(), K)

        padded_boxes = torch.zeros((K, 4), dtype=torch.float32)
        padded_labels = torch.full((K,), fill_value=self.background_id, dtype=torch.long)
        mask = torch.zeros((K,), dtype=torch.bool)

        if N > 0:
            padded_boxes[:N] = boxes[:N]
            padded_labels[:N] = labels[:N]
            mask[:N] = True

        target = {
            "boxes": padded_boxes,
            "labels": padded_labels,
            "mask": mask,
            "meta": {
                "image_id": s.image_id,
                "path": s.image_path,
                "orig_size": (orig_h, orig_w),
                "resized_size": (self.image_size, self.image_size),
            },
        }
        return x, target


def collate_part3(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    boxes = torch.stack([t["boxes"] for t in targets], dim=0)
    labels = torch.stack([t["labels"] for t in targets], dim=0)
    mask = torch.stack([t["mask"] for t in targets], dim=0)
    meta = [t["meta"] for t in targets]

    return images, {"boxes": boxes, "labels": labels, "mask": mask, "meta": meta}
