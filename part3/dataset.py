from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter, GaussianBlur


@dataclass
class Part3Sample:
    image_id: str
    image_path: str
    width: int
    height: int
    boxes: List[List[float]]     # xyxy in original image pixels OR already padded xyxy
    labels: List[int]            # class ids (may include bg_id for padded)
    mask: List[bool] | None = None  # optional if present in JSON


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
    base = min(w, h)
    s = random.uniform(scale_min, scale_max)
    crop = int(round(base * s))
    crop = max(64, min(crop, base))

    max_left = w - crop
    max_top = h - crop
    left = 0 if max_left <= 0 else random.randint(0, max_left)
    top = 0 if max_top <= 0 else random.randint(0, max_top)
    return left, top, crop


def _crop_boxes_xyxy(boxes: torch.Tensor, left: int, top: int, crop: int) -> torch.Tensor:
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
    Reads train/val/test.json produced by the *relaxed* VOC builder.

    Each sample may already contain padded boxes/labels/mask length K,
    OR may contain variable-length boxes/labels (older script). This loader
    supports both and will always output fixed K.

    Returns:
      image: FloatTensor [3,S,S] in [0,1]
      target:
        boxes: [K,4] float (xyxy in resized pixels)
        labels: [K] long (includes background)
        mask: [K] bool (True = real object slot)
        meta: dict
    """

    def __init__(
        self,
        index_json: str | Path,
        classes_json: str | Path,
        image_size: int = 448,
        max_objects: int = 3,
        augment: bool = False,
        aug_scale_min: float = 0.60,
        aug_scale_max: float = 1.00,
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

        self._jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)
        self._blur = GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))

        classes_obj = _load_json(self.classes_json)
        self.classes: List[str] = classes_obj["classes"]
        self.bg_id: int = int(classes_obj.get("bg_id", len(self.classes) - 1))
        self.num_classes_total: int = len(self.classes)

        items = _load_json(self.index_json)
        self.samples: List[Part3Sample] = []
        for it in items:
            self.samples.append(
                Part3Sample(
                    image_id=it["image_id"],
                    image_path=it["image_path"],
                    width=int(it.get("width", 0)),
                    height=int(it.get("height", 0)),
                    boxes=it["boxes"],
                    labels=it["labels"],
                    mask=it.get("mask", None),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_to_k(
        self,
        boxes: torch.Tensor,   # [N,4]
        labels: torch.Tensor,  # [N]
        mask: torch.Tensor,    # [N] (True)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = self.max_objects
        padded_boxes = torch.zeros((K, 4), dtype=torch.float32)
        padded_labels = torch.full((K,), fill_value=self.bg_id, dtype=torch.long)
        padded_mask = torch.zeros((K,), dtype=torch.bool)

        N = min(int(labels.numel()), K)
        if N > 0:
            padded_boxes[:N] = boxes[:N]
            padded_labels[:N] = labels[:N]
            padded_mask[:N] = mask[:N]
        return padded_boxes, padded_labels, padded_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")

        orig_w, orig_h = img.size
        if s.width > 0 and s.height > 0:
            orig_w, orig_h = s.width, s.height

        boxes = torch.tensor(s.boxes, dtype=torch.float32)   # [N,4]
        labels = torch.tensor(s.labels, dtype=torch.long)    # [N]

        if s.mask is None:
            # assume first N are real objects
            mask = torch.ones((labels.numel(),), dtype=torch.bool)
        else:
            mask = torch.tensor(s.mask, dtype=torch.bool)

        # if JSON is already padded length K, we can still apply augmentations
        # but we must be careful: augmentations must ignore padded slots.
        # We'll only augment real boxes (mask True), then re-pad.
        real_idx = mask.nonzero(as_tuple=False).squeeze(1)
        real_boxes = boxes[real_idx] if real_idx.numel() > 0 else boxes.new_zeros((0, 4))
        real_labels = labels[real_idx] if real_idx.numel() > 0 else labels.new_zeros((0,), dtype=torch.long)

        if self.augment and real_labels.numel() > 0:
            # Random square crop (retry to keep at least 1 object)
            for _ in range(5):
                left, top, crop = _random_square_crop_params(orig_w, orig_h, self.aug_scale_min, self.aug_scale_max)
                boxes_c = _crop_boxes_xyxy(real_boxes, left, top, crop)
                boxes_c, labels_c = _filter_valid_boxes_xyxy(boxes_c, real_labels, min_wh=2.0)
                if labels_c.numel() > 0:
                    img = TF.crop(img, top, left, crop, crop)
                    real_boxes, real_labels = boxes_c, labels_c
                    orig_w, orig_h = crop, crop
                    break

            # Horizontal flip
            if random.random() < self.aug_flip_p:
                img = TF.hflip(img)
                if real_boxes.numel() > 0:
                    x1 = real_boxes[:, 0].clone()
                    x2 = real_boxes[:, 2].clone()
                    real_boxes[:, 0] = (orig_w - 1) - x2
                    real_boxes[:, 2] = (orig_w - 1) - x1

            # Color jitter
            if random.random() < self.aug_jitter_p:
                img = self._jitter(img)

            # Blur
            if random.random() < self.aug_blur_p:
                img = self._blur(img)

        # Resize image
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Resize boxes
        real_boxes = _resize_boxes_xyxy(real_boxes, orig_w, orig_h, self.image_size, self.image_size)
        real_boxes, real_labels = _filter_valid_boxes_xyxy(real_boxes, real_labels, min_wh=2.0)

        # Sort by area desc to keep deterministic slot ordering when padding
        if real_labels.numel() > 0:
            areas = (real_boxes[:, 2] - real_boxes[:, 0]).clamp(min=0) * (real_boxes[:, 3] - real_boxes[:, 1]).clamp(min=0)
            order = torch.argsort(areas, descending=True)
            real_boxes = real_boxes[order]
            real_labels = real_labels[order]

        real_mask = torch.ones((real_labels.numel(),), dtype=torch.bool)
        padded_boxes, padded_labels, padded_mask = self._pad_to_k(real_boxes, real_labels, real_mask)

        target = {
            "boxes": padded_boxes,
            "labels": padded_labels,
            "mask": padded_mask,
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