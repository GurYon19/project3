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
    ):
        self.index_json = Path(index_json)
        self.classes_json = Path(classes_json)
        self.image_size = int(image_size)
        self.max_objects = int(max_objects)

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

        # resize image
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        x = TF.to_tensor(img)  # [3,H,W] float in [0,1]

        boxes = torch.tensor(s.boxes, dtype=torch.float32)   # [N,4] in orig pixels
        labels = torch.tensor(s.labels, dtype=torch.long)    # [N]

        boxes = _resize_boxes_xyxy(boxes, orig_w, orig_h, self.image_size, self.image_size)

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
