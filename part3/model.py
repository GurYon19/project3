# part3/model.py
"""
Part 3 fixed-capacity multi-object detector (K=3 slots, C=4 classes).

Design (high-level):
- Backbone: MobileNetV3-Small (reuse models/backbone.py)
- Head: fixed "set prediction" head producing K=3 slots per image:
    pred_boxes:      (B, K, 4)  normalized xyxy in [0,1]
    pred_obj_logits: (B, K)     objectness logits (slot used vs empty)
    pred_cls_logits: (B, K, C)  class logits for used slots

Core idea:
- Extract a compact global-but-informative feature vector from the backbone feature map.
- For each slot k, concatenate that vector with a learnable slot embedding q_k.
- Run a shared MLP to produce slot outputs.

This matches the project constraint: fixed capacity (max_objects=3).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

import config
from models.backbone import get_backbone  # expected existing helper


@dataclass
class ModelOutput:
    pred_boxes: torch.Tensor      # (B, K, 4) in [0,1] xyxy
    pred_obj_logits: torch.Tensor # (B, K)
    pred_cls_logits: torch.Tensor # (B, K, C)


class SlotHead(nn.Module):
    """
    Shared slot head MLP.
    Input:  (B, K, D) where D = feat_dim + slot_dim
    Output: boxes (B,K,4), obj_logits (B,K), cls_logits (B,K,C)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.box_head = nn.Linear(hidden_dim, 4)
        self.obj_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, num_classes)

        # init: encourage low objectness at start (optional but helps stability)
        nn.init.constant_(self.obj_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, K, D)
        """
        h = self.mlp(x)                          # (B, K, H)
        box_raw = self.box_head(h)               # (B, K, 4)
        obj_logits = self.obj_head(h).squeeze(-1)  # (B, K)
        cls_logits = self.cls_head(h)            # (B, K, C)

        # Boxes: constrain to [0,1]
        # We output normalized coordinates and will let the loss handle geometry.
        box_norm = torch.sigmoid(box_raw)

        # Ensure xyxy ordering (x1<=x2, y1<=y2) in a differentiable way:
        x1y1 = torch.minimum(box_norm[..., 0:2], box_norm[..., 2:4])
        x2y2 = torch.maximum(box_norm[..., 0:2], box_norm[..., 2:4])
        boxes = torch.cat([x1y1, x2y2], dim=-1)

        return boxes, obj_logits, cls_logits


class Part3Detector(nn.Module):
    def __init__(
        self,
        num_classes: int | None = None,
        max_objects: int | None = None,
        backbone_name: str | None = None,
        pretrained: bool | None = None,
        backbone_out_features: int | None = None,
        image_size: int | None = None,
        slot_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        cfg = config.PART3_CONFIG
        self.num_classes = int(num_classes) if num_classes is not None else int(cfg["num_classes"])
        self.K = int(max_objects) if max_objects is not None else int(cfg["max_objects"])
        self.image_size = int(image_size) if image_size is not None else int(config.IMAGE_SIZE)

        bb_name = backbone_name or config.BACKBONE
        bb_pretrained = bool(pretrained) if pretrained is not None else bool(config.PRETRAINED)
        bb_out = int(backbone_out_features) if backbone_out_features is not None else int(config.BACKBONE_OUT_FEATURES)

        # Backbone should output a feature map of shape (B, C, H, W)
        self.backbone = get_backbone(bb_name)
        if not bb_pretrained:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone_out = bb_out

        # Light conv "neck" to keep some spatial cues and stabilize training
        self.neck = nn.Sequential(
            nn.Conv2d(self.backbone_out, self.backbone_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.backbone_out),
            nn.ReLU(inplace=True),
        )

        # Pool to small grid then flatten (keeps more info than global avg only)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        feat_dim = self.backbone_out * 7 * 7

        # Learnable slot embeddings (K, slot_dim)
        self.slot_embed = nn.Parameter(torch.randn(self.K, slot_dim) * 0.02)

        # Shared MLP head operates on concatenated [feat || slot]
        self.head = SlotHead(in_dim=feat_dim + slot_dim, hidden_dim=hidden_dim, num_classes=self.num_classes)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        x: (B, 3, H, W) ImageNet normalized
        """
        feat = self.backbone(x)   # expected (B, C, h, w)
        feat = self.neck(feat)
        pooled = self.pool(feat)  # (B, C, 7, 7)
        B = pooled.shape[0]
        vec = pooled.flatten(1)   # (B, feat_dim)

        # Expand to (B, K, feat_dim)
        vec_k = vec.unsqueeze(1).expand(B, self.K, vec.shape[-1])

        # Slot embeddings -> (B, K, slot_dim)
        slot = self.slot_embed.unsqueeze(0).expand(B, self.K, self.slot_embed.shape[-1])

        # Concatenate and predict
        inp = torch.cat([vec_k, slot], dim=-1)  # (B, K, feat_dim+slot_dim)
        boxes, obj_logits, cls_logits = self.head(inp)

        return ModelOutput(pred_boxes=boxes, pred_obj_logits=obj_logits, pred_cls_logits=cls_logits)


def _smoke_test(device: torch.device, batch_size: int = 2) -> None:
    """
    Quick forward-pass test to verify shapes and dtype.
    """
    from torch.utils.data import DataLoader
    from part3.dataset import Part3CocoDataset, collate_fn

    ds = Part3CocoDataset(split="train")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    imgs, targets, metas = next(iter(dl))

    model = Part3Detector().to(device)
    model.eval()

    imgs = imgs.to(device)
    with torch.no_grad():
        out = model(imgs)

    print("[SMOKE TEST]")
    print("  imgs:", tuple(imgs.shape))
    print("  pred_boxes:", tuple(out.pred_boxes.shape), out.pred_boxes.min().item(), out.pred_boxes.max().item())
    print("  pred_obj_logits:", tuple(out.pred_obj_logits.shape))
    print("  pred_cls_logits:", tuple(out.pred_cls_logits.shape))
    print("  target boxes:", tuple(targets["boxes"].shape))
    print("  target labels:", tuple(targets["labels"].shape))
    print("  target mask:", tuple(targets["mask"].shape))
    print("  meta[0]:", metas[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    if args.device is None:
        device = config.DEVICE
    else:
        device = torch.device(args.device)

    _smoke_test(device=device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
