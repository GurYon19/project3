from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class ModelConfig:
    image_size: int = 448
    max_objects: int = 3
    num_classes_total: int = 4          # includes background
    bg_id: int = 3
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    dropout: float = 0.1
    hidden: int = 512


class FixedSlotDetector(nn.Module):
    """
    Simple fixed-slot detector:
      - MobileNetV3 backbone -> feature map
      - global pooling -> vector
      - MLP predicts K boxes and K class logits

    Output:
      boxes:  [B,K,4] xyxy in pixel coords (0..S-1)
      logits: [B,K,C] (C includes background)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.S = int(cfg.image_size)
        self.K = int(cfg.max_objects)
        self.C = int(cfg.num_classes_total)

        if cfg.backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if cfg.pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            self.backbone = net.features
            backbone_out = 576
        elif cfg.backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if cfg.pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            self.backbone = net.features
            backbone_out = 960
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        # 1x1 conv projects backbone channels → hidden dim (keeps spatial dims)
        self.proj = nn.Conv2d(backbone_out, cfg.hidden, kernel_size=1)

        # K learnable slot query vectors — each slot attends to different image regions
        self.slot_queries = nn.Parameter(torch.randn(self.K, cfg.hidden) * 0.02)

        # Per-slot MLP after attention pooling
        self.slot_mlp = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
        )

        # Per-slot heads (applied independently to each slot's feature)
        self.fc_box = nn.Linear(cfg.hidden, 4)
        self.fc_cls = nn.Linear(cfg.hidden, self.C)

        # init
        nn.init.normal_(self.fc_box.weight, std=0.01)
        nn.init.constant_(self.fc_box.bias, 0.0)
        nn.init.normal_(self.fc_cls.weight, std=0.01)
        nn.init.constant_(self.fc_cls.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B,3,S,S]
        f = self.backbone(x)               # [B, backbone_out, h, w]
        f = self.proj(f)                   # [B, hidden, h, w]
        B, C, H, W = f.shape
        f_flat = f.view(B, C, H * W)      # [B, hidden, HW]

        # Per-slot attention: each slot query attends to different spatial regions
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, hidden]
        attn = torch.bmm(q, f_flat) / (C ** 0.5)              # [B, K, HW]
        attn = F.softmax(attn, dim=-1)
        slot_feats = torch.bmm(attn, f_flat.permute(0, 2, 1)) # [B, K, hidden]
        slot_feats = self.slot_mlp(slot_feats)                 # [B, K, hidden]

        box_raw = self.fc_box(slot_feats)   # [B, K, 4]
        cls_raw = self.fc_cls(slot_feats)   # [B, K, C]

        # convert box_raw to xyxy pixels
        # box_raw -> sigmoid -> (cx,cy,w,h) in [0,1]
        t = torch.sigmoid(box_raw)
        cx, cy, w, h2 = t[..., 0], t[..., 1], t[..., 2], t[..., 3]

        # constrain sizes a bit to avoid zero-area boxes
        w = 0.05 + 0.95 * w
        h2 = 0.05 + 0.95 * h2

        x1 = (cx - 0.5 * w) * (self.S - 1)
        y1 = (cy - 0.5 * h2) * (self.S - 1)
        x2 = (cx + 0.5 * w) * (self.S - 1)
        y2 = (cy + 0.5 * h2) * (self.S - 1)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes = boxes.clamp(0, self.S - 1)

        return {"boxes": boxes, "logits": cls_raw}