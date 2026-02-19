# part3/model.py
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class FixedSlotDetector(nn.Module):
    """
    Fixed-slot detector:
      - Predicts K slots per image (K=max_objects=3)
      - Each slot predicts:
          boxes:  (x1, y1, x2, y2)  -> shape [B, K, 4] in PIXELS on resized image
          logits: (C+1) classes     -> shape [B, K, C+1] (includes background)

    Key stability feature:
      - Box outputs are constrained to valid ranges using sigmoid + ordering,
        which prevents NaNs in IoU/CIoU computations.
    """

    def __init__(
        self,
        num_classes: int = 3,        # person, car, dog
        max_objects: int = 3,        # fixed capacity K
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: int = 448,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.num_logits = num_classes + 1  # + background
        self.image_size = int(image_size)

        # ---- Backbone ----
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)

        self.backbone = backbone.features  # nn.Sequential
        self.backbone_out_channels = 576   # MobileNetV3-Small final channels

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Neck ----
        self.neck = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Shared MLP ----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.box_head = nn.Linear(hidden_dim, max_objects * 4)              # K * 4
        self.cls_head = nn.Linear(hidden_dim, max_objects * self.num_logits)  # K * (C+1)

        nn.init.normal_(self.box_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.box_head.bias, 0.0)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0.0)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict:
              boxes:  [B, K, 4] (x1,y1,x2,y2) pixel coords on resized image
              logits: [B, K, C+1]
        """
        B = x.shape[0]

        feats = self.backbone(x)                 # [B, 576, h, w]
        feats = self.neck(feats)                 # [B, hidden_dim, h, w]
        pooled = self.pool(feats).flatten(1)     # [B, hidden_dim]

        z = self.mlp(pooled)                     # [B, hidden_dim]

        box_raw = self.box_head(z).view(B, self.max_objects, 4)                 # [B,K,4]
        logits = self.cls_head(z).view(B, self.max_objects, self.num_logits)    # [B,K,C+1]

        # ---- Constrain boxes ----
        # Interpret raw as normalized xyxy in [0,1] then order corners and scale to pixels.
        b = torch.sigmoid(box_raw)  # [B,K,4] in [0,1]
        x1n, y1n, x2n, y2n = b.unbind(dim=-1)

        x1 = torch.minimum(x1n, x2n) * (self.image_size - 1)
        y1 = torch.minimum(y1n, y2n) * (self.image_size - 1)
        x2 = torch.maximum(x1n, x2n) * (self.image_size - 1)
        y2 = torch.maximum(y1n, y2n) * (self.image_size - 1)

        # avoid degenerate boxes
        x2 = torch.maximum(x2, x1 + 1.0)
        y2 = torch.maximum(y2, y1 + 1.0)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return {"boxes": boxes, "logits": logits}
