from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def kaiming_init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class FixedSlotDetector(nn.Module):
    """
    Fixed-slot detector (K slots per image):
      - boxes:  [B, K, 4] in pixels (x1,y1,x2,y2) on resized image
      - logits: [B, K, C+1] (includes background)

    Notes:
      - Backbone: MobileNetV3-Large features (960 channels)
      - Neck: 1x1 bottleneck 960 -> hidden_dim
      - Pool: global avg pool
      - Head: shared MLP, then box + class heads

    Training stability:
      - Box head outputs are raw (unbounded) during training.
      - During inference, boxes are clamped to [0, image_size-1] and ordered.
        (Avoids the sigmoid+SmoothL1 saturation trap.)
    """

    def __init__(
        self,
        num_classes: int = 20,
        max_objects: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: int = 448,
        constrain_boxes_in_train: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_objects = int(max_objects)
        self.num_logits = self.num_classes + 1  # + background
        self.image_size = int(image_size)
        self.constrain_boxes_in_train = bool(constrain_boxes_in_train)

        # ---- Backbone (MobileNetV3-Large) ----
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            backbone = mobilenet_v3_large(weights=weights)
        else:
            backbone = mobilenet_v3_large(weights=None)

        self.backbone = backbone.features  # nn.Sequential
        self.backbone_out_channels = 960

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Neck: 1x1 bottleneck (960 -> hidden_dim) ----
        self.neck = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Pool to a single vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Shared MLP (BN before ReLU, Linear bias removed) ----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mlp.apply(kaiming_init_linear)

        # Heads
        self.box_head = nn.Linear(hidden_dim, self.max_objects * 4)
        self.cls_head = nn.Linear(hidden_dim, self.max_objects * self.num_logits)

        # Small init for heads is fine
        nn.init.normal_(self.box_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.box_head.bias, 0.0)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0.0)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _order_and_clamp_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        boxes: [B,K,4] raw xyxy in pixels (may be unordered/out-of-range)
        returns ordered/clamped boxes, min size 1px
        """
        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        x1o = torch.minimum(x1, x2)
        y1o = torch.minimum(y1, y2)
        x2o = torch.maximum(x1, x2)
        y2o = torch.maximum(y1, y2)

        lo = 0.0
        hi = float(self.image_size - 1)

        x1o = x1o.clamp(lo, hi)
        y1o = y1o.clamp(lo, hi)
        x2o = x2o.clamp(lo, hi)
        y2o = y2o.clamp(lo, hi)

        # avoid degenerate boxes
        x2o = torch.maximum(x2o, x1o + 1.0)
        y2o = torch.maximum(y2o, y1o + 1.0)

        return torch.stack([x1o, y1o, x2o, y2o], dim=-1)

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

        feats = self.backbone(x)                 # [B, 960, h, w]
        feats = self.neck(feats)                 # [B, hidden_dim, h, w]
        pooled = self.pool(feats).flatten(1)     # [B, hidden_dim]

        z = self.mlp(pooled)                     # [B, hidden_dim]

        box_raw = self.box_head(z).view(B, self.max_objects, 4)
        logits = self.cls_head(z).view(B, self.max_objects, self.num_logits)

        # box_raw is pixel-space prediction. To keep training gradients healthy,
        # we do NOT apply sigmoid by default.
        # If your loss needs valid boxes (e.g., IoU loss), set constrain_boxes_in_train=True.
        if (not self.training) or self.constrain_boxes_in_train:
            boxes = self._order_and_clamp_xyxy(box_raw)
        else:
            boxes = box_raw

        return {"boxes": boxes, "logits": logits}