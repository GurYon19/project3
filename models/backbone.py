"""
Backbone model loader for MobileNetV3 (Small/Large).
Handles feature extraction and pretrained weight loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torchvision import models
import requests


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    out_channels: int


BACKBONE_SPECS = {
    "mobilenet_v3_small": BackboneSpec("mobilenet_v3_small", out_channels=576),
    "mobilenet_v3_large": BackboneSpec("mobilenet_v3_large", out_channels=960),
}


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 backbone for feature extraction.

    - Returns feature maps from `model.features`
    - Supports small/large variants
    - `out_channels` matches the last feature stage:
        small: 576
        large: 960
    """

    def __init__(
        self,
        variant: str = "mobilenet_v3_large",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        if variant not in BACKBONE_SPECS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(BACKBONE_SPECS.keys())}"
            )

        spec = BACKBONE_SPECS[variant]
        self.variant = spec.name
        self.out_channels = spec.out_channels

        # Load torchvision model
        if spec.name == "mobilenet_v3_small":
            if pretrained:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                model = models.mobilenet_v3_small(weights=weights)
            else:
                model = models.mobilenet_v3_small(weights=None)
        else:  # mobilenet_v3_large
            if pretrained:
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                model = models.mobilenet_v3_large(weights=weights)
            else:
                model = models.mobilenet_v3_large(weights=None)

        self.model = model
        self.features = model.features  # nn.Sequential

        if freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            feats: [B, out_channels, H/32, W/32] (approx; depends on input size)
        """
        return self.features(x)

    def freeze(self):
        """Freeze all backbone parameters."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone parameters for fine-tuning.
        If num_layers is provided, only the last N feature blocks are unfrozen.
        """
        if num_layers is None:
            for p in self.features.parameters():
                p.requires_grad = True
            return

        layers: List[nn.Module] = list(self.features.children())
        for layer in layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def get_backbone(
    variant: str = "mobilenet_v3_large",
    pretrained: bool = True,
    freeze: bool = True,
) -> MobileNetV3Backbone:
    return MobileNetV3Backbone(variant=variant, pretrained=pretrained, freeze=freeze)


def get_classification_model(variant: str = "mobilenet_v3_large", pretrained: bool = True) -> nn.Module:
    """
    Get the full MobileNetV3 classification model (Part 1 usage).
    """
    if variant == "mobilenet_v3_small":
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            return models.mobilenet_v3_small(weights=weights)
        return models.mobilenet_v3_small(weights=None)

    if variant == "mobilenet_v3_large":
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            return models.mobilenet_v3_large(weights=weights)
        return models.mobilenet_v3_large(weights=None)

    raise ValueError(f"Unknown variant '{variant}'")


def get_imagenet_labels() -> list:
    """
    Download and return ImageNet class labels (1000).
    """
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        import json
        r = requests.get(url, timeout=10)
        return json.loads(r.text)
    except Exception:
        return [f"class_{i}" for i in range(1000)]