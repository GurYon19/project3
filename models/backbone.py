"""
Backbone model loader for MobileNetV3-Small.
Handles feature extraction and pretrained weight loading.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import requests


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Small backbone for feature extraction.
    
    Extracts features from the network before the classification head.
    Output channels: 576 (from the last convolutional layer)
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.model = models.mobilenet_v3_small(weights=weights)
        else:
            self.model = models.mobilenet_v3_small(weights=None)
        
        # Extract feature extractor (everything except classifier)
        self.features = self.model.features
        
        # Output channels from last conv layer
        self.out_channels = 576
        
        if freeze:
            self.freeze()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Feature maps (B, 576, H/32, W/32)
        """
        return self.features(x)
    
    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone parameters for fine-tuning.
        
        Args:
            num_layers: If specified, only unfreeze last N layers
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N layers
            layers = list(self.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable
        }


def get_backbone(pretrained: bool = True, freeze: bool = True) -> MobileNetV3Backbone:
    """
    Factory function to create MobileNetV3 backbone.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        freeze: Whether to freeze backbone initially
        
    Returns:
        MobileNetV3Backbone instance
    """
    return MobileNetV3Backbone(pretrained=pretrained, freeze=freeze)


def get_classification_model(pretrained: bool = True) -> nn.Module:
    """
    Get the full MobileNetV3-Small classification model for Part 1.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Complete MobileNetV3-Small model with classifier
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)
    
    return model


def get_imagenet_labels() -> list:
    """
    Download and return ImageNet class labels.
    
    Returns:
        List of 1000 ImageNet class names
    """
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    
    try:
        import json
        response = requests.get(url, timeout=10)
        labels = json.loads(response.text)
        return labels
    except:
        # Fallback: return generic labels
        return [f"class_{i}" for i in range(1000)]
