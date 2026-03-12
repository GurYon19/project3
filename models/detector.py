"""
Object Detection Model Wrapper.
"""
import torch
import torch.nn as nn

from .backbone import get_backbone
from .heads import SingleObjectHead, MultiObjectHead


class ObjectDetector(nn.Module):
    """Complete object detector: Backbone + Pooling + Head."""
    
    def __init__(self, phase: int = 2, pretrained: bool = True, freeze_backbone: bool = True,
                 num_classes: int = 20, max_objects: int = 3, model_name: str = 'small'):
        super().__init__()
        
        self.phase = phase
        self.backbone = get_backbone(pretrained=pretrained, freeze=freeze_backbone, model_name=model_name)
        in_features = self.backbone.out_channels  # 576 for small
        
        # 4x4 spatial pooling preserves spatial information (top-left vs bottom-right)
        self.output_grid_size = 4
        self.pool = nn.AdaptiveAvgPool2d(self.output_grid_size)
        
        flat_features = in_features * (self.output_grid_size ** 2)  # 576 * 16 = 9216
        
        if phase == 2:
            self.head = SingleObjectHead(in_features=flat_features)
        else:
            self.head = MultiObjectHead(in_features=flat_features, num_classes=num_classes, max_objects=max_objects)
    
    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        pooled = self.pool(features)
        flat = pooled.view(pooled.size(0), -1)  # Flatten
        
        out = self.head(flat)
        # Clamp to [0, 1] only at inference for bounding box coordinates
        if not self.training and self.phase == 2:
            out = out.clamp(0.0, 1.0)
            
        return out
    
    def freeze_backbone(self):
        self.backbone.freeze()
    
    def unfreeze_backbone(self, num_layers: int = None):
        self.backbone.unfreeze(num_layers)
    
    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list:
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.head.parameters(), 'lr': head_lr}
        ]
    
    def count_parameters(self) -> dict:
        backbone_params = self.backbone.count_parameters()
        head_total = sum(p.numel() for p in self.head.parameters())
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        return {
            'backbone': backbone_params,
            'head': {'total': head_total, 'trainable': head_trainable},
            'total': backbone_params['total'] + head_total,
            'total_trainable': backbone_params['trainable'] + head_trainable
        }


def create_detector(phase: int, config: dict = None) -> ObjectDetector:
    if config is None:
        config = {}
    
    return ObjectDetector(
        phase=phase,
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', True),
        num_classes=config.get('num_classes', 20),
        max_objects=config.get('max_objects', 3),
        model_name=config.get('model_name', 'small')
    )

