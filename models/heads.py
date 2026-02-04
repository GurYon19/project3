"""
Detection Heads for Object Detection.
Phase 2: SingleObjectHead - Predicts one bounding box
Phase 3: MultiObjectHead - Predicts multiple boxes with class labels
"""
import torch
import torch.nn as nn


class SingleObjectHead(nn.Module):
    """
    Detection head for single-object detection (Part 2).
    
    Input: Flattened feature vector from backbone (B, 576)
    Output: Bounding box coordinates (B, 4) normalized to [0, 1]
    
    Bounding Box Format: (x_center, y_center, width, height)
    """
    
    def __init__(self, in_features: int = 576):
        super().__init__()
        
        # Only using basic PyTorch building blocks as required
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 4),
            nn.Sigmoid()  # Constrains output to [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiObjectHead(nn.Module):
    """
    Detection head for multi-object detection (Part 3).
    
    Output per object: 4 coords + num_classes + 1 objectness
    """
    
    def __init__(self, in_features: int = 576, num_classes: int = 20, max_objects: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        self.shared = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
        )
        
        self.box_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_objects * 4),
            nn.Sigmoid()
        )
        
        self.class_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_objects * num_classes),
        )
        
        self.obj_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, max_objects),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict:
        B = x.size(0)
        features = self.shared(x)
        
        boxes = self.box_branch(features).view(B, self.max_objects, 4)
        classes = self.class_branch(features).view(B, self.max_objects, self.num_classes)
        objectness = self.obj_branch(features).view(B, self.max_objects)
        
        return {'boxes': boxes, 'classes': classes, 'objectness': objectness}


def get_detection_head(phase: int, **kwargs) -> nn.Module:
    if phase == 2:
        return SingleObjectHead(**kwargs)
    elif phase == 3:
        return MultiObjectHead(**kwargs)
    else:
        raise ValueError(f"Invalid phase: {phase}")
