"""
Data augmentation and preprocessing transforms.
Handles synchronized transformations for images and bounding boxes.
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random
import numpy as np
from typing import Tuple, Optional


class DetectionTransform:
    """
    Transform for detection that applies synchronized transformations
    to both images and bounding boxes.
    """
    
    def __init__(
        self,
        image_size: int = 448,
        is_training: bool = True,
        h_flip_prob: float = 0.0,
        color_jitter: bool = False
    ):
        self.image_size = image_size
        self.is_training = is_training
        self.h_flip_prob = h_flip_prob
        self.color_jitter = color_jitter
        
        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(
        self, 
        image: Image.Image, 
        boxes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transforms to image and boxes.
        
        Args:
            image: PIL Image
            boxes: (N, 4) tensor of boxes in [x_center, y_center, w, h] format
            
        Returns:
            Transformed image tensor and boxes
        """
        # Convert to tensor first
        image = TF.to_tensor(image)
        
        if self.is_training and boxes is not None:
            # Random horizontal flip
            if random.random() < self.h_flip_prob:
                image = TF.hflip(image)
                # Flip boxes: x_center = 1 - x_center
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
        
        # Resize
        image = TF.resize(image, [self.image_size, self.image_size])
        
        # Color jitter (only for training)
        if self.is_training and self.color_jitter:
            image = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )(image)
        
        # Normalize
        image = self.normalize(image)
        
        return image, boxes


class InferenceTransform:
    """Simple transform for inference (no augmentation)."""
    
    def __init__(self, image_size: int = 448):
        self.image_size = image_size
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Transform image for inference."""
        image = TF.to_tensor(image)
        image = TF.resize(image, [self.image_size, self.image_size])
        image = self.normalize(image)
        return image


def get_transforms(is_training: bool = True, image_size: int = 448) -> DetectionTransform:
    """
    Get detection transforms.
    
    Args:
        is_training: Whether for training or validation
        image_size: Target image size
        
    Returns:
        DetectionTransform instance
    """
    return DetectionTransform(
        image_size=image_size,
        is_training=is_training,
        h_flip_prob=0.0,
        color_jitter=False
    )


def get_inference_transform(image_size: int = 448) -> InferenceTransform:
    """
    Get inference transform.
    
    Args:
        image_size: Target image size
        
    Returns:
        InferenceTransform instance
    """
    return InferenceTransform(image_size=image_size)
