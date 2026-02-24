"""
Configuration for Object Detection Project.
"""
import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "datasets"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for dir_path in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# -------------------------
# Model (GLOBAL DEFAULTS)
# -------------------------
# Switch to MobileNetV3-Large
BACKBONE = "mobilenet_v3_large"
BACKBONE_OUT_FEATURES = 960
PRETRAINED = True

# Input resolution
IMAGE_SIZE = 448

# Optional "neck" bottleneck for detection heads
# (prevents parameter blow-up when backbone is large)
NECK_CHANNELS = 256

# Part 2 Config - (keep if you still want it around)
PART2_CONFIG = {
    "num_classes": 1,
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 2e-3,
    "coord_weight": 2.0,
    "weight_decay": 1e-4,
    "freeze_backbone": True,
    "unfreeze_epoch": 5,
}

# Part 3 Config
PART3_CONFIG = {
    "num_classes": 20,      # VOC has 20 classes (plus background handled internally)
    "max_objects": 3,       # fixed capacity K
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "freeze_backbone": True,
    "unfreeze_epoch": 30,

    # model dims
    "backbone": BACKBONE,
    "backbone_out_features": BACKBONE_OUT_FEATURES,
    "neck_channels": NECK_CHANNELS,
    "image_size": IMAGE_SIZE,

    # loss weights (your existing plan)
    "lambda_box": 5.0,
    "lambda_cls": 1.0,
    "lambda_obj": 1.0,
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# VOC Classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]