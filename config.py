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

PART3_DATA_DIR = PROJECT_ROOT / "data" / "part3"
PART3_TRAIN_DIR = PART3_DATA_DIR / "train"
PART3_VALID_DIR = PART3_DATA_DIR / "valid"
PART3_TEST_DIR  = PART3_DATA_DIR / "test"

for dir_path in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model
BACKBONE = "mobilenet_v3_small"
BACKBONE_OUT_FEATURES = 576
PRETRAINED = True
IMAGE_SIZE = 448  # Increased for maximum detail

# Part 2 Config - Pure DIoU, Head-Only Training
PART2_CONFIG = {
    "num_classes": 1,
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 2e-3,  # Increased LR
    "coord_weight": 2.0,    # Light L1 penalty for center drift
    "weight_decay": 1e-4,
    "freeze_backbone": True,  # Keep backbone frozen initially
    "unfreeze_epoch": 5,  # Unfreeze top layers at epoch 5
}

# Part 3 Config
PART3_CONFIG = {
    "num_classes": 4,         # person, car, dog, cat
    "class_names": ["person", "car", "dog", "cat"],  
    "max_objects": 3,

    "data_root": PART3_DATA_DIR,
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,

    "freeze_backbone": True,
    "unfreeze_epoch": 30,

    "lambda_box": 5.0,
    "lambda_cls": 1.0,
    "lambda_obj": 1.0,

    # Inference thresholds 
    "score_thresh": 0.25,
    "nms_iou_thresh": 0.5,
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
