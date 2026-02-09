# Part 3: Multi-Object Detection - Agent Context Document

## Purpose
This document provides everything needed for an AI agent to start working on Part 3 (Multi-Object Detection) seamlessly. Part 1 and Part 2 are complete.

---

## 1. Project Summary

**Course**: Computer Vision  
**Project**: Object Detection with Deep Learning  
**Backbone**: MobileNetV3-Small (selected via ID digit-sum = 9)

### Status:
| Part | Task | Status | Points |
|------|------|--------|--------|
| Part 1 | Classification Backbone Analysis | ✅ Complete | 10 |
| Part 2 | Single Object Detection (1 class, 1 object/image) | ✅ Complete | 40 |
| Part 3 | Multi-Object Detection (multi-class, multi-object/image) | ❌ **TODO** | 50 |

### Part 2 Results:
- Best Validation IoU: **91.2%**
- Input Resolution: 448×448
- Loss Function: CIoU + L1 regularization

---

## 2. Project Structure

```
project3/
├── config.py                 # Global configuration (backbone, image size, hyperparams)
├── requirements.txt          # Python dependencies
├── main.py                   # Main entry point (if used)
├── README.md                 # Project documentation
├── colab_runner.ipynb        # Google Colab notebook for training
├── Final_Report.txt          # Part 1 + Part 2 merged report
├── Project3_guidelines-3(2).txt  # Course requirements document
├── check_json.py             # Utility to check dataset JSON stats
│
├── models/
│   ├── __init__.py           # Module exports
│   ├── backbone.py           # MobileNetV3-Small backbone loader
│   ├── detector.py           # Part 2 single-object detector
│   └── heads.py              # Detection head implementations
│
├── data/
│   ├── __init__.py           # Module exports
│   ├── dataset.py            # COCO dataset loader
│   └── transforms.py         # Image augmentation/preprocessing
│
├── utils/
│   ├── __init__.py           # Module exports
│   ├── loss.py               # CIoU, DIoU, GIoU loss implementations
│   ├── metrics.py            # IoU calculation utilities
│   └── visualization.py      # Drawing/plotting utilities
│
├── tools/                    # Utility scripts for dataset management
│   ├── analyze_dataset.py    # Dataset statistics
│   ├── check_missing_images.py
│   ├── ciou_reference.py     # CIoU reference implementation
│   ├── clean_unused_images.py
│   ├── count_dataset.py
│   ├── delete_trash_images.py
│   ├── filter_single_object.py
│   ├── restore_backup.py
│   └── sync_annotations.py
│
├── part1/
│   ├── train.py              # Classification inference demo
│   ├── report.txt            # Part 1 report
│   ├── images/               # Sample images for demo
│   └── outputs/              # Demo output images
│
├── part2/
│   ├── train.py              # Part 2 training script
│   ├── trainer.py            # Part 2 training loop
│   ├── inference.py          # Run inference on images
│   ├── visualize_worst.py    # Visualize worst predictions
│   └── report.txt            # Part 2 report
│
├── part3/                    # ← **CREATE PART 3 HERE** (currently empty)
│   └── (empty - to be built)
│
├── datasets/
│   └── part2/                # Part 2 tiger dataset (DO NOT USE FOR PART 3)
│       ├── train/            # 5,765 images
│       ├── valid/            # 374 images
│       └── test/             # 378 images
│
├── checkpoints/
│   └── part2/
│       └── best_model.pth    # Trained Part 2 model
│
├── logs/                     # TensorBoard logs
│
└── outputs/                  # Inference outputs, visualizations
    └── worst_predictions/    # Part 2 worst prediction images
```

---

## 3. What Needs to Be Built for Part 3

Part 3 requires a **completely fresh implementation** (no code reuse from Part 2).

### Required Components:

| Component | Description |
|-----------|-------------|
| `part3/train.py` | Main training script for multi-object detection |
| `part3/trainer.py` | Training loop handling variable objects per image |
| `part3/model.py` | Detection head for multi-class, multi-object output |
| `part3/dataset.py` | Dataset loader supporting multiple bounding boxes per image |
| `part3/loss.py` | Loss function for multi-object matching (e.g., Hungarian matching) |
| `part3/inference.py` | Inference script with NMS (Non-Maximum Suppression) |
| `part3/evaluate.py` | mAP calculation for evaluation |

### Key Challenges for Part 3:
1. **Variable number of objects per image** - Need padding/collation strategy
2. **Multi-class classification** - Each box has class label + coordinates
3. **Object matching during training** - Match predictions to ground truth
4. **NMS during inference** - Remove duplicate detections
5. **mAP evaluation** - Standard COCO/VOC evaluation metric

---

## 4. Key Technical Decisions (Principles from Part 2)

These are the **design principles** that worked well - apply similar thinking to Part 3:

| Decision | Rationale |
|----------|-----------|
| **448×448 input** | Higher resolution preserves small object detail |
| **Spatial pooling (not global)** | Preserve spatial information for localization |
| **CIoU loss** | Better than L1/L2 for bounding box regression |
| **Progressive unfreezing** | Train head first, then fine-tune backbone |
| **Cosine LR schedule** | Smooth convergence without manual tuning |

---

## 5. How to Run

### Verify Part 2 Works:
```bash
# Run Part 1 demo
python part1/train.py

# Run Part 2 inference
python part2/inference.py --folder datasets/part2/test --checkpoint checkpoints/part2/best_model.pth
```

### Part 3 (Once Built):
```bash
# Training
python part3/train.py --data-dir datasets/part3 --epochs 100

# Inference
python part3/inference.py --image test.jpg --checkpoint checkpoints/part3/best_model.pth

# Evaluation
python part3/evaluate.py --data-dir datasets/part3/test --checkpoint checkpoints/part3/best_model.pth
```

---

## 6. Constraints

These constraints come from the project guidelines:

| Constraint | Details |
|------------|---------|
| **No external detection libraries** | Cannot use YOLO, Detectron2, MMDetection, etc. |
| **PyTorch primitives only** | Conv2d, Linear, ReLU, pooling, etc. |
| **Must use MobileNetV3-Small backbone** | Selected via ID digit-sum |
| **Must justify all design decisions** | Report should explain why each choice was made |

---

## 7. Quick Reference

### Config File Location:
`config.py` - Contains `PART3_CONFIG` dictionary (update as needed)

### Current Part 3 Config (in config.py):
```python
PART3_CONFIG = {
    "num_classes": 20,        # Update based on dataset
    "max_objects": 3,         # Maximum objects per image
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "freeze_backbone": True,
    "unfreeze_epoch": 30,
    "lambda_box": 5.0,        # Box regression loss weight
    "lambda_cls": 1.0,        # Classification loss weight
    "lambda_obj": 1.0,        # Objectness loss weight
}
```

### Key Files to Read First:
1. `Project3_guidelines-3(2).txt` - Course requirements
2. `config.py` - Configuration structure
3. `models/backbone.py` - How backbone is loaded

---

## 8. Next Steps for Agent

1. **Choose/Download Part 3 Dataset**
   - Multi-class, multi-object dataset (e.g., Pascal VOC, custom Roboflow dataset)
   - NOT the Part 2 tiger dataset

2. **Create `part3/` directory structure**

3. **Implement dataset loader with collation for variable objects**

4. **Design detection head architecture**
   - Option A: Anchor-based (like SSD)
   - Option B: Anchor-free (like FCOS/CenterNet)

5. **Implement training loop with object matching**

6. **Add NMS and mAP evaluation**

---

**Document Created**: 2026-02-09  
**Last Verified**: 2026-02-09  
**Verification Status**: ✅ Matches actual project structure
