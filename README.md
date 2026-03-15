# Deep Learning Object Detection Project

## Quick Start Guide

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pillow matplotlib opencv-python tensorboard tqdm requests
```

### 2. Run Part 1 (Classification Demo)

**Step 1**: Add sample images to `part1_images/` directory
- Add 3-5 images (JPG, PNG, etc.)
- Examples: cat.jpg, dog.jpg, car.jpg, etc.

**Step 2**: Run the classification script
```bash
python part1_classification.py
```

This will:
- Load MobileNetV3-Small with ImageNet weights
- Run inference on your images
- Display top-5 predictions for each image
- Save results to `outputs/part1_classification_results.png`

---

### 3. Prepare for Part 2 (Single Object Detection)

**Dataset Options:**
1. **COCO (Dogs)** - Recommended
   - Download from: https://cocodataset.org/
   - Filter for single-object images of dogs
   
2. **Roboflow** - Easiest
   - Browse: https://public.roboflow.com/object-detection
   - Download a single-class dataset (e.g., "Person Detection")

**Directory Structure:**
```
datasets/part2/
├── train/
│   ├── images/
│   └── annotations.json  (COCO format)
└── valid/
    ├── images/
    └── annotations.json
```

**Run Training:**
```bash
python -m part2.train --data-dir datasets/part2
```

---

### 4. Part 3 — Multi-Class, Multi-Object Detection

**Task:** Detect up to K=3 objects per image across 3 classes: `person`, `car`, `dog` (plus background).

**Architecture:** `FixedSlotDetector` — MobileNetV3-Small backbone with per-slot spatial attention heads. Each of the K=3 output slots attends to a different spatial region of the feature map, enabling co-detection of multiple classes in the same image.

#### Dataset

Uses PASCAL VOC 2007+2012 filtered to images containing person, car, or dog (relaxed: images with up to K objects total, at least one foreground object).

Build the dataset index (run once after downloading VOC):
```bash
python -m tools.build_voc_part3_k3_relaxed
```

Expected structure:
```
datasets/part3_voc_k3_relaxed/
├── classes.json       # {"classes": ["person","car","dog","__background__"], "bg_id": 3}
├── train.json
├── val.json
└── test.json
```

#### Training

**Standard run (pretrained backbone, 30 epochs):**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.train \
  --data-dir datasets/part3_voc_k3_relaxed \
  --image-size 448 \
  --max-objects 3 \
  --epochs 30 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --backbone mobilenet_v3_small \
  --pretrained \
  --w-cls 1.0 \
  --w-box 5.0 \
  --tag run1 \
  --out-dir checkpoints/part3_spatial
```

**With focal loss (helps with class imbalance):**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.train \
  --data-dir datasets/part3_voc_k3_relaxed \
  --image-size 448 \
  --max-objects 3 \
  --epochs 30 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --backbone mobilenet_v3_small \
  --pretrained \
  --w-cls 1.0 \
  --w-box 5.0 \
  --use-focal \
  --focal-gamma 2.0 \
  --tag spatial_attn_focal \
  --out-dir checkpoints/part3_spatial
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir checkpoints/part3_spatial/run1/tb
```

#### Evaluation (mAP@0.5)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.evaluate \
  --checkpoint checkpoints/part3_spatial/run1/best.pth \
  --data-dir datasets/part3_voc_k3_relaxed \
  --split test \
  --conf-thresh 0.25 \
  --tag run1
```

Results are written to `outputs/part3/metrics_<tag>.json`.

#### Inference

**On a single image:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.inference \
  --checkpoint checkpoints/part3_spatial/spatial_attn_focal/best.pth \
  --classes-json datasets/part3_voc_k3_relaxed/classes.json \
  --image path/to/image.jpg \
  --conf-thresh 0.20
```

**On a video (with temporal smoothing):**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.inference \
  --checkpoint checkpoints/part3_spatial/spatial_attn_focal/best.pth \
  --classes-json datasets/part3_voc_k3_relaxed/classes.json \
  --video path/to/video.mp4 \
  --conf-thresh 0.20 \
  --bg-gate 0.5 \
  --ema-alpha 0.4 \
  --out outputs/part3/infer_focal_final
```

Key inference flags:
- `--conf-thresh` — minimum class confidence to show a box (lower = more detections, more noise)
- `--bg-gate` — suppress slot if background probability exceeds this value
- `--ema-alpha` — EMA blend factor for box smoothing: 0=frozen, 1=no smoothing (lower = smoother but laggier)
- `--ema-decay` — frames without detection before a tracked box disappears

#### Architecture: Spatial Slot Attention

The key architectural improvement over a naive GAP (Global Average Pool) baseline is **per-slot spatial attention**:

```
Backbone features: [B, 576, h, w]
     ↓ 1×1 conv
Projected features: [B, 512, h, w]  →  flattened to [B, 512, H×W]
     ↓ dot-product attention (K learnable slot queries × spatial positions)
Slot features: [B, K, 512]   (each slot attends to a different region)
     ↓ per-slot MLP + box/class heads
Output: boxes [B, K, 4],  logits [B, K, C]
```

With GAP, all K slots share the same pooled feature vector, so the model cannot distinguish multiple objects. With spatial attention, each slot query specializes to a different part of the image, enabling true multi-class co-detection.

**Results — ablation across training configurations (test set, conf-thresh=0.25):**

| Model | person | car | dog | mAP@0.5 |
|-------|--------|-----|-----|---------|
| GAP baseline (no pretrained) | — | — | — | 0.150 |
| GAP + pretrained + ImageNet norm | ~0.39 | ~0.20 | ~0.31 | 0.354 |
| Spatial attention (gamma=0) | 0.446 | 0.200 | 0.472 | 0.376 |
| Spatial attention + focal (gamma=2.0) | 0.476 | 0.225 | 0.493 | **0.398** |
| Spatial attention + focal (gamma=3.0) | 0.446 | 0.188 | 0.478 | 0.371 |
| Spatial attention + focal (gamma=2.0) + aug_scale_min=0.40 | 0.425 | 0.203 | 0.482 | 0.370 |

Best checkpoint: `checkpoints/part3_spatial/spatial_attn_focal/best.pth` (focal gamma=2.0)

---

## Project Structure

```
project3/
├── part1/
│   ├── train.py              # Part 1 classification script
│   ├── report.txt
│   └── images/               # Sample images for inference demo
├── part2/
│   ├── train.py              # Part 2 training
│   ├── trainer.py
│   ├── inference.py
│   └── visualize_worst.py
├── part3/
│   ├── model.py              # FixedSlotDetector (spatial attention architecture)
│   ├── dataset.py            # Part3VOCDataset — VOC loader with augmentation
│   ├── loss.py               # FixedSlotLoss — Hungarian matching + CIoU + focal CE
│   ├── trainer.py            # Trainer — training loop, validation, checkpointing
│   ├── train.py              # Training entry point (python -m part3.train)
│   ├── evaluate.py           # mAP@0.5 evaluation (python -m part3.evaluate)
│   ├── inference.py          # Image/folder/video inference (python -m part3.inference)
│   └── run_inference_batch.py
├── tools/
│   ├── build_voc_part3_k3_relaxed.py  # Build dataset index from VOC
│   ├── check_class_dist.py
│   ├── filter_single_object.py
│   └── ...                   # Other dataset utilities
├── models/
│   ├── backbone.py
│   ├── heads.py
│   └── detector.py
├── utils/
│   ├── loss.py
│   ├── metrics.py
│   └── visualization.py
├── data/
│   ├── dataset.py
│   └── transforms.py
├── datasets/
│   └── part3_voc_k3_relaxed/ # Dataset index JSONs (train/val/test + classes.json)
├── checkpoints/
│   └── part3_spatial/<tag>/
│       ├── best.pth          # Best checkpoint by val mIoU
│       ├── last.pth
│       ├── summary.json      # Full training history
│       └── tb/               # TensorBoard logs
├── outputs/
│   └── part3/                # Evaluation results and inference outputs
├── videos/                   # Input/output videos
├── config.py
├── requirements.txt
└── README.md
```

---

## Key Features

### ✨ Design Highlights
- **Composition over Inheritance**: Swappable detection heads
- **Only Basic PyTorch Blocks**: Conv2d, Linear, ReLU, BatchNorm, Dropout
- **TensorBoard Integration**: Real-time training monitoring
- **Checkpoint Management**: Auto-save best model + periodic checkpoints
- **Early Stopping**: Prevents overfitting
- **Dynamic Backbone Unfreezing**: Fine-tune after initial training
- **Multi-Format Support**: COCO JSON, YOLO txt, Pascal VOC XML

### 📊 Training Features
- **Differential Learning Rates**: Separate LR for backbone vs head
- **Gradient Clipping**: Prevents exploding gradients
- **Data Augmentation**: Horizontal flip, color jitter (synchronized with bboxes)
- **Validation Metrics**: IoU, detection rate, mAP@0.5, mAP@0.75

---

## Next Steps

### Immediate Actions:
1. ✅ **Install dependencies** (see above)
2. ✅ **Add images to `part1_images/`** for classification demo
3. ✅ **Run Part 1** to verify setup
4. 📥 **Download datasets** for Part 2 and Part 3
5. 🚀 **Start training!**

### For Part 1 Report:
- Run `part1_classification.py`
- Analyze MobileNetV3-Small architecture
- Document:
  - Total parameters: ~2.5M
  - Input size: 224x224
  - Output: 1000 classes (ImageNet)
  - Model size: ~10 MB

### For Part 2 & 3:
- Monitor with TensorBoard: `tensorboard --logdir checkpoints/part3_spatial/<tag>/tb`
- Best models saved to `checkpoints/`
- Part 3 training: `python -m part3.train` (see Part 3 section above for full command)
- Part 3 inference: `python -m part3.inference` (see Part 3 section above for full command)

---

## Troubleshooting

**Import Errors?**
```bash
pip install torch torchvision --upgrade
```

**No GPU?**
- Training will use CPU automatically
- Expect slower training times

**Dataset Format Issues?**
- Check annotation format matches (COCO JSON, YOLO txt, or VOC XML)
- Verify image paths are correct
- Ensure bbox format is correct

---

## Contact & Support

For questions about the project structure or implementation, refer to:
- `Project3_guidelines-3(2).txt` - Original requirements
- `tips.txt` - Additional guidance
- Code comments in each module

**Good luck with your project! 🚀**
