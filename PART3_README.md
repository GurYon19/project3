# Part 3 — VOC Relaxed (K=3) — Complete Runbook

This document contains the **exact order of commands** to run for Part 3 using:

- Classes: `person, car, dog`
- Fixed capacity: `K=3`
- Background class included (`bg_id = 3`)
- Dataset: `datasets/part3_voc_k3_relaxed`
- Checkpoints: `checkpoints/part3`
- Outputs: `outputs/part3`

---

# 0️⃣ Confirm Branch

```bash
git status
git branch
```

Expected:
- Branch: `part3-voc-relaxed-k3`
- Working tree clean (or only intentional changes)

---

# 1️⃣ Build / Filter Dataset (RELAXED + K=3)

```bash
python tools/build_voc_part3_k3_relaxed.py \
  --voc-root "/Users/yehudafrist/RUNI/computer_vision/project3/datasets/pascal_voc" \
  --out-dir datasets/part3_voc_k3_relaxed \
  --classes person car dog \
  --max-objects 3 \
  --selection-strategy prefer_then_area \
  --seed 42
```

Expected output:
- Creates:
  - `train.json`
  - `val.json`
  - `test.json`
  - `classes.json`
  - `_stats.json`
- Console prints:
  - kept images
  - truncated_images > 0
  - padded_images > 0 (normal for VOC)

---

# 2️⃣ Confirm classes.json

```bash
cat datasets/part3_voc_k3_relaxed/classes.json
```

Expected:
```json
{
  "classes": ["person", "car", "dog", "__background__"],
  "bg_id": 3
}
```

---

# 3️⃣ Confirm dataset.py Works

```bash
python -c "from part3.dataset import Part3VOCDataset; d=Part3VOCDataset('datasets/part3_voc_k3_relaxed/train.json','datasets/part3_voc_k3_relaxed/classes.json'); x,t=d[0]; print(x.shape,t['boxes'].shape,t['labels'],t['mask'])"
```

Expected output example:

```
torch.Size([3, 448, 448]) torch.Size([3, 4]) tensor([0, 3, 3]) tensor([ True, False, False])
```

Meaning:
- 1 real object
- 2 padded background slots

---

# 4️⃣ Confirm Dataset Contains Multi-Object Samples

```bash
python - <<'PY'
from part3.dataset import Part3VOCDataset
d=Part3VOCDataset('datasets/part3_voc_k3_relaxed/train.json','datasets/part3_voc_k3_relaxed/classes.json')
counts={0:0,1:0,2:0,3:0}
for i in range(500):
    _,t=d[i]
    n=int(t["mask"].sum().item())
    counts[n]+=1
print("mask True histogram:", counts)
PY
```

Expected:
- Non-zero counts for `2` and `3`

---

# 5️⃣ Sanity Check loss.py

```bash
python -c "import torch; from part3.loss import FixedSlotLoss; L=FixedSlotLoss(4,bg_id=3); out={'boxes':torch.rand(2,3,4)*448,'logits':torch.randn(2,3,4)}; tgt={'boxes':torch.rand(2,3,4)*448,'labels':torch.randint(0,4,(2,3)),'mask':torch.tensor([[1,0,0],[1,1,0]]).bool()}; print(L(out,tgt).keys())"
```

Expected:

```
dict_keys(['loss', 'loss_cls', 'loss_box'])
```

---

# 6️⃣ Confirm model.py Forward Pass

```bash
python - <<'PY'
import torch
from part3.model import FixedSlotDetector, ModelConfig
cfg=ModelConfig(image_size=448,max_objects=3,num_classes_total=4,bg_id=3,backbone="mobilenet_v3_small",pretrained=False)
m=FixedSlotDetector(cfg)
x=torch.rand(2,3,448,448)
y=m(x)
print("boxes:", y["boxes"].shape, "logits:", y["logits"].shape)
PY
```

Expected:

```
boxes: torch.Size([2, 3, 4]) logits: torch.Size([2, 3, 4])
```

---

# 7️⃣ Training Sanity Check (Short Run)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.train \
  --data-dir datasets/part3_voc_k3_relaxed \
  --epochs 3 \
  --batch-size 16 \
  --num-workers 0 \
  --tag voc_k3_relaxed_sanity \
  --use-focal
```

Expected:
- Training loss printed
- Validation mAP printed
- Checkpoints:
  - `checkpoints/part3/voc_k3_relaxed_sanity/last.pth`
  - `checkpoints/part3/voc_k3_relaxed_sanity/best.pth`

---

# 8️⃣ Full Training

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
  --use-focal \
  --tag voc_k3_relaxed_run1
```

Expected:
- TensorBoard logs in:
  - `checkpoints/part3/voc_k3_relaxed_run1/tb`
- Best checkpoint:
  - `checkpoints/part3/voc_k3_relaxed_run1/best.pth`

---

# 9️⃣ View TensorBoard

```bash
tensorboard --logdir checkpoints/part3_relaxed/voc_k3_relaxed_run1/tb
```

Open the URL shown in terminal.

Look for:
- train/loss
- train/loss_cls
- train/loss_box
- val/mAP@0.5

---

# 🔟 Evaluate on Test Split

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.evaluate \
  --data-dir datasets/part3_voc_k3_relaxed \
  --split test \
  --checkpoint checkpoints/part3_relaxed/voc_k3_relaxed_run1/best.pth \
  --conf-thresh 0.35 \
  --topk 3 \
  --batch-size 32 \
  --num-workers 0 \
  --out-dir outputs/part3 \
  --tag voc_k3_relaxed_run1_best
```

Expected:
- Console prints AP per class
- File written:
  - `outputs/part3/metrics_voc_k3_relaxed_run1_test.json`

---

# 1️⃣1️⃣ Threshold Sweep (Recommended)

```bash
for thr in 0.10 0.15 0.20 0.25 0.30; do
  python -m part3.evaluate \
    --data-dir datasets/part3_voc_k3_relaxed \
    --split test \
    --checkpoint checkpoints/part3_relaxed/voc_k3_relaxed_run1/best.pth \
    --conf-thresh $thr \
    --topk 3 \
    --batch-size 32 \
    --num-workers 0 \
    --out-dir outputs/part3 \
    --tag voc_k3_relaxed_run1_test_thr${thr}
done
```

Expected:
- Multiple metric JSON files saved

---

# 1️⃣2️⃣ Inference on Test Images

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m part3.run_inference_batch \
  --index-json datasets/part3_voc_k3_relaxed/test.json \
  --checkpoint checkpoints/part3_relaxed/voc_k3_relaxed_run1/best.pth \
  --classes-json datasets/part3_voc_k3_relaxed/classes.json \
  --image-size 448 \
  --max-objects 3 \
  --conf-thresh 0.15 \
  --topk 3 \
  --out-dir outputs/part3/infer_voc_k3_relaxed_test20 \
  --n 20 \
  --seed 42
```

Expected:
- outputs/part3/infer_voc_k3_relaxed_test20/*

---

# 1️⃣3️⃣ Inference on Video (External Requirement)

Remove audio (recommended):

```bash
ffmpeg -i videos/input_video.mp4 -an -vcodec copy videos/input_video_noaudio.mp4
```

Run inference:

```bash
python -m part3.inference \
  --checkpoint checkpoints/part3_realxed/voc_k3_relaxed_run1/best.pth \
  --classes-json datasets/part3_voc_k3_relaxed/classes.json \
  --video videos/input_video_noaudio.mp4 \
  --conf-thresh 0.20 \
  --out outputs/part3/video_run1_thr020
```

Expected:
- `outputs/part3/video_run1_thr020/video_out.mp4`

Important:
- The video must show at least **two classes in the same frame**.

---

# ✅ Final Checklist

✔ Dataset built (relaxed, padded, truncated)  
✔ classes.json contains background class  
✔ Dataset returns correct mask behavior  
✔ Loss works  
✔ Model forward shapes correct  
✔ Training runs  
✔ mAP@0.5 computed  
✔ External video inference works  

---

# 🔧 If Results Look Poor

1. Lower confidence threshold (0.10–0.20).
2. Verify dataset stats:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   p=Path("datasets/part3_voc_k3_relaxed/_stats.json")
   print(json.loads(p.read_text())["per_class_object_counts_selected"])
   PY
   ```
3. Confirm `bg_id` consistency everywhere.
4. Ensure `car` and `dog` appear in validation/test splits.

---

This completes the full Part 3 execution pipeline.