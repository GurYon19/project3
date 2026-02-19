- Filter dataset:

python tools/filter_voc_for_part3.py \
  --voc-root datasets/pascal_voc \
  --classes person car dog \
  --max-objects 3

  output: /Users/yehudafrist/RUNI/computer_vision/project3/datasets/part3/*

- Confirm model.py works:

python -c "import torch; from part3.model import FixedSlotDetector; m=FixedSlotDetector(); x=torch.randn(2,3,448,448); y=m(x); print(y['boxes'].shape, y['logits'].shape)"

  output: torch.Size([2, 3, 4]) torch.Size([2, 3, 4])

- Confirm dataset.py:

python -c "from part3.dataset import Part3VOCDataset, collate_part3; from torch.utils.data import DataLoader; ds=Part3VOCDataset('datasets/part3/train.json','datasets/part3/classes.json'); dl=DataLoader(ds,batch_size=2,collate_fn=collate_part3); x,t=next(iter(dl)); print(x.shape, t['boxes'].shape, t['labels'].shape, t['mask'].shape, t['labels'].max().item())"

  output: torch.Size([2, 3, 448, 448]) torch.Size([2, 3, 4]) torch.Size([2, 3]) torch.Size([2, 3]) 3

- Sanity check for loss.py:

python -c "import torch; from part3.model import FixedSlotDetector; from part3.loss import FixedSlotLoss; \
m=FixedSlotDetector(); crit=FixedSlotLoss(num_classes=3); \
x=torch.randn(2,3,448,448); out=m(x); \
t={'boxes':torch.zeros(2,3,4), 'labels':torch.full((2,3),3,dtype=torch.long), 'mask':torch.zeros(2,3,dtype=torch.bool)}; \
print(crit(out,t).keys())"

  output: dict_keys(['loss', 'loss_cls', 'loss_box'])

- Training sanity check:

python -m part3.train \
  --data-dir datasets/part3 \
  --epochs 3 \
  --batch-size 16 \
  --freeze-backbone \
  --num-workers 0

output example:
[DEVICE] mps
[DATA] train=7079 val=1517 classes=['person', 'car', 'dog'] bg_id=3
[E000] train loss=3.9836 (cls=0.5912, box=0.6785) | val loss=3.7454 miou=0.3421
[E001] train loss=3.6915 (cls=0.4574, box=0.6468) | val loss=3.5789 miou=0.3324
[E002] train loss=3.5856 (cls=0.4195, box=0.6332) | val loss=3.5365 miou=0.3298

- Run full training:

python -m part3.train \
  --data-dir datasets/part3 \
  --image-size 448 \
  --max-objects 3 \
  --epochs 60 \
  --batch-size 16 \
  --freeze-backbone \
  --unfreeze-epoch 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --log-dir logs/part3_run1 \
  --ckpt-dir checkpoints/part3_run1

- Resume if interrupted: 

python -m part3.train \
  --data-dir datasets/part3 \
  --image-size 448 \
  --max-objects 3 \
  --epochs 60 \
  --batch-size 16 \
  --freeze-backbone \
  --unfreeze-epoch 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --log-dir logs/part3_run1 \
  --ckpt-dir checkpoints/part3_run1 \
  --resume checkpoints/part3_run1/last.pth

- View TensorBoard curves:

tensorboard --logdir logs/part3_run1
Then open the url and download curves.

- Run inference on image: 

python -m part3.inference \
  --checkpoint checkpoints/part3/best.pth \
  --image /Users/yehudafrist/RUNI/computer_vision/project3/datasets/pascal_voc/JPEGImages/2007_000027.jpg \
  --conf-thresh 0.35 \
  --out-dir outputs/part3

- Test evalute.py:

python -m part3.evaluate \
  --data-dir datasets/part3 \
  --split test \
  --checkpoint checkpoints/part3_run1/best.pth \
  --conf-thresh 0.35 \
  --topk 3 \
  --batch-size 32 \
  --num-workers 0 \
  --out-dir outputs/part3 \
  --tag run1_best

output: outputs/part3/run1_best_test_ap50.json

- Implemented weighted loss, re-train:

python -m part3.train \
  --data-dir datasets/part3 \
  --epochs 60 \
  --batch-size 16 \
  --freeze-backbone \
  --unfreeze-epoch 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --log-dir logs/part3_run2_weighted \
  --ckpt-dir checkpoints/part3_run2_weighted

- Then evaluate:

python -m part3.evaluate \
  --data-dir datasets/part3 \
  --split test \
  --checkpoint checkpoints/part3_run2_weighted/best.pth \
  --conf-thresh 0.35 \
  --topk 3 \
  --batch-size 32 \
  --num-workers 0 \
  --out-dir outputs/part3 \
  --tag run2_weighted_best

- Weighted loss didnt help, run inference on image from run1:

python -m part3.inference \
  --checkpoint checkpoints/part3_run1/best.pth \
  --classes-json datasets/part3/classes.json \
  --image-size 448 \
  --max-objects 3 \
  --conf-thresh 0.35 \
  --topk 3 \
  --image /path/to/some_test_image.jpg \
  --out-dir outputs/part3/infer_run1

output: outputs/part3/infer_run1/<image>_pred.jpg

- Grab n random test images and run inference on all of them

python -m part3.run_inference_batch \
  --index-json datasets/part3/test.json \
  --checkpoint checkpoints/part3_run1/best.pth \
  --out-dir outputs/part3/infer_run1_test30 \
  --n 30 \
  --seed 42 \
  --conf-thresh 0.35

outputs: utputs/part3/infer_run1_test30/*

Todo:

2) Run inference on a handful of images (for qualitative results)

After you have mAP, generate visuals:

10–20 test images

show success and failure cases

include 2–4 examples in the report

Also, this is a quick sanity check that predictions “look right”.

✅ Do this right after evaluation.

3) External video inference (required deliverable)

Once inference works well, do the external video demo:

short clip with person + car or person + dog

save predicted video to outputs/part3/

✅ Do this after image inference.

4) Only then: address class imbalance / augmentation (as an improvement experiment)

Right now you have a strong baseline. Don’t change training until you have baseline metrics.

After evaluation you’ll know:

if car/dog AP is low (likely) → imbalance is hurting

if everything is decent → keep it simple

Then pick one improvement (not 5 things), rerun a shorter training, compare:

Recommended single improvement:

balanced sampling or class-weighted CE (cheap and effective)
OR

horizontal flip + color jitter augmentation (also cheap and safe)

But don’t do this until you have baseline evaluation.