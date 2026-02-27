- Filter dataset:

python tools/build_voc_part3_k3_relaxed.py \
  --voc-root "/Users/yehudafrist/RUNI/computer_vision/project3/datasets/pascal_voc" \
  --out-dir datasets/part3_voc_k3_relaxed \
  --classes person car dog \
  --max-objects 3 \
  --selection-strategy prefer_then_area \
  --seed 42

  output: /Users/yehudafrist/RUNI/computer_vision/project3/datasets/part3_voc_k3_relaxed/*

- Confirm model.py works:

  output: 

- Confirm dataset.py:

python -c "from part3.dataset import Part3VOCDataset, collate_part3; d=Part3VOCDataset('datasets/part3_voc_k3_relaxed/train.json','datasets/part3_voc_k3_relaxed/classes.json',augment=True); x,t=d[0]; print(x.shape,t['boxes'].shape,t['labels'].shape,t['mask'])"

  output: torch.Size([3, 448, 448]) torch.Size([3, 4]) torch.Size([3]) tensor([ True, False, False])

- Sanity check for loss.py:

python -c "import torch; from part3.loss import FixedSlotLoss; L=FixedSlotLoss(4,bg_id=3); out={'boxes':torch.rand(2,3,4)*448,'logits':torch.randn(2,3,4)}; tgt={'boxes':torch.rand(2,3,4)*448,'labels':torch.randint(0,4,(2,3)),'mask':torch.tensor([[1,0,0],[1,1,0]]).bool()}; print(L(out,tgt).keys())"

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

- Download video to videos folder
- Remove sound using:
ffmpeg -i videos/input_video.mp4 -an -vcodec copy videos/input_video_noaudio.mp4

- Run inference on the video:

KMP_DUPLICATE_LIB_OK=TRUE python -m part3.inference \
  --checkpoint checkpoints/part3_run1/best.pth \
  --classes-json datasets/part3/classes.json \
  --image-size 448 \
  --max-objects 3 \
  --conf-thresh 0.25 \
  --topk 3 \
  --video videos/input_video_noaudio.mp4 \
  --out-dir outputs/part3/video_run1_thr025

- Run on k9 video:

KMP_DUPLICATE_LIB_OK=TRUE python -m part3.inference \
  --checkpoint checkpoints/part3_run5_aug/best.pth \
  --classes-json datasets/part3/classes.json \
  --image-size 448 \
  --max-objects 3 \
  --conf-thresh 0.25 \
  --topk 3 \
  --video videos/k9_video_input_448_letterbox_noaudio.mp4 \
  --out-dir outputs/part3/k9_video_run5_aug_thr025