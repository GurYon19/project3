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

  output:

- Training sanity check:

python -m part3.train \
  --data-dir datasets/part3 \
  --epochs 3 \
  --batch-size 16 \
  --freeze-backbone \
  --num-workers 0

  output:


