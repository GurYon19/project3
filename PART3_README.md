Filter dataset:

python tools/filter_voc_for_part3.py \
  --voc-root datasets/pascal_voc \
  --classes person car dog \
  --max-objects 3
