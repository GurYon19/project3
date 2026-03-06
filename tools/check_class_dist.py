import json
from pathlib import Path

for split in ['train', 'val', 'test']:
    data = json.loads(Path(f'datasets/part3_voc_k3_relaxed/{split}.json').read_text())
    classes_info = json.loads(Path('datasets/part3_voc_k3_relaxed/classes.json').read_text())
    classes = classes_info['classes']
    bg_id = classes_info['bg_id']

    counts = {c: 0 for c in classes if c != '__background__'}
    n_images = {c: 0 for c in classes if c != '__background__'}
    for item in data:
        seen = set()
        for label, mask in zip(item['labels'], item['mask']):
            if mask and label != bg_id:
                counts[classes[label]] += 1
                seen.add(classes[label])
        for c in seen:
            n_images[c] += 1
    print(f'{split}: object_counts={counts}')
    print(f'       images_with_class={n_images}')
    print(f'       total_images={len(data)}')
    print()
