"""Quick count of training images"""
import json
from pathlib import Path

def count_single_objects():
    """Count images with exactly 1 object"""
    ann_path = Path("datasets/part2/train/_annotations.coco.json")
    
    with open(ann_path) as f:
        data = json.load(f)
    
    # Count annotations per image
    img_obj_count = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_obj_count[img_id] = img_obj_count.get(img_id, 0) + 1
    
    total = len(data['images'])
    single = sum(1 for count in img_obj_count.values() if count == 1)
    multi = total - single
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Total images:          {total}")
    print(f"Single-object images:  {single} ({100*single/total:.1f}%)")
    print(f"Multi-object images:   {multi} ({100*multi/total:.1f}%)")
    print(f"\n{'='*60}")
    print(f"RECOMMENDED SPLITS (for single-object images)")
    print(f"{'='*60}")
    
    # Recommended sizes
    rec_train = 800
    rec_val = 200
    rec_test = 200
    
    # Use minimum of available and recommended
    use_train = min(single - 400, rec_train)  # Reserve 400 for val+test
    use_val = min((single - use_train) // 2, rec_val)
    use_test = min(single - use_train - use_val, rec_test)
    
    print(f"Training:   {use_train} (recommended: {rec_train})")
    print(f"Validation: {use_val} (recommended: {rec_val})")
    print(f"Test:       {use_test} (recommended: {rec_test})")
    print(f"Total:      {use_train + use_val + use_test}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    count_single_objects()
