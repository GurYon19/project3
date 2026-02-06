"""
Filter dataset to only images with exactly 1 object.
RESPECTS SPLITS: Does not mix train/valid/test.
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict
import os

def filter_split(data_dir: str, split_name: str, auto_confirm: bool = False):
    """Filter a specific split (train/valid/test) to keep only single-object images."""
    split_dir = Path(data_dir) / split_name
    annotations_file = split_dir / '_annotations.coco.json'
    
    if not annotations_file.exists():
        print(f"Skipping {split_name} (no annotations found)")
        return

    print(f"\nProcessing {split_name}...")
    
    with open(annotations_file, 'r') as f:
        coco = json.load(f)
    
    # helper: group annotations by image
    img_to_annos = defaultdict(list)
    for ann in coco['annotations']:
        img_to_annos[ann['image_id']].append(ann)
        
    # Find valid images (exactly 1 annotation)
    valid_img_ids = set()
    single_obj_count = 0
    multi_obj_count = 0
    zero_obj_count = 0
    
    for img in coco['images']:
        img_id = img['id']
        ann_count = len(img_to_annos[img_id])
        
        if ann_count == 1:
            valid_img_ids.add(img_id)
            single_obj_count += 1
        elif ann_count > 1:
            multi_obj_count += 1
        else:
            zero_obj_count += 1
            
    total = len(coco['images'])
    print(f"  Total images: {total}")
    print(f"  Single Object (Keep): {single_obj_count}")
    print(f"  Multi Object (Remove): {multi_obj_count}")
    print(f"  Zero Object (Remove): {zero_obj_count}")
    
    if not auto_confirm:
        confirm = input("  Apply changes? (y/n): ")
        if confirm.lower() != 'y':
            print("  Cancelled.")
            return

    # Filter Data
    new_images = [img for img in coco['images'] if img['id'] in valid_img_ids]
    new_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in valid_img_ids]
    
    # Save Backup
    shutil.copy(annotations_file, annotations_file.with_suffix('.json.bak'))
    
    # Save New JSON
    coco['images'] = new_images
    coco['annotations'] = new_annotations
    
    with open(annotations_file, 'w') as f:
        json.dump(coco, f, indent=2)
        
    print(f"  ✅ Updated JSON: {annotations_file}")
    
    # Optional: Delete unused images files?
    # Usually safer to just filter JSON, but if user wants to save space:
    # We will just report for now.

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='datasets/part2')
    parser.add_argument('--auto', action='store_true', help='Skip confirmation')
    args = parser.parse_args()
    
    print(f"Filtering Single Object Images in: {args.data_dir}")
    print(f"Mode: IN-PLACE (Respecting splits)")
    
    for split in ['train', 'valid', 'test']:
        filter_split(args.data_dir, split, args.auto)

if __name__ == "__main__":
    main()
