"""
Filter dataset to only images with exactly 1 object.
Recreated V3 Robust Version.
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

def analyze_and_filter_dataset(data_dir: str, auto_confirm: bool = False):
    data_path = Path(data_dir)
    
    print("="*60)
    print("DATASET FILTERING: Single-Object Only")
    print("="*60)
    
    # 1. Collect Valid Images
    all_single_object_data = []
    
    # Standard category to unify everything
    unified_categories = [{"id": 0, "name": "Tiger", "supercategory": "none"}]
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        annotations_file = split_dir / '_annotations.coco.json'
        
        if not annotations_file.exists():
            continue
            
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
            
        # Count objects
        image_object_count = defaultdict(int)
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_object_count[ann['image_id']] += 1
            image_annotations[ann['image_id']].append(ann)
            
        # Filter strictly single object
        for img in coco_data['images']:
            img_id = img['id']
            img_path = split_dir / img['file_name']
            
            # Check 1: Is single object?
            if image_object_count[img_id] != 1:
                continue
                
            # Check 2: Does file exist?
            if not img_path.exists():
                continue
                
            # Store data 
            all_single_object_data.append({
                'image': img,
                'annotation': image_annotations[img_id][0], 
                'source_split': split,
                'filename': img['file_name']
            })

    total_valid = len(all_single_object_data)
    print(f"\nFound {total_valid} valid single-object images.")
        
    if total_valid == 0:
        print("\n❌ ERROR: No valid single-object images found!")
        return

    # 3. Create Backup & Prepare for Write
    print("\nCreating backup and restructuring...")
    
    backup_dir = data_path.parent / f"{data_path.name}_backup"
    
    if not backup_dir.exists():
        print(f"Creating backup: {backup_dir}")
        shutil.copytree(data_path, backup_dir)
    else:
        print(f"Using existing backup: {backup_dir}")
    
    # Use 100% of available data
    # (User can subsample during training if they want)
    target_total = total_valid
    
    # Split Strategy: 70/15/15
    train_size = int(target_total * 0.70)
    val_size = int(target_total * 0.15)
    test_size = target_total - train_size - val_size
    
    print(f"Split Strategy: 70/15/15")
    print(f"  Train: {train_size}")
    print(f"  Valid: {val_size}")
    print(f"  Test:  {test_size}")
    
    if not auto_confirm:
        resp = input("\nProceed? (y/n): ").lower().strip()
        if resp != 'y': return

    # Shuffle Data
    random.seed(42)
    random.shuffle(all_single_object_data)
    
    # Create Slices
    data_splits = {
        'train': all_single_object_data[:train_size],
        'valid': all_single_object_data[train_size:train_size+val_size],
        'test':  all_single_object_data[train_size+val_size:train_size+val_size+test_size]
    }
    
    # 4. Write New Dataset
    # We will clear the 'part2' folder contents
    for split_subfolder in ['train', 'valid', 'test']:
        d = data_path / split_subfolder
        if d.exists():
            for f in d.glob('*'):
                try: f.unlink() 
                except: pass
        else:
            d.mkdir(parents=True, exist_ok=True)

    # Copy files FROM BACKUP to avoid self-deletion issues
    for split_name, entries in data_splits.items():
        if not entries: continue
        
        dest_dir = data_path / split_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Writing {split_name.upper()} ({len(entries)} images)...")
        
        new_coco = {
            'images': [],
            'annotations': [],
            'categories': unified_categories
        }
        
        for idx, entry in enumerate(entries):
            fname = entry['filename']
            src_split = entry['source_split']
            
            # The Source is explicitly the BACKUP directory
            src_path = backup_dir / src_split / fname
            dst_path = dest_dir / fname
            
            if not src_path.exists():
                print(f"  ⚠️ Warning: Source lost? {src_path}")
                continue
                
            shutil.copy2(src_path, dst_path)
            
            # Build COCO
            img_info = entry['image'].copy()
            img_info['id'] = idx
            new_coco['images'].append(img_info)
            
            ann = entry['annotation'].copy()
            ann['id'] = idx
            ann['image_id'] = idx
            ann['category_id'] = 0
            new_coco['annotations'].append(ann)
            
        with open(dest_dir / '_annotations.coco.json', 'w') as f:
            json.dump(new_coco, f)
            
    print("\n✅ Done! Dataset filtered successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/part2')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    
    analyze_and_filter_dataset(args.data_dir, args.auto)
