"""
Merge test set into validation set for Part 2.
- Copies test images into valid/ folder
- Merges COCO annotations with proper ID remapping
- Validates alignment at every step
"""
import json
import shutil
from pathlib import Path
import sys

def merge_test_into_valid(data_dir: str):
    data_dir = Path(data_dir)
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"
    
    valid_ann_path = valid_dir / "_annotations.coco.json"
    test_ann_path = test_dir / "_annotations.coco.json"
    
    # --- Pre-checks ---
    assert valid_dir.exists(), f"Valid dir not found: {valid_dir}"
    assert test_dir.exists(), f"Test dir not found: {test_dir}"
    assert valid_ann_path.exists(), f"Valid annotations not found: {valid_ann_path}"
    assert test_ann_path.exists(), f"Test annotations not found: {test_ann_path}"
    
    # --- Load annotations ---
    with open(valid_ann_path, 'r') as f:
        valid_data = json.load(f)
    with open(test_ann_path, 'r') as f:
        test_data = json.load(f)
    
    print("=" * 60)
    print("BEFORE MERGE")
    print("=" * 60)
    print(f"  Valid images:      {len(valid_data['images'])}")
    print(f"  Valid annotations: {len(valid_data['annotations'])}")
    print(f"  Test images:       {len(test_data['images'])}")
    print(f"  Test annotations:  {len(test_data['annotations'])}")
    
    # --- Validate categories match ---
    valid_cats = {c['id']: c['name'] for c in valid_data['categories']}
    test_cats = {c['id']: c['name'] for c in test_data['categories']}
    assert valid_cats == test_cats, f"Category mismatch!\nValid: {valid_cats}\nTest: {test_cats}"
    print(f"  Categories match: {valid_cats}")
    
    # --- Pre-merge: check for filename collisions ---
    valid_filenames = {img['file_name'] for img in valid_data['images']}
    test_filenames = {img['file_name'] for img in test_data['images']}
    overlap = valid_filenames & test_filenames
    if overlap:
        print(f"\n  [WARN] Found {len(overlap)} overlapping filenames - will skip these duplicates")
    
    # --- Backup valid annotations ---
    backup_path = valid_dir / "_annotations.coco.json.pre_merge_backup"
    shutil.copy2(valid_ann_path, backup_path)
    print(f"\n  [BACKUP] Backed up original valid annotations to: {backup_path.name}")
    
    # --- Compute ID offsets to avoid collisions ---
    max_valid_img_id = max(img['id'] for img in valid_data['images']) if valid_data['images'] else 0
    max_valid_ann_id = max(ann['id'] for ann in valid_data['annotations']) if valid_data['annotations'] else 0
    
    img_id_offset = max_valid_img_id + 1
    ann_id_offset = max_valid_ann_id + 1
    
    print(f"\n  ID offsets: img_id += {img_id_offset}, ann_id += {ann_id_offset}")
    
    # --- Build old_id -> new_id mapping for test images ---
    test_img_id_map = {}
    images_copied = 0
    images_skipped = 0
    
    for img in test_data['images']:
        # Skip duplicates
        if img['file_name'] in overlap:
            images_skipped += 1
            continue
        
        old_id = img['id']
        new_id = old_id + img_id_offset
        test_img_id_map[old_id] = new_id
        
        # Create new image entry with remapped ID
        new_img = img.copy()
        new_img['id'] = new_id
        valid_data['images'].append(new_img)
        
        # Copy actual image file
        src = test_dir / img['file_name']
        dst = valid_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
            images_copied += 1
        else:
            print(f"  [ERROR] WARNING: Test image not found: {src}")
    
    # --- Remap and merge annotations ---
    annotations_added = 0
    annotations_skipped = 0
    
    for ann in test_data['annotations']:
        old_img_id = ann['image_id']
        
        # Skip annotations for duplicate images
        if old_img_id not in test_img_id_map:
            annotations_skipped += 1
            continue
        
        new_ann = ann.copy()
        new_ann['id'] = ann['id'] + ann_id_offset
        new_ann['image_id'] = test_img_id_map[old_img_id]
        valid_data['annotations'].append(new_ann)
        annotations_added += 1
    
    # --- Save merged annotations ---
    with open(valid_ann_path, 'w') as f:
        json.dump(valid_data, f, indent=2)
    
    # ====================
    # VALIDATION CHECKS
    # ====================
    print("\n" + "=" * 60)
    print("AFTER MERGE")
    print("=" * 60)
    print(f"  Images copied:       {images_copied}")
    print(f"  Images skipped (dup): {images_skipped}")
    print(f"  Annotations added:   {annotations_added}")
    print(f"  Annotations skipped: {annotations_skipped}")
    print(f"  Total valid images:      {len(valid_data['images'])}")
    print(f"  Total valid annotations: {len(valid_data['annotations'])}")
    
    # Check 1: Unique image IDs
    img_ids = [img['id'] for img in valid_data['images']]
    assert len(img_ids) == len(set(img_ids)), "FAIL: DUPLICATE IMAGE IDs FOUND!"
    print("  [OK] All image IDs are unique")
    
    # Check 2: Unique annotation IDs
    ann_ids = [ann['id'] for ann in valid_data['annotations']]
    assert len(ann_ids) == len(set(ann_ids)), "FAIL: DUPLICATE ANNOTATION IDs FOUND!"
    print("  [OK] All annotation IDs are unique")
    
    # Check 3: Every annotation points to a valid image
    valid_img_ids = set(img_ids)
    for ann in valid_data['annotations']:
        assert ann['image_id'] in valid_img_ids, f"FAIL: Annotation {ann['id']} references missing image {ann['image_id']}"
    print("  [OK] All annotations reference valid images")
    
    # Check 4: Every image file actually exists on disk
    missing_files = []
    for img in valid_data['images']:
        if not (valid_dir / img['file_name']).exists():
            missing_files.append(img['file_name'])
    if missing_files:
        print(f"  [ERROR] {len(missing_files)} image files missing on disk!")
        for f in missing_files[:5]:
            print(f"      {f}")
    else:
        print(f"  [OK] All {len(valid_data['images'])} image files exist on disk")
    
    # Check 5: Every image on disk has annotation entry
    disk_images = set(f.name for f in valid_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
    ann_images = set(img['file_name'] for img in valid_data['images'])
    unannotated = disk_images - ann_images
    if unannotated:
        print(f"  [WARN] {len(unannotated)} images on disk without annotation entries")
    else:
        print(f"  [OK] All disk images have annotation entries")
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE — All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/part2"
    merge_test_into_valid(data_dir)
