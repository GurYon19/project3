"""
Remove images from dataset based on files in worst_predictions folder.
Also updates COCO annotations to remove references to deleted images.
"""
import os
import json
from pathlib import Path

def extract_original_filename(worst_filename: str) -> str:
    """Extract original image filename from worst prediction filename.
    
    Format: worst_XX_iouX.XX_ORIGINALNAME.jpg
    """
    # Remove prefix like "worst_01_iou0.00_"
    parts = worst_filename.split('_', 3)  # Split into at most 4 parts
    if len(parts) >= 4:
        return parts[3]  # The original filename is the 4th part
    return None

def remove_images_from_dataset(worst_dir: str, data_dir: str):
    """Remove images marked for deletion from dataset."""
    
    worst_path = Path(worst_dir)
    data_path = Path(data_dir)
    
    # Get list of images to delete (exclude JSON files)
    to_delete = []
    for f in worst_path.glob('worst_*.jpg'):
        original_name = extract_original_filename(f.name)
        if original_name:
            to_delete.append(original_name)
    
    print(f"Found {len(to_delete)} images to remove:")
    for name in to_delete:
        print(f"  - {name}")
    
    # Find and delete from train and valid folders
    deleted_count = 0
    deleted_files = []
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        for original_name in to_delete:
            img_path = split_dir / original_name
            if img_path.exists():
                print(f"  Deleting: {split}/{original_name}")
                os.remove(img_path)
                deleted_count += 1
                deleted_files.append((split, original_name))
    
    print(f"\n✅ Deleted {deleted_count} images from dataset")
    
    # Update annotations
    print("\nUpdating annotations...")
    for split in ['train', 'valid', 'test']:
        annotations_file = data_path / split / '_annotations.coco.json'
        if not annotations_file.exists():
            continue
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        original_count = len(coco_data['images'])
        
        # Get names of files deleted from this split
        deleted_in_split = {name for s, name in deleted_files if s == split}
        
        if not deleted_in_split:
            continue
        
        # Filter images
        valid_images = []
        removed_ids = set()
        
        for img in coco_data['images']:
            if img['file_name'] not in deleted_in_split:
                valid_images.append(img)
            else:
                removed_ids.add(img['id'])
                print(f"  Removed from {split} annotations: {img['file_name']}")
        
        # Filter annotations
        valid_annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] not in removed_ids
        ]
        
        # Update COCO data
        coco_data['images'] = valid_images
        coco_data['annotations'] = valid_annotations
        
        # Save
        with open(annotations_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        new_count = len(coco_data['images'])
        print(f"  {split}: {original_count} → {new_count} images")
    
    print(f"\n✅ Annotations updated successfully!")
    print(f"\nSummary:")
    print(f"  Images removed: {deleted_count}")
    print(f"  Annotations cleaned: ✓")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove bad images from dataset")
    parser.add_argument('--worst-dir', type=str, default='outputs/worst_predictions',
                       help='Directory containing worst prediction images')
    parser.add_argument('--data-dir', type=str, default='datasets/part2',
                       help='Dataset directory')
    
    args = parser.parse_args()
    remove_images_from_dataset(args.worst_dir, args.data_dir)
