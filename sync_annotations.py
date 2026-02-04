"""
Script to sync COCO annotations with existing images.
Removes annotations for images that no longer exist.
"""
import json
from pathlib import Path

def sync_annotations(data_dir: str):
    """Remove annotations for missing images."""
    data_path = Path(data_dir)
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        annotations_file = split_dir / '_annotations.coco.json'
        
        if not annotations_file.exists():
            print(f"Skipping {split}: No annotations file found")
            continue
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        original_images = len(coco_data['images'])
        original_annotations = len(coco_data['annotations'])
        
        # Get list of existing image files
        existing_files = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            existing_files.update(f.name for f in split_dir.glob(ext))
        
        print(f"\n{split.upper()}:")
        print(f"  Images in folder: {len(existing_files)}")
        print(f"  Images in annotations: {original_images}")
        
        # Filter images that exist
        valid_images = []
        valid_image_ids = set()
        removed_images = []
        
        for img in coco_data['images']:
            if img['file_name'] in existing_files:
                valid_images.append(img)
                valid_image_ids.add(img['id'])
            else:
                removed_images.append(img['file_name'])
        
        # Filter annotations for valid images
        valid_annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] in valid_image_ids
        ]
        
        # Update COCO data
        coco_data['images'] = valid_images
        coco_data['annotations'] = valid_annotations
        
        # Save updated annotations
        with open(annotations_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  Removed {len(removed_images)} missing image references")
        print(f"  Final images: {len(valid_images)}")
        print(f"  Final annotations: {len(valid_annotations)}")
        
        if removed_images and len(removed_images) <= 10:
            print(f"  Removed files: {removed_images}")

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/part2"
    print(f"Syncing annotations in: {data_dir}")
    sync_annotations(data_dir)
    print("\n✅ Annotations synced with existing images!")
