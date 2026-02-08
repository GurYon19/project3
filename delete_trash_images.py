"""
Script to delete 'Trash' images from the dataset.
Usage:
1. Run inference -> outputs images to a folder.
2. Manually MOVE the BAD images to a new 'trash' folder.
3. Run this script pointing to that 'trash' folder.
   It will PERMANENTLY delete the corresponding original images from the dataset.
"""
import os
import json
import shutil
from pathlib import Path

def extract_original_filename(trash_filename: str) -> str:
    """
    Extract original image filename.
    Inference script prefixes output with 'pred_'.
    We must remove it to find the original.
    """
    name = trash_filename
    if name.startswith("pred_"):
        name = name[5:]  # Remove 'pred_'
    return name

def delete_trash(trash_dir: str, data_dir: str):
    trash_path = Path(trash_dir)
    data_path = Path(data_dir)
    
    if not trash_path.exists():
        print(f"❌ Trash folder not found: {trash_dir}")
        return

    # 1. Identify files to delete
    files_to_delete = []
    print(f"Scanning {trash_path}...")
    for f in trash_path.iterdir():
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            orig_name = extract_original_filename(f.name)
            files_to_delete.append(orig_name)
    
    if not files_to_delete:
        print("No images found in trash folder.")
        return
        
    print(f"Found {len(files_to_delete)} images marked for deletion.")
    
    confirm = input("Are you sure you want to delete these from the dataset? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return

    # 2. Iterate through splits and delete
    deleted_count = 0
    deleted_files_per_split = {'train': set(), 'valid': set(), 'test': set()}
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        for target_name in files_to_delete:
            target_path = split_dir / target_name
            if target_path.exists():
                print(f"  Deleting: {split}/{target_name}")
                os.remove(target_path)
                deleted_count += 1
                deleted_files_per_split[split].add(target_name)
    
    print(f"\n✅ Deleted {deleted_count} files.")
    
    # 3. Clean Annotations
    print("\nUpdating annotations...")
    for split in ['train', 'valid', 'test']:
        deleted_names = deleted_files_per_split[split]
        if not deleted_names:
            continue
            
        anno_file = data_path / split / '_annotations.coco.json'
        if not anno_file.exists():
            continue
            
        with open(anno_file, 'r') as f:
            coco = json.load(f)
            
        # Filter images
        valid_images = []
        removed_ids = set()
        
        for img in coco['images']:
            if img['file_name'] in deleted_names:
                removed_ids.add(img['id'])
            else:
                valid_images.append(img)
                
        # Filter annotations
        valid_annos = [a for a in coco['annotations'] if a['image_id'] not in removed_ids]
        
        # Save
        coco['images'] = valid_images
        coco['annotations'] = valid_annos
        
        with open(anno_file, 'w') as f:
            json.dump(coco, f, indent=2)
            
        print(f"  Updated {split} JSON: Removed {len(removed_ids)} entries.")

    print("\n🎉 Cleanup Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trash-dir', type=str, required=True, help='Folder containing bad images to delete')
    parser.add_argument('--data-dir', type=str, default='datasets/part2', help='Dataset root')
    args = parser.parse_args()
    
    delete_trash(args.trash_dir, args.data_dir)
