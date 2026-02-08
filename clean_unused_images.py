"""
Script to physically delete image files that are not referenced in the JSON annotations.
This cleans up the folder after filtering the dataset (e.g., removing multi-object images from JSON).
"""
import os
import json
from pathlib import Path

def clean_unused_files(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    print(f"\nCleaning unused images in: {data_dir}")
    
    # 1. Load valid filenames from JSON
    anno_file = data_path / '_annotations.coco.json'
    if not anno_file.exists():
        print(f"❌ No annotations file found in {data_dir}")
        return
        
    with open(anno_file, 'r') as f:
        coco = json.load(f)
        
    valid_filenames = set()
    for img in coco['images']:
        valid_filenames.add(img['file_name'])
        
    print(f"  Valid images in JSON: {len(valid_filenames)}")
    
    # 2. Scan folder and identify orphans
    files_in_folder = []
    orphaned_files = []
    
    # Scan for image files
    files_found_set = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for f in data_path.glob(ext):
            files_found_set.add(f)
            
    # Check for orphans
    for f in files_found_set:
        if f.name not in valid_filenames:
            orphaned_files.append(f)
                
    total_files = len(files_found_set)
    orphan_count = len(orphaned_files)
    
    print(f"  Total image files found: {total_files}")
    print(f"  Orphaned files (Not in JSON): {orphan_count}")
    
    if orphan_count == 0:
        print("  ✅ Folder is clean. No files to delete.")
        return

    # 3. Confirm Deletion
    confirm = input(f"  ⚠️ Delete {orphan_count} unused files? (y/n): ")
    if confirm.lower() != 'y':
        print("  Cancelled.")
        return
        
    # 4. Delete
    deleted = 0
    for f in orphaned_files:
        try:
            os.remove(f)
            deleted += 1
            if deleted % 100 == 0:
                print(f"    Deleted {deleted}/{orphan_count}...", end='\r')
        except Exception as e:
            print(f"    Error deleting {f.name}: {e}")
            
    print(f"\n  ✅ Successfully deleted {deleted} files.")
    print(f"  Remaining files: {total_files - deleted}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='datasets/part2', help='Dataset root')
    args = parser.parse_args()
    
    root = Path(args.data_dir)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        clean_unused_files(str(root / split))

if __name__ == "__main__":
    main()
