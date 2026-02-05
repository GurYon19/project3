"""
Clean annotations by removing references to missing images.
Creates backup before modifying.
"""
import json
import shutil
from pathlib import Path

def clean_annotations(data_dir: str = 'datasets/part2', auto_confirm: bool = False):
    """Remove missing images and their annotations from COCO files."""
    data_path = Path(data_dir)
    
    print("="*70)
    print("CLEANING ANNOTATIONS - REMOVING MISSING IMAGES")
    print("="*70)
    
    total_removed = 0
    total_kept = 0
    changes = {}
    
    # First, analyze what needs to be cleaned
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        annotations_file = split_dir / '_annotations.coco.json'
        
        if not annotations_file.exists():
            print(f"\n⚠️ {split.upper()}: No annotations file found")
            continue
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Check which images exist
        existing_images = []
        missing_images = []
        missing_image_ids = set()
        
        for img in coco_data['images']:
            img_path = split_dir / img['file_name']
            
            if img_path.exists():
                existing_images.append(img)
            else:
                missing_images.append(img)
                missing_image_ids.add(img['id'])
        
        # Count annotations for missing images
        removed_annotations = [ann for ann in coco_data['annotations'] 
                              if ann['image_id'] in missing_image_ids]
        kept_annotations = [ann for ann in coco_data['annotations'] 
                           if ann['image_id'] not in missing_image_ids]
        
        changes[split] = {
            'total_images': len(coco_data['images']),
            'existing_images': existing_images,
            'missing_images': missing_images,
            'kept_annotations': kept_annotations,
            'removed_annotations': removed_annotations,
            'categories': coco_data.get('categories', []),
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', [])
        }
        
        total_removed += len(missing_images)
        total_kept += len(existing_images)
        
        # Print analysis
        if len(missing_images) > 0:
            print(f"\n{split.upper()}:")
            print(f"  Total images:      {len(coco_data['images'])}")
            print(f"  ✅ Existing:       {len(existing_images)}")
            print(f"  ❌ Missing:        {len(missing_images)}")
            print(f"  Annotations to remove: {len(removed_annotations)}")
        else:
            print(f"\n{split.upper()}: ✅ All {len(coco_data['images'])} images exist")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images:  {total_kept + total_removed}")
    print(f"✅ Keep:       {total_kept}")
    print(f"❌ Remove:     {total_removed}")
    
    if total_removed == 0:
        print("\n✅ No missing images found - nothing to clean!")
        return
    
    # Confirm
    if not auto_confirm:
        print("\n" + "="*70)
        print("This will:")
        print("  1. Backup current annotations to *_backup.json")
        print("  2. Update annotations to remove missing images")
        print("="*70)
        response = input("\nProceed? (y/n): ").strip().lower()
        
        if response != 'y':
            print("Cancelled.")
            return
    else:
        print("\n✅ Auto-confirmed (--auto flag)")
    
    # Clean annotations
    print("\nCleaning annotations...")
    
    for split, change_info in changes.items():
        if len(change_info['missing_images']) == 0:
            continue
        
        split_dir = data_path / split
        annotations_file = split_dir / '_annotations.coco.json'
        backup_file = split_dir / '_annotations.coco_backup.json'
        
        # Backup original
        print(f"\n{split.upper()}:")
        print(f"  📦 Backing up to: {backup_file.name}")
        shutil.copy2(annotations_file, backup_file)
        
        # Create clean COCO data
        clean_coco = {
            'images': change_info['existing_images'],
            'annotations': change_info['kept_annotations'],
            'categories': change_info['categories']
        }
        
        # Add optional fields if they exist
        if change_info['info']:
            clean_coco['info'] = change_info['info']
        if change_info['licenses']:
            clean_coco['licenses'] = change_info['licenses']
        
        # Save cleaned annotations
        with open(annotations_file, 'w') as f:
            json.dump(clean_coco, f, indent=2)
        
        print(f"  ✅ Removed {len(change_info['missing_images'])} images")
        print(f"  ✅ Removed {len(change_info['removed_annotations'])} annotations")
        print(f"  ✅ Kept {len(change_info['existing_images'])} images")
    
    print("\n" + "="*70)
    print("✅ ANNOTATIONS CLEANED SUCCESSFULLY!")
    print("="*70)
    print("\nBackups saved with suffix: _backup.json")
    print("You can now run the filtering script.")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean annotations by removing missing images")
    parser.add_argument('--data-dir', type=str, default='datasets/part2',
                       help='Dataset directory')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-confirm without prompting')
    
    args = parser.parse_args()
    clean_annotations(args.data_dir, auto_confirm=args.auto)
