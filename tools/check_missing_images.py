"""
Check for missing images in the dataset.
"""
import json
from pathlib import Path

def check_missing_images(data_dir: str = 'datasets/part2'):
    """Find all images referenced in annotations but missing from filesystem."""
    data_path = Path(data_dir)
    
    print("="*70)
    print("CHECKING FOR MISSING IMAGES")
    print("="*70)
    
    all_missing = []
    total_images = 0
    total_found = 0
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        annotations_file = split_dir / '_annotations.coco.json'
        
        if not annotations_file.exists():
            print(f"\n⚠️ {split.upper()}: No annotations file")
            continue
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        missing_in_split = []
        found_in_split = 0
        
        for img in coco_data['images']:
            img_path = split_dir / img['file_name']
            total_images += 1
            
            if not img_path.exists():
                missing_in_split.append({
                    'filename': img['file_name'],
                    'path': str(img_path),
                    'split': split
                })
            else:
                found_in_split += 1
                total_found += 1
        
        # Print split summary
        total_in_split = len(coco_data['images'])
        missing_count = len(missing_in_split)
        
        if missing_count > 0:
            print(f"\n{split.upper()}: ❌ {missing_count}/{total_in_split} missing ({100*missing_count/total_in_split:.1f}%)")
            all_missing.extend(missing_in_split)
        else:
            print(f"\n{split.upper()}: ✅ {total_in_split}/{total_in_split} found (100%)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images in annotations: {total_images}")
    print(f"Found:   {total_found} ({100*total_found/total_images:.1f}%)")
    print(f"Missing: {len(all_missing)} ({100*len(all_missing)/total_images:.1f}%)")
    
    # Print all missing
    if all_missing:
        print("\n" + "="*70)
        print(f"ALL MISSING IMAGES ({len(all_missing)} total)")
        print("="*70)
        
        # Group by split
        by_split = {}
        for item in all_missing:
            split = item['split']
            if split not in by_split:
                by_split[split] = []
            by_split[split].append(item['filename'])
        
        for split, filenames in sorted(by_split.items()):
            print(f"\n{split.upper()} ({len(filenames)} missing):")
            print("-" * 70)
            for filename in sorted(filenames):
                print(f"  ⚠️ {filename}")
        
        # Save to file
        output_file = data_path / 'missing_images.txt'
        with open(output_file, 'w') as f:
            f.write(f"Missing Images Report\n")
            f.write(f"{'='*70}\n")
            f.write(f"Total missing: {len(all_missing)}\n\n")
            
            for split, filenames in sorted(by_split.items()):
                f.write(f"\n{split.upper()} ({len(filenames)} missing):\n")
                f.write("-" * 70 + "\n")
                for filename in sorted(filenames):
                    f.write(f"{filename}\n")
        
        print(f"\n{'='*70}")
        print(f"📄 Full list saved to: {output_file}")
        print(f"{'='*70}")
    else:
        print("\n✅ No missing images!")
    
    return all_missing

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for missing images")
    parser.add_argument('--data-dir', type=str, default='datasets/part2',
                       help='Dataset directory')
    
    args = parser.parse_args()
    check_missing_images(args.data_dir)
