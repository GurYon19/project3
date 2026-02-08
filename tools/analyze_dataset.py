"""
Simple analysis: Check how many images have exactly 1 object.
Don't modify anything, just report.
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_dataset(data_dir: str):
    """Analyze dataset for single-object images."""
    data_path = Path(data_dir)
    
    print("="*70)
    print("DATASET ANALYSIS: Single vs Multi-Object Images")
    print("="*70)
    
    total_stats = {
        'single': 0,
        'multi': 0,
        'total': 0
    }
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_path / split
        annotations_file = split_dir / '_annotations.coco.json'
        
        if not annotations_file.exists():
            continue
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Count objects per image
        image_object_count = defaultdict(int)
        for ann in coco_data['annotations']:
            image_object_count[ann['image_id']] += 1
        
        # Categorize
        single_object = sum(1 for count in image_object_count.values() if count == 1)
        multi_object = sum(1 for count in image_object_count.values() if count > 1)
        total = len(coco_data['images'])
        
        print(f"\n{split.upper()}:")
        print(f"  Total images:        {total:4d}")
        print(f"  Single-object:       {single_object:4d} ({100*single_object/total if total else 0:5.1f}%)")
        print(f"  Multi-object:        {multi_object:4d} ({100*multi_object/total if total else 0:5.1f}%)")
        print(f"  No annotation:       {total - single_object - multi_object:4d}")
        
        total_stats['single'] += single_object
        total_stats['multi'] += multi_object
        total_stats['total'] += total
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"  Total images:        {total_stats['total']:4d}")
    print(f"  Single-object:       {total_stats['single']:4d} ({100*total_stats['single']/total_stats['total'] if total_stats['total'] else 0:5.1f}%)")
    print(f"  Multi-object:        {total_stats['multi']:4d} ({100*total_stats['multi']/total_stats['total'] if total_stats['total'] else 0:5.1f}%)")
    print("="*70)
    
    # Recommendations
    print("\n📊 RECOMMENDATIONS:")
    single = total_stats['single']
    
    if single >= 1000:
        print(f"  ✅ Excellent dataset size ({single} single-object images)")
        print(f"  Recommended split: 70/15/15 (train/val/test)")
        rec_train, rec_val, rec_test = int(single*0.70), int(single*0.15), int(single*0.15)
    elif single >= 500:
        print(f"  ✅ Good dataset size ({single} single-object images)")
        print(f"  Recommended split: 70/15/15 (train/val/test)")
        rec_train, rec_val, rec_test = int(single*0.70), int(single*0.15), int(single*0.15)
    elif single >= 300:
        print(f"  ⚠️  Small dataset ({single} single-object images)")
        print(f"  Recommended split: 80/10/10 (train/val/test)")
        rec_train, rec_val, rec_test = int(single*0.80), int(single*0.10), int(single*0.10)
    else:
        print(f"  ⚠️  Very small dataset ({single} single-object images)")
        print(f"  Recommended split: 80/20/0 (train/val, no test)")
        rec_train, rec_val, rec_test = int(single*0.80), int(single*0.20), 0
    
    print(f"\n  Suggested sizes:")
    print(f"    Train: {rec_train} images")
    print(f"    Val:   {rec_val} images")
    if rec_test > 0:
        print(f"    Test:  {rec_test} images")
    
    # Check current split
    print(f"\n💡 CURRENT STATUS:")
    print(f"  Your current dataset already has good single-object images.")
    print(f"  If multi-object images ({total_stats['multi']}) are causing issues,")
    print(f"  you can manually remove them or use a filtering script.")

if __name__ == "__main__":
    analyze_dataset("datasets/part2")
