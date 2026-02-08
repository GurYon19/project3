"""
Visualize worst predictions from the trained model.
Identifies images where the model struggles most.
"""
import torch
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEVICE, IMAGE_SIZE
from models.detector import create_detector
from data.transforms import get_transforms
from utils.loss import calculate_iou


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint."""
    model = create_detector(phase=2, config={'pretrained': False, 'freeze_backbone': False})
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def cxwh_to_xyxy(box, img_w, img_h):
    """Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]


def visualize_prediction(image_path, pred_box, gt_box, iou, output_path):
    """Draw prediction and ground truth on image."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    img_w, img_h = img.size
    
    # Convert boxes to pixel coordinates
    pred_xyxy = cxwh_to_xyxy(pred_box, img_w, img_h)
    gt_xyxy = cxwh_to_xyxy(gt_box, img_w, img_h)
    
    # Draw ground truth (green)
    draw.rectangle(gt_xyxy, outline='green', width=3)
    draw.text((gt_xyxy[0], gt_xyxy[1] - 15), 'GT', fill='green')
    
    # Draw prediction (red)
    draw.rectangle(pred_xyxy, outline='red', width=3)
    draw.text((pred_xyxy[0], pred_xyxy[1] - 15), f'Pred IoU={iou:.2f}', fill='red')
    
    # Add filename
    draw.text((10, 10), Path(image_path).name, fill='white')
    
    img.save(output_path)
    return img


def analyze_worst_predictions(data_dir: str, checkpoint_path: str, output_dir: str, top_k: int = 20, split: str = 'valid'):
    """Find and visualize worst predictions."""
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations_file = data_path / split / '_annotations.coco.json'
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image id to info mapping
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    img_id_to_ann = {}
    for ann in coco_data['annotations']:
        img_id_to_ann[ann['image_id']] = ann
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path)
    transform = get_transforms(is_training=False, image_size=IMAGE_SIZE)
    
    # Evaluate each image
    results = []
    print(f"\nEvaluating {len(img_id_to_info)} {split} images...")
    
    for img_id, img_info in img_id_to_info.items():
        if img_id not in img_id_to_ann:
            continue
            
        img_path = data_path / split / img_info['file_name']
        if not img_path.exists():
            continue
        
        # Get ground truth
        ann = img_id_to_ann[img_id]
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        img_w, img_h = img_info['width'], img_info['height']
        
        # Convert to normalized CXWH
        gt_box = [
            (bbox[0] + bbox[2]/2) / img_w,  # cx
            (bbox[1] + bbox[3]/2) / img_h,  # cy
            bbox[2] / img_w,                 # w
            bbox[3] / img_h                  # h
        ]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image_tensor, _ = transform(image, None)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            pred_box = model(image_tensor).squeeze(0).cpu().numpy()
        
        # Calculate IoU
        pred_tensor = torch.tensor(pred_box).unsqueeze(0)
        gt_tensor = torch.tensor(gt_box).unsqueeze(0)
        iou = calculate_iou(pred_tensor, gt_tensor).item()
        
        results.append({
            'image_path': str(img_path),
            'pred_box': pred_box.tolist(),
            'gt_box': gt_box,
            'iou': iou,
            'filename': img_info['file_name']
        })
    
    # Sort by IoU (worst first)
    results.sort(key=lambda x: x['iou'])
    
    # Save analysis
    print(f"\n{'='*60}")
    print(f"WORST {top_k} PREDICTIONS (Lowest IoU)")
    print('='*60)
    
    worst_images = []
    for i, result in enumerate(results[:top_k]):
        print(f"\n{i+1}. {result['filename']}")
        print(f"   IoU: {result['iou']:.4f}")
        print(f"   GT:   [{', '.join(f'{x:.3f}' for x in result['gt_box'])}]")
        print(f"   Pred: [{', '.join(f'{x:.3f}' for x in result['pred_box'])}]")
        
        # Visualize
        out_file = output_path / f"worst_{i+1:02d}_iou{result['iou']:.2f}_{Path(result['filename']).stem}.jpg"
        visualize_prediction(result['image_path'], result['pred_box'], result['gt_box'], result['iou'], out_file)
        worst_images.append(result['filename'])
    
    print(f"\n{'='*60}")
    print(f"BEST {top_k} PREDICTIONS (Highest IoU)")
    print('='*60)
    
    for i, result in enumerate(results[-top_k:]):
        print(f"\n{len(results) - top_k + i + 1}. {result['filename']}")
        print(f"   IoU: {result['iou']:.4f}")
    
    # Summary statistics
    ious = [r['iou'] for r in results]
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print('='*60)
    print(f"Total images: {len(results)}")
    print(f"Mean IoU: {sum(ious)/len(ious):.4f}")
    print(f"Median IoU: {sorted(ious)[len(ious)//2]:.4f}")
    print(f"Min IoU: {min(ious):.4f}")
    print(f"Max IoU: {max(ious):.4f}")
    print(f"IoU < 0.5: {sum(1 for iou in ious if iou < 0.5)} images ({100*sum(1 for iou in ious if iou < 0.5)/len(ious):.1f}%)")
    print(f"IoU > 0.7: {sum(1 for iou in ious if iou > 0.7)} images ({100*sum(1 for iou in ious if iou > 0.7)/len(ious):.1f}%)")
    
    # Save results to JSON
    with open(output_path / 'prediction_analysis.json', 'w') as f:
        json.dump({
            'worst_images': worst_images,
            'statistics': {
                'total': len(results),
                'mean_iou': sum(ious)/len(ious),
                'min_iou': min(ious),
                'max_iou': max(ious)
            },
            'all_results': results
        }, f, indent=2)
    
    print(f"\n✅ Visualizations saved to: {output_path}")
    print(f"✅ Analysis saved to: {output_path / 'prediction_analysis.json'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze worst predictions")
    parser.add_argument('--data-dir', type=str, default='datasets/part2', help='Dataset directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/part2/best_model.pth', help='Model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/worst_predictions', help='Output directory')
    parser.add_argument('--top-k', type=int, default=20, help='Number of worst predictions to show')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'], help='Dataset split to analyze')
    
    args = parser.parse_args()
    analyze_worst_predictions(args.data_dir, args.checkpoint, args.output, args.top_k, args.split)
