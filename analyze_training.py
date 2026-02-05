"""
TensorBoard Log Analyzer - Identify problematic batches and training patterns
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import json

def analyze_tensorboard_logs(log_dir: str):
    """Analyze TensorBoard logs for training insights."""
    
    # Find event files
    log_path = Path(log_dir)
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    
    if not event_files:
        print("No TensorBoard event files found!")
        return
    
    print(f"Found {len(event_files)} event file(s)")
    
    for event_file in event_files:
        print(f"\n{'='*60}")
        print(f"Analyzing: {event_file.name}")
        print('='*60)
        
        # Load events
        ea = event_accumulator.EventAccumulator(str(event_file.parent))
        ea.Reload()
        
        # Get available tags
        print("\n📊 Available Metrics:")
        for tag_type, tags in ea.Tags().items():
            if tags:
                print(f"  {tag_type}: {tags}")
        
        # Analyze scalars
        if 'scalars' in ea.Tags() and ea.Tags()['scalars']:
            scalars = ea.Tags()['scalars']
            
            # Training batch loss analysis
            if 'Loss/train_batch' in scalars:
                batch_losses = ea.Scalars('Loss/train_batch')
                print(f"\n📈 Batch Loss Analysis:")
                print(f"  Total batches logged: {len(batch_losses)}")
                
                # Find outlier batches (high loss)
                losses = [(e.step, e.value) for e in batch_losses]
                avg_loss = sum(l[1] for l in losses) / len(losses)
                std_loss = (sum((l[1] - avg_loss)**2 for l in losses) / len(losses))**0.5
                
                threshold = avg_loss + 2 * std_loss
                outliers = [(step, loss) for step, loss in losses if loss > threshold]
                
                print(f"  Average batch loss: {avg_loss:.4f}")
                print(f"  Std deviation: {std_loss:.4f}")
                print(f"  Outlier threshold (avg + 2std): {threshold:.4f}")
                print(f"  Outlier batches (high loss): {len(outliers)}")
                
                if outliers:
                    print(f"\n⚠️  PROBLEMATIC BATCHES (loss > {threshold:.3f}):")
                    for step, loss in sorted(outliers, key=lambda x: -x[1])[:10]:
                        epoch = step // 46  # Approximate epoch (46 batches/epoch based on your logs)
                        batch = step % 46
                        print(f"    Step {step} (Epoch ~{epoch}, Batch ~{batch}): Loss = {loss:.4f}")
            
            # Validation metrics
            if 'Metrics/mean_iou' in scalars:
                iou_data = ea.Scalars('Metrics/mean_iou')
                print(f"\n📊 IoU Progression:")
                for e in iou_data:
                    marker = "  ⭐ BEST" if e.value == max(x.value for x in iou_data) else ""
                    print(f"  Epoch {e.step}: IoU = {e.value:.4f}{marker}")
                
                best_iou = max(e.value for e in iou_data)
                best_epoch = [e.step for e in iou_data if e.value == best_iou][0]
                print(f"\n🏆 Best IoU: {best_iou:.4f} at Epoch {best_epoch}")
            
            # Loss curves
            if 'Loss/train_epoch' in scalars and 'Loss/val' in scalars:
                train_losses = ea.Scalars('Loss/train_epoch')
                val_losses = ea.Scalars('Loss/val')
                
                print(f"\n📉 Loss Curve Summary:")
                print(f"  Training: {train_losses[0].value:.4f} → {train_losses[-1].value:.4f}")
                print(f"  Validation: {val_losses[0].value:.4f} → {val_losses[-1].value:.4f}")
                
                # Check for overfitting
                gap = val_losses[-1].value - train_losses[-1].value
                print(f"  Final gap (val - train): {gap:.4f}")
                if gap > 0.1:
                    print("  ⚠️ Potential overfitting detected!")
                else:
                    print("  ✅ No significant overfitting")
            
            # Learning rate analysis
            lr_tags = [t for t in scalars if 'LearningRate' in t]
            if lr_tags:
                print(f"\n📈 Learning Rate Schedule:")
                for tag in lr_tags:
                    lr_data = ea.Scalars(tag)
                    print(f"  {tag}:")
                    for e in lr_data[-5:]:  # Last 5 entries
                        print(f"    Epoch {e.step}: {e.value:.8f}")

def identify_bad_images(log_dir: str, data_dir: str = "datasets/part2"):
    """Try to identify which images might be in problematic batches."""
    print("\n" + "="*60)
    print("🔍 Attempting to identify problematic images...")
    print("="*60)
    
    # This would require tracking during training - for now, provide guidance
    print("""
To identify specific problematic images, you would need to:

1. Add image path logging during training:
   - Log image paths when loss > threshold
   - Save to a file for review

2. Visual inspection of outliers:
   - Check for poor annotations (wrong boxes)
   - Check for unclear/blurry images
   - Check for multiple objects (not single-object)
   - Check for occluded/partial objects

3. Common issues in object detection datasets:
   - Label noise (wrong bounding boxes)
   - Ambiguous objects
   - Extreme aspect ratios
   - Very small or very large objects
""")

if __name__ == "__main__":
    log_dir = "logs/part2"
    analyze_tensorboard_logs(log_dir)
    identify_bad_images(log_dir)
