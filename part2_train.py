"""
Part 2: Single Object Detection Training Script
"""
import torch
import argparse
from pathlib import Path
import sys
import random

sys.path.insert(0, str(Path(__file__).parent))

from config import PART2_CONFIG, DEVICE, IMAGE_SIZE
from models.detector import create_detector
from data.dataset import SingleObjectDataset, get_dataloader
from data.transforms import get_transforms
from trainer import create_trainer


def train_part2(args):
    """Train single-object detection model."""
    print("="*60)
    print("Part 2: Single Object Detection Training")
    print("="*60)
    
    # Setup
    config = PART2_CONFIG.copy()
    config['device'] = str(DEVICE)
    config['log_dir'] = str(Path('logs') / 'part2')
    config['checkpoint_dir'] = str(Path('checkpoints') / 'part2')
    
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
        
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Data
    print(f"\nLoading dataset from: {args.data_dir}")
    train_transform = get_transforms(is_training=True, image_size=IMAGE_SIZE)
    val_transform = get_transforms(is_training=False, image_size=IMAGE_SIZE)
    
    train_dataset = SingleObjectDataset(
        images_dir=Path(args.data_dir) / 'train',
        annotations_file=Path(args.data_dir) / 'train' / '_annotations.coco.json',
        transform=train_transform,
        annotation_format=args.format
    )
    
    val_dataset = SingleObjectDataset(
        images_dir=Path(args.data_dir) / 'valid',
        annotations_file=Path(args.data_dir) / 'valid' / '_annotations.coco.json',
        transform=val_transform,
        annotation_format=args.format
    )
    
    print(f"  Train samples (full): {len(train_dataset)}")
    print(f"  Valid samples (full): {len(val_dataset)}")
    
    # Apply subset if specified
    if args.subset and args.subset < 1.0:
        from torch.utils.data import Subset
        train_size = int(len(train_dataset) * args.subset)
        
        train_indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = Subset(train_dataset, train_indices)
        
        print(f"  Using {args.subset*100:.0f}% of TRAINING data: {len(train_dataset)} samples")
        print(f"  Validation set kept full: {len(val_dataset)} samples")
    
    train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    print(f"\nCreating model...")
    model = create_detector(phase=2, config={
        'pretrained': True,
        'freeze_backbone': config['freeze_backbone']
    })
    
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['total_trainable']:,}")
    
    # Trainer
    trainer = create_trainer(model, phase=2, config=config)
    
    if args.resume:
        print(f"\n>>> RESUMING training from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(args.resume)
            print(f"    Resumed at epoch {trainer.current_epoch}")
        except FileNotFoundError:
            print(f"❌ Checkpoint '{args.resume}' not found in {config['checkpoint_dir']}")
            print("Beginning fresh training...")
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=15,
        unfreeze_epoch=config.get('unfreeze_epoch')
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best model saved to: {config['checkpoint_dir']}/best_model.pth")
    print("="*60)


def demo_with_synthetic_data():
    """Demo training with synthetic data if no dataset is provided."""
    print("\n" + "="*60)
    print("Running DEMO with synthetic data")
    print("="*60)
    
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create synthetic dataset
    num_samples = 100
    images = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    boxes = torch.rand(num_samples, 4) * 0.5 + 0.25  # Boxes centered in image
    
    train_dataset = TensorDataset(images[:80], boxes[:80])
    val_dataset = TensorDataset(images[80:], boxes[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = create_detector(phase=2, config={'pretrained': True, 'freeze_backbone': True})
    
    print(f"\nModel parameters: {model.count_parameters()}")
    
    # Create trainer
    config = {
        'device': str(DEVICE),
        'learning_rate': 1e-3,
        'log_dir': 'logs/part2_demo',
        'checkpoint_dir': 'checkpoints/part2_demo'
    }
    
    trainer = create_trainer(model, phase=2, config=config)
    
    # Train for just a few epochs to verify everything works
    trainer.train(train_loader, val_loader, epochs=5, early_stopping_patience=10)
    
    print("\nDemo complete! The training pipeline is working correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2: Single Object Detection")
    parser.add_argument('--data-dir', type=str, default=None, help="Path to dataset directory")
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate")
    parser.add_argument('--format', type=str, default='coco', choices=['coco', 'yolo'], 
                       help="Annotation format")
    parser.add_argument('--subset', type=float, default=None, 
                       help="Fraction of dataset to use (e.g., 0.4 for 40%)")
    parser.add_argument('--demo', action='store_true', help="Run demo with synthetic data")
    
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    if args.demo or args.data_dir is None:
        demo_with_synthetic_data()
    else:
        train_part2(args)
