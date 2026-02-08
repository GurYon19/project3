"""
Training Engine for Object Detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm


class Trainer:
    """Training engine for Part 2 and Part 3."""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                 scheduler=None, device: str = 'cpu', phase: int = 2,
                 log_dir: str = 'logs', checkpoint_dir: str = 'checkpoints', class_names: list = None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.phase = phase
        self.class_names = class_names or []
        
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_metric = 0
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        
        # Track high-loss samples for analysis
        self.high_loss_threshold = 0.6  # Flag losses above this
        self.high_loss_samples = []
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            if self.phase == 2:
                targets = targets.to(self.device)
            else:
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log high-loss batches for analysis
            if loss.item() > self.high_loss_threshold:
                self.high_loss_samples.append({
                    'epoch': self.current_epoch,
                    'batch': batch_idx,
                    'loss': loss.item()
                })
            
            if batch_idx % 10 == 0:
                global_step = self.current_epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/train_epoch', avg_loss, self.current_epoch)
        
        # Log learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            name = param_group.get('name', f'group_{i}')
            self.writer.add_scalar(f'LearningRate/{name}', param_group['lr'], self.current_epoch)
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        for images, targets in dataloader:
            images = images.to(self.device)
            
            if self.phase == 2:
                targets = targets.to(self.device)
            else:
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
            
            total_loss += loss_dict['loss'].item()
            total_iou += loss_dict.get('mean_iou', torch.tensor(0)).item()
            
            # Track loss components
            if 'coord_loss' in loss_dict:
                if not hasattr(self, '_val_coord_loss'):
                    self._val_coord_loss = 0
                    self._val_iou_loss = 0
                self._val_coord_loss += loss_dict['coord_loss'].item()
                self._val_iou_loss += loss_dict['iou_loss'].item()
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        mean_iou = total_iou / num_batches
        
        # Log main metrics
        self.writer.add_scalar('Loss/val', avg_loss, self.current_epoch)
        self.writer.add_scalar('Metrics/mean_iou', mean_iou, self.current_epoch)
        
        # Log loss components if available
        if hasattr(self, '_val_coord_loss'):
            self.writer.add_scalar('Loss/val_coord', self._val_coord_loss / num_batches, self.current_epoch)
            self.writer.add_scalar('Loss/val_iou', self._val_iou_loss / num_batches, self.current_epoch)
            self._val_coord_loss = 0
            self._val_iou_loss = 0
        
        # Log detection rate at IoU thresholds
        self.writer.add_scalar('Metrics/detection_rate_50', 1.0 if mean_iou > 0.5 else 0.0, self.current_epoch)
        
        return {'loss': avg_loss, 'mean_iou': mean_iou}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
              early_stopping_patience: int = 15, unfreeze_epoch: int = None):
        print(f"\n{'='*60}")
        print(f"Starting training for Phase {self.phase}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            if unfreeze_epoch is not None and epoch == unfreeze_epoch:
                print(f"\n>>> Unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone(num_layers=3)
                
                # Robust Logic: "Wake up" the existing backbone group
                target_backbone_lr = self.config.get('learning_rate', 0.001) * 0.1
                
                backbone_group_found = False
                for param_group in self.optimizer.param_groups:
                    if param_group.get('name') == 'backbone':
                        param_group['lr'] = target_backbone_lr
                        backbone_group_found = True
                        print(f"    Backbone group un-muted! New LR: {target_backbone_lr:.6f}")
                
                if not backbone_group_found:
                    print("⚠️ Warning: Could not find 'backbone' param group to unfreeze!")
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}, IoU={val_metrics['mean_iou']:.4f}")
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            if val_metrics['mean_iou'] > self.best_metric:
                self.best_metric = val_metrics['mean_iou']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  >>> New best model! IoU: {val_metrics['mean_iou']:.4f}")
            else:
                self.epochs_without_improvement += 1
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', val_metrics)
            
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}!")
                break
        
        self.save_checkpoint('final_model.pth', val_metrics)
        self.writer.close()
        
        # Report high-loss samples
        if self.high_loss_samples:
            print(f"\n⚠️ High-loss batches detected: {len(self.high_loss_samples)}")
            # Save to file for analysis
            import json
            with open(self.log_dir / 'high_loss_batches.json', 'w') as f:
                json.dump(self.high_loss_samples, f, indent=2)
            print(f"   Saved to: {self.log_dir / 'high_loss_batches.json'}")
        
        print(f"\nTraining complete! Best IoU: {self.best_metric:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Dict = None):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Try to save with error handling (file lock issues on Windows)
        try:
            torch.save(checkpoint, self.checkpoint_dir / filename)
        except RuntimeError as e:
            print(f"  ⚠️ Warning: Could not save {filename}: {e}")
            # Try alternative filename
            alt_filename = f"backup_{filename}"
            try:
                torch.save(checkpoint, self.checkpoint_dir / alt_filename)
                print(f"  ✅ Saved as {alt_filename} instead")
            except RuntimeError:
                print(f"  ❌ Could not save checkpoint, continuing training...")
    
    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
            


def create_trainer(model: nn.Module, phase: int, config: Dict, class_names: list = None) -> Trainer:
    from utils.loss import get_loss_function
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if phase == 2:
        criterion = get_loss_function(phase, coord_weight=config.get('coord_weight', 0.0))
    else:
        criterion = get_loss_function(phase, lambda_box=config.get('lambda_box', 5.0),
                                       lambda_cls=config.get('lambda_cls', 1.0),
                                       lambda_obj=config.get('lambda_obj', 1.0))
    
    # Check if backbone should remain frozen
    freeze_backbone = config.get('freeze_backbone', False)
    unfreeze_epoch = config.get('unfreeze_epoch', None)
    
    head_lr = config.get('learning_rate', 1e-3)
    
    # Robust Initialization: Always include backbone params
    if freeze_backbone and unfreeze_epoch is None:
        # PERMANENTLY FROZEN
        # Backbone is frozen and won't be unfrozen. We can omit it efficiently.
        param_groups = [
            {'params': model.head.parameters(), 'lr': head_lr, 'name': 'head'}
        ]
        print(f"  Optimizer: Head Only (Backbone permanently frozen)")
        
    elif freeze_backbone and unfreeze_epoch is not None:
        # TEMPORARILY FROZEN (Silent Init)
        # Initialize backbone with LR=0.0 so correct params are tracked but not updated
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': 0.0, 'name': 'backbone'},
            {'params': model.head.parameters(), 'lr': head_lr, 'name': 'head'}
        ]
        print(f"  Optimizer: Hybrid Init (Backbone LR=0.0 -> waiting for epoch {unfreeze_epoch})")
        
    else:
        # FULLY UNFROZEN (Standard)
        backbone_lr = head_lr * 0.02
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': backbone_lr, 'name': 'backbone'},
            {'params': model.head.parameters(), 'lr': head_lr, 'name': 'head'}
        ]
        print(f"  Optimizer: Full Training (Backbone LR={backbone_lr:.6f})")
    
    optimizer = optim.AdamW(param_groups, weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.8)
    
    # Store config in trainer for reference
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                   device=device, phase=phase, log_dir=config.get('log_dir', 'logs'),
                   checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'), class_names=class_names)
    trainer.config = config # Inject config
    return trainer
