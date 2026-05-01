"""
=============================================================================
 train.py — Training Loop for DeepLabV3+ Terrain Risk Model
 Stage 2: Deep Learning Risk Heatmap Training
=============================================================================
 Reference: Pre-Defense Report §5
 
 Loss:  L = L_BCE + L_Dice
 Optimizer: AdamW with CosineAnnealingLR
 Features: Early stopping, checkpoint saving, CSV logging
=============================================================================
"""

import os
import csv
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    EARLY_STOPPING_PATIENCE, DICE_SMOOTH,
    CHECKPOINTS_DIR, LOGS_DIR, BATCH_SIZE
)
from model import build_model, CombinedLoss
from dataset import create_dataloaders


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute evaluation metrics for risk estimation.
    
    Parameters
    ----------
    pred : torch.Tensor, shape (B, 1, H, W)
    target : torch.Tensor, shape (B, 1, H, W)
    
    Returns
    -------
    metrics : dict
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - pearson: Pearson correlation coefficient
    """
    with torch.no_grad():
        pred_flat = pred.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        mae = np.mean(np.abs(pred_flat - target_flat))
        rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
        
        # Pearson correlation
        if pred_flat.std() > 1e-8 and target_flat.std() > 1e-8:
            pearson = np.corrcoef(pred_flat, target_flat)[0, 1]
        else:
            pearson = 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'pearson': float(pearson)
        }


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training when validation loss doesn't improve for `patience` epochs.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch. Returns average losses and metrics."""
    model.train()
    
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with auxiliary output
        outputs = model.get_aux_output(images)
        pred = outputs['out']
        aux = outputs['aux']
        
        # Compute loss
        losses = criterion(pred, labels, aux)
        total_loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        running_loss += losses['total'].item()
        running_bce += losses['bce'].item()
        running_dice += losses['dice'].item()
        
        metrics = compute_metrics(pred.detach(), labels)
        running_mae += metrics['mae']
        running_rmse += metrics['rmse']
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f"{running_loss/n_batches:.4f}",
            'mae': f"{running_mae/n_batches:.4f}"
        })
    
    return {
        'loss': running_loss / max(n_batches, 1),
        'bce': running_bce / max(n_batches, 1),
        'dice': running_dice / max(n_batches, 1),
        'mae': running_mae / max(n_batches, 1),
        'rmse': running_rmse / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate the model. Returns average losses and metrics."""
    model.eval()
    
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    running_pearson = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"  Val   Epoch {epoch+1}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        pred = model(images)
        
        # Compute loss (no aux for validation)
        losses = criterion(pred, labels)
        
        # Accumulate
        running_loss += losses['total'].item()
        running_bce += losses['bce'].item()
        running_dice += losses['dice'].item()
        
        metrics = compute_metrics(pred, labels)
        running_mae += metrics['mae']
        running_rmse += metrics['rmse']
        running_pearson += metrics['pearson']
        n_batches += 1
    
    return {
        'loss': running_loss / max(n_batches, 1),
        'bce': running_bce / max(n_batches, 1),
        'dice': running_dice / max(n_batches, 1),
        'mae': running_mae / max(n_batches, 1),
        'rmse': running_rmse / max(n_batches, 1),
        'pearson': running_pearson / max(n_batches, 1),
    }


def train(
    max_tiles: int = None,
    epochs: int = None,
    batch_size: int = None,
    resume_from: str = None
):
    """
    Full training pipeline.
    
    Parameters
    ----------
    max_tiles : int or None
        Limit tiles for quick demo.
    epochs : int or None
        Override epoch count.
    batch_size : int or None
        Override batch size.
    resume_from : str or None
        Path to checkpoint to resume from.
    
    Returns
    -------
    history : dict
        Training history with losses and metrics per epoch.
    """
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    
    print("=" * 70)
    print(" TRAINING: DeepLabV3+ MobileNetV3-Large for Terrain Risk Estimation")
    print("=" * 70)
    
    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        max_tiles=max_tiles
    )
    
    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(pretrained=True)
    
    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────
    criterion = CombinedLoss(dice_smooth=DICE_SMOOTH)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[TRAIN] Resumed from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}")
    
    # ── CSV Logger ──────────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"training_log_{timestamp}.csv"
    log_fields = [
        'epoch', 'lr',
        'train_loss', 'train_bce', 'train_dice', 'train_mae', 'train_rmse',
        'val_loss', 'val_bce', 'val_dice', 'val_mae', 'val_rmse', 'val_pearson',
        'time_sec'
    ]
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
    
    # ── Training Loop ───────────────────────────────────────────────────
    history = {k: [] for k in log_fields}
    
    print(f"\n{'Epoch':>5} {'LR':>10} {'Train Loss':>11} {'Val Loss':>11} "
          f"{'Val MAE':>9} {'Val RMSE':>9} {'Pearson':>9} {'Time':>7}")
    print("─" * 80)
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, DEVICE, epoch
        )
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        row = {
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_metrics['loss'],
            'train_bce': train_metrics['bce'],
            'train_dice': train_metrics['dice'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'val_loss': val_metrics['loss'],
            'val_bce': val_metrics['bce'],
            'val_dice': val_metrics['dice'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_pearson': val_metrics['pearson'],
            'time_sec': epoch_time
        }
        
        for k, v in row.items():
            history[k].append(v)
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(row)
        
        # Print epoch summary
        print(f"{epoch+1:>5} {current_lr:>10.2e} "
              f"{train_metrics['loss']:>11.4f} {val_metrics['loss']:>11.4f} "
              f"{val_metrics['mae']:>9.4f} {val_metrics['rmse']:>9.4f} "
              f"{val_metrics['pearson']:>9.4f} {epoch_time:>6.1f}s")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = CHECKPOINTS_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"       ✓ Best model saved (val_loss={best_val_loss:.4f})")
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, CHECKPOINTS_DIR / "latest_checkpoint.pth")
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            print(f"\n[EARLY STOPPING] No improvement for "
                  f"{EARLY_STOPPING_PATIENCE} epochs. Stopping.")
            break
    
    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f" Best validation loss: {best_val_loss:.4f}")
    print(f" Model checkpoint:     {CHECKPOINTS_DIR / 'best_model.pth'}")
    print(f" Training log:         {log_path}")
    print(f"{'='*70}")
    
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train terrain risk model")
    parser.add_argument('--max-tiles', type=int, default=None,
                        help="Limit tiles for quick demo")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    history = train(
        max_tiles=args.max_tiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from=args.resume
    )
