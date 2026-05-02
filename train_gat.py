"""
=============================================================================
 train_gat.py — Training Loop for GATv2 Terrain Risk Model
 Stage 2b: Graph Attention Network Training
=============================================================================
 Reference: Pre-Defense Report §5b

 Loss:  L = MSE + 0.5·SmoothL1
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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    DEVICE, GAT_LEARNING_RATE, WEIGHT_DECAY,
    GAT_EPOCHS, EARLY_STOPPING_PATIENCE,
    GAT_CHECKPOINTS_DIR, GAT_LOGS_DIR, GAT_BATCH_SIZE
)
from gat_model import build_gat_model, GraphRiskLoss
from graph_dataset import create_graph_dataloaders


def compute_graph_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> dict:
    """
    Compute evaluation metrics for node-level risk estimation.

    Parameters
    ----------
    pred : torch.Tensor, shape [N, 1] or [N]
    target : torch.Tensor, shape [N]

    Returns
    -------
    metrics : dict with 'mae', 'rmse', 'pearson' keys.
    """
    with torch.no_grad():
        pred_flat = pred.squeeze(-1).cpu().numpy()
        target_flat = target.cpu().numpy()

        mae = np.mean(np.abs(pred_flat - target_flat))
        rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

        if pred_flat.std() > 1e-8 and target_flat.std() > 1e-8:
            pearson = np.corrcoef(pred_flat, target_flat)[0, 1]
        else:
            pearson = 0.0

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'pearson': float(pearson),
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""

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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train GATv2 for one epoch. Returns average losses and metrics."""
    model.train()

    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  GAT Train Epoch {epoch+1}", leave=False)
    for batch in pbar:
        batch = batch.to(device)

        # Forward pass
        pred = model(batch.x, batch.edge_index)

        # Compute loss
        losses = criterion(pred, batch.y)
        total_loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate
        running_loss += losses['total'].item()
        running_mse += losses['mse'].item()

        metrics = compute_graph_metrics(pred.detach(), batch.y)
        running_mae += metrics['mae']
        running_rmse += metrics['rmse']
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{running_loss/n_batches:.4f}",
            'mae': f"{running_mae/n_batches:.4f}"
        })

    return {
        'loss': running_loss / max(n_batches, 1),
        'mse': running_mse / max(n_batches, 1),
        'mae': running_mae / max(n_batches, 1),
        'rmse': running_rmse / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Validate the GATv2 model. Returns average losses and metrics."""
    model.eval()

    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    running_pearson = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  GAT Val   Epoch {epoch+1}", leave=False)
    for batch in pbar:
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index)
        losses = criterion(pred, batch.y)

        running_loss += losses['total'].item()
        running_mse += losses['mse'].item()

        metrics = compute_graph_metrics(pred, batch.y)
        running_mae += metrics['mae']
        running_rmse += metrics['rmse']
        running_pearson += metrics['pearson']
        n_batches += 1

    return {
        'loss': running_loss / max(n_batches, 1),
        'mse': running_mse / max(n_batches, 1),
        'mae': running_mae / max(n_batches, 1),
        'rmse': running_rmse / max(n_batches, 1),
        'pearson': running_pearson / max(n_batches, 1),
    }


def train_gat(
    max_tiles: int = None,
    epochs: int = None,
    batch_size: int = None,
    resume_from: str = None
):
    """
    Full GATv2 training pipeline.

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
    epochs = epochs or GAT_EPOCHS
    batch_size = batch_size or GAT_BATCH_SIZE

    print("=" * 70)
    print(" TRAINING: GATv2 Graph Attention Network for Terrain Risk")
    print("=" * 70)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = create_graph_dataloaders(
        batch_size=batch_size,
        max_tiles=max_tiles,
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = build_gat_model()

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────
    criterion = GraphRiskLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=GAT_LEARNING_RATE,
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
        print(f"[GAT-TRAIN] Resumed from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}")

    # ── CSV Logger ──────────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = GAT_LOGS_DIR / f"gat_training_log_{timestamp}.csv"
    log_fields = [
        'epoch', 'lr',
        'train_loss', 'train_mse', 'train_mae', 'train_rmse',
        'val_loss', 'val_mse', 'val_mae', 'val_rmse', 'val_pearson',
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
            'train_mse': train_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_pearson': val_metrics['pearson'],
            'time_sec': epoch_time,
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
            checkpoint_path = GAT_CHECKPOINTS_DIR / "best_gat_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"       ✓ Best GAT model saved (val_loss={best_val_loss:.4f})")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, GAT_CHECKPOINTS_DIR / "latest_gat_checkpoint.pth")

        # Early stopping check
        if early_stopping(val_metrics['loss']):
            print(f"\n[EARLY STOPPING] No improvement for "
                  f"{EARLY_STOPPING_PATIENCE} epochs. Stopping.")
            break

    print(f"\n{'='*70}")
    print(f" GATv2 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f" Best validation loss: {best_val_loss:.4f}")
    print(f" Model checkpoint:     {GAT_CHECKPOINTS_DIR / 'best_gat_model.pth'}")
    print(f" Training log:         {log_path}")
    print(f"{'='*70}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GATv2 terrain risk model")
    parser.add_argument('--max-tiles', type=int, default=None,
                        help="Limit tiles for quick demo")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    history = train_gat(
        max_tiles=args.max_tiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from=args.resume,
    )
