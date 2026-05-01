"""
=============================================================================
 fusion.py — Heatmap Fusion: Physics + Deep Learning
 Stage 3: Weighted Combination of Risk Maps
=============================================================================
 Reference: Pre-Defense Report §6

 H_final(x,y) = α · H_learned + (1-α) · H_physics
 where α ∈ [0.6, 0.8], default α = 0.7
=============================================================================
"""

import numpy as np
import glob
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from config import (
    DEVICE, FUSION_ALPHA, FUSION_SMOOTH_SIGMA,
    TILE_DIRS, PSEUDO_LABELS_DIR, DL_PREDICTIONS_DIR,
    FUSED_MAPS_DIR, CHECKPOINTS_DIR,
    TILE_PATTERN, MAX_TILES
)


def fuse_heatmaps(
    h_learned: np.ndarray,
    h_physics: np.ndarray,
    alpha: float = None,
    smooth_sigma: float = None
) -> np.ndarray:
    """
    §6 — Weighted heatmap fusion.

        H_final(x,y) = α · H_learned + (1-α) · H_physics

    Parameters
    ----------
    h_learned : np.ndarray
        Deep learning risk map in [0, 1], shape (H, W).
    h_physics : np.ndarray
        Physics-based risk map in [0, 1], shape (H, W).
    alpha : float
        DL weight (0.6–0.8). Higher = trust DL more.
    smooth_sigma : float
        Gaussian smoothing sigma for the fused output.

    Returns
    -------
    h_final : np.ndarray
        Fused risk map in [0, 1], shape (H, W).
    """
    alpha = alpha if alpha is not None else FUSION_ALPHA
    smooth_sigma = smooth_sigma if smooth_sigma is not None else FUSION_SMOOTH_SIGMA

    # Weighted combination
    h_final = alpha * h_learned + (1.0 - alpha) * h_physics

    # Post-processing: Gaussian smoothing
    if smooth_sigma > 0:
        h_final = gaussian_filter(h_final, sigma=smooth_sigma)

    # Clamp to [0, 1]
    h_final = np.clip(h_final, 0.0, 1.0)

    return h_final.astype(np.float32)


def fuse_all_tiles(
    tile_dirs: list = None,
    label_dir: Path = None,
    predictions_dir: Path = None,
    output_dir: Path = None,
    alpha: float = None,
    max_tiles: int = None
) -> dict:
    """
    Fuse physics pseudo-labels with DL predictions for all available tiles.

    For tiles without DL predictions, falls back to physics risk only.

    Parameters
    ----------
    tile_dirs : list of Path
    label_dir : Path
        Directory containing physics pseudo-labels (.npy).
    predictions_dir : Path
        Directory containing DL prediction maps (.npy).
    output_dir : Path
        Directory to save fused maps.
    alpha : float
        Fusion weight for DL component.
    max_tiles : int or None

    Returns
    -------
    stats : dict
    """
    tile_dirs = tile_dirs or TILE_DIRS
    label_dir = label_dir or PSEUDO_LABELS_DIR
    predictions_dir = predictions_dir or DL_PREDICTIONS_DIR
    output_dir = output_dir or FUSED_MAPS_DIR
    alpha = alpha if alpha is not None else FUSION_ALPHA
    max_tiles = max_tiles or MAX_TILES

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect tiles that have physics labels
    all_tiles = []
    for td in tile_dirs:
        tiles = sorted(glob.glob(str(Path(td) / TILE_PATTERN)))
        all_tiles.extend(tiles)

    if max_tiles is not None:
        all_tiles = all_tiles[:max_tiles]

    print(f"[FUSION] Fusing {len(all_tiles)} tiles with α={alpha}")

    stats = {
        'total': len(all_tiles),
        'fused_with_dl': 0,
        'physics_only': 0,
        'failed': 0,
        'mean_risk': [],
    }

    for tile_path in tqdm(all_tiles, desc="Fusing heatmaps"):
        try:
            tile_name = Path(tile_path).stem
            out_path = output_dir / f"{tile_name}.npy"

            if out_path.exists():
                stats['fused_with_dl'] += 1
                continue

            # Load physics pseudo-label
            phys_path = label_dir / f"{tile_name}.npy"
            if not phys_path.exists():
                stats['failed'] += 1
                continue

            h_physics = np.load(str(phys_path)).astype(np.float32)

            # Load DL prediction if available
            dl_path = predictions_dir / f"{tile_name}.npy"
            if dl_path.exists():
                h_learned = np.load(str(dl_path)).astype(np.float32)
                h_final = fuse_heatmaps(h_learned, h_physics, alpha=alpha)
                stats['fused_with_dl'] += 1
            else:
                # Fallback: smooth physics-only map
                h_final = gaussian_filter(h_physics, sigma=FUSION_SMOOTH_SIGMA)
                h_final = np.clip(h_final, 0.0, 1.0).astype(np.float32)
                stats['physics_only'] += 1

            np.save(str(out_path), h_final)
            stats['mean_risk'].append(float(h_final.mean()))

        except Exception as e:
            print(f"\n[WARNING] Failed {tile_path}: {e}")
            stats['failed'] += 1

    print(f"\n{'='*60}")
    print(f" FUSION COMPLETE")
    print(f"{'='*60}")
    print(f" Fused with DL:   {stats['fused_with_dl']}")
    print(f" Physics only:    {stats['physics_only']}")
    print(f" Failed:          {stats['failed']}")
    if stats['mean_risk']:
        print(f" Mean final risk: {np.mean(stats['mean_risk']):.4f}")
    print(f"{'='*60}")

    return stats


@torch.no_grad()
def predict_tiles(
    model: nn.Module,
    tile_dirs: list = None,
    output_dir: Path = None,
    batch_size: int = 4,
    max_tiles: int = None
):
    """
    Run trained model inference on all tiles and save DL predictions.

    Parameters
    ----------
    model : nn.Module
        Trained TerrainRiskModel.
    tile_dirs : list of Path
    output_dir : Path
    batch_size : int
    max_tiles : int or None
    """
    import cv2
    from torch.utils.data import DataLoader, TensorDataset

    tile_dirs = tile_dirs or TILE_DIRS
    output_dir = output_dir or DL_PREDICTIONS_DIR
    max_tiles = max_tiles or MAX_TILES
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(DEVICE)

    # Collect all tile paths
    all_tiles = []
    for td in tile_dirs:
        tiles = sorted(glob.glob(str(Path(td) / TILE_PATTERN)))
        all_tiles.extend(tiles)

    if max_tiles is not None:
        all_tiles = all_tiles[:max_tiles]

    print(f"[PREDICT] Running inference on {len(all_tiles)} tiles...")

    for tile_path in tqdm(all_tiles, desc="DL Inference"):
        try:
            tile_name = Path(tile_path).stem
            out_path = output_dir / f"{tile_name}.npy"

            if out_path.exists():
                continue

            # Load image
            img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_t = torch.from_numpy(
                img.astype(np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)

            # Predict
            pred = model(img_t)  # (1, 1, H, W)
            pred_np = pred.squeeze().cpu().numpy()  # (H, W)

            np.save(str(out_path), pred_np.astype(np.float32))

        except Exception as e:
            print(f"\n[WARNING] Inference failed {tile_path}: {e}")

    print(f"[PREDICT] Done. Saved to {output_dir}")


def load_best_model():
    """Load the best trained model from checkpoint."""
    from model import build_model

    checkpoint_path = CHECKPOINTS_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run train.py first."
        )

    model = build_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[FUSION] Loaded model from {checkpoint_path}")
    print(f"[FUSION] Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    return model


if __name__ == "__main__":
    # Test fusion formula with known values from §6 example
    h_learned = np.full((512, 512), 0.7, dtype=np.float32)
    h_physics = np.full((512, 512), 0.5, dtype=np.float32)
    alpha = 0.7

    h_final = fuse_heatmaps(h_learned, h_physics, alpha=alpha, smooth_sigma=0)
    expected = alpha * 0.7 + (1 - alpha) * 0.5  # = 0.49 + 0.15 = 0.64
    print(f"Fusion test:")
    print(f"  H_learned = 0.7, H_physics = 0.5, α = {alpha}")
    print(f"  Expected:  {expected:.4f}")
    print(f"  Got:       {h_final.mean():.4f}")
    assert abs(h_final.mean() - expected) < 1e-4, "Fusion formula mismatch!"
    print("  ✓ Fusion formula verified!")
