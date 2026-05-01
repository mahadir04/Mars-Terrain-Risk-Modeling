"""
=============================================================================
 pseudo_labels.py — Physics-Based Pseudo-Label Generation
 Generates training targets from physics risk maps (no ground truth needed)
=============================================================================
"""

import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from config import (
    TILE_DIRS, PSEUDO_LABELS_DIR, PHYSICS_MAPS_DIR,
    WEIGHT_SLOPE, WEIGHT_ROUGHNESS, WEIGHT_DEPTH,
    SOBEL_KSIZE, ROUGHNESS_WINDOW, DEPTH_WINDOW,
    PSEUDO_LABEL_SIGMA, MAX_TILES, TILE_PATTERN
)
from physics_features import load_and_preprocess, compute_physics_risk_map


def generate_pseudo_labels(
    tile_dirs: list = None,
    output_dir: Path = None,
    physics_dir: Path = None,
    max_tiles: int = None,
    save_physics: bool = True
):
    """
    Generate pseudo-labels for all tiles using physics-based risk estimation.
    
    For each tile:
    1. Compute H_physics (slope + roughness + depth)
    2. Apply Gaussian smoothing for label consistency
    3. Save as .npy float32 files
    
    Parameters
    ----------
    tile_dirs : list of Path
        Directories containing tile images.
    output_dir : Path
        Directory to save pseudo-labels.
    physics_dir : Path
        Directory to save physics risk maps.
    max_tiles : int or None
        Maximum number of tiles to process.
    save_physics : bool
        Whether to also save individual physics feature maps.
    """
    tile_dirs = tile_dirs or TILE_DIRS
    output_dir = output_dir or PSEUDO_LABELS_DIR
    physics_dir = physics_dir or PHYSICS_MAPS_DIR
    max_tiles = max_tiles or MAX_TILES
    
    output_dir.mkdir(parents=True, exist_ok=True)
    physics_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all tile paths
    all_tiles = []
    for td in tile_dirs:
        tiles = sorted(glob.glob(str(td / TILE_PATTERN)))
        all_tiles.extend(tiles)
    
    if max_tiles is not None:
        all_tiles = all_tiles[:max_tiles]
    
    print(f"[PSEUDO-LABELS] Processing {len(all_tiles)} tiles...")
    print(f"[PSEUDO-LABELS] Physics weights: slope={WEIGHT_SLOPE}, "
          f"roughness={WEIGHT_ROUGHNESS}, depth={WEIGHT_DEPTH}")
    print(f"[PSEUDO-LABELS] Gaussian sigma: {PSEUDO_LABEL_SIGMA}")
    
    stats = {
        'total': len(all_tiles),
        'processed': 0,
        'failed': 0,
        'mean_risk': [],
        'max_risk': [],
    }
    
    for tile_path in tqdm(all_tiles, desc="Generating pseudo-labels"):
        try:
            tile_name = Path(tile_path).stem
            
            # Check if already processed
            label_path = output_dir / f"{tile_name}.npy"
            if label_path.exists():
                stats['processed'] += 1
                continue
            
            # Load and compute physics features
            image = load_and_preprocess(tile_path)
            result = compute_physics_risk_map(
                image,
                w_slope=WEIGHT_SLOPE,
                w_roughness=WEIGHT_ROUGHNESS,
                w_depth=WEIGHT_DEPTH,
                sobel_ksize=SOBEL_KSIZE,
                roughness_window=ROUGHNESS_WINDOW,
                depth_window=DEPTH_WINDOW
            )
            
            # Apply Gaussian smoothing for label consistency
            pseudo_label = gaussian_filter(
                result['combined'], sigma=PSEUDO_LABEL_SIGMA
            )
            
            # Re-normalize after smoothing
            pl_min, pl_max = pseudo_label.min(), pseudo_label.max()
            if pl_max - pl_min > 1e-8:
                pseudo_label = (pseudo_label - pl_min) / (pl_max - pl_min)
            
            pseudo_label = pseudo_label.astype(np.float32)
            
            # Save pseudo-label
            np.save(str(label_path), pseudo_label)
            
            # Save individual physics maps if requested
            if save_physics:
                physics_subdir = physics_dir / tile_name
                physics_subdir.mkdir(parents=True, exist_ok=True)
                np.save(str(physics_subdir / "slope.npy"), result['slope'])
                np.save(str(physics_subdir / "roughness.npy"), result['roughness'])
                np.save(str(physics_subdir / "depth.npy"), result['depth'])
                np.save(str(physics_subdir / "combined.npy"), result['combined'])
            
            stats['processed'] += 1
            stats['mean_risk'].append(float(pseudo_label.mean()))
            stats['max_risk'].append(float(pseudo_label.max()))
            
        except Exception as e:
            print(f"\n[WARNING] Failed: {tile_path}: {e}")
            stats['failed'] += 1
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f" PSEUDO-LABEL GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f" Total tiles:     {stats['total']}")
    print(f" Processed:       {stats['processed']}")
    print(f" Failed:          {stats['failed']}")
    if stats['mean_risk']:
        print(f" Mean risk:       {np.mean(stats['mean_risk']):.4f}")
        print(f" Max risk (avg):  {np.mean(stats['max_risk']):.4f}")
    print(f" Output dir:      {output_dir}")
    print(f"{'='*60}")
    
    return stats


if __name__ == "__main__":
    stats = generate_pseudo_labels()
