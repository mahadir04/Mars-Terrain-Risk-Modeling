"""
=============================================================================
 graph_dataset.py — PyTorch Geometric Dataset for Mars Terrain Graphs
 Loads tiles, computes physics features, and returns graph Data objects.
=============================================================================
 Reference: Pre-Defense Report §5b

 Each tile produces one graph:
   - Nodes: N×N patches with [slope, roughness, depth, intensity, x, y]
   - Edges: 8-connected grid neighbours
   - Target: Per-node risk from pseudo-labels
=============================================================================
"""

import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import random_split

try:
    from torch_geometric.data import Dataset as PyGDataset, Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    raise ImportError(
        "torch_geometric is required. "
        "Install with: pip install torch-geometric"
    )

from config import (
    TILE_DIRS, PSEUDO_LABELS_DIR, PHYSICS_MAPS_DIR,
    TILE_PATTERN, MAX_TILES, NODE_GRID_SIZE,
    VAL_RATIO, RANDOM_SEED, GAT_BATCH_SIZE,
    WEIGHT_SLOPE, WEIGHT_ROUGHNESS, WEIGHT_DEPTH,
    SOBEL_KSIZE, ROUGHNESS_WINDOW, DEPTH_WINDOW
)
from graph_utils import GridGraphBuilder
from physics_features import (
    compute_slope_map, compute_roughness_map, compute_depth_map
)


class MarsTerrainGraphDataset(PyGDataset):
    """
    PyTorch Geometric Dataset that converts Mars terrain tiles into graphs.

    For each tile:
      1. Load grayscale image → compute slope, roughness, depth
      2. Load pseudo-label (target risk)
      3. Convert to a graph using GridGraphBuilder

    Parameters
    ----------
    tile_dirs : list of Path
        Directories containing tile PNG images.
    label_dir : Path
        Directory containing pseudo-label .npy files.
    max_tiles : int or None
        Maximum number of tiles to use.
    grid_size : int
        Grid resolution for nodification.
    """

    def __init__(
        self,
        tile_dirs: list = None,
        label_dir: Path = None,
        max_tiles: int = None,
        grid_size: int = None,
    ):
        self._tile_dirs = tile_dirs or TILE_DIRS
        self._label_dir = Path(label_dir or PSEUDO_LABELS_DIR)
        self._grid_size = grid_size or NODE_GRID_SIZE

        # Build sample list: (tile_path, label_path)
        self._samples = []
        for td in self._tile_dirs:
            tiles = sorted(glob.glob(str(Path(td) / TILE_PATTERN)))
            for tile_path in tiles:
                tile_name = Path(tile_path).stem
                label_path = self._label_dir / f"{tile_name}.npy"
                if label_path.exists():
                    self._samples.append((tile_path, str(label_path)))

        # Limit number of tiles
        _max = max_tiles or MAX_TILES
        if _max is not None:
            self._samples = self._samples[:_max]

        # Shared graph builder (edge_index is the same for all tiles)
        self._graph_builder = GridGraphBuilder(grid_size=self._grid_size)

        print(f"[GAT-DATASET] Loaded {len(self._samples)} tile-label pairs "
              f"(grid={self._grid_size}×{self._grid_size}, "
              f"nodes={self._graph_builder.num_nodes}, "
              f"edges={self._graph_builder.num_edges})")

        # PyG Dataset requires calling super().__init__ after setup
        super().__init__()

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> Data:
        """
        Load tile at index `idx` and return a graph Data object.

        Returns
        -------
        data : torch_geometric.data.Data
            Graph with node features, edges, and target risk.
        """
        tile_path, label_path = self._samples[idx]

        # ── Load image ──────────────────────────────────────────────────
        img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {tile_path}")
        image = img.astype(np.float32) / 255.0

        # ── Compute physics features on-the-fly ─────────────────────────
        slope = compute_slope_map(image, ksize=SOBEL_KSIZE)
        roughness = compute_roughness_map(image, window_size=ROUGHNESS_WINDOW)
        depth = compute_depth_map(image, window_size=DEPTH_WINDOW)

        # ── Load pseudo-label target ────────────────────────────────────
        target = np.load(label_path).astype(np.float32)

        # ── Build graph ─────────────────────────────────────────────────
        data = self._graph_builder.build_graph(
            image=image,
            slope=slope,
            roughness=roughness,
            depth=depth,
            target=target,
        )

        return data


def create_graph_dataloaders(
    tile_dirs: list = None,
    label_dir: Path = None,
    batch_size: int = None,
    val_ratio: float = None,
    max_tiles: int = None,
    grid_size: int = None,
) -> Tuple:
    """
    Create train and validation DataLoaders for graph-based training.

    Parameters
    ----------
    tile_dirs : list of Path
    label_dir : Path
    batch_size : int
    val_ratio : float
    max_tiles : int or None
    grid_size : int

    Returns
    -------
    train_loader, val_loader : tuple of PyGDataLoader
    """
    batch_size = batch_size or GAT_BATCH_SIZE
    val_ratio = val_ratio or VAL_RATIO

    # Full dataset
    full_dataset = MarsTerrainGraphDataset(
        tile_dirs=tile_dirs,
        label_dir=label_dir,
        max_tiles=max_tiles,
        grid_size=grid_size,
    )

    # Split into train and validation
    total = len(full_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"[GAT-DATASET] Train: {train_size} | Val: {val_size}")

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


# ─────────────────────────── CLI Test ───────────────────────────────────────
if __name__ == "__main__":
    print("Testing MarsTerrainGraphDataset...")

    ds = MarsTerrainGraphDataset(max_tiles=5)
    print(f"Dataset size: {len(ds)}")

    if len(ds) > 0:
        data = ds[0]
        print(f"\nSample graph:")
        print(f"  x (features):   {data.x.shape}")
        print(f"  edge_index:     {data.edge_index.shape}")
        print(f"  y (target):     {data.y.shape}")
        print(f"  Feature range:  [{data.x.min():.3f}, {data.x.max():.3f}]")
        print(f"  Target range:   [{data.y.min():.3f}, {data.y.max():.3f}]")

        # Test DataLoader
        train_loader, val_loader = create_graph_dataloaders(max_tiles=5)
        for batch in train_loader:
            print(f"\nBatch info:")
            print(f"  x:          {batch.x.shape}")
            print(f"  edge_index: {batch.edge_index.shape}")
            print(f"  y:          {batch.y.shape}")
            print(f"  batch:      {batch.batch.shape}")
            break

        print("\n✓ Graph dataset test passed!")
    else:
        print("No samples found. Run pseudo_labels.py first!")
