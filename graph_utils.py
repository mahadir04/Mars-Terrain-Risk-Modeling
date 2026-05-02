"""
=============================================================================
 graph_utils.py — Terrain Map → Graph Conversion ("Nodification")
 Converts 2D pixel grids into graph structures for GATv2
=============================================================================
 Reference: Pre-Defense Report §5b

 Process:
   1. Divide 512×512 tile into N×N grid of patches
   2. Each patch → 1 node with features [slope, roughness, depth, intensity, x, y]
   3. Connect 8-neighbors → edge_index
   4. Pseudo-label risk → node-level target y
=============================================================================
"""

import numpy as np
import torch
from typing import Tuple, Optional

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        "torch_geometric is required for GATv2. "
        "Install with: pip install torch-geometric"
    )

from config import NODE_GRID_SIZE, TILE_SIZE


class GridGraphBuilder:
    """
    Converts 2D terrain feature maps into a graph for GATv2.

    The 512×512 tile is divided into an N×N grid of patches.
    Each patch becomes one node in the graph with a feature vector
    derived from the physics-based maps (slope, roughness, depth)
    and the raw grayscale intensity.

    Parameters
    ----------
    grid_size : int
        Number of patches along each axis. Default 32 → 1024 nodes.
    tile_size : int
        Original tile resolution in pixels. Default 512.
    """

    def __init__(self, grid_size: int = None, tile_size: int = None):
        self.grid_size = grid_size or NODE_GRID_SIZE
        self.tile_size = tile_size or TILE_SIZE
        self.patch_size = self.tile_size // self.grid_size  # e.g. 512/32 = 16px

        # Pre-compute edge_index (static for all tiles of the same grid)
        self._edge_index = self._build_grid_edges()

    # ─────────────────── Edge Construction ───────────────────────────────

    def _build_grid_edges(self) -> torch.Tensor:
        """
        Build 8-connected grid edges for an N×N grid.

        Each node (r, c) is connected to its 8 neighbours:
            (r±1, c), (r, c±1), (r±1, c±1)

        Returns
        -------
        edge_index : torch.Tensor, shape [2, E]
            COO-format edge list (undirected).
        """
        N = self.grid_size
        src, dst = [], []

        # 8 directions: up, down, left, right, and 4 diagonals
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
        ]

        for r in range(N):
            for c in range(N):
                node_id = r * N + c
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        neighbor_id = nr * N + nc
                        src.append(node_id)
                        dst.append(neighbor_id)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index

    @property
    def edge_index(self) -> torch.Tensor:
        """Static edge connectivity for the grid (shared across all tiles)."""
        return self._edge_index

    @property
    def num_nodes(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def num_edges(self) -> int:
        return self._edge_index.shape[1]

    # ─────────────────── Node Feature Extraction ────────────────────────

    def _extract_patch_features(self, feature_map: np.ndarray) -> np.ndarray:
        """
        Downsample a 2D map to N×N by averaging over patches.

        Parameters
        ----------
        feature_map : np.ndarray, shape (H, W)
            Any 2D feature map (slope, roughness, depth, or image).

        Returns
        -------
        patch_values : np.ndarray, shape (N*N,)
            Mean value per patch, flattened in row-major order.
        """
        N = self.grid_size
        P = self.patch_size
        H, W = feature_map.shape

        # Ensure the map is the right size (pad/crop if needed)
        if H != self.tile_size or W != self.tile_size:
            # Resize to tile_size × tile_size
            import cv2
            feature_map = cv2.resize(
                feature_map, (self.tile_size, self.tile_size),
                interpolation=cv2.INTER_AREA
            )

        # Reshape into (N, P, N, P) and take the mean per patch
        patches = feature_map.reshape(N, P, N, P)
        patch_means = patches.mean(axis=(1, 3))  # shape (N, N)

        return patch_means.flatten()  # shape (N*N,)

    def _positional_encoding(self) -> np.ndarray:
        """
        Generate normalized (x, y) position for each node.

        Returns
        -------
        positions : np.ndarray, shape (N*N, 2)
            Columns: [normalized_x, normalized_y] in [0, 1].
        """
        N = self.grid_size
        positions = np.zeros((N * N, 2), dtype=np.float32)
        for r in range(N):
            for c in range(N):
                node_id = r * N + c
                positions[node_id, 0] = c / (N - 1)  # x (column)
                positions[node_id, 1] = r / (N - 1)  # y (row)
        return positions

    # ─────────────────── Main Conversion ────────────────────────────────

    def build_graph(
        self,
        image: np.ndarray,
        slope: np.ndarray,
        roughness: np.ndarray,
        depth: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> Data:
        """
        Convert terrain maps into a PyTorch Geometric Data object.

        Parameters
        ----------
        image : np.ndarray, shape (H, W)
            Grayscale terrain image in [0, 1].
        slope : np.ndarray, shape (H, W)
            Normalized slope map in [0, 1].
        roughness : np.ndarray, shape (H, W)
            Normalized roughness map in [0, 1].
        depth : np.ndarray, shape (H, W)
            Normalized depth map in [0, 1].
        target : np.ndarray or None, shape (H, W)
            Pseudo-label risk map in [0, 1]. If None, no target is set.

        Returns
        -------
        data : torch_geometric.data.Data
            Graph with:
            - x:          Node features [N*N, 6]
            - edge_index: Edge connectivity [2, E]
            - y:          Target risk per node [N*N] (if target provided)
            - num_nodes:  N*N
        """
        # Extract per-patch features
        feat_intensity = self._extract_patch_features(image)
        feat_slope = self._extract_patch_features(slope)
        feat_roughness = self._extract_patch_features(roughness)
        feat_depth = self._extract_patch_features(depth)
        feat_pos = self._positional_encoding()

        # Stack into feature matrix: [slope, roughness, depth, intensity, x, y]
        node_features = np.stack([
            feat_slope,
            feat_roughness,
            feat_depth,
            feat_intensity,
            feat_pos[:, 0],  # x
            feat_pos[:, 1],  # y
        ], axis=1).astype(np.float32)  # shape (N*N, 6)

        x = torch.from_numpy(node_features)

        # Build Data object
        data = Data(
            x=x,
            edge_index=self._edge_index.clone(),
        )

        # Add target if available
        if target is not None:
            target_nodes = self._extract_patch_features(target)
            data.y = torch.from_numpy(target_nodes.astype(np.float32))

        return data

    def graph_to_map(self, node_values: np.ndarray) -> np.ndarray:
        """
        Upsample node-level predictions back to a 2D risk map.

        Uses nearest-neighbor upsampling to reconstruct the full-resolution
        risk map from node-level GATv2 predictions.

        Parameters
        ----------
        node_values : np.ndarray, shape (N*N,)
            Risk score per node.

        Returns
        -------
        risk_map : np.ndarray, shape (tile_size, tile_size)
            Upsampled risk map.
        """
        N = self.grid_size
        P = self.patch_size

        # Reshape to grid
        grid = node_values.reshape(N, N)

        # Upsample: repeat each value over its patch
        risk_map = np.repeat(np.repeat(grid, P, axis=0), P, axis=1)

        return risk_map.astype(np.float32)


# ─────────────────────────── CLI Test ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" Graph Utils — Terrain Nodification Test")
    print("=" * 60)

    builder = GridGraphBuilder()

    print(f"\n  Grid size:    {builder.grid_size}×{builder.grid_size}")
    print(f"  Patch size:   {builder.patch_size}×{builder.patch_size} px")
    print(f"  Total nodes:  {builder.num_nodes}")
    print(f"  Total edges:  {builder.num_edges}")

    # Create dummy maps
    dummy_image = np.random.rand(512, 512).astype(np.float32)
    dummy_slope = np.random.rand(512, 512).astype(np.float32)
    dummy_roughness = np.random.rand(512, 512).astype(np.float32)
    dummy_depth = np.random.rand(512, 512).astype(np.float32)
    dummy_target = np.random.rand(512, 512).astype(np.float32)

    # Build graph
    data = builder.build_graph(
        dummy_image, dummy_slope, dummy_roughness, dummy_depth, dummy_target
    )

    print(f"\n  Graph Data object:")
    print(f"    x (features):    {data.x.shape}  → {data.x.dtype}")
    print(f"    edge_index:      {data.edge_index.shape}")
    print(f"    y (target):      {data.y.shape}  → {data.y.dtype}")
    print(f"    Feature range:   [{data.x.min():.3f}, {data.x.max():.3f}]")
    print(f"    Target range:    [{data.y.min():.3f}, {data.y.max():.3f}]")

    # Test reconstruction
    fake_preds = np.random.rand(builder.num_nodes).astype(np.float32)
    risk_map = builder.graph_to_map(fake_preds)
    print(f"\n  Reconstructed map: {risk_map.shape}")
    print(f"    Range: [{risk_map.min():.3f}, {risk_map.max():.3f}]")

    print("\n  ✓ Graph construction test passed!")
