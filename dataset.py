"""
=============================================================================
 dataset.py — PyTorch Dataset & DataLoader for Mars Terrain Tiles
=============================================================================
"""

import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random

from config import (
    TILE_DIRS, PSEUDO_LABELS_DIR, TILE_PATTERN,
    TILE_SIZE, BATCH_SIZE, NUM_WORKERS,
    VAL_RATIO, RANDOM_SEED, MAX_TILES
)


class MarsTerrainDataset(Dataset):
    """
    PyTorch Dataset for Mars terrain risk estimation.
    
    Loads 512×512 grayscale tiles and corresponding pseudo-label risk maps.
    Applies data augmentation during training.
    
    Parameters
    ----------
    tile_dirs : list of Path
        Directories containing tile PNG images.
    label_dir : Path
        Directory containing pseudo-label .npy files.
    transform : bool
        Whether to apply data augmentation.
    max_tiles : int or None
        Maximum number of tiles to use.
    """
    
    def __init__(
        self,
        tile_dirs: list = None,
        label_dir: Path = None,
        transform: bool = True,
        max_tiles: int = None
    ):
        super().__init__()
        tile_dirs = tile_dirs or TILE_DIRS
        label_dir = label_dir or PSEUDO_LABELS_DIR
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        # Collect all tile paths that have corresponding pseudo-labels
        self.samples = []
        for td in tile_dirs:
            tiles = sorted(glob.glob(str(Path(td) / TILE_PATTERN)))
            for tile_path in tiles:
                tile_name = Path(tile_path).stem
                label_path = self.label_dir / f"{tile_name}.npy"
                if label_path.exists():
                    self.samples.append((tile_path, str(label_path)))
        
        # Limit number of tiles if specified
        max_tiles = max_tiles or MAX_TILES
        if max_tiles is not None:
            self.samples = self.samples[:max_tiles]
        
        print(f"[DATASET] Loaded {len(self.samples)} tile-label pairs "
              f"(transform={transform})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_path, label_path = self.samples[idx]
        
        # Load grayscale image
        image = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load: {tile_path}")
        image = image.astype(np.float32) / 255.0
        
        # Load pseudo-label
        label = np.load(label_path).astype(np.float32)
        
        # Convert to tensors: (1, H, W)
        image_t = torch.from_numpy(image).unsqueeze(0)
        label_t = torch.from_numpy(label).unsqueeze(0)
        
        # Apply augmentations
        if self.transform:
            image_t, label_t = self._augment(image_t, label_t)
        
        return image_t, label_t
    
    def _augment(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized random augmentations to image and label.
        
        Augmentations:
        - Random horizontal flip (50%)
        - Random vertical flip (50%)
        - Random rotation (0°, 90°, 180°, 270°)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)
        
        # Random 90° rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[1, 2])
        
        return image, label


def create_dataloaders(
    tile_dirs: list = None,
    label_dir: Path = None,
    batch_size: int = None,
    val_ratio: float = None,
    max_tiles: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Parameters
    ----------
    tile_dirs : list of Path
    label_dir : Path
    batch_size : int
    val_ratio : float
    max_tiles : int or None
    
    Returns
    -------
    train_loader, val_loader : tuple of DataLoader
    """
    batch_size = batch_size or BATCH_SIZE
    val_ratio = val_ratio or VAL_RATIO
    
    # Full dataset (with augmentation)
    full_dataset = MarsTerrainDataset(
        tile_dirs=tile_dirs,
        label_dir=label_dir,
        transform=True,
        max_tiles=max_tiles
    )
    
    # Split into train and validation
    total = len(full_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Disable augmentation for validation subset
    # (We wrap with a thin dataset that disables transforms)
    val_dataset.dataset = MarsTerrainDataset(
        tile_dirs=tile_dirs,
        label_dir=label_dir,
        transform=False,
        max_tiles=max_tiles
    )
    
    print(f"[DATASET] Train: {train_size} | Val: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    ds = MarsTerrainDataset(max_tiles=10)
    print(f"Dataset size: {len(ds)}")
    
    if len(ds) > 0:
        img, lbl = ds[0]
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Label shape: {lbl.shape}, dtype: {lbl.dtype}")
        print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Label range: [{lbl.min():.3f}, {lbl.max():.3f}]")
    else:
        print("No samples found. Run pseudo_labels.py first!")
