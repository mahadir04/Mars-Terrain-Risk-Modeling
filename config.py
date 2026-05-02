"""
=============================================================================
 config.py — Central Configuration
 Physics + Learning-Based Terrain Risk Modeling for Mars Rover Path Planning
=============================================================================
"""

import os
from pathlib import Path

# ─────────────────────────── Dataset Paths ──────────────────────────────────
DATASET_ROOT = Path(r"D:\original-image-slices-512x512")
TILE_DIRS = [
    DATASET_ROOT / "sliced_tiles_1",
    DATASET_ROOT / "sliced_tiles_2",
]

# ─────────────────────────── Output Paths ───────────────────────────────────
PROJECT_ROOT = Path(r"d:\PRE_Defanse")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PHYSICS_MAPS_DIR = OUTPUT_DIR / "physics_maps"
PSEUDO_LABELS_DIR = OUTPUT_DIR / "pseudo_labels"
DL_PREDICTIONS_DIR = OUTPUT_DIR / "dl_predictions"
FUSED_MAPS_DIR = OUTPUT_DIR / "fused_maps"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create all output directories
for d in [PHYSICS_MAPS_DIR, PSEUDO_LABELS_DIR, DL_PREDICTIONS_DIR,
          FUSED_MAPS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Image Properties ───────────────────────────────
TILE_SIZE = 512          # px
IMG_CHANNELS = 1         # grayscale
TILE_PATTERN = "tile_x*_y*_pos*.png"

# ─────────────────────────── Physics Feature Extraction ─────────────────────
# §4.1  Slope Map — Sobel gradient magnitude
SOBEL_KSIZE = 3          # Sobel kernel size (3 or 5)

# §4.2  Roughness Map — Local variance
ROUGHNESS_WINDOW = 15    # Window size for local variance

# §4.3  Relative Depth Map — Shadow-based
DEPTH_WINDOW = 31        # Window size for local mean

# §4.4  Combined Physics Risk
WEIGHT_SLOPE = 0.4       # w1
WEIGHT_ROUGHNESS = 0.3   # w2
WEIGHT_DEPTH = 0.3       # w3

# Gaussian smoothing for pseudo-labels (keep small to avoid blurring)
PSEUDO_LABEL_SIGMA = 0.5

# ─────────────────────────── Model Architecture ────────────────────────────
# §5  DeepLabV3+ with MobileNetV3-Large
MODEL_NAME = "deeplabv3_mobilenet_v3_large"
PRETRAINED_BACKBONE = True
NUM_CLASSES = 1          # Single-channel risk output

# ─────────────────────────── Training ──────────────────────────────────────
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3 
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5

# Train/Val split
VAL_RATIO = 0.2
RANDOM_SEED = 42

# Loss weights
DICE_SMOOTH = 1.0        # Smoothing factor for Dice loss

# ─────────────────────────── GATv2 Graph Model ─────────────────────────────
# §5b  Graph Attention Network v2 for terrain risk estimation
NODE_GRID_SIZE = 32          # Divide 512×512 tile into 32×32 grid → 1024 nodes
GAT_INPUT_DIM = 6            # Node features: [slope, roughness, depth, intensity, x, y]
GAT_HIDDEN_DIM = 64          # Hidden channels per attention head
GAT_OUTPUT_DIM = 1           # Single risk score per node
GAT_NUM_HEADS = 4            # Multi-head attention heads
GAT_NUM_LAYERS = 3           # Number of GATv2Conv layers
GAT_DROPOUT = 0.2            # Dropout rate for attention and features
GAT_LEARNING_RATE = 5e-4     # Learning rate for GAT training
GAT_EPOCHS = 30              # Training epochs for GAT
GAT_BATCH_SIZE = 32          # Graphs per batch (each tile = 1 graph)

# GAT output directories
GAT_CHECKPOINTS_DIR = OUTPUT_DIR / "gat_checkpoints"
GAT_PREDICTIONS_DIR = OUTPUT_DIR / "gat_predictions"
GAT_LOGS_DIR = OUTPUT_DIR / "gat_logs"
for d in [GAT_CHECKPOINTS_DIR, GAT_PREDICTIONS_DIR, GAT_LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Heatmap Fusion ────────────────────────────────
# §6  Fusion weight: H_final = α·H_learned + (1-α)·H_physics
FUSION_ALPHA = 0.7

# Post-processing: 0 = disabled (use bilateral filter in fusion.py instead)
FUSION_SMOOTH_SIGMA = 0

# ─────────────────────────── Visualization ─────────────────────────────────
RISK_CMAP = "inferno"          # Colormap for risk heatmaps
FEATURE_CMAP = "magma"         # Colormap for individual features
TERRAIN_CMAP = "gray"          # Colormap for original terrain
FIG_DPI = 150                  # Figure resolution
NUM_SAMPLE_TILES = 8           # Tiles to visualize in galleries

# ─────────────────────────── Pipeline Control ──────────────────────────────
# Maximum tiles to process (set to None for all)
MAX_TILES = None               # e.g., 500 for quick demo

# Device selection
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Dataset dirs: {[str(d) for d in TILE_DIRS]}")
print(f"[CONFIG] Output dir: {OUTPUT_DIR}")
