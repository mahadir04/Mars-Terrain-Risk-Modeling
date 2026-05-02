# 🚀 Mars Rover Terrain Risk Modeling: A Hybrid Physics & Graph Attention Approach

**Physics + Learning-Based Terrain Risk Modeling for Safe Mars Rover Path Planning**

This repository contains the full source code for a professional-grade, hybrid terrain risk modeling system designed for autonomous Mars rover navigation. By fusing deterministic physics-based feature extraction with advanced deep learning (both CNNs and Graph Attention Networks), this pipeline generates a unified, continuous terrain risk field $H(x, y) \in [0, 1]$ that is both physically interpretable and semantically intelligent.

---

## 🎯 Project Motivation & Overview

Navigating the Martian surface is incredibly dangerous. Rovers face steep cliffs, hidden craters, and rough, rocky terrain. Traditional path planning relies either purely on simple heuristics (which miss complex hazards) or purely on black-box neural networks (which are hard to interpret and trust). 

This project bridges that gap. We calculate exact physical parameters (like slope and roughness) and use them to supervise advanced deep learning models (DeepLabV3+ and GATv2). The final product is a robust heat map that tells the rover exactly where it is safe to drive.

---

## 🏗️ System Architecture

The pipeline is fully automated and consists of four primary stages.

### 1. Stage 1: Physics-Based Feature Extraction (`physics_features.py`)
Instead of relying on human annotations, we extract physically meaningful hazard maps directly from orbital imagery using classic computer vision techniques:
*   **Slope Map ($S$)**: Terrain gradient magnitude is computed using Sobel operators. High gradients represent steep inclines that a rover cannot climb.
*   **Roughness Map ($R$)**: Surface irregularity is captured via local windowed variance. High variance indicates rocky or uneven terrain.
*   **Relative Depth Map ($D$)**: Shadow-based depression detection identifies craters and pits by comparing local pixels to a wider regional mean.
*   **Output**: These are linearly combined into a physics risk map $H_{physics}$, which serves as a "pseudo-label" to train our deep learning models without manual labeling.

### 2. Stage 2a: CNN-Based Semantic Estimation (`train.py`, `model.py`)
To capture complex, large-scale semantic hazards that simple physics equations might miss, we use a state-of-the-art image segmentation model:
*   **Architecture**: DeepLabV3+ (Decoder) with a MobileNetV3-Large (Encoder). This ensures the model is lightweight enough for potential edge-device deployment.
*   **Data Augmentation**: To prevent overfitting, we apply dynamic photometric augmentations (random brightness and contrast shifts by ±20%) as well as geometric augmentations (rotations, flips).
*   **Loss Function**: A combined Binary Cross-Entropy (BCE) + Dice Loss ensures the model accurately predicts both widespread risk areas and sharp hazard boundaries.

### 3. Stage 2b: Graph Attention Network (`gat_model.py`, `train_gat.py`)
To model how hazards propagate spatially (e.g., a steep slope is more dangerous if it ends in a deep crater), we implemented a Graph Attention Network (GATv2):
*   **Nodification**: The $512 \times 512$ pixel grid is downsampled into a $32 \times 32$ graph of patches (1,024 nodes).
*   **Features**: Each node contains a vector `[Slope, Roughness, Depth, Visual Intensity, X, Y]`.
*   **Connectivity**: Nodes are connected to their 8 neighbours (horizontal, vertical, diagonal).
*   **Dynamic Attention**: Using `GATv2Conv`, the model learns to calculate attention weights between neighbours. It learns *which* surrounding terrain features contribute most to the central node's risk score.

### 4. Stage 3: Heatmap Fusion (`fusion.py`)
The final risk map is a weighted blending of the deterministic physics maps and the learned predictions:
$$H_{final}(x, y) = \alpha \cdot H_{learned} + (1 - \alpha) \cdot H_{physics}$$
We apply a **Bilateral Filter** to the final output to smooth out noise while strictly preserving sharp geographical boundaries (like crater rims).

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mahadir04/Mars-Terrain-Risk-Modeling.git
   cd Mars-Terrain-Risk-Modeling
   ```

2. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note on PyTorch Geometric**: `torch-geometric` requires PyTorch to be installed first. Depending on your CUDA version, you may need to install the scatter/sparse dependencies manually. Please refer to the [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## 🚀 Usage Guide

The entire end-to-end system is orchestrated via `run_pipeline.py`. 

### Quick Demo (Recommended for Testing)
Run a short training loop on a small subset of tiles to verify your environment is working:
```powershell
python run_pipeline.py --max-tiles 100 --epochs 3 --gat-epochs 5
```

### Full Pipeline Execution
Process all tiles, generate pseudo-labels, train the CNN, train the GATv2 model, fuse the results, and generate all thesis figures:
```powershell
python run_pipeline.py
```

### Modular Execution
You can skip stages you have already completed. For example, to skip CNN training and only train the Graph model:
```powershell
python run_pipeline.py --skip-train --gat-epochs 30
```
Or, if you just want to regenerate the publication figures from existing checkpoints:
```powershell
python run_pipeline.py --figures-only
```

---

## 📂 Repository Structure

```text
📦 Mars-Terrain-Risk-Modeling
┣ 📂 outputs/                 # Auto-generated outputs
┃ ┣ 📂 checkpoints/           # Best CNN weights (.pth)
┃ ┣ 📂 gat_checkpoints/       # Best GATv2 weights (.pth)
┃ ┣ 📂 figures/               # Final PNG figures for thesis
┃ ┣ 📂 fused_maps/            # Final risk matrices (.npy)
┃ ┗ 📂 logs/                  # Training history CSVs
┣ 📜 config.py                # Global hyperparameters (Learning Rates, grid sizes, etc.)
┣ 📜 run_pipeline.py          # Main execution script
┣ 📜 dataset.py               # CNN Data loading and photometric augmentation
┣ 📜 physics_features.py      # Sobel, Variance, and Depth equations
┣ 📜 pseudo_labels.py         # Physics target generation
┣ 📜 model.py                 # DeepLabV3+ CNN architecture
┣ 📜 train.py                 # CNN Training loop w/ early stopping
┣ 📜 graph_utils.py           # Terrain "Nodification" logic (Grid to Graph)
┣ 📜 graph_dataset.py         # PyTorch Geometric dataset loader
┣ 📜 gat_model.py             # GATv2 architecture & attention mechanisms
┣ 📜 train_gat.py             # GATv2 Training loop
┣ 📜 fusion.py                # Alpha-blending and bilateral filtering
┣ 📜 visualize.py             # Matplotlib figure generation
┗ 📜 requirements.txt         # Python dependencies
```

---

## 🧠 Deep Dive: The "Nodification" Process

To apply Graph Neural Networks to satellite imagery, we must translate pixels into nodes. 

1. **Downsampling**: A $512 \times 512$ tile has over 260,000 pixels. We divide this into a $32 \times 32$ grid.
2. **Feature Aggregation**: For each $16 \times 16$ pixel patch, we calculate the mean Slope, Roughness, Depth, and Visual Brightness.
3. **Graph Construction**: This patch becomes a single node, retaining spatial context via normalized $(x,y)$ coordinates, and passes messages to its 8 physical neighbours.

```
512×512 Image Tile              32×32 Graph (1,024 nodes)
┌──────────────────┐           ┌──────────────────┐
│ ░░░░░░░░░░░░░░░░ │           │ ●──●──●──●──● ...│
│ ░░░░░░░░░░░░░░░░ │  ──────>  │ │╲ │╲ │╲ │╲ │   │
│ ░░░░░░░░░░░░░░░░ │ Nodify    │ ●──●──●──●──● ...│
│ ░░░░░░░░░░░░░░░░ │           │ │╲ │╲ │╲ │╲ │   │
│ ░░░░░░░░░░░░░░░░ │           │ ●──●──●──●──● ...│
└──────────────────┘           └──────────────────┘
```

---

## 🛠️ Configuration & Tuning

Almost all parameters can be tweaked inside `config.py` without touching the core logic. Key variables include:
*   `WEIGHT_DECAY` and `EARLY_STOPPING_PATIENCE`: Tuned to heavily penalize overfitting.
*   `NODE_GRID_SIZE`: Change the resolution of the GATv2 graph (default 32).
*   `FUSION_ALPHA`: Determines how much to trust the Neural Network vs the pure Physics equations (default 0.7).
