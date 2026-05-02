# 🚀 Mars Rover Terrain Risk Modeling

**Physics + Learning-Based Terrain Risk Modeling for Safe Mars Rover Path Planning**

This repository contains a professional-grade hybrid terrain risk modeling system designed for autonomous Mars rover navigation. It fuses physics-based feature extraction with deep learning-based risk estimation to generate a unified, continuous terrain risk field $H(x, y) \in [0, 1]$.

---

## 🏗️ System Architecture

The pipeline consists of four primary stages:

### 1. Stage 1: Physics-Based Feature Extraction
Calculates physically meaningful hazard maps directly from orbital imagery:
*   **Slope Map**: Terrain gradient magnitude using Sobel operators.
*   **Roughness Map**: Surface irregularity via local windowed variance.
*   **Relative Depth Map**: Shadow-based depression detection (craters, pits).
*   **Output**: A combined physics risk map $H_{physics}$ used for pseudo-label supervision.

### 2. Stage 2a: CNN-Based Deep Learning Estimation
A CNN-based approach to capture complex semantic hazards:
*   **Architecture**: MobileNetV3-Large (Encoder) + DeepLabV3+ (Decoder).
*   **Input**: 512×512 grayscale Mars terrain tiles.
*   **Loss Function**: Combined Binary Cross-Entropy (BCE) + Dice Loss.
*   **Supervision**: Self-supervised via physics-based pseudo-labels.

### 3. Stage 2b: GATv2 Graph Attention Network
A graph-based approach that models spatial terrain relationships via attention:
*   **Nodification**: Each 512×512 tile is divided into a 32×32 grid of patches (1,024 nodes).
*   **Node Features**: Each node has 6 features: `[slope, roughness, depth, intensity, x, y]`.
*   **Edge Connectivity**: 8-connected grid (horizontal, vertical, diagonal neighbours).
*   **Architecture**: GATv2Conv × 3 layers, 4 attention heads, 64 hidden channels.
*   **Loss Function**: Combined MSE + Smooth L1 for node-level risk regression.
*   **Advantage**: GATv2 uses *dynamic attention* (Brody et al., ICLR 2022), learning which neighbouring terrain patches are most relevant for risk propagation. For example, a crater rim node can "attend to" nearby slope nodes to propagate danger signals.

### 4. Stage 3: Heatmap Fusion
Weighted blending of physics and learned maps:
$$H_{final}(x, y) = \alpha \cdot H_{learned} + (1 - \alpha) \cdot H_{physics}$$
The resulting fused map provides a robust risk estimate that balances physical interpretability with semantic intelligence.

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   cd d:\PRE_Defanse
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: `torch-geometric` requires PyTorch to be installed first. If you encounter issues, install it manually following the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## 🚀 Usage

The entire system is orchestrated via `run_pipeline.py`.

### Full Pipeline Run
Processes all tiles, trains both CNN and GATv2 models, and generates visualizations:
```powershell
python run_pipeline.py
```

### Quick Demo Mode
Process a small subset (100 tiles) and run a short training (3 epochs) to verify the setup:
```powershell
python run_pipeline.py --max-tiles 100 --epochs 3 --gat-epochs 5
```

### Resume/Skip Training
If you already have trained checkpoints and just want to generate results/figures:
```powershell
python run_pipeline.py --skip-train --skip-gat
```

### GATv2 Only
Skip CNN training and only train the GATv2 graph model:
```powershell
python run_pipeline.py --skip-train --gat-epochs 30
```

### Figures Only
Regenerate all 7 publication-quality figures without re-running any data processing:
```powershell
python run_pipeline.py --figures-only
```

### Train GATv2 Standalone
You can also train GATv2 independently:
```powershell
python train_gat.py --max-tiles 500 --epochs 20
```

---

## 📂 Project Structure

### Core Pipeline
*   `config.py`: Central hub for hyperparameters, weights ($\alpha$), and file paths.
*   `physics_features.py`: Logic for Slope, Roughness, and Depth extraction.
*   `pseudo_labels.py`: Script to generate training targets from physics maps.
*   `run_pipeline.py`: Main entry point to run all stages sequentially.

### CNN Path (Stage 2a)
*   `dataset.py`: PyTorch Dataset/DataLoader with synchronized augmentations.
*   `model.py`: DeepLabV3+ with MobileNetV3-Large implementation.
*   `train.py`: Full CNN training loop with early stopping and logging.

### GATv2 Path (Stage 2b)
*   `graph_utils.py`: **Terrain "Nodification"** — converts 512×512 pixel maps into graph structures (nodes, edges, features).
*   `graph_dataset.py`: PyTorch Geometric Dataset that produces `Data` objects for GATv2.
*   `gat_model.py`: GATv2 model architecture with multi-head attention and batch normalization.
*   `train_gat.py`: GATv2 training loop with early stopping, CSV logging, and checkpointing.

### Fusion & Visualization
*   `fusion.py`: Logic for weighted heatmap combination and post-processing.
*   `visualize.py`: Script to generate the 7 research figures for the pre-defense.

---

## 🧠 GATv2 — How it Works

### Step 1: Nodification
The 512×512 terrain tile is divided into a **32×32 grid** of patches (each 16×16 pixels). Each patch becomes a **node** in the graph.

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

### Step 2: Feature Extraction
Each node gets a feature vector of size **6**:

| Feature     | Source            | Meaning                              |
|:------------|:------------------|:-------------------------------------|
| Slope       | Sobel gradient    | Terrain steepness                    |
| Roughness   | Local variance    | Surface irregularity                 |
| Depth       | Shadow detection  | Depression/crater depth              |
| Intensity   | Raw grayscale     | Visual brightness                    |
| x, y        | Grid position     | Spatial location (normalized 0–1)    |

### Step 3: Graph Attention
GATv2Conv layers compute **attention weights** between connected nodes, learning which neighbours are most important for risk prediction. This is more expressive than standard GAT because attention is computed *after* the linear transformation.

### Step 4: Upsampling
Node-level predictions are upsampled back to the original 512×512 resolution.

---

## 📊 Outputs

All results are saved in the `outputs/` directory:
*   `outputs/figures/`: Publication-ready PNG figures (Tile Gallery, Training Curves, Fusion Panels).
*   `outputs/checkpoints/`: Best and latest CNN model weights (`.pth`).
*   `outputs/gat_checkpoints/`: Best and latest GATv2 model weights (`.pth`).
*   `outputs/logs/`: Timestamped CSV training logs (CNN).
*   `outputs/gat_logs/`: Timestamped CSV training logs (GATv2).
*   `outputs/fused_maps/`: Final risk heatmaps for every tile (`.npy`).

---

## 🛠️ Requirements

*   Python 3.10+
*   PyTorch (CUDA recommended)
*   PyTorch Geometric (for GATv2)
*   OpenCV
*   scikit-image
*   Matplotlib
*   NumPy
*   SciPy
