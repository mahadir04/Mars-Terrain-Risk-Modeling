# 🚀 Mars Rover Terrain Risk Modeling

**Physics + Learning-Based Terrain Risk Modeling for Safe Mars Rover Path Planning**

This repository contains a professional-grade hybrid terrain risk modeling system designed for autonomous Mars rover navigation. It fuses physics-based feature extraction with deep learning-based risk estimation to generate a unified, continuous terrain risk field $H(x, y) \in [0, 1]$.

---

## 🏗️ System Architecture

The pipeline consists of three primary stages:

### 1. Stage 1: Physics-Based Feature Extraction
Calculates physically meaningful hazard maps directly from orbital imagery:
*   **Slope Map**: Terrain gradient magnitude using Sobel operators.
*   **Roughness Map**: Surface irregularity via local windowed variance.
*   **Relative Depth Map**: Shadow-based depression detection (craters, pits).
*   **Output**: A combined physics risk map $H_{physics}$ used for pseudo-label supervision.

### 2. Stage 2: Deep Learning Estimation
A CNN-based approach to capture complex semantic hazards:
*   **Architecture**: MobileNetV3-Large (Encoder) + DeepLabV3+ (Decoder).
*   **Input**: 512x512 grayscale Mars terrain tiles.
*   **Loss Function**: Combined Binary Cross-Entropy (BCE) + Dice Loss.
*   **Supervision**: Self-supervised via physics-based pseudo-labels.

### 3. Stage 3: Heatmap Fusion
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

---

## 🚀 Usage

The entire system is orchestrated via `run_pipeline.py`.

### Full Pipeline Run
Processes all tiles, trains the model, and generates visualizations:
```powershell
python run_pipeline.py
```

### Quick Demo Mode
Process a small subset (100 tiles) and run a short training (3 epochs) to verify the setup:
```powershell
python run_pipeline.py --max-tiles 100 --epochs 3
```

### Resume/Skip Training
If you already have a trained checkpoint and just want to generate results/figures:
```powershell
python run_pipeline.py --skip-train
```

### Figures Only
Regenerate all 7 publication-quality figures without re-running any data processing:
```powershell
python run_pipeline.py --figures-only
```

---

## 📂 Project Structure

*   `config.py`: Central hub for hyperparameters, weights ($\alpha$), and file paths.
*   `physics_features.py`: Logic for Slope, Roughness, and Depth extraction.
*   `pseudo_labels.py`: Script to generate training targets from physics maps.
*   `dataset.py`: PyTorch Dataset/DataLoader with synchronized augmentations.
*   `model.py`: DeepLabV3+ with MobileNetV3-Large implementation.
*   `train.py`: Full training loop with early stopping and logging.
*   `fusion.py`: Logic for weighted heatmap combination and post-processing.
*   `visualize.py`: Script to generate the 7 research figures for the pre-defense.
*   `run_pipeline.py`: Main entry point to run the stages sequentially.

---

## 📊 Outputs

All results are saved in the `outputs/` directory:
*   `outputs/figures/`: Publication-ready PNG figures (Tile Gallery, Training Curves, Fusion Panels).
*   `outputs/checkpoints/`: Best and latest model weights (`.pth`).
*   `outputs/logs/`: Timestamped CSV training logs.
*   `outputs/fused_maps/`: Final risk heatmaps for every tile (`.npy`).

---

## 🛠️ Requirements

*   Python 3.10+
*   PyTorch (CUDA recommended)
*   OpenCV
*   scikit-image
*   Matplotlib
*   NumPy
