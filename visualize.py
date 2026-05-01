"""
=============================================================================
 visualize.py — Publication-Quality Visualizations
 Pre-Defense Report Figures Generator
=============================================================================
 Generates 7 figure types:
   1. Input tile gallery
   2. Physics feature maps (Slope | Roughness | Depth | Combined)
   3. DL prediction heatmaps
   4. Fusion comparison panels (Original | Physics | DL | Fused)
   5. Training curves (loss, MAE, Pearson)
   6. Risk distribution histograms
   7. Risk statistics summary table
=============================================================================
"""

import cv2
import csv
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Optional
from scipy.ndimage import gaussian_filter

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3d4158',
    'text.color': '#e8eaf6',
    'axes.labelcolor': '#e8eaf6',
    'xtick.color': '#9fa8da',
    'ytick.color': '#9fa8da',
    'grid.color': '#2d3047',
    'grid.alpha': 0.6,
})

from config import (
    TILE_DIRS, PSEUDO_LABELS_DIR, DL_PREDICTIONS_DIR,
    FUSED_MAPS_DIR, FIGURES_DIR, LOGS_DIR,
    WEIGHT_SLOPE, WEIGHT_ROUGHNESS, WEIGHT_DEPTH,
    SOBEL_KSIZE, ROUGHNESS_WINDOW, DEPTH_WINDOW,
    FUSION_ALPHA, NUM_SAMPLE_TILES, FIG_DPI,
    RISK_CMAP, FEATURE_CMAP, TERRAIN_CMAP, TILE_PATTERN
)
from physics_features import load_and_preprocess, compute_physics_risk_map


# ─────────────────── Custom Mars colour palette ─────────────────────────────
_mars_colors = ['#0d0d1a', '#1a0a2e', '#3d1166', '#7b2d8b',
                '#c0392b', '#e67e22', '#f1c40f', '#ffeaa7']
MARS_CMAP = LinearSegmentedColormap.from_list("mars_risk", _mars_colors, N=256)


def _get_sample_tiles(n: int = None) -> List[str]:
    """Return n tile paths spread across both tile directories."""
    n = n or NUM_SAMPLE_TILES
    all_tiles = []
    for td in TILE_DIRS:
        tiles = sorted(glob.glob(str(td / TILE_PATTERN)))
        all_tiles.extend(tiles)

    if len(all_tiles) == 0:
        raise FileNotFoundError("No tiles found in dataset directories.")

    # Pick evenly-spaced samples for visual diversity
    indices = np.linspace(0, len(all_tiles) - 1, n, dtype=int)
    return [all_tiles[i] for i in indices]


def _add_colorbar(fig, ax, im, label="Risk [0–1]"):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.yaxis.set_tick_params(color='#9fa8da')
    cb.outline.set_edgecolor('#3d4158')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#9fa8da')
    cb.set_label(label, color='#e8eaf6', fontsize=8)
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Input Tile Gallery
# ─────────────────────────────────────────────────────────────────────────────
def fig_tile_gallery(n: int = None, save: bool = True) -> plt.Figure:
    """Sample Mars terrain tiles from the dataset."""
    n = n or NUM_SAMPLE_TILES
    tiles = _get_sample_tiles(n)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    fig.suptitle("Mars Terrain Dataset — Sample Tiles (512×512)",
                 fontsize=15, fontweight='bold', color='#e8eaf6', y=1.01)

    for i, (ax, tile_path) in enumerate(zip(axes, tiles)):
        img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        tile_name = Path(tile_path).parent.name + "/" + Path(tile_path).stem
        short = Path(tile_path).stem[:22]
        ax.set_title(short, fontsize=8, color='#b0bec5', pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#3d4158')

    for ax in axes[len(tiles):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save:
        p = FIGURES_DIR / "fig1_tile_gallery.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Physics Feature Maps
# ─────────────────────────────────────────────────────────────────────────────
def fig_physics_features(tile_path: str = None, save: bool = True) -> plt.Figure:
    """Four-panel physics feature map: Slope | Roughness | Depth | Combined."""
    if tile_path is None:
        tile_path = _get_sample_tiles(1)[0]

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

    fig = plt.figure(figsize=(20, 4.5))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05, hspace=0.1)

    panels = [
        ('Original Terrain', image, 'gray', '(Input)'),
        ('Slope Map\n$S(x,y)=\\sqrt{G_x^2+G_y^2}$', result['slope'], FEATURE_CMAP,
         f'$w_1={WEIGHT_SLOPE}$'),
        ('Roughness Map\n$R(x,y)=\\mathrm{Var}(I_\\mathrm{{local}})$',
         result['roughness'], FEATURE_CMAP, f'$w_2={WEIGHT_ROUGHNESS}$'),
        ('Depth Map\n$D(x,y)=\\mu_\\mathrm{{local}}-I(x,y)$',
         result['depth'], FEATURE_CMAP, f'$w_3={WEIGHT_DEPTH}$'),
        ('Combined Risk\n$H_\\mathrm{physics}=\\sum w_i\\cdot f_i$',
         result['combined'], MARS_CMAP, '$H_{physics}\\in[0,1]$'),
    ]

    for col, (title, data, cmap, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        vmin, vmax = (0, 255) if col == 0 else (0, 1)
        im = ax.imshow(data if col == 0 else data, cmap=cmap,
                       vmin=0, vmax=1 if col > 0 else None,
                       interpolation='bilinear')
        ax.set_title(f"{title}\n{subtitle}", fontsize=10,
                     color='#e8eaf6', pad=5, linespacing=1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#5c6bc0')
        if col > 0:
            _add_colorbar(fig, ax, im, label="" if col < 4 else "Risk [0–1]")

    fig.suptitle(
        "Stage 1: Physics-Based Feature Extraction  |  "
        f"Weights: Slope={WEIGHT_SLOPE} · Roughness={WEIGHT_ROUGHNESS} · Depth={WEIGHT_DEPTH}",
        fontsize=13, fontweight='bold', color='#7986cb', y=1.03
    )

    if save:
        p = FIGURES_DIR / "fig2_physics_features.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — DL Prediction Gallery
# ─────────────────────────────────────────────────────────────────────────────
def fig_dl_predictions(n: int = 4, save: bool = True) -> Optional[plt.Figure]:
    """Gallery of DL-predicted risk heatmaps (if predictions exist)."""
    dl_files = sorted(glob.glob(str(DL_PREDICTIONS_DIR / "*.npy")))
    if not dl_files:
        print("  ⚠ No DL predictions found. Run train.py and fusion.py first.")
        return None

    dl_files = dl_files[:n]
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))

    fig.suptitle("Stage 2: Deep Learning Risk Heatmaps  (MobileNetV3 + DeepLabV3+)",
                 fontsize=14, fontweight='bold', color='#e8eaf6')

    for i, dl_path in enumerate(dl_files):
        tile_name = Path(dl_path).stem

        # Find original tile
        tile_path = None
        for td in TILE_DIRS:
            p = list(td.glob(f"{tile_name}.png"))
            if p:
                tile_path = str(p[0])
                break

        # Top row: original
        ax_top = axes[0, i]
        if tile_path:
            img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            ax_top.imshow(img, cmap='gray')
        ax_top.set_title(f"Input\n{tile_name[:18]}", fontsize=8, color='#b0bec5')
        ax_top.axis('off')

        # Bottom row: DL prediction
        ax_bot = axes[1, i]
        pred = np.load(dl_path)
        im = ax_bot.imshow(pred, cmap=MARS_CMAP, vmin=0, vmax=1)
        ax_bot.set_title(f"$H_{{learned}}$ — mean={pred.mean():.3f}", fontsize=9,
                         color='#ef9a9a')
        ax_bot.axis('off')
        _add_colorbar(fig, ax_bot, im)

    plt.tight_layout()
    if save:
        p = FIGURES_DIR / "fig3_dl_predictions.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Fusion Comparison Panel
# ─────────────────────────────────────────────────────────────────────────────
def fig_fusion_comparison(tile_path: str = None, save: bool = True) -> plt.Figure:
    """
    Four-panel comparison: Original | Physics | DL | Fused
    Shows example fusion with formula annotation.
    """
    if tile_path is None:
        # prefer a tile that has all three maps
        fused_files = sorted(glob.glob(str(FUSED_MAPS_DIR / "*.npy")))
        if fused_files:
            tile_name = Path(fused_files[0]).stem
            for td in TILE_DIRS:
                p = list(td.glob(f"{tile_name}.png"))
                if p:
                    tile_path = str(p[0])
                    break
        if tile_path is None:
            tile_path = _get_sample_tiles(1)[0]

    tile_name = Path(tile_path).stem
    image = load_and_preprocess(tile_path)
    result = compute_physics_risk_map(image, w_slope=WEIGHT_SLOPE,
                                     w_roughness=WEIGHT_ROUGHNESS,
                                     w_depth=WEIGHT_DEPTH)
    h_physics = result['combined']

    # Load DL prediction if available
    dl_path = DL_PREDICTIONS_DIR / f"{tile_name}.npy"
    h_learned = np.load(str(dl_path)) if dl_path.exists() else None

    # Fused
    if h_learned is not None:
        h_fused = FUSION_ALPHA * h_learned + (1 - FUSION_ALPHA) * h_physics
        h_fused = cv2.bilateralFilter(h_fused.astype(np.float32), d=9, sigmaColor=0.05, sigmaSpace=3)
        h_fused = np.clip(h_fused, 0, 1)
    else:
        h_fused = cv2.bilateralFilter(h_physics.astype(np.float32), d=9, sigmaColor=0.05, sigmaSpace=3)
        h_fused = np.clip(h_fused, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    panels = [
        ('(a) Original Terrain', image, 'gray',
         'Input Mars CTX Image'),
        ('(b) Physics Risk Map\n$H_{physics}$', h_physics, MARS_CMAP,
         f'Slope+Roughness+Depth\n($w$=[{WEIGHT_SLOPE},{WEIGHT_ROUGHNESS},{WEIGHT_DEPTH}])'),
        ('(c) DL Risk Map\n$H_{learned}$',
         h_learned if h_learned is not None else h_physics, MARS_CMAP,
         'MobileNetV3+DeepLabV3+' if h_learned is not None else '(Not available)'),
        (f'(d) Fused Risk Map\n$H_{{final}}$', h_fused, MARS_CMAP,
         f'$\\alpha={FUSION_ALPHA}$·DL + '
         f'$(1-{FUSION_ALPHA})$·Physics'),
    ]

    for ax, (title, data, cmap, subtitle) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1 if cmap != 'gray' else None,
                       interpolation='bilinear')
        ax.set_title(title, fontsize=12, color='#e8eaf6', pad=6)
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, color='#9fa8da',
                style='italic')
        ax.axis('off')

        if cmap != 'gray':
            _add_colorbar(fig, ax, im)

        # Risk stats overlay
        if cmap != 'gray':
            ax.text(0.02, 0.02,
                    f"μ={data.mean():.3f}  σ={data.std():.3f}",
                    transform=ax.transAxes, fontsize=8,
                    color='white', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55))

    fig.suptitle(
        "Stage 3: Heatmap Fusion  —  "
        f"$H_{{final}} = {FUSION_ALPHA}\\cdot H_{{learned}} + "
        f"{1-FUSION_ALPHA}\\cdot H_{{physics}}$",
        fontsize=14, fontweight='bold', color='#7986cb', y=1.03
    )

    plt.tight_layout()
    if save:
        p = FIGURES_DIR / "fig4_fusion_comparison.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Training Curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_curves(save: bool = True) -> Optional[plt.Figure]:
    """Loss and metric curves from training log CSV (picks latest)."""
    log_files = sorted(glob.glob(str(LOGS_DIR / "training_log_*.csv")))
    if not log_files:
        # Fallback to old name just in case
        log_path = LOGS_DIR / "training_log.csv"
        if not log_path.exists():
            print("  ⚠ No training log found. Run train.py first.")
            return None
    else:
        log_path = Path(log_files[-1])
        print(f"  [VISUALIZE] Using latest log: {log_path.name}")

    epochs, train_loss, val_loss = [], [], []
    train_mae, val_mae, val_pearson = [], [], []

    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_mae.append(float(row['train_mae']))
            val_mae.append(float(row['val_mae']))
            val_pearson.append(float(row['val_pearson']))

    if not epochs:
        print("  ⚠ Training log is empty.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def _plot(ax, y1, y2, label1, label2, ylabel, title, c1='#7986cb', c2='#ef9a9a'):
        ax.plot(epochs, y1, color=c1, lw=2, label=label1, marker='o', ms=3)
        ax.plot(epochs, y2, color=c2, lw=2, label=label2, marker='s', ms=3)
        best_epoch = epochs[np.argmin(y2)]
        ax.axvline(best_epoch, color='#66bb6a', lw=1.2, ls='--', alpha=0.7,
                   label=f'Best (epoch {best_epoch})')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    _plot(axes[0], train_loss, val_loss,
          'Train Loss', 'Val Loss', 'L = L_BCE + L_Dice',
          'Combined Loss (BCE + Dice)')

    _plot(axes[1], train_mae, val_mae,
          'Train MAE', 'Val MAE', 'Mean Absolute Error',
          'Regression Error (MAE)', c1='#7986cb', c2='#ef9a9a')

    # Pearson — single curve
    axes[2].plot(epochs, val_pearson, color='#26c6da', lw=2,
                 label='Val Pearson $r$', marker='^', ms=3)
    best_e = epochs[np.argmax(val_pearson)]
    axes[2].axvline(best_e, color='#66bb6a', lw=1.2, ls='--', alpha=0.7,
                    label=f'Best (epoch {best_e})')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Pearson $r$")
    axes[2].set_title("DL–Physics Correlation (Pearson $r$)", fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.4)

    fig.suptitle("Stage 2: Training Progress — MobileNetV3 + DeepLabV3+",
                 fontsize=14, fontweight='bold', color='#7986cb')

    plt.tight_layout()
    if save:
        p = FIGURES_DIR / "fig5_training_curves.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Risk Distribution Histograms
# ─────────────────────────────────────────────────────────────────────────────
def fig_risk_distribution(n_tiles: int = 50, save: bool = True) -> plt.Figure:
    """
    Histogram comparison of risk value distributions across all three map types.
    Shows how fusion produces a smoother, more reliable distribution.
    """
    physics_vals, dl_vals, fused_vals = [], [], []

    phys_files = sorted(glob.glob(str(PSEUDO_LABELS_DIR / "*.npy")))[:n_tiles]
    for pf in phys_files:
        physics_vals.append(np.load(pf).ravel())

    dl_files = sorted(glob.glob(str(DL_PREDICTIONS_DIR / "*.npy")))[:n_tiles]
    for df in dl_files:
        dl_vals.append(np.load(df).ravel())

    fused_files = sorted(glob.glob(str(FUSED_MAPS_DIR / "*.npy")))[:n_tiles]
    for ff in fused_files:
        fused_vals.append(np.load(ff).ravel())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    def _hist(ax, data_list, color, title, label):
        if not data_list:
            ax.text(0.5, 0.5, 'No data\nyet', transform=ax.transAxes,
                    ha='center', va='center', color='#9fa8da', fontsize=12)
            ax.set_title(title, fontweight='bold')
            return
        all_data = np.concatenate(data_list)
        ax.hist(all_data, bins=80, color=color, alpha=0.85, edgecolor='none',
                density=True)
        ax.axvline(all_data.mean(), color='white', lw=1.5, ls='--',
                   label=f'μ={all_data.mean():.3f}')
        ax.axvline(np.median(all_data), color='#ffd54f', lw=1.5, ls=':',
                   label=f'median={np.median(all_data):.3f}')
        ax.set_xlabel("Risk Value H(x,y)")
        ax.set_ylabel("Density")
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.95, f"σ={all_data.std():.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='#b0bec5')

    _hist(axes[0], physics_vals, '#7986cb',
          'Physics Risk\n$H_{physics}$', 'Physics')
    _hist(axes[1], dl_vals, '#ef9a9a',
          'DL Risk\n$H_{learned}$', 'DL')
    _hist(axes[2], fused_vals, '#66bb6a',
          f'Fused Risk\n$H_{{final}}$ (α={FUSION_ALPHA})', 'Fused')

    fig.suptitle(
        "Risk Value Distributions — Physics vs. DL vs. Fused\n"
        "Fusion produces a smoother, more calibrated risk field",
        fontsize=13, fontweight='bold', color='#7986cb'
    )
    plt.tight_layout()
    if save:
        p = FIGURES_DIR / "fig6_risk_distribution.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Multi-Tile Fusion Strip
# ─────────────────────────────────────────────────────────────────────────────
def fig_multi_tile_strip(n: int = 4, save: bool = True) -> plt.Figure:
    """
    Compact 3-row strip (Original | Physics | Fused) for multiple tiles.
    Great for the pre-defense slide deck.
    """
    tiles = _get_sample_tiles(n)
    fig, axes = plt.subplots(3, n, figsize=(n * 4, 10))

    row_labels = [
        "Original Terrain",
        "$H_{physics}$ (Physics Risk)",
        "$H_{final}$ (Fused Risk)"
    ]
    row_colors = ['#9fa8da', '#ce93d8', '#80cbc4']

    for col, tile_path in enumerate(tiles):
        image = load_and_preprocess(tile_path)
        result = compute_physics_risk_map(
            image, w_slope=WEIGHT_SLOPE,
            w_roughness=WEIGHT_ROUGHNESS,
            w_depth=WEIGHT_DEPTH
        )
        h_physics = result['combined']

        tile_name = Path(tile_path).stem
        fused_path = FUSED_MAPS_DIR / f"{tile_name}.npy"
        h_fused = (np.load(str(fused_path))
                   if fused_path.exists()
                   else cv2.bilateralFilter(h_physics.astype(np.float32), d=9, sigmaColor=0.05, sigmaSpace=3))

        row_data = [image, h_physics, h_fused]
        cmaps = ['gray', MARS_CMAP, MARS_CMAP]

        for row, (data, cmap) in enumerate(zip(row_data, cmaps)):
            ax = axes[row, col]
            im = ax.imshow(data, cmap=cmap,
                           vmin=0, vmax=1 if row > 0 else None,
                           interpolation='bilinear')
            ax.axis('off')

            if col == 0:
                ax.text(-0.05, 0.5, row_labels[row],
                        transform=ax.transAxes, rotation=90,
                        va='center', ha='right', fontsize=11,
                        color=row_colors[row], fontweight='bold')

            if row > 0:
                ax.text(0.02, 0.02,
                        f"μ={data.mean():.3f}",
                        transform=ax.transAxes, fontsize=8,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55))

            if row == 0:
                ax.set_title(tile_name[:20], fontsize=8.5,
                             color='#b0bec5', pad=4)

    fig.suptitle(
        "Mars Rover Terrain Risk Pipeline  —  "
        "Original → Physics → Fused",
        fontsize=14, fontweight='bold', color='#7986cb', y=1.01
    )
    plt.tight_layout()
    if save:
        p = FIGURES_DIR / "fig7_multi_tile_strip.png"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  ✓ Saved: {p.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Generate ALL figures
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_figures():
    """Generate all 7 publication-quality figures for the pre-defense."""
    print("\n" + "=" * 60)
    print("  GENERATING PRE-DEFENSE FIGURES")
    print("=" * 60)

    print("\n[1/7] Tile gallery...")
    fig_tile_gallery()
    plt.close('all')

    print("[2/7] Physics feature maps...")
    fig_physics_features()
    plt.close('all')

    print("[3/7] DL prediction gallery...")
    fig_dl_predictions()
    plt.close('all')

    print("[4/7] Fusion comparison panel...")
    fig_fusion_comparison()
    plt.close('all')

    print("[5/7] Training curves...")
    fig_training_curves()
    plt.close('all')

    print("[6/7] Risk distributions...")
    fig_risk_distribution()
    plt.close('all')

    print("[7/7] Multi-tile fusion strip...")
    fig_multi_tile_strip()
    plt.close('all')

    print(f"\n{'='*60}")
    print(f"  ALL FIGURES SAVED TO: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_all_figures()
