"""
=============================================================================
 run_pipeline.py — End-to-End Pipeline Orchestrator
 Physics + Learning-Based Terrain Risk Modeling for Mars Rover Navigation
=============================================================================
 Steps:
   1. Generate physics features and pseudo-labels for all tiles
   2. Train DeepLabV3+ on pseudo-labels
   3. Run inference on all tiles → DL predictions
   4. Fuse physics + DL heatmaps → H_final
   5. Generate all publication figures
   6. Print final statistics report
=============================================================================
"""

import time
import argparse
import numpy as np
from pathlib import Path

from config import (
    DEVICE, TILE_DIRS, PSEUDO_LABELS_DIR,
    DL_PREDICTIONS_DIR, FUSED_MAPS_DIR, FIGURES_DIR,
    CHECKPOINTS_DIR, EPOCHS, BATCH_SIZE, FUSION_ALPHA,
    GAT_CHECKPOINTS_DIR, GAT_PREDICTIONS_DIR, GAT_EPOCHS, GAT_BATCH_SIZE
)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   MARS ROVER TERRAIN RISK MODELING PIPELINE                      ║
║   Physics + Deep Learning Hybrid Terrain Risk Estimation         ║
╠══════════════════════════════════════════════════════════════════╣
║   Stage 1  │ Physics Feature Extraction (Slope/Roughness/Depth)  ║
║   Stage 2a │ DeepLabV3+ MobileNetV3-Large CNN Training           ║
║   Stage 2b │ GATv2 Graph Attention Network Training              ║
║   Stage 3  │ Heatmap Fusion  H = α·DL + (1-α)·Physics           ║
╚══════════════════════════════════════════════════════════════════╝
""")


def step_1_pseudo_labels(max_tiles: int = None, skip: bool = False):
    """Step 1: Generate physics-based pseudo-labels."""
    print("\n" + "─" * 70)
    print("  STEP 1: Physics Feature Extraction & Pseudo-Label Generation")
    print("─" * 70)

    if skip:
        import glob
        n = len(glob.glob(str(PSEUDO_LABELS_DIR / "*.npy")))
        print(f"  [SKIP] Found {n} existing pseudo-labels in {PSEUDO_LABELS_DIR}")
        return

    from pseudo_labels import generate_pseudo_labels
    t0 = time.time()
    stats = generate_pseudo_labels(max_tiles=max_tiles)
    print(f"  Done in {time.time()-t0:.1f}s")
    return stats


def step_2_train(max_tiles: int = None, epochs: int = None,
                 batch_size: int = None, skip: bool = False,
                 resume: str = None):
    """Step 2: Train the DeepLabV3+ model."""
    print("\n" + "─" * 70)
    print("  STEP 2: Deep Learning Training (MobileNetV3 + DeepLabV3+)")
    print("─" * 70)

    if skip:
        best = CHECKPOINTS_DIR / "best_model.pth"
        if best.exists():
            print(f"  [SKIP] Found checkpoint: {best}")
        else:
            print("  [SKIP] No checkpoint found — skipping training entirely")
        return

    from train import train
    t0 = time.time()
    history = train(
        max_tiles=max_tiles,
        epochs=epochs,
        batch_size=batch_size,
        resume_from=resume
    )
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return history


def step_2b_train_gat(max_tiles: int = None, epochs: int = None,
                      batch_size: int = None, skip: bool = False,
                      resume: str = None):
    """Step 2b: Train the GATv2 graph attention model."""
    print("\n" + "─" * 70)
    print("  STEP 2b: GATv2 Graph Attention Network Training")
    print("─" * 70)

    if skip:
        best = GAT_CHECKPOINTS_DIR / "best_gat_model.pth"
        if best.exists():
            print(f"  [SKIP] Found GAT checkpoint: {best}")
        else:
            print("  [SKIP] No GAT checkpoint found — skipping GATv2 training")
        return

    from train_gat import train_gat
    t0 = time.time()
    history = train_gat(
        max_tiles=max_tiles,
        epochs=epochs,
        batch_size=batch_size,
        resume_from=resume
    )
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return history


def step_3_inference(max_tiles: int = None, skip: bool = False):
    """Step 3: Run inference on all tiles."""
    print("\n" + "─" * 70)
    print("  STEP 3: DL Inference on All Tiles")
    print("─" * 70)

    best = CHECKPOINTS_DIR / "best_model.pth"
    if not best.exists():
        print("  [SKIP] No trained model found. Run step 2 first.")
        return

    if skip:
        import glob
        n = len(glob.glob(str(DL_PREDICTIONS_DIR / "*.npy")))
        print(f"  [SKIP] Found {n} existing DL predictions")
        return

    from fusion import load_best_model, predict_tiles
    t0 = time.time()
    model = load_best_model()
    predict_tiles(model=model, max_tiles=max_tiles)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")


def step_4_fusion(max_tiles: int = None, alpha: float = None):
    """Step 4: Fuse physics + DL predictions."""
    print("\n" + "─" * 70)
    print(f"  STEP 4: Heatmap Fusion  (α={alpha or FUSION_ALPHA})")
    print("─" * 70)

    from fusion import fuse_all_tiles
    t0 = time.time()
    stats = fuse_all_tiles(max_tiles=max_tiles, alpha=alpha)
    print(f"  Done in {time.time()-t0:.1f}s")
    return stats


def step_5_figures():
    """Step 5: Generate all publication figures."""
    print("\n" + "─" * 70)
    print("  STEP 5: Generating Pre-Defense Figures")
    print("─" * 70)

    from visualize import generate_all_figures
    t0 = time.time()
    generate_all_figures()
    print(f"  Done in {time.time()-t0:.1f}s")


def step_6_report():
    """Step 6: Print final statistics report."""
    import glob

    print("\n" + "═" * 70)
    print("  FINAL PIPELINE REPORT")
    print("═" * 70)

    # Count outputs
    n_tiles_total = sum(
        len(list(td.glob("tile_*.png"))) for td in TILE_DIRS
    )
    n_labels = len(glob.glob(str(PSEUDO_LABELS_DIR / "*.npy")))
    n_preds = len(glob.glob(str(DL_PREDICTIONS_DIR / "*.npy")))
    n_fused = len(glob.glob(str(FUSED_MAPS_DIR / "*.npy")))
    n_figs = len(glob.glob(str(FIGURES_DIR / "*.png")))

    has_model = (CHECKPOINTS_DIR / "best_model.pth").exists()
    has_gat_model = (GAT_CHECKPOINTS_DIR / "best_gat_model.pth").exists()

    # Compute risk statistics on fused maps
    fused_files = glob.glob(str(FUSED_MAPS_DIR / "*.npy"))[:200]
    if fused_files:
        all_risk = np.concatenate([np.load(f).ravel() for f in fused_files])
        risk_mean = all_risk.mean()
        risk_std = all_risk.std()
        risk_high = (all_risk > 0.7).mean() * 100  # % high-risk pixels
        risk_low = (all_risk < 0.3).mean() * 100   # % safe pixels
    else:
        risk_mean = risk_std = risk_high = risk_low = 0.0

    print(f"""
  Dataset
    Total tiles in dataset:     {n_tiles_total:>8,}
    Pseudo-labels generated:    {n_labels:>8,}
    DL predictions:             {n_preds:>8,}
    Fused maps:                 {n_fused:>8,}

  Models
    CNN checkpoint:             {'✓ Found' if has_model else '✗ Not found'}
    Architecture (CNN):         MobileNetV3-Large + DeepLabV3+
    GAT checkpoint:             {'✓ Found' if has_gat_model else '✗ Not found'}
    Architecture (GAT):         GATv2Conv × 3 layers, 4 heads

  Risk Statistics  (from {len(fused_files)} tiles)
    Mean risk H_final:          {risk_mean:>8.4f}
    Std deviation:              {risk_std:>8.4f}
    High-risk pixels (>0.7):    {risk_high:>7.1f}%
    Safe pixels (<0.3):         {risk_low:>7.1f}%

  Output Figures:               {n_figs:>8}
    Location:                   {FIGURES_DIR}

  Device used:                  {DEVICE}
""")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Mars Rover Terrain Risk Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all tiles, 30 epochs)
  python run_pipeline.py

  # Quick demo (500 tiles, 5 epochs)
  python run_pipeline.py --max-tiles 500 --epochs 5

  # Skip training (use existing checkpoint)
  python run_pipeline.py --skip-train

  # Physics + figures only (no DL)
  python run_pipeline.py --skip-train --skip-inference

  # Just regenerate figures
  python run_pipeline.py --figures-only
"""
    )
    parser.add_argument('--max-tiles', type=int, default=None,
                        help="Max tiles to process (None=all, e.g. 500 for demo)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Training epochs (default from config)")
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None,
                        help=f"Fusion weight α (default={FUSION_ALPHA})")
    parser.add_argument('--skip-labels', action='store_true',
                        help="Skip pseudo-label generation if already done")
    parser.add_argument('--skip-train', action='store_true',
                        help="Skip CNN training (use existing checkpoint)")
    parser.add_argument('--skip-gat', action='store_true',
                        help="Skip GATv2 training (use existing checkpoint)")
    parser.add_argument('--skip-inference', action='store_true',
                        help="Skip DL inference step")
    parser.add_argument('--figures-only', action='store_true',
                        help="Only regenerate figures (skip all other steps)")
    parser.add_argument('--gat-epochs', type=int, default=None,
                        help="Training epochs for GATv2")
    parser.add_argument('--gat-batch-size', type=int, default=None,
                        help="Batch size for GATv2")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    print_banner()
    print(f"  Max tiles:  {args.max_tiles or 'ALL'}")
    print(f"  Device:     {DEVICE}")
    print(f"  Fusion α:   {args.alpha or FUSION_ALPHA}")

    t_total = time.time()

    if args.figures_only:
        step_5_figures()
        step_6_report()
    else:
        step_1_pseudo_labels(
            max_tiles=args.max_tiles,
            skip=args.skip_labels
        )
        step_2_train(
            max_tiles=args.max_tiles,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip=args.skip_train,
            resume=args.resume
        )
        step_2b_train_gat(
            max_tiles=args.max_tiles,
            epochs=args.gat_epochs,
            batch_size=args.gat_batch_size,
            skip=args.skip_gat,
        )
        step_3_inference(
            max_tiles=args.max_tiles,
            skip=args.skip_inference or args.skip_train
        )
        step_4_fusion(
            max_tiles=args.max_tiles,
            alpha=args.alpha
        )
        step_5_figures()
        step_6_report()

    elapsed = (time.time() - t_total) / 60
    print(f"\n  ✓ Pipeline complete in {elapsed:.1f} minutes\n")


if __name__ == "__main__":
    main()
