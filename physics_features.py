"""
=============================================================================
 physics_features.py — Physics-Based Feature Extraction
 Stage 1: Slope, Roughness, and Relative Depth Maps
=============================================================================
 Reference: Pre-Defense Report §4
=============================================================================
"""

import cv2
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def compute_slope_map(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    §4.1 — Slope Map (Terrain Gradient Magnitude)
    
    Computes the gradient magnitude using the Sobel operator:
        S(x,y) = √(Gx² + Gy²)
    
    High slope values indicate steep terrain → dangerous for rover traversal.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), dtype float32 in [0, 1].
    ksize : int
        Sobel kernel size (3 or 5).
    
    Returns
    -------
    slope : np.ndarray
        Normalized slope map in [0, 1], shape (H, W).
    """
    # Compute gradients in x and y directions
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Gradient magnitude
    slope = np.sqrt(Gx**2 + Gy**2)
    
    # Normalize to [0, 1]
    s_min, s_max = slope.min(), slope.max()
    if s_max - s_min > 1e-8:
        slope = (slope - s_min) / (s_max - s_min)
    else:
        slope = np.zeros_like(slope)
    
    return slope.astype(np.float32)


def compute_roughness_map(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    §4.2 — Roughness Map (Surface Irregularity via Local Variance)
    
    Computes local variance within a sliding window:
        R(x,y) = Var(I_local) = (1/N) Σ(Ii - μ)²
    
    Higher variance → rougher surface → harder to traverse.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), dtype float32 in [0, 1].
    window_size : int
        Size of the local window for variance computation.
    
    Returns
    -------
    roughness : np.ndarray
        Normalized roughness map in [0, 1], shape (H, W).
    """
    # Local mean: E[X]
    local_mean = uniform_filter(image.astype(np.float64), size=window_size)
    
    # Local mean of squares: E[X²]
    local_sq_mean = uniform_filter(image.astype(np.float64)**2, size=window_size)
    
    # Variance: E[X²] - (E[X])²
    roughness = local_sq_mean - local_mean**2
    roughness = np.maximum(roughness, 0)  # Numerical stability
    
    # Normalize to [0, 1]
    r_min, r_max = roughness.min(), roughness.max()
    if r_max - r_min > 1e-8:
        roughness = (roughness - r_min) / (r_max - r_min)
    else:
        roughness = np.zeros_like(roughness)
    
    return roughness.astype(np.float32)


def compute_depth_map(image: np.ndarray, window_size: int = 31) -> np.ndarray:
    """
    §4.3 — Relative Depth Map (Shadow-Based Depression Detection)
    
    Darker areas relative to local mean indicate depressions or craters:
        D(x,y) = μ_local - I(x,y)
    
    Large positive difference → possible crater or depression.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), dtype float32 in [0, 1].
    window_size : int
        Window size for local mean computation.
    
    Returns
    -------
    depth : np.ndarray
        Normalized depth map in [0, 1], shape (H, W).
    """
    # Local mean
    local_mean = uniform_filter(image.astype(np.float64), size=window_size)
    
    # Relative depth (positive = darker than surroundings)
    depth = local_mean - image.astype(np.float64)
    depth = np.maximum(depth, 0)  # Only depressions (darker regions)
    
    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-8:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)
    
    return depth.astype(np.float32)


def compute_physics_risk_map(
    image: np.ndarray,
    w_slope: float = 0.4,
    w_roughness: float = 0.3,
    w_depth: float = 0.3,
    sobel_ksize: int = 3,
    roughness_window: int = 15,
    depth_window: int = 31
) -> dict:
    """
    §4.4 — Combined Physics Risk Map
    
    Computes all three physics features and combines them:
        H_physics(x,y) = w1·S + w2·R + w3·D
    
    Normalized to [0, 1].
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), dtype float32 in [0, 1].
    w_slope, w_roughness, w_depth : float
        Feature weights (should sum to 1.0).
    sobel_ksize, roughness_window, depth_window : int
        Parameters for each feature extractor.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'slope': Normalized slope map
        - 'roughness': Normalized roughness map
        - 'depth': Normalized depth map
        - 'combined': Combined physics risk map H_physics ∈ [0, 1]
    """
    # Compute individual features
    slope = compute_slope_map(image, ksize=sobel_ksize)
    roughness = compute_roughness_map(image, window_size=roughness_window)
    depth = compute_depth_map(image, window_size=depth_window)
    
    # Weighted combination
    combined = w_slope * slope + w_roughness * roughness + w_depth * depth
    
    # Final normalization to [0, 1]
    c_min, c_max = combined.min(), combined.max()
    if c_max - c_min > 1e-8:
        combined = (combined - c_min) / (c_max - c_min)
    else:
        combined = np.zeros_like(combined)
    
    return {
        'slope': slope,
        'roughness': roughness,
        'depth': depth,
        'combined': combined.astype(np.float32)
    }


def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load a terrain tile and convert to float32 grayscale in [0, 1].
    
    Parameters
    ----------
    image_path : str or Path
        Path to the PNG tile.
    
    Returns
    -------
    image : np.ndarray
        Grayscale image, shape (H, W), dtype float32 in [0, 1].
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img.astype(np.float32) / 255.0


# ─────────────────────────── CLI Test ───────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import TILE_DIRS, SOBEL_KSIZE, ROUGHNESS_WINDOW, DEPTH_WINDOW
    from config import WEIGHT_SLOPE, WEIGHT_ROUGHNESS, WEIGHT_DEPTH
    
    # Load a sample tile
    import glob
    sample_tiles = sorted(glob.glob(str(TILE_DIRS[0] / "tile_x0005_y0040_*.png")))
    if not sample_tiles:
        sample_tiles = sorted(glob.glob(str(TILE_DIRS[0] / "tile_*.png")))
    
    sample_path = sample_tiles[0]
    print(f"Processing: {sample_path}")
    
    image = load_and_preprocess(sample_path)
    result = compute_physics_risk_map(
        image,
        w_slope=WEIGHT_SLOPE,
        w_roughness=WEIGHT_ROUGHNESS,
        w_depth=WEIGHT_DEPTH,
        sobel_ksize=SOBEL_KSIZE,
        roughness_window=ROUGHNESS_WINDOW,
        depth_window=DEPTH_WINDOW
    )
    
    # Quick visualization
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ['Original', 'Slope', 'Roughness', 'Depth', 'Combined Risk']
    maps = [image, result['slope'], result['roughness'], result['depth'], result['combined']]
    cmaps = ['gray', 'magma', 'magma', 'magma', 'inferno']
    
    for ax, title, data, cmap in zip(axes, titles, maps, cmaps):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Stage 1: Physics-Based Feature Extraction", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/physics_features_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Done! Saved to outputs/figures/physics_features_test.png")
