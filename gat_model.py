"""
=============================================================================
 gat_model.py — GATv2 (Graph Attention Network v2) for Terrain Risk
 Stage 2b: Graph-Based Deep Learning Risk Estimation
=============================================================================
 Reference: Pre-Defense Report §5b

 Architecture:
   Input:   Node features [slope, roughness, depth, intensity, x, y]
   Layers:  Multiple GATv2Conv layers with multi-head attention
   Output:  Per-node risk score ∈ [0, 1]  (sigmoid activation)

 GATv2 Advantage over GAT:
   - "Dynamic attention": attention is computed AFTER the linear
     transformation, making it strictly more expressive than GATv1.
   - Reference: Brody et al., "How Attentive are Graph Attention
     Networks?" (ICLR 2022)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, BatchNorm
except ImportError:
    raise ImportError(
        "torch_geometric is required for GATv2. "
        "Install with: pip install torch-geometric"
    )

from config import (
    DEVICE, GAT_INPUT_DIM, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM,
    GAT_NUM_HEADS, GAT_NUM_LAYERS, GAT_DROPOUT
)


class TerrainGATv2Model(nn.Module):
    """
    Graph Attention Network v2 for per-node terrain risk estimation.

    Architecture:
        Input Linear → [GATv2Conv + BatchNorm + ELU + Dropout] × L → Output Linear → Sigmoid

    Each GATv2Conv layer uses multi-head attention to learn which
    neighbouring terrain patches are most relevant for predicting
    risk at the current node.

    Parameters
    ----------
    in_channels : int
        Number of input node features (default: 6).
    hidden_channels : int
        Hidden dimension per attention head (default: 64).
    out_channels : int
        Output dimension per node (default: 1 = risk score).
    num_heads : int
        Number of attention heads (default: 4).
    num_layers : int
        Number of GATv2Conv layers (default: 3).
    dropout : float
        Dropout rate for attention weights and features (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = None,
        out_channels: int = None,
        num_heads: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()

        in_channels = in_channels or GAT_INPUT_DIM
        hidden_channels = hidden_channels or GAT_HIDDEN_DIM
        out_channels = out_channels or GAT_OUTPUT_DIM
        num_heads = num_heads or GAT_NUM_HEADS
        num_layers = num_layers or GAT_NUM_LAYERS
        dropout = dropout or GAT_DROPOUT

        self.dropout = dropout
        self.num_layers = num_layers

        # ── Input projection ────────────────────────────────────────────
        # Project raw features to hidden dimension
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ── GATv2 Layers ────────────────────────────────────────────────
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # First layer takes hidden_channels, subsequent layers take
            # hidden_channels * num_heads (because of head concatenation)
            if i == 0:
                layer_in = hidden_channels
            else:
                layer_in = hidden_channels * num_heads

            # Last GATv2 layer: average heads instead of concatenating
            if i == num_layers - 1:
                gat = GATv2Conv(
                    in_channels=layer_in,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,       # Average instead of concatenate
                    dropout=dropout,
                    add_self_loops=True,
                    share_weights=False,
                )
            else:
                gat = GATv2Conv(
                    in_channels=layer_in,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=True,        # Concatenate heads
                    dropout=dropout,
                    add_self_loops=True,
                    share_weights=False,
                )

            self.gat_layers.append(gat)

            # BatchNorm dimension depends on concat mode
            if i == num_layers - 1:
                self.batch_norms.append(BatchNorm(hidden_channels))
            else:
                self.batch_norms.append(BatchNorm(hidden_channels * num_heads))

        # ── Output head ─────────────────────────────────────────────────
        # Map from hidden_channels → 1 (risk score)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through GATv2.

        Parameters
        ----------
        x : torch.Tensor, shape [N, in_channels]
            Node feature matrix.
        edge_index : torch.Tensor, shape [2, E]
            Graph connectivity in COO format.
        return_attention : bool
            If True, also return attention weights from the last layer.

        Returns
        -------
        risk : torch.Tensor, shape [N, 1]
            Per-node risk score in [0, 1].
        attention_weights : tuple (optional)
            (edge_index, alpha) from last GATv2 layer.
        """
        # Input projection
        h = self.input_proj(x)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GATv2 message passing layers
        attn_weights = None
        for i in range(self.num_layers):
            if return_attention and i == self.num_layers - 1:
                h, attn_weights = self.gat_layers[i](
                    h, edge_index, return_attention_weights=True
                )
            else:
                h = self.gat_layers[i](h, edge_index)

            h = self.batch_norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Output prediction
        risk = self.output_head(h)
        risk = self.sigmoid(risk)

        if return_attention and attn_weights is not None:
            return risk, attn_weights
        return risk


# ─────────────────────────── Loss Function ──────────────────────────────────

class GraphRiskLoss(nn.Module):
    """
    Combined MSE + Smooth L1 loss for node-level risk regression.

    L = λ₁·MSE(pred, target) + λ₂·SmoothL1(pred, target)
    """

    def __init__(self, mse_weight: float = 1.0, smooth_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss.

        Parameters
        ----------
        pred : torch.Tensor, shape [N, 1] or [N]
        target : torch.Tensor, shape [N]

        Returns
        -------
        losses : dict with 'total', 'mse', 'smooth_l1' keys.
        """
        pred = pred.squeeze(-1)  # [N]

        mse_loss = self.mse(pred, target)
        smooth_loss = self.smooth_l1(pred, target)
        total = self.mse_weight * mse_loss + self.smooth_weight * smooth_loss

        return {
            'total': total,
            'mse': mse_loss,
            'smooth_l1': smooth_loss,
        }


def build_gat_model(**kwargs) -> TerrainGATv2Model:
    """
    Factory function to create and prepare the GATv2 model.

    Returns
    -------
    model : TerrainGATv2Model
        Model moved to the configured device.
    """
    model = TerrainGATv2Model(**kwargs)
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GAT-MODEL] Architecture: GATv2Conv × {model.num_layers}")
    print(f"[GAT-MODEL] Total parameters:     {total_params:,}")
    print(f"[GAT-MODEL] Trainable parameters: {trainable_params:,}")
    print(f"[GAT-MODEL] Device: {DEVICE}")

    return model


# ─────────────────────────── CLI Test ───────────────────────────────────────
if __name__ == "__main__":
    from graph_utils import GridGraphBuilder
    import numpy as np

    print("=" * 60)
    print(" GATv2 Model — Architecture Test")
    print("=" * 60)

    # Build model
    model = build_gat_model()
    print(f"\n{model}")

    # Create a dummy graph
    builder = GridGraphBuilder()
    dummy_image = np.random.rand(512, 512).astype(np.float32)
    dummy_slope = np.random.rand(512, 512).astype(np.float32)
    dummy_roughness = np.random.rand(512, 512).astype(np.float32)
    dummy_depth = np.random.rand(512, 512).astype(np.float32)
    dummy_target = np.random.rand(512, 512).astype(np.float32)

    data = builder.build_graph(
        dummy_image, dummy_slope, dummy_roughness, dummy_depth, dummy_target
    )

    # Move to device
    data = data.to(DEVICE)

    # Forward pass
    with torch.no_grad():
        risk = model(data.x, data.edge_index)
    print(f"\nInput:  x={data.x.shape}, edges={data.edge_index.shape}")
    print(f"Output: risk={risk.shape}")
    print(f"Range:  [{risk.min():.4f}, {risk.max():.4f}]")

    # Test with attention weights
    with torch.no_grad():
        risk, (edge_idx, alpha) = model(
            data.x, data.edge_index, return_attention=True
        )
    print(f"\nAttention weights: {alpha.shape}")
    print(f"  Mean α: {alpha.mean():.4f}, Max α: {alpha.max():.4f}")

    # Test loss
    criterion = GraphRiskLoss()
    losses = criterion(risk, data.y)
    print(f"\nLoss: total={losses['total']:.4f}, "
          f"mse={losses['mse']:.4f}, smooth_l1={losses['smooth_l1']:.4f}")

    print("\n✓ GATv2 model test passed!")
