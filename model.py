"""
=============================================================================
 model.py — DeepLabV3+ with MobileNetV3-Large for Terrain Risk Estimation
 Stage 2: Deep Learning Risk Heatmap Prediction
=============================================================================
 Reference: Pre-Defense Report §5
 
 Architecture:
   Encoder:  MobileNetV3-Large (pretrained on ImageNet)
   Decoder:  DeepLabV3+ with ASPP (Atrous Spatial Pyramid Pooling)
   Output:   H_learned(x,y) ∈ [0, 1]  (single-channel sigmoid)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

from config import PRETRAINED_BACKBONE, DEVICE


class TerrainRiskModel(nn.Module):
    """
    DeepLabV3+ with MobileNetV3-Large backbone for pixel-wise risk estimation.
    
    Modifications from standard DeepLabV3:
    1. Input: 1-channel grayscale → replicate to 3 channels for pretrained weights
    2. Output: 21 classes → 1 channel (continuous risk score)
    3. Final activation: Sigmoid → output ∈ [0, 1]
    
    Parameters
    ----------
    pretrained : bool
        Whether to use ImageNet-pretrained backbone weights.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained DeepLabV3 with MobileNetV3-Large
        if pretrained and PRETRAINED_BACKBONE:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self.model = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=True)
            print("[MODEL] Loaded pretrained MobileNetV3-Large backbone")
        else:
            self.model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
            print("[MODEL] Initialized from scratch (no pretrained weights)")
        
        # Modify the classifier head: 21 classes → 1 channel
        # DeepLabV3 classifier is at model.classifier[-1]
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(
            in_channels, 1, kernel_size=1
        )
        
        # Also modify the auxiliary classifier if it exists
        if self.model.aux_classifier is not None:
            aux_in_channels = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(
                aux_in_channels, 1, kernel_size=1
            )
        
        # Sigmoid for [0, 1] output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) — grayscale.
        
        Returns
        -------
        risk_map : torch.Tensor
            Risk heatmap of shape (B, 1, H, W), values in [0, 1].
        """
        # Replicate grayscale to 3 channels for pretrained backbone
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Get input spatial dimensions
        input_h, input_w = x.shape[2], x.shape[3]
        
        # Forward through DeepLabV3
        output = self.model(x)
        logits = output['out']  # (B, 1, H', W')
        
        # Upsample to input resolution if needed
        if logits.shape[2] != input_h or logits.shape[3] != input_w:
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode='bilinear', align_corners=False
            )
        
        # Apply sigmoid activation
        risk_map = self.sigmoid(logits)
        
        return risk_map
    
    def get_aux_output(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning both main and auxiliary outputs (for training).
        
        Returns
        -------
        outputs : dict
            - 'out': Main output (B, 1, H, W)
            - 'aux': Auxiliary output (B, 1, H, W) or None
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        input_h, input_w = x.shape[2], x.shape[3]
        
        output = self.model(x)
        
        main = output['out']
        if main.shape[2] != input_h or main.shape[3] != input_w:
            main = F.interpolate(
                main, size=(input_h, input_w),
                mode='bilinear', align_corners=False
            )
        main = self.sigmoid(main)
        
        aux = None
        if 'aux' in output and output['aux'] is not None:
            aux = output['aux']
            if aux.shape[2] != input_h or aux.shape[3] != input_w:
                aux = F.interpolate(
                    aux, size=(input_h, input_w),
                    mode='bilinear', align_corners=False
                )
            aux = self.sigmoid(aux)
        
        return {'out': main, 'aux': aux}


# ─────────────────────────── Loss Functions ─────────────────────────────────

class DiceLoss(nn.Module):
    """
    §5 — Dice Loss for overlap measurement.
    
        L_Dice = 1 - (2|P∩G|) / (|P| + |G| + smooth)
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    §5 — Combined Loss: L = L_BCE + L_Dice
    
    Binary Cross-Entropy + Dice Loss for training the risk estimation model.
    """
    
    def __init__(self, dice_smooth: float = 1.0, aux_weight: float = 0.4):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=dice_smooth)
        self.aux_weight = aux_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_pred: torch.Tensor = None
    ) -> dict:
        """
        Compute combined loss.
        
        Returns dict with 'total', 'bce', 'dice', and optionally 'aux' losses.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        total_loss = bce_loss + dice_loss
        
        result = {
            'bce': bce_loss,
            'dice': dice_loss,
            'total': total_loss
        }
        
        # Auxiliary loss for deep supervision
        if aux_pred is not None:
            aux_bce = self.bce(aux_pred, target)
            aux_dice = self.dice(aux_pred, target)
            aux_loss = aux_bce + aux_dice
            result['aux'] = aux_loss
            result['total'] = total_loss + self.aux_weight * aux_loss
        
        return result


def build_model(pretrained: bool = True) -> TerrainRiskModel:
    """
    Factory function to create and prepare the model.
    
    Returns
    -------
    model : TerrainRiskModel
        Model moved to the configured device.
    """
    model = TerrainRiskModel(pretrained=pretrained)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters:     {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    print(f"[MODEL] Device: {DEVICE}")
    
    return model


# ─────────────────────────── CLI Test ───────────────────────────────────────
if __name__ == "__main__":
    print("Building model...")
    model = build_model(pretrained=True)
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 512, 512).to(DEVICE)
    print(f"\nInput shape:  {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with aux output
    outputs = model.get_aux_output(dummy_input)
    print(f"\nMain output:  {outputs['out'].shape}")
    if outputs['aux'] is not None:
        print(f"Aux output:   {outputs['aux'].shape}")
    
    # Test loss
    criterion = CombinedLoss()
    target = torch.rand(2, 1, 512, 512).to(DEVICE)
    losses = criterion(outputs['out'], target, outputs['aux'])
    print(f"\nTotal loss: {losses['total']:.4f}")
    print(f"BCE loss:   {losses['bce']:.4f}")
    print(f"Dice loss:  {losses['dice']:.4f}")
    if 'aux' in losses:
        print(f"Aux loss:   {losses['aux']:.4f}")
    
    print("\n✓ Model test passed!")
