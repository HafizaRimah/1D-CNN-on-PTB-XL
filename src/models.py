"""
Neural network architectures for 12-lead ECG classification on PTB-XL:

1. EarlyFusionCNN (Baseline):
   Passes all 12 leads together into the input layer (in_channels=12).
   Standard 1D convolutional feature extractor + global pooling + classifier.

2. LeadAttentionCNN (Lead-Fusion Variant):
   Extracts lead-specific representations using a shared 1D CNN backbone (in_channels=1).
   Computes learnable lead-wise attention weights alpha in R^12 (interpretable).
   Fuses representations via weighted sum before the final classification head.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    Standard 1D Convolutional Block:
    Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
        pool_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class EarlyFusionCNN(nn.Module):
    """
    Baseline Model: Early Fusion 1D CNN.
    
    Treats all 12 ECG leads as input channels from the very first layer.
    Input shape:  (Batch_Size, 12, 1000)
    Output shape: (Batch_Size, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, 12, 1000) -> (B, 32, 500)
            ConvBlock1D(in_channels, base_filters, kernel_size=7, padding=3, pool_size=2, dropout=dropout),
            # Block 2: (B, 32, 500) -> (B, 64, 250)
            ConvBlock1D(base_filters, base_filters * 2, kernel_size=7, padding=3, pool_size=2, dropout=dropout),
            # Block 3: (B, 64, 250) -> (B, 128, 125)
            ConvBlock1D(base_filters * 2, base_filters * 4, kernel_size=5, padding=2, pool_size=2, dropout=dropout),
            # Block 4: (B, 128, 125) -> (B, 256, 62)
            ConvBlock1D(base_filters * 4, base_filters * 8, kernel_size=5, padding=2, pool_size=2, dropout=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 256, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_filters * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 12, 1000)
        feats = self.features(x)                    # (B, 256, 62)
        pooled = self.global_pool(feats).squeeze(-1) # (B, 256)
        logits = self.classifier(pooled)            # (B, 5)
        return logits


class LeadAttentionModule(nn.Module):
    """
    Self-attention / scoring module for fusing 12 lead representations.
    
    Given 12 lead feature vectors of dimension D:
    score_i = v^T tanh(W * feat_i + b)
    alpha = softmax(scores) in R^12
    fused = sum(alpha_i * feat_i) in R^D
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, lead_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lead_features shape: (B, 12, D)
        scores = self.attn_net(lead_features)          # (B, 12, 1)
        weights = F.softmax(scores, dim=1)             # (B, 12, 1), sum over leads = 1
        fused = torch.sum(weights * lead_features, dim=1) # (B, D)
        return fused, weights.squeeze(-1)              # fused: (B, D), weights: (B, 12)


class LeadAttentionCNN(nn.Module):
    """
    Lead-Fusion Variant: Lead-Wise Attention 1D CNN.

    1. Processes each of the 12 leads independently through a shared 1D CNN backbone (in_channels=1).
    2. Computes dynamic attention weights alpha across the 12 leads.
    3. Fuses the 12 lead vectors into a single patient representation via weighted average.
    4. Applies the classification head.
    
    Input shape:  (Batch_Size, 12, 1000)
    Output:
        logits: (Batch_Size, num_classes)
        attn_weights: (Batch_Size, 12) - per-lead attention importance (interpretable)
    """

    def __init__(
        self,
        num_leads: int = 12,
        num_classes: int = 5,
        base_filters: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_leads = num_leads

        # Shared 1D CNN backbone inspecting single-lead waveforms (in_channels=1)
        self.backbone = nn.Sequential(
            # Block 1: (B*12, 1, 1000) -> (B*12, 32, 500)
            ConvBlock1D(1, base_filters, kernel_size=7, padding=3, pool_size=2, dropout=dropout),
            # Block 2: (B*12, 32, 500) -> (B*12, 64, 250)
            ConvBlock1D(base_filters, base_filters * 2, kernel_size=7, padding=3, pool_size=2, dropout=dropout),
            # Block 3: (B*12, 64, 250) -> (B*12, 128, 125)
            ConvBlock1D(base_filters * 2, base_filters * 4, kernel_size=5, padding=2, pool_size=2, dropout=dropout),
            # Block 4: (B*12, 128, 125) -> (B*12, 256, 62)
            ConvBlock1D(base_filters * 4, base_filters * 8, kernel_size=5, padding=2, pool_size=2, dropout=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        feature_dim = base_filters * 8  # 256

        # Learnable attention fusion across leads
        self.attention_fusion = LeadAttentionModule(feature_dim=feature_dim, hidden_dim=64)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (B, 12, 1000)
        B, L, T = x.shape

        # Reshape to treat each lead independently: (B * 12, 1, 1000)
        x_reshaped = x.view(B * L, 1, T)

        # Extract features per lead with shared backbone
        feats = self.backbone(x_reshaped)                    # (B * 12, 256, 62)
        pooled = self.global_pool(feats).view(B, L, -1)      # (B, 12, 256)

        # Fuse 12 leads via attention
        fused, attn_weights = self.attention_fusion(pooled)  # fused: (B, 256), attn: (B, 12)

        # Final classification logits
        logits = self.classifier(fused)                     # (B, 5)

        if return_attention:
            return logits, attn_weights
        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
