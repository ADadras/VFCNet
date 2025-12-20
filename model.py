"""VFCNet: Vector Flow Composition Network.

This module implements the core VFCNet architecture for photographic image
composition classification using Gradient Vector Flow (GVF) features.

Architecture Overview:
1. DINOv3 Backbone: Extracts multi-level visual features (low/mid/high level)
2. GVF Extractor: Computes divergence, curl, and magnitude from flow field
3. Dual-Stream Attention: Fuses baseline GVF and saliency-enhanced GVF
4. Classifier Head: MLP for composition classification
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# =============================================================================
# Multi-Level DINOv3 Feature Extraction
# =============================================================================

class DINOv3MultiLevelWrapper(nn.Module):
    """Wrapper around DINOv3 that extracts features from multiple transformer blocks.
    
    Extracts features from early (low-level), middle, and late (high-level) blocks
    to capture different levels of visual abstraction:
    - Low-level (blocks 2-3): edges, textures, local patterns
    - Mid-level (blocks 5-6): parts, shapes, spatial structure
    - High-level (blocks 10-11): semantic concepts (with optional dropout)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        low_level_blocks: Tuple[int, ...] = (2, 3),
        mid_level_blocks: Tuple[int, ...] = (5, 6),
        high_level_blocks: Tuple[int, ...] = (10, 11),
        high_level_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.low_level_blocks = low_level_blocks
        self.mid_level_blocks = mid_level_blocks
        self.high_level_blocks = high_level_blocks
        self.high_level_dropout = nn.Dropout(high_level_dropout)
        
        # Get embedding dimension from backbone
        self.embed_dim = getattr(backbone, 'embed_dim', 768)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-level features from input image.
        
        Args:
            x: Input tensor of shape (B, 3, H, W), ImageNet-normalized
            
        Returns:
            Dictionary with keys 'low', 'mid', 'high', 'combined' containing
            CLS token features from different transformer block levels.
        """
        all_blocks = sorted(set(
            self.low_level_blocks + self.mid_level_blocks + self.high_level_blocks
        ))
        
        # Use DINOv3's intermediate layer extraction
        if hasattr(self.backbone, 'get_intermediate_layers'):
            intermediate = self.backbone.get_intermediate_layers(
                x, n=all_blocks, reshape=False, return_class_token=True
            )
            block_to_cls = {}
            for block_idx, (patch_tokens, cls_token) in zip(all_blocks, intermediate):
                block_to_cls[block_idx] = cls_token
        else:
            # Fallback: use final features
            final_feat = self.backbone(x)
            if isinstance(final_feat, dict):
                final_feat = final_feat.get('x_norm_clstoken', final_feat.get('x'))
            return {
                'low': final_feat,
                'mid': final_feat,
                'high': self.high_level_dropout(final_feat),
                'combined': final_feat,
            }
        
        def pool_blocks(blocks):
            feats = [block_to_cls[b] for b in blocks if b in block_to_cls]
            if not feats:
                return torch.zeros(x.size(0), self.embed_dim, device=x.device)
            return torch.stack(feats, dim=0).mean(dim=0)
        
        low_feat = pool_blocks(self.low_level_blocks)
        mid_feat = pool_blocks(self.mid_level_blocks)
        high_feat = pool_blocks(self.high_level_blocks)
        
        # Apply dropout to high-level features to reduce semantic dependency
        high_feat_dropped = self.high_level_dropout(high_feat)
        
        return {
            'low': low_feat,
            'mid': mid_feat,
            'high': high_feat_dropped,
            'combined': torch.cat([low_feat, mid_feat, high_feat_dropped], dim=-1),
        }


# =============================================================================
# GVF Feature Extraction
# =============================================================================

class MultiScaleGVFExtractor(nn.Module):
    """Extract GVF features at multiple spatial scales.
    
    Computes differential properties (divergence, curl, magnitude) at different
    resolutions to capture both local and global flow patterns. This is the core
    feature extractor that converts GVF fields into discriminative embeddings.
    
    Key insight from ablation study:
    - Divergence captures convergence/divergence patterns (-8.6% if removed)
    - Magnitude captures flow strength distribution (-7.3% if removed)
    - Curl captures rotational patterns (-4.1% if removed)
    """
    
    def __init__(
        self,
        output_dim: int = 128,
        scales: Tuple[int, ...] = (1, 2, 4),
        use_divergence: bool = True,
        use_curl: bool = True,
        use_magnitude: bool = True,
        pooling: str = "avg",
    ):
        super().__init__()
        self.scales = scales
        self.use_divergence = use_divergence
        self.use_curl = use_curl
        self.use_magnitude = use_magnitude
        
        # Count active channels
        self.num_channels = sum([use_divergence, use_curl, use_magnitude])
        if self.num_channels == 0:
            raise ValueError("At least one of divergence, curl, or magnitude must be enabled")
        
        # Per-scale convolutional feature extractors
        self.scale_convs = nn.ModuleList()
        for _ in scales:
            pool_layer = nn.AdaptiveAvgPool2d(4) if pooling == "avg" else nn.AdaptiveMaxPool2d(4)
            self.scale_convs.append(nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                pool_layer,
                nn.Flatten(),
            ))
        
        # Statistics: 4 per feature per scale
        num_stats = 4 * self.num_channels * len(scales)
        conv_features = 32 * 4 * 4 * len(scales)
        self.proj = nn.Linear(conv_features + num_stats, output_dim)
    
    def _compute_differential_features(
        self, gvf_u: torch.Tensor, gvf_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute divergence, curl, and magnitude from GVF components.
        
        Args:
            gvf_u: Horizontal flow component (B, H, W)
            gvf_v: Vertical flow component (B, H, W)
            
        Returns:
            Tuple of (divergence, curl, magnitude) tensors
        """
        # Divergence: ∂u/∂x + ∂v/∂y (measures convergence/divergence)
        du_dx = gvf_u[:, :, 2:] - gvf_u[:, :, :-2]  # Central difference
        dv_dy = gvf_v[:, 2:, :] - gvf_v[:, :-2, :]
        
        # Curl: ∂v/∂x - ∂u/∂y (measures rotation)
        dv_dx = gvf_v[:, :, 2:] - gvf_v[:, :, :-2]
        du_dy = gvf_u[:, 2:, :] - gvf_u[:, :-2, :]
        
        # Crop to common size
        min_h = min(du_dx.size(1), dv_dy.size(1), dv_dx.size(1), du_dy.size(1))
        min_w = min(du_dx.size(2), dv_dy.size(2), dv_dx.size(2), du_dy.size(2))
        
        divergence = du_dx[:, :min_h, :min_w] + dv_dy[:, :min_h, :min_w]
        curl = dv_dx[:, :min_h, :min_w] - du_dy[:, :min_h, :min_w]
        magnitude = torch.sqrt(
            gvf_u[:, :min_h, :min_w]**2 + gvf_v[:, :min_h, :min_w]**2 + 1e-6
        )
        
        return divergence, curl, magnitude
    
    def forward(self, gvf_u: torch.Tensor, gvf_v: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale GVF features.
        
        Args:
            gvf_u: Horizontal GVF component (B, H, W)
            gvf_v: Vertical GVF component (B, H, W)
            
        Returns:
            Feature tensor of shape (B, output_dim)
        """
        all_conv_features = []
        all_stats = []
        
        for scale_idx, scale in enumerate(self.scales):
            # Downsample if scale > 1
            if scale > 1:
                u_scaled = F.avg_pool2d(gvf_u.unsqueeze(1), kernel_size=scale).squeeze(1)
                v_scaled = F.avg_pool2d(gvf_v.unsqueeze(1), kernel_size=scale).squeeze(1)
            else:
                u_scaled, v_scaled = gvf_u, gvf_v
            
            div, curl, mag = self._compute_differential_features(u_scaled, v_scaled)
            
            # Build channel stack based on configuration
            channels = []
            stat_tensors = []
            
            if self.use_divergence:
                channels.append(div)
                stat_tensors.extend([
                    div.mean(dim=(1, 2)),
                    div.std(dim=(1, 2)),
                    (div > 0).float().mean(dim=(1, 2)),  # Convergence ratio
                    (div < 0).float().mean(dim=(1, 2)),  # Divergence ratio
                ])
            
            if self.use_curl:
                channels.append(curl)
                stat_tensors.extend([
                    curl.mean(dim=(1, 2)),
                    curl.std(dim=(1, 2)),
                    (curl > 0).float().mean(dim=(1, 2)),  # CCW rotation ratio
                    (curl < 0).float().mean(dim=(1, 2)),  # CW rotation ratio
                ])
            
            if self.use_magnitude:
                channels.append(mag)
                stat_tensors.extend([
                    mag.mean(dim=(1, 2)),
                    mag.std(dim=(1, 2)),
                    mag.max(dim=1)[0].max(dim=1)[0],
                    (mag > mag.mean(dim=(1, 2), keepdim=True)).float().mean(dim=(1, 2)),
                ])
            
            field_stack = torch.stack(channels, dim=1)
            all_conv_features.append(self.scale_convs[scale_idx](field_stack))
            all_stats.append(torch.stack(stat_tensors, dim=1))
        
        combined = torch.cat([
            torch.cat(all_conv_features, dim=1),
            torch.cat(all_stats, dim=1)
        ], dim=1)
        
        return self.proj(combined)


# =============================================================================
# Dual-Stream Attention Module
# =============================================================================

class DualStreamGVFAttention(nn.Module):
    """Attention-based fusion of baseline and saliency-enhanced GVF streams.
    
    This module implements the dual-stream attention mechanism that learns to
    weight the contribution of baseline GVF (edge-only) vs saliency-enhanced GVF.
    
    Key insight: The attention mechanism provides robustness - even with suboptimal
    saliency, performance degrades only 5.9-9.7% vs 10-22% for single-stream models.
    """
    
    def __init__(
        self,
        vf_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=vf_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, vf_dim))
        
    def forward(
        self,
        baseline_feat: torch.Tensor,
        saliency_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse baseline and saliency GVF features via attention.
        
        Args:
            baseline_feat: Features from baseline GVF (B, vf_dim)
            saliency_feat: Features from saliency-enhanced GVF (B, vf_dim)
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        B = baseline_feat.size(0)
        
        # Stack as key-value pairs
        gvf_kv = torch.stack([baseline_feat, saliency_feat], dim=1)  # (B, 2, vf_dim)
        
        # Expand learnable query
        query = self.query.expand(B, -1, -1)  # (B, 1, vf_dim)
        
        # Attend to both streams
        attended, weights = self.attention(query, gvf_kv, gvf_kv)
        
        return attended.squeeze(1), weights


# =============================================================================
# Complete VFCNet Model
# =============================================================================

class VFCNet(nn.Module):
    """Vector Flow Composition Network for image composition classification.
    
    Architecture:
    1. DINOv3 backbone extracts multi-level features (low/mid/high)
    2. Two parallel GVF extractors process baseline and saliency-enhanced GVF
    3. Dual-stream attention fuses the two GVF representations
    4. Final classifier combines DINOv3 features with attended GVF features
    
    Args:
        embed_dim: DINOv3 embedding dimension (768 for ViT-B)
        num_classes: Number of composition classes (9 for KU-PCP)
        vf_dim: Vector field feature dimension
        hidden_dim: Classifier hidden dimension
        dropout: Dropout rate
        gvf_scales: Multi-scale pyramid scales
        use_divergence: Include divergence features
        use_curl: Include curl features
        use_magnitude: Include magnitude features
        num_attention_heads: Number of attention heads for GVF fusion
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 9,
        vf_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        gvf_scales: Tuple[int, ...] = (1, 2, 4),
        use_divergence: bool = True,
        use_curl: bool = True,
        use_magnitude: bool = True,
        num_attention_heads: int = 4,
        pooling: str = "avg",
    ):
        super().__init__()
        
        # Store config for debugging/introspection
        self.num_classes = num_classes
        self.use_saliency = True  # Dual-stream always uses saliency
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.vf_dim = vf_dim
        
        # GVF feature extractors for both streams
        self.baseline_gvf_extractor = MultiScaleGVFExtractor(
            output_dim=vf_dim,
            scales=gvf_scales,
            use_divergence=use_divergence,
            use_curl=use_curl,
            use_magnitude=use_magnitude,
            pooling=pooling,
        )
        self.saliency_gvf_extractor = MultiScaleGVFExtractor(
            output_dim=vf_dim,
            scales=gvf_scales,
            use_divergence=use_divergence,
            use_curl=use_curl,
            use_magnitude=use_magnitude,
            pooling=pooling,
        )
        
        # Dual-stream attention for GVF fusion
        self.gvf_attention = DualStreamGVFAttention(
            vf_dim=vf_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Feature fusion (DINO low + mid + high + GVF)
        in_dim = embed_dim * 3 + vf_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.fused_dim = hidden_dim
    
    def forward(
        self,
        dino_features: Dict[str, torch.Tensor],
        baseline_gvf_u: torch.Tensor,
        baseline_gvf_v: torch.Tensor,
        saliency_gvf_u: torch.Tensor,
        saliency_gvf_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VFCNet.
        
        Args:
            dino_features: Dict with 'low', 'mid', 'high' DINOv3 features
            baseline_gvf_u: Baseline GVF horizontal component (B, H, W)
            baseline_gvf_v: Baseline GVF vertical component (B, H, W)
            saliency_gvf_u: Saliency-enhanced GVF horizontal component (B, H, W)
            saliency_gvf_v: Saliency-enhanced GVF vertical component (B, H, W)
            
        Returns:
            Tuple of (logits, fused_features, attention_weights)
        """
        low = dino_features['low']
        mid = dino_features['mid']
        high = dino_features['high']
        
        # Extract GVF features from both streams
        baseline_feat = self.baseline_gvf_extractor(baseline_gvf_u, baseline_gvf_v)
        saliency_feat = self.saliency_gvf_extractor(saliency_gvf_u, saliency_gvf_v)
        
        # Fuse via attention
        gvf_feat, attn_weights = self.gvf_attention(baseline_feat, saliency_feat)
        
        # Combine all features
        concat = torch.cat([low, mid, high, gvf_feat], dim=-1)
        fused = self.fusion(concat)
        logits = self.classifier(fused)
        
        return logits, fused, attn_weights
    
    def get_embeddings(
        self,
        dino_features: Dict[str, torch.Tensor],
        baseline_gvf_u: torch.Tensor,
        baseline_gvf_v: torch.Tensor,
        saliency_gvf_u: torch.Tensor,
        saliency_gvf_v: torch.Tensor,
    ) -> torch.Tensor:
        """Extract fused embeddings for evaluation (e.g., CDA metric).
        
        Returns:
            Fused feature embeddings of shape (B, fused_dim)
        """
        _, fused, _ = self.forward(
            dino_features, baseline_gvf_u, baseline_gvf_v,
            saliency_gvf_u, saliency_gvf_v
        )
        return fused


# =============================================================================
# Single-Stream Variant (for ablation)
# =============================================================================

class VFCNetSingleStream(nn.Module):
    """Single-stream VFCNet variant (no saliency-enhanced GVF).
    
    This simplified variant uses only the baseline GVF without the dual-stream
    attention mechanism. Useful for ablation studies.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 9,
        vf_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        gvf_scales: Tuple[int, ...] = (1, 2, 4),
        use_divergence: bool = True,
        use_curl: bool = True,
        use_magnitude: bool = True,
        pooling: str = "avg",
    ):
        super().__init__()
        
        self.gvf_extractor = MultiScaleGVFExtractor(
            output_dim=vf_dim,
            scales=gvf_scales,
            use_divergence=use_divergence,
            use_curl=use_curl,
            use_magnitude=use_magnitude,
            pooling=pooling,
        )
        
        in_dim = embed_dim * 3 + vf_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.fused_dim = hidden_dim
    
    def forward(
        self,
        dino_features: Dict[str, torch.Tensor],
        gvf_u: torch.Tensor,
        gvf_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            dino_features: Dict with 'low', 'mid', 'high' DINOv3 features
            gvf_u: GVF horizontal component (B, H, W)
            gvf_v: GVF vertical component (B, H, W)
            
        Returns:
            Tuple of (logits, fused_features)
        """
        low = dino_features['low']
        mid = dino_features['mid']
        high = dino_features['high']
        
        gvf_feat = self.gvf_extractor(gvf_u, gvf_v)
        concat = torch.cat([low, mid, high, gvf_feat], dim=-1)
        fused = self.fusion(concat)
        logits = self.classifier(fused)
        
        return logits, fused


def build_vfcnet(config) -> VFCNet:
    """Build VFCNet from configuration.
    
    Args:
        config: VFCNetConfig or ModelConfig instance
        
    Returns:
        VFCNet model instance
    """
    model_config = config.model if hasattr(config, 'model') else config
    
    return VFCNet(
        embed_dim=model_config.embed_dim,
        num_classes=model_config.num_classes,
        vf_dim=model_config.vf_dim,
        hidden_dim=model_config.hidden_dim,
        dropout=model_config.dropout,
        gvf_scales=model_config.gvf_scales,
        use_divergence=model_config.use_divergence,
        use_curl=model_config.use_curl,
        use_magnitude=model_config.use_magnitude,
        num_attention_heads=model_config.num_attention_heads,
        pooling=model_config.pooling,
    )
