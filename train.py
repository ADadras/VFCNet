"""Training script for VFCNet  

Implements the best model from the ablation study:
- Dual-stream GVF architecture with attention fusion
- 10 GVF iterations (optimal from ablation)
- Intensity gradient GVF with saliency force
- All differential features (divergence, curl, magnitude)

Usage:
    python train.py --config config.yaml
    python train.py --epochs 30 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from PIL import Image

from config import VFCNetConfig, PathConfig, GVFConfig, ModelConfig, TrainingConfig
from model import VFCNet, build_vfcnet
from gvf import compute_intensity_gradient_gvf, compute_saliency_enhanced_gvf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# GVF Computation
# =============================================================================

def compute_laplacian(field: np.ndarray) -> np.ndarray:
    """Compute Laplacian using finite differences."""
    return cv2.Laplacian(field.astype(np.float32), cv2.CV_32F)


def compute_gvf_diffusion(fx: np.ndarray, fy: np.ndarray, mu: float = 0.15, 
                          iterations: int = 10) -> tuple:
    """Standard GVF diffusion from gradient field."""
    sq_mag = fx**2 + fy**2
    u = fx.copy().astype(np.float32)
    v = fy.copy().astype(np.float32)
    for _ in range(iterations):
        lap_u = compute_laplacian(u)
        lap_v = compute_laplacian(v)
        u = u + mu * lap_u - (u - fx) * sq_mag
        v = v + mu * lap_v - (v - fy) * sq_mag
    u = u / (np.abs(u).max() + 1e-6)
    v = v / (np.abs(v).max() + 1e-6)
    return u, v


def compute_intensity_gvf(gray: np.ndarray, mu: float = 0.15, iterations: int = 10) -> tuple:
    """GVF from raw intensity gradients."""
    fy, fx = np.gradient(gray.astype(np.float32))
    return compute_gvf_diffusion(fx, fy, mu, iterations)


def compute_intensity_gvf_with_saliency_force(gray: np.ndarray, saliency: np.ndarray,
                                               mu: float = 0.15, beta: float = 0.1,
                                               iterations: int = 10) -> tuple:
    """GVF from intensity gradients with saliency force term."""
    fy, fx = np.gradient(gray.astype(np.float32))
    sq_mag = fx**2 + fy**2
    sal_grad_y, sal_grad_x = np.gradient(saliency.astype(np.float32))
    u = fx.copy().astype(np.float32)
    v = fy.copy().astype(np.float32)
    for _ in range(iterations):
        lap_u = compute_laplacian(u)
        lap_v = compute_laplacian(v)
        u = u + mu * lap_u - (u - fx) * sq_mag + beta * sal_grad_x
        v = v + mu * lap_v - (v - fy) * sq_mag + beta * sal_grad_y
    u = u / (np.abs(u).max() + 1e-6)
    v = v / (np.abs(v).max() + 1e-6)
    return u, v


# =============================================================================
# DINOv3 Multi-Level Wrapper
# =============================================================================

class DINOv3MultiLevelWrapper(nn.Module):
    """Wrapper around DINOv3 that extracts features from multiple transformer blocks.
    
    Uses return_class_token=True to extract proper CLS tokens from specific blocks,
    matching the FlowNet implementation.
    """
    
    def __init__(self, backbone: nn.Module, 
                 low_level_blocks: tuple = (2, 3),
                 mid_level_blocks: tuple = (5, 6),
                 high_level_blocks: tuple = (10, 11),
                 high_level_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.low_level_blocks = low_level_blocks
        self.mid_level_blocks = mid_level_blocks
        self.high_level_blocks = high_level_blocks
        self.high_level_dropout = nn.Dropout(high_level_dropout) if high_level_dropout > 0 else nn.Identity()
        
        # Pre-compute all blocks we need to extract
        self.all_blocks = list(sorted(set(low_level_blocks) | set(mid_level_blocks) | set(high_level_blocks)))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get intermediate features from DINOv3 using return_class_token=True
        # This returns a list of (patch_tokens, cls_token) tuples for each block
        intermediate = self.backbone.get_intermediate_layers(
            x, n=self.all_blocks, reshape=False, return_class_token=True
        )
        
        # Build mapping from block index to CLS token
        block_to_cls = {}
        for block_idx, (patch_tokens, cls_token) in zip(self.all_blocks, intermediate):
            block_to_cls[block_idx] = cls_token  # (B, embed_dim)
        
        # Pool features from each level
        low_features = [block_to_cls[idx] for idx in self.low_level_blocks]
        low = torch.stack(low_features, dim=0).mean(dim=0)
        
        mid_features = [block_to_cls[idx] for idx in self.mid_level_blocks]
        mid = torch.stack(mid_features, dim=0).mean(dim=0)
        
        high_features = [block_to_cls[idx] for idx in self.high_level_blocks]
        high = torch.stack(high_features, dim=0).mean(dim=0)
        high = self.high_level_dropout(high)
        
        return {'low': low, 'mid': mid, 'high': high}


# =============================================================================
# Dataset
# =============================================================================

def _generate_cache_if_needed(cache_dir: Path, rgb_dir: Path, split: str, config: VFCNetConfig):
    """Generate saliency cache if it doesn't exist.
    
    Args:
        cache_dir: Where cache should be stored
        rgb_dir: Where RGB images are located
        split: "train" or "test"
        config: VFCNet configuration
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and list(cache_dir.glob("*.pt")):
        return  # Cache already exists
    
    print(f"\nCache not found at {cache_dir}")
    print("Generating saliency cache (this may take a while on first run)...")
    
    try:
        from generate_cache import generate_kupcp_cache
        generate_kupcp_cache(config, split, overwrite=False)
    except ImportError as e:
        raise RuntimeError(
            f"Cache not found and cannot generate: {e}\n"
            "Please either:\n"
            "  1. Install deepgaze-pytorch: pip install deepgaze-pytorch\n"
            "  2. Or manually generate cache: python generate_cache.py --dataset kupcp"
        )


class KUPCPDualStreamDataset(Dataset):
    """KUPCP with dual-stream GVF (baseline + saliency-force).
    
    Uses cached saliency maps and computes GVF on-the-fly.
    If cache doesn't exist, it will be generated automatically.
    """
    
    def __init__(self, cache_dir: Path, rgb_dir: Path, gvf_size: int = 56,
                 mu: float = 0.15, beta: float = 0.1, iterations: int = 10,
                 config: Optional[VFCNetConfig] = None, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.rgb_dir = Path(rgb_dir)
        self.gvf_size = gvf_size
        self.mu = mu
        self.beta = beta
        self.iterations = iterations
        
        # Auto-generate cache if needed
        if config is not None:
            _generate_cache_if_needed(self.cache_dir, self.rgb_dir, split, config)
        
        self.cache_files = sorted(self.cache_dir.glob("*.pt"))
        if not self.cache_files:
            raise RuntimeError(
                f"No cached tensors in {self.cache_dir}\n"
                "Generate cache first: python generate_cache.py --dataset kupcp"
            )
        
        self.rgb_files = []
        for cf in self.cache_files:
            stem = cf.stem
            for ext in ['.jpg', '.jpeg', '.png']:
                rgb_path = self.rgb_dir / f"{stem}{ext}"
                if rgb_path.exists():
                    self.rgb_files.append(rgb_path)
                    break
            else:
                self.rgb_files.append(self.rgb_dir / f"{stem}.jpg")
        
        print(f"KUPCPDualStreamDataset: {len(self.cache_files)} samples, mu={mu}, iter={iterations}")
    
    def __len__(self):
        return len(self.cache_files)
    
    def __getitem__(self, idx):
        cache_data = torch.load(self.cache_files[idx], map_location="cpu", weights_only=False)
        inputs_normalized = cache_data["inputs"]
        multi_label = cache_data["multi_label"]
        
        # Denormalize saliency (ImageNet normalization)
        saliency_raw = inputs_normalized[0].numpy() * 0.229 + 0.485
        saliency_raw = np.clip(saliency_raw, 0, 1)
        
        # Load grayscale
        rgb_path = self.rgb_files[idx]
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = saliency_raw.copy()
        
        # Compute baseline GVF (intensity gradient only)
        u_base, v_base = compute_intensity_gvf(gray, self.mu, self.iterations)
        
        # Compute saliency-enhanced GVF
        gvf_u_sal, gvf_v_sal = compute_intensity_gvf_with_saliency_force(
            gray, saliency_raw, self.mu, self.beta, self.iterations)
        
        # Rescale to [0, 1] and resize
        base_u_small = cv2.resize((u_base + 1) * 0.5, (self.gvf_size, self.gvf_size))
        base_v_small = cv2.resize((v_base + 1) * 0.5, (self.gvf_size, self.gvf_size))
        sal_u_small = cv2.resize((gvf_u_sal + 1) * 0.5, (self.gvf_size, self.gvf_size))
        sal_v_small = cv2.resize((gvf_v_sal + 1) * 0.5, (self.gvf_size, self.gvf_size))
        
        # DINOv3 input: average GVF stacked with saliency
        avg_u = ((u_base + 1) * 0.5 + (gvf_u_sal + 1) * 0.5) / 2
        avg_v = ((v_base + 1) * 0.5 + (gvf_v_sal + 1) * 0.5) / 2
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        raw_stack = np.stack([saliency_raw, avg_u, avg_v], axis=0)
        normalized = (raw_stack - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        
        return {
            "inputs": torch.from_numpy(normalized).float(),
            "baseline_gvf_u": torch.from_numpy(base_u_small).float(),
            "baseline_gvf_v": torch.from_numpy(base_v_small).float(),
            "saliency_gvf_u": torch.from_numpy(sal_u_small).float(),
            "saliency_gvf_v": torch.from_numpy(sal_v_small).float(),
            "multi_label": multi_label,
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    backbone: nn.Module,
    head: VFCNet,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    freeze_backbone: bool = False,
) -> float:
    """Train for one epoch."""
    head.train()
    if freeze_backbone:
        backbone.eval()
    else:
        backbone.train()
    
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs = batch["inputs"].to(device)
        targets = batch["multi_label"].to(device)
        base_u = batch["baseline_gvf_u"].to(device)
        base_v = batch["baseline_gvf_v"].to(device)
        sal_u = batch["saliency_gvf_u"].to(device)
        sal_v = batch["saliency_gvf_v"].to(device)
        
        with autocast("cuda", enabled=True):
            features = backbone(inputs)
            logits, _, _ = head(features, base_u, base_v, sal_u, sal_v)
            loss = criterion(logits, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    backbone: nn.Module,
    head: VFCNet,
    val_loader: DataLoader,
) -> Tuple[float, float]:
    """Validate model and compute F1 score."""
    head.eval()
    backbone.eval()
    
    tp = fp = fn = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["multi_label"].to(device)
            base_u = batch["baseline_gvf_u"].to(device)
            base_v = batch["baseline_gvf_v"].to(device)
            sal_u = batch["saliency_gvf_u"].to(device)
            sal_v = batch["saliency_gvf_v"].to(device)
            
            features = backbone(inputs)
            logits, _, _ = head(features, base_u, base_v, sal_u, sal_v)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            tp += (preds * targets).sum().item()
            fp += (preds * (1 - targets)).sum().item()
            fn += ((1 - preds) * targets).sum().item()
    
    f1 = (2 * tp) / max(1e-6, 2 * tp + fp + fn)
    return f1


import random

def train(config: VFCNetConfig, args):
    """Main training loop."""
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Config: mu={config.gvf.mu}, iterations={config.gvf.iterations}, beta={config.gvf.beta}")
    
    # Load DINOv3 backbone
    print("Loading DINOv3 backbone...")
    dinov3 = torch.hub.load(
        config.paths.dino_repo, 
        config.paths.dino_model, 
        source="local",
        weights=config.paths.dino_weights,
    ).to(device)
    dinov3.eval()
    
    backbone = DINOv3MultiLevelWrapper(
        backbone=dinov3,
        low_level_blocks=(2, 3),
        mid_level_blocks=(5, 6),
        high_level_blocks=(10, 11),
        high_level_dropout=0.0,
    ).to(device)
    
    # Build model
    head = build_vfcnet(config).to(device)
    print(f"VFCNet head: {sum(p.numel() for p in head.parameters())} parameters")
    
    # Build datasets (will auto-generate cache if needed)
    train_dataset = KUPCPDualStreamDataset(
        cache_dir=config.paths.cache_root / "train",
        rgb_dir=config.paths.kupcp_root / "train_img",
        gvf_size=56,
        mu=config.gvf.mu,
        beta=config.gvf.beta,
        iterations=config.gvf.iterations,
        config=config,
        split="train",
    )
    val_dataset = KUPCPDualStreamDataset(
        cache_dir=config.paths.cache_root / "test",
        rgb_dir=config.paths.kupcp_root / "test_img",
        gvf_size=56,
        mu=config.gvf.mu,
        beta=config.gvf.beta,
        iterations=config.gvf.iterations,
        config=config,
        split="test",
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer
    freeze_backbone = args.freeze_backbone
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        params = list(head.parameters())
    else:
        for param in backbone.parameters():
            param.requires_grad = True
        params = list(backbone.parameters()) + list(head.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")
    
    # Training loop
    best_f1 = 0.0
    best_state = {"head": None, "backbone": None}
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            backbone, head, train_loader, optimizer, criterion, scaler, freeze_backbone
        )
        f1 = validate(backbone, head, val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, F1={f1:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state["head"] = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            if not freeze_backbone:
                best_state["backbone"] = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
            print(f"  -> New best F1: {best_f1:.4f}")
    
    # Restore best model
    head.load_state_dict(best_state["head"])
    if best_state["backbone"]:
        backbone.load_state_dict(best_state["backbone"])
    
    # Save checkpoint
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"vfcnet_seed{args.seed}_f1_{best_f1:.4f}.pt"
    
    torch.save({
        "head": best_state["head"],
        "backbone": best_state["backbone"],
        "config": {
            "mu": config.gvf.mu,
            "beta": config.gvf.beta,
            "iterations": config.gvf.iterations,
        },
        "best_f1": best_f1,
    }, checkpoint_path)
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    return head, backbone, best_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Train VFCNet")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze DINOv3 backbone")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--gvf_iterations", type=int, default=10, help="GVF iterations")
    parser.add_argument("--gvf_mu", type=float, default=0.15, help="GVF mu parameter")
    parser.add_argument("--gvf_beta", type=float, default=0.1, help="GVF saliency beta")
    parser.add_argument("--seed", type=int, default=1111, help="Random seed (1111 for exp reproduction)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load or create config
    config = VFCNetConfig()
    
    # Override from args
    config.gvf.iterations = args.gvf_iterations
    config.gvf.mu = args.gvf_mu
    config.gvf.beta = args.gvf_beta
    
    train(config, args)


if __name__ == "__main__":
    main()
