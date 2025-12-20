"""Generate saliency cache for VFCNet training and evaluation.

This script generates the saliency cache files required by VFCNet.
It uses DeepGaze II for saliency prediction and creates .pt files
containing pre-computed saliency maps and labels.

Usage:
    # Generate KUPCP cache (training)
    python generate_cache.py --dataset kupcp --split train
    python generate_cache.py --dataset kupcp --split test
    
    # Generate PICD cache (evaluation)
    python generate_cache.py --dataset picd

Requirements:
    pip install deepgaze-pytorch scipy
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
from tqdm.auto import tqdm

from config import VFCNetConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global saliency model (lazy loaded)
_saliency_model = None
_centerbias_template = None


def load_saliency_model(centerbias_path: Optional[str] = None):
    """Load DeepGaze IIE model and centerbias template."""
    global _saliency_model, _centerbias_template
    
    if _saliency_model is None:
        try:
            from deepgaze_pytorch import DeepGazeIIE
        except ImportError:
            raise ImportError(
                "deepgaze-pytorch is required for cache generation. "
                "Install with: pip install deepgaze-pytorch"
            )
        
        print("Loading DeepGaze IIE model...")
        _saliency_model = DeepGazeIIE(pretrained=True).to(device).eval()
        
        # Load or create centerbias template
        if centerbias_path and Path(centerbias_path).exists():
            _centerbias_template = np.load(centerbias_path)
            print(f"Loaded centerbias from {centerbias_path}")
        else:
            # Create default centerbias (gaussian centered)
            print("Creating default centerbias template...")
            size = 1024
            y, x = np.mgrid[0:size, 0:size]
            center = size / 2
            sigma = size / 4
            _centerbias_template = -((x - center)**2 + (y - center)**2) / (2 * sigma**2)
            _centerbias_template = _centerbias_template.astype(np.float32)
        
        print(f"Saliency model loaded on {device}")
    
    return _saliency_model, _centerbias_template


def _resize_centerbias(template: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize centerbias template to match image dimensions."""
    scale_y = h / template.shape[0]
    scale_x = w / template.shape[1]
    centerbias = zoom(template, (scale_y, scale_x), order=0, mode="nearest")
    centerbias = centerbias - logsumexp(centerbias)
    return centerbias


def compute_saliency(image: np.ndarray, centerbias_path: Optional[str] = None) -> np.ndarray:
    """Compute saliency map using DeepGaze IIE.
    
    Args:
        image: RGB image as numpy array (H, W, 3) in range [0, 255]
        centerbias_path: Optional path to centerbias template
        
    Returns:
        Saliency map as numpy array (H, W) in range [0, 1]
    """
    model, cb_template = load_saliency_model(centerbias_path)
    
    h, w = image.shape[:2]
    
    # Convert to tensor (expects float in [0, 255])
    image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Resize centerbias to match image
    centerbias = _resize_centerbias(cb_template, h, w)
    centerbias_tensor = torch.from_numpy(centerbias).unsqueeze(0).float().to(device)
    
    # Compute saliency
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            log_density = model(image_tensor, centerbias_tensor)
    
    saliency = log_density.exp().detach().squeeze().cpu().numpy()
    
    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


def prepare_cache_tensor(image: np.ndarray, saliency: np.ndarray) -> torch.Tensor:
    """Prepare the 3-channel input tensor for caching.
    
    The tensor contains:
    - Channel 0: Saliency map (normalized with ImageNet stats)
    - Channel 1: Saliency map (duplicate for compatibility)
    - Channel 2: Saliency map (duplicate for compatibility)
    
    All channels are normalized with ImageNet mean/std.
    """
    # Resize saliency to 224x224
    saliency_resized = cv2.resize(saliency, (224, 224))
    
    # Stack to 3 channels
    saliency_3ch = np.stack([saliency_resized] * 3, axis=0)
    
    # Normalize with ImageNet stats
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    
    normalized = (saliency_3ch - imagenet_mean) / imagenet_std
    
    return torch.from_numpy(normalized).float()


# =============================================================================
# KUPCP Dataset Processing
# =============================================================================

# KUPCP composition classes
KUPCP_CLASSES = [
    "RoT", "Hor", "Sym", "Dia", "Cur", "VL", "Tri", "Pat", "BoC"
]

def parse_kupcp_labels(label_str: str) -> torch.Tensor:
    """Parse KUPCP multi-label string to tensor.
    
    KUPCP labels are like: "1,0,0,1,0,0,0,0,0"
    """
    parts = label_str.strip().split(',')
    labels = [int(p) for p in parts]
    return torch.tensor(labels, dtype=torch.float32)


def load_kupcp_split(kupcp_root: Path, split: str) -> List[Dict]:
    """Load KUPCP train or test split.
    
    Args:
        kupcp_root: Path to KUPCP dataset
        split: "train" or "test"
        
    Returns:
        List of dicts with 'image_path', 'labels', 'image_name'
    """
    if split == "train":
        img_dir = kupcp_root / "train_img"
        label_file = kupcp_root / "train_label.txt"
    else:
        img_dir = kupcp_root / "test_img"
        label_file = kupcp_root / "test_label.txt"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    samples = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                img_name = parts[0]
                label_str = parts[1]
                
                # Find image file
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    candidate = img_dir / f"{img_name}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path is None:
                    # Try without extension (filename might include it)
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        continue
                
                samples.append({
                    "image_path": img_path,
                    "labels": parse_kupcp_labels(label_str),
                    "image_name": img_name,
                })
    
    print(f"Loaded {len(samples)} samples from KUPCP {split}")
    return samples


def generate_kupcp_cache(config: VFCNetConfig, split: str, overwrite: bool = False):
    """Generate saliency cache for KUPCP dataset.
    
    Args:
        config: VFCNet configuration
        split: "train" or "test"
        overwrite: Whether to overwrite existing cache files
    """
    kupcp_root = Path(config.paths.kupcp_root)
    cache_root = Path(config.paths.cache_root)
    
    # Create cache directory
    cache_dir = cache_root / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    samples = load_kupcp_split(kupcp_root, split)
    
    print(f"\nGenerating cache for {len(samples)} images...")
    print(f"Output directory: {cache_dir}")
    
    for i, sample in enumerate(tqdm(samples, desc=f"KUPCP {split}")):
        # Output filename (4-digit zero-padded)
        out_path = cache_dir / f"{i+1:04d}.pt"
        
        if out_path.exists() and not overwrite:
            continue
        
        # Load image
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            print(f"Warning: Could not load {sample['image_path']}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 for saliency computation
        image_resized = cv2.resize(image, (224, 224))
        
        # Compute saliency
        saliency = compute_saliency(image_resized, config.paths.centerbias_path)
        
        # Prepare tensor
        inputs = prepare_cache_tensor(image_resized, saliency)
        
        # Save cache
        cache_data = {
            "inputs": inputs,
            "multi_label": sample["labels"],
            "image_name": sample["image_name"],
        }
        torch.save(cache_data, out_path)
    
    print(f"Cache generation complete: {cache_dir}")


# =============================================================================
# PICD Dataset Processing
# =============================================================================

# PICD composition classes (matching the folder structure)
PICD_CLASSES = [
    "DENSE", "DIA", "DIFFUSE", "HORI2", "HORI3", "LINE_VERTI2", "LINE_VERTI3",
    "LINE_VERTI_MANY", "PATTERN", "PERSPECTIVE", "POINT_1_ROT", "POINT_MULTI_DIA",
    "POINT_MULTI_HORI", "POINT_MULTI_TRI", "POINT_MULTI_VERTI", "POINT_SHAPE_CENT",
    "SCATTER", "SHAPE_VERTI_AVERAGE", "SHAPE_VERTI_MID", "SHAPE_VERTI_ONESIDE",
    "SPECIAL_C", "SPECIAL_O", "SPECIAL_S", "SPECIAL_TRIANGLE"
]


def load_picd_images(picd_root: Path) -> List[Dict]:
    """Load all PICD images from subfolders.
    
    PICD structure:
        picd_root/
        ├── single_labels/
        │   ├── HORI2/
        │   │   ├── 000001.jpg
        │   └── ...
        ├── multi_labels/
        │   └── ...
        └── public/
            └── ...
    """
    samples = []
    
    for subset in ["single_labels", "multi_labels", "public"]:
        subset_dir = picd_root / subset
        if not subset_dir.exists():
            continue
        
        for class_dir in subset_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            for img_path in class_dir.glob("*.jpg"):
                samples.append({
                    "image_path": img_path,
                    "class_name": class_name,
                    "subset": subset,
                    "image_name": img_path.name,
                })
    
    print(f"Found {len(samples)} images in PICD")
    return samples


def generate_picd_cache(config: VFCNetConfig, overwrite: bool = False):
    """Generate saliency cache for PICD dataset.
    
    Args:
        config: VFCNet configuration
        overwrite: Whether to overwrite existing cache files
    """
    picd_root = Path(config.paths.picd_root)
    cache_root = Path(config.paths.picd_cache_root)
    
    # Load all images
    samples = load_picd_images(picd_root)
    
    print(f"\nGenerating cache for {len(samples)} images...")
    print(f"Output directory: {cache_root}")
    
    for sample in tqdm(samples, desc="PICD"):
        # Mirror the source structure in cache
        rel_path = sample["image_path"].relative_to(picd_root)
        out_path = cache_root / rel_path.parent / f"{rel_path.stem}.pt"
        
        if out_path.exists() and not overwrite:
            continue
        
        # Create output directory
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            print(f"Warning: Could not load {sample['image_path']}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 for saliency computation
        image_resized = cv2.resize(image, (224, 224))
        
        # Compute saliency
        saliency = compute_saliency(image_resized, config.paths.centerbias_path)
        
        # Prepare tensor (PICD doesn't have multi-label, use class index)
        inputs = prepare_cache_tensor(image_resized, saliency)
        
        # For PICD, we don't have multi-label, just store inputs
        cache_data = {
            "inputs": inputs,
            "multi_label": torch.zeros(9),  # Placeholder
            "image_name": sample["image_name"],
            "class_name": sample["class_name"],
        }
        torch.save(cache_data, out_path)
    
    print(f"Cache generation complete: {cache_root}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate saliency cache for VFCNet")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["kupcp", "picd"], help="Dataset to process")
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "test"], help="Split for KUPCP")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing cache files")
    parser.add_argument("--kupcp_root", type=str, default=None,
                       help="Override KUPCP root path")
    parser.add_argument("--picd_root", type=str, default=None,
                       help="Override PICD root path")
    parser.add_argument("--cache_root", type=str, default=None,
                       help="Override cache output path")
    parser.add_argument("--centerbias", type=str, default=None,
                       help="Path to centerbias .npy file (optional)")
    args = parser.parse_args()
    
    # Load config
    config = VFCNetConfig()
    
    # Override paths from args
    if args.kupcp_root:
        config.paths.kupcp_root = Path(args.kupcp_root)
    if args.picd_root:
        config.paths.picd_root = Path(args.picd_root)
    if args.cache_root:
        if args.dataset == "kupcp":
            config.paths.cache_root = Path(args.cache_root)
        else:
            config.paths.picd_cache_root = Path(args.cache_root)
    if args.centerbias:
        config.paths.centerbias_path = Path(args.centerbias)
    
    print(f"Device: {device}")
    
    if args.dataset == "kupcp":
        if args.split is None:
            # Generate both splits
            generate_kupcp_cache(config, "train", args.overwrite)
            generate_kupcp_cache(config, "test", args.overwrite)
        else:
            generate_kupcp_cache(config, args.split, args.overwrite)
    else:
        generate_picd_cache(config, args.overwrite)


if __name__ == "__main__":
    main()
