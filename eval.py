"""Self-contained evaluation script for VFCNet on PICD.

Replicates the exact CDA evaluation logic from PICD_eval.py for standalone use.

Evaluates trained VFCNet on PICD dataset with:
- CDA-1: Composition Discrimination Accuracy (5 seeds, 12 triplets per pair)
- CDA-2: Semantic Robustness (2760 triplets)
- 50-triplet stable estimates for both
- DBI (Davies-Bouldin Index)
- Silhouette Score
- Per-class accuracy

Usage:
    python eval.py --checkpoint output/checkpoints/vfcnet_best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from config import VFCNetConfig
from model import VFCNet, build_vfcnet


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
# PICD Dataset with Dual-Stream GVF
# =============================================================================

def _generate_picd_cache_if_needed(cache_dir: Path, rgb_root: Path, config):
    """Generate PICD saliency cache if it doesn't exist.
    
    Args:
        cache_dir: Where cache should be stored
        rgb_root: Where RGB images are located
        config: VFCNet configuration or None
    """
    cache_dir = Path(cache_dir)
    
    # Check if cache exists by looking for .pt files in subdirectories
    has_cache = False
    for subdir in ["single_labels", "multi_labels", "public"]:
        subpath = cache_dir / subdir
        if subpath.exists():
            for cat_dir in subpath.iterdir():
                if cat_dir.is_dir() and list(cat_dir.glob("*.pt")):
                    has_cache = True
                    break
        if has_cache:
            break
    
    if has_cache:
        return  # Cache already exists
    
    print(f"\nPICD cache not found at {cache_dir}")
    print("Generating saliency cache (this may take a while on first run)...")
    
    try:
        from generate_cache import generate_picd_cache
        if config is None:
            from config import VFCNetConfig
            config = VFCNetConfig()
            config.paths.picd_root = rgb_root
            config.paths.picd_cache_root = cache_dir
        generate_picd_cache(config, overwrite=False)
    except ImportError as e:
        raise RuntimeError(
            f"Cache not found and cannot generate: {e}\n"
            "Please either:\n"
            "  1. Install deepgaze-pytorch: pip install deepgaze-pytorch\n"
            "  2. Or manually generate cache: python generate_cache.py --dataset picd"
        )


class PICDDualStreamDataset(Dataset):
    """PICD with dual-stream GVF for evaluation.
    
    Uses all subdirs: single_labels, multi_labels, public (matching original).
    If cache doesn't exist, it will be generated automatically.
    """
    
    def __init__(self, cache_dir: Path, rgb_root: Path,
                 gvf_size: int = 56, mu: float = 0.15, beta: float = 0.1,
                 iterations: int = 10, gvf_mode: str = "intensity",
                 config=None, auto_generate: bool = True):
        self.cache_dir = Path(cache_dir)
        self.rgb_root = Path(rgb_root)
        self.gvf_size = gvf_size
        self.mu = mu
        self.beta = beta
        self.iterations = iterations
        self.gvf_mode = gvf_mode
        
        # Auto-generate cache if needed
        if auto_generate:
            _generate_picd_cache_if_needed(self.cache_dir, self.rgb_root, config)
        
        # Build PICD metadata for semantic labels from CSV
        self._picd_meta = {}
        labels_csv = Path(__file__).parent / "labels_PICD.csv"
        if labels_csv.exists():
            import csv
            with open(labels_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_id = row["img_id"].strip()
                    folder = row.get("folder_name", "").strip()
                    yolo_raw = row.get("yolo_labels", "").strip()
                    yolo_labels = [l.strip() for l in yolo_raw.split(",") if l.strip()] if yolo_raw else []
                    self._picd_meta[(img_id, folder)] = {"yolo_labels": yolo_labels}
            print(f"Loaded {len(self._picd_meta)} label entries from {labels_csv}")
        else:
            print(f"Warning: labels_PICD.csv not found at {labels_csv}")
        
        # Find cache files from ALL subdirs
        self.cache_files = []
        for subdir in ["single_labels", "multi_labels", "public"]:
            subpath = self.cache_dir / subdir
            if subpath.exists():
                for cat_dir in subpath.iterdir():
                    if cat_dir.is_dir():
                        self.cache_files.extend(list(cat_dir.glob("*.pt")))
        
        if not self.cache_files:
            raise RuntimeError(
                f"No cached tensors found in {self.cache_dir}\n"
                "Generate cache first: python generate_cache.py --dataset picd"
            )
        
        self.classes = sorted(set(cf.parent.name for cf in self.cache_files))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Build data list with semantic labels (yolo_labels)
        self.data = []
        for cf in self.cache_files:
            parts = cf.relative_to(self.cache_dir).parts
            cls_name = parts[-2] if len(parts) >= 2 else "unknown"
            
            # Find RGB path
            rgb_path = None
            for subset in ["single_labels", "multi_labels", "public"]:
                candidate = self.rgb_root / subset / cls_name / cf.name.replace(".pt", ".jpg")
                if candidate.exists():
                    rgb_path = candidate
                    break
            
            image_name = cf.stem + ".jpg"
            meta = self._picd_meta.get((image_name, cls_name)) or self._picd_meta.get((image_name, ""))
            yolo_labels = meta.get("yolo_labels", []) if meta else []
            
            self.data.append({
                "cache_file": cf,
                "rgb_path": rgb_path,
                "label": cls_name,
                "class_id": self.class_to_idx.get(cls_name, -1),
                "image_name": image_name,
                "folder_name": cls_name,
                "yolo_labels": yolo_labels,
            })
        
        # Count samples with semantic labels
        n_semantic = sum(1 for d in self.data if d["yolo_labels"])
        print(f"PICDDualStreamDataset: {len(self.data)} samples, mode={gvf_mode}")
        print(f"Semantic labels: {n_semantic}/{len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        cache_data = torch.load(sample["cache_file"], map_location="cpu", weights_only=False)
        inputs_normalized = cache_data["inputs"]
        
        saliency_raw = inputs_normalized[0].numpy() * 0.229 + 0.485
        saliency_raw = np.clip(saliency_raw, 0, 1)
        
        # Load grayscale
        rgb_path = sample["rgb_path"]
        if rgb_path and Path(rgb_path).exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = saliency_raw.copy()
        
        # Compute GVF
        u_base, v_base = compute_intensity_gvf(gray, self.mu, self.iterations)
        gvf_u_sal, gvf_v_sal = compute_intensity_gvf_with_saliency_force(
            gray, saliency_raw, self.mu, self.beta, self.iterations)
        
        base_u_small = cv2.resize((u_base + 1) * 0.5, (self.gvf_size, self.gvf_size))
        base_v_small = cv2.resize((v_base + 1) * 0.5, (self.gvf_size, self.gvf_size))
        sal_u_small = cv2.resize((gvf_u_sal + 1) * 0.5, (self.gvf_size, self.gvf_size))
        sal_v_small = cv2.resize((gvf_v_sal + 1) * 0.5, (self.gvf_size, self.gvf_size))
        
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
            "class_id": sample["class_id"],
            "label": sample["label"],
            "image_name": sample["image_name"],
        }


# =============================================================================
# CDA Evaluation (exact copy from PICD_eval.py)
# =============================================================================

@dataclass
class Sample:
    """Lightweight holder for dataset metadata."""
    index: int
    class_id: int
    class_name: str
    image_name: str
    semantic_label: Optional[str] = None


@dataclass
class Triplet:
    """Indices for two positives (same composition) and one negative."""
    pos_a: int
    pos_b: int
    neg: int
    positive_class: str
    negative_class: str


def _ensure_tensor(value: Any) -> torch.Tensor:
    """Convert to tensor if needed."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _sample_two(indices: Sequence[int], rng: random.Random) -> Tuple[int, int]:
    """Sample two indices from a sequence."""
    if len(indices) >= 2:
        a, b = rng.sample(list(indices), 2)
    else:
        a = b = indices[0]
    return a, b


def _predict_outlier(vectors: Sequence[torch.Tensor]) -> int:
    """Predict which of 3 vectors is the outlier.
    
    Finds the pair with smallest pairwise distance, returns the index
    of the remaining vector (the outlier).
    """
    stack = torch.stack(list(vectors))
    dist = torch.cdist(stack, stack, p=2)
    
    best_pair = None
    best_dist = float("inf")
    for i in range(3):
        for j in range(i + 1, 3):
            d = dist[i, j].item()
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)
    
    if best_pair is None:
        return 2
    
    outlier = {0, 1, 2} - set(best_pair)
    return outlier.pop()


def get_semantic_labels(dataset: PICDDualStreamDataset) -> Dict[str, str]:
    """Extract semantic labels from dataset for CDA-2."""
    semantic_labels = {}
    for s in dataset.data:
        name = s.get("image_name", "")
        yolo = s.get("yolo_labels", [])
        if name and yolo:
            semantic_labels[name] = yolo[0]
    return semantic_labels


class PICDEvaluator:
    """Computes CDA for VFCNet - exact copy of PICD_eval.py logic."""

    def __init__(
        self,
        dataset: PICDDualStreamDataset,
        *,
        num_triplets_per_pair: int = 12,
        num_semantic_triplets: int = 2760,
        seed: int = 42,
        semantic_labels: Optional[Dict[str, str]] = None,
    ):
        self.dataset = dataset
        self.num_triplets_per_pair = num_triplets_per_pair
        self.num_semantic_triplets = num_semantic_triplets
        self.seed = seed
        self.semantic_labels = semantic_labels or {}
        self.class_names = list(dataset.classes) if hasattr(dataset, 'classes') else None
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Sample]:
        """Collect sample metadata from dataset."""
        samples = []
        for idx, sample_meta in enumerate(self.dataset.data):
            class_id = sample_meta.get('class_id', 0)
            class_name = self._class_name(class_id)
            image_name = sample_meta.get("image_name", f"sample_{idx}")
            semantic_label = self.semantic_labels.get(image_name)
            samples.append(Sample(idx, class_id, class_name, image_name, semantic_label))
        return samples

    def _class_name(self, class_id: int) -> str:
        """Get class name from class id."""
        if self.class_names and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def _has_semantic_labels(self) -> bool:
        """Check if any samples have semantic labels for CDA-2."""
        return any(s.semantic_label is not None for s in self.samples)

    def _build_triplets(self) -> Tuple[List[Triplet], Dict[str, Any]]:
        """Build triplets for CDA-1."""
        rng = random.Random(self.seed)
        by_class: Dict[int, List[int]] = {}
        for sample in self.samples:
            by_class.setdefault(sample.class_id, []).append(sample.index)

        class_ids = sorted(by_class.keys())
        triplets: List[Triplet] = []
        skipped: Dict[str, int] = {}

        for pos_id in class_ids:
            pos_indices = by_class[pos_id]
            if len(pos_indices) < 2:
                skipped[f"pos_{pos_id}"] = skipped.get(f"pos_{pos_id}", 0) + 23
                continue
            for neg_id in class_ids:
                if pos_id == neg_id:
                    continue
                neg_indices = by_class[neg_id]
                if not neg_indices:
                    skipped[f"neg_{neg_id}"] = skipped.get(f"neg_{neg_id}", 0) + 1
                    continue
                for _ in range(self.num_triplets_per_pair):
                    pos_a, pos_b = _sample_two(pos_indices, rng)
                    neg = rng.choice(neg_indices)
                    triplets.append(Triplet(
                        pos_a=pos_a,
                        pos_b=pos_b,
                        neg=neg,
                        positive_class=self._class_name(pos_id),
                        negative_class=self._class_name(neg_id),
                    ))
        return triplets, skipped

    def _build_semantic_triplets(self) -> Tuple[List[Triplet], Dict[str, Any]]:
        """Build triplets for CDA-2: Robustness towards Semantic Interference.
        
        Triplet structure:
        - P1, P2: Same composition category, DIFFERENT semantic labels
        - N: Different composition category, SAME semantic label as P1 or P2
        """
        rng = random.Random(self.seed + 1)  # Different seed from CDA-1
        
        by_class: Dict[int, List[Sample]] = {}
        by_semantic: Dict[str, List[Sample]] = {}
        
        for sample in self.samples:
            if sample.semantic_label is None:
                continue
            by_class.setdefault(sample.class_id, []).append(sample)
            by_semantic.setdefault(sample.semantic_label, []).append(sample)
        
        triplets: List[Triplet] = []
        skipped: Dict[str, int] = {"no_diverse_positives": 0, "no_semantic_negative": 0}
        
        class_ids = sorted(by_class.keys())
        target_per_class = max(1, self.num_semantic_triplets // max(1, len(class_ids)))
        
        for pos_class_id in class_ids:
            pos_samples = by_class[pos_class_id]
            class_triplets: List[Triplet] = []
            
            pos_by_semantic: Dict[str, List[Sample]] = {}
            for s in pos_samples:
                pos_by_semantic.setdefault(s.semantic_label, []).append(s)
            
            semantic_labels_in_class = list(pos_by_semantic.keys())
            if len(semantic_labels_in_class) < 2:
                skipped["no_diverse_positives"] += 1
                continue
            
            attempts = 0
            max_attempts = target_per_class * 10
            
            while len(class_triplets) < target_per_class and attempts < max_attempts:
                attempts += 1
                
                sem1, sem2 = rng.sample(semantic_labels_in_class, 2)
                p1 = rng.choice(pos_by_semantic[sem1])
                p2 = rng.choice(pos_by_semantic[sem2])
                
                target_semantic = rng.choice([sem1, sem2])
                neg_candidates = [
                    s for s in by_semantic.get(target_semantic, [])
                    if s.class_id != pos_class_id
                ]
                
                if not neg_candidates:
                    skipped["no_semantic_negative"] += 1
                    continue
                
                neg = rng.choice(neg_candidates)
                
                class_triplets.append(Triplet(
                    pos_a=p1.index,
                    pos_b=p2.index,
                    neg=neg.index,
                    positive_class=self._class_name(pos_class_id),
                    negative_class=self._class_name(neg.class_id),
                ))
            
            triplets.extend(class_triplets)
        
        if len(triplets) > self.num_semantic_triplets:
            triplets = rng.sample(triplets, self.num_semantic_triplets)
        
        return triplets, skipped

    def _cda(
        self,
        triplets: List[Triplet],
        embeddings: Dict[int, torch.Tensor],
        *,
        show_progress: bool = True,
        desc: str = "CDA",
    ) -> Dict[str, Any]:
        """Compute CDA score."""
        correct = 0
        per_class: Dict[str, Dict[str, int]] = {}
        iterator = triplets
        if show_progress:
            iterator = tqdm(iterator, desc=desc, leave=False)

        for triplet in iterator:
            idxs = [triplet.pos_a, triplet.pos_b, triplet.neg]
            vecs = [_ensure_tensor(embeddings[i]) for i in idxs]
            pred_idx = _predict_outlier(vecs)
            is_correct = idxs[pred_idx] == triplet.neg
            correct += int(is_correct)

            stats = per_class.setdefault(triplet.positive_class, {"correct": 0, "total": 0})
            stats["correct"] += int(is_correct)
            stats["total"] += 1

        per_class_cda = {
            label: stats["correct"] / stats["total"]
            for label, stats in per_class.items()
            if stats["total"] > 0
        }
        return {
            "avg_cda": correct / max(1, len(triplets)),
            "per_class_cda": per_class_cda,
        }

    def _evaluate_with_embeddings(
        self,
        embeddings: Dict[int, torch.Tensor],
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate CDA with pre-computed embeddings."""
        triplets_cda1, _ = self._build_triplets()
        metrics_cda1 = self._cda(triplets_cda1, embeddings, show_progress=show_progress, desc="CDA-1")
        
        metrics = {
            "cda_score": metrics_cda1["avg_cda"],
            "avg_cda1": metrics_cda1["avg_cda"],
            "num_triplets_cda1": len(triplets_cda1),
        }
        
        if self._has_semantic_labels():
            triplets_cda2, _ = self._build_semantic_triplets()
            if triplets_cda2:
                metrics_cda2 = self._cda(triplets_cda2, embeddings, show_progress=show_progress, desc="CDA-2")
                metrics["cda_score_semantic_robustness"] = metrics_cda2["avg_cda"]
                metrics["avg_cda2"] = metrics_cda2["avg_cda"]
                metrics["num_triplets_cda2"] = len(triplets_cda2)
                metrics["per_class_cda2"] = metrics_cda2["per_class_cda"]
            else:
                metrics["cda_score_semantic_robustness"] = None
        else:
            metrics["cda_score_semantic_robustness"] = None
        
        metrics["per_class_cda1"] = metrics_cda1["per_class_cda"]
        return metrics


# =============================================================================
# Full Evaluation Pipeline (matching eval_reviewer_ablations.py)
# =============================================================================

def compute_all_metrics(embeddings: Dict[int, torch.Tensor], 
                        dataset: PICDDualStreamDataset,
                        semantic_labels: Dict[str, str]) -> Dict[str, Any]:
    """Compute CDA, DBI, Silhouette, and per-class metrics.
    
    Matches the exact logic from eval_reviewer_ablations.py compute_all_metrics.
    """
    def evaluator_factory(seed=42, num_triplets_per_pair=12):
        return PICDEvaluator(dataset, semantic_labels=semantic_labels, seed=seed, 
                            num_triplets_per_pair=num_triplets_per_pair)
    
    print("\n5-seed variance (12 triplets each):")
    cda_results = []
    for seed in [42, 43, 44, 45, 46]:
        evaluator = evaluator_factory(seed=seed, num_triplets_per_pair=12)
        metrics = evaluator._evaluate_with_embeddings(embeddings)
        cda_results.append({
            "seed": seed,
            "cda1": metrics["cda_score"],
            "cda2": metrics.get("cda_score_semantic_robustness"),
        })
        print(f"  Seed {seed}: CDA-1={metrics['cda_score']:.4f}, CDA-2={metrics.get('cda_score_semantic_robustness', 'N/A')}")
    
    cda1_vals = [r["cda1"] for r in cda_results]
    cda2_vals = [r["cda2"] for r in cda_results if r["cda2"] is not None]
    
    # CDA with 50 triplets
    print("\n50-triplet stable estimate:")
    evaluator_50 = evaluator_factory(seed=42, num_triplets_per_pair=50)
    metrics_50 = evaluator_50._evaluate_with_embeddings(embeddings)
    cda1_50 = metrics_50["cda_score"]
    cda2_50 = metrics_50.get("cda_score_semantic_robustness")
    print(f"  CDA-1 (50 triplets): {cda1_50:.4f}")
    if cda2_50:
        print(f"  CDA-2 (50 triplets): {cda2_50:.4f}")
    
    # Clustering metrics (matching original: cosine for silhouette)
    X, labels = [], []
    for idx in range(len(dataset)):
        if idx in embeddings:
            X.append(embeddings[idx].numpy())
            sample = dataset.data[idx]
            labels.append(sample.get("class_id", -1))
    
    X = np.array(X)
    labels = np.array(labels)
    valid_mask = labels >= 0
    X_valid, labels_valid = X[valid_mask], labels[valid_mask]
    
    try:
        dbi = davies_bouldin_score(X_valid, labels_valid)
    except:
        dbi = None
    try:
        sil = silhouette_score(X_valid, labels_valid, metric='cosine')
    except:
        sil = None
    
    results = {
        "cda1_mean": float(np.mean(cda1_vals)),
        "cda1_std": float(np.std(cda1_vals)),
        "cda2_mean": float(np.mean(cda2_vals)) if cda2_vals else None,
        "cda2_std": float(np.std(cda2_vals)) if cda2_vals else None,
        "cda1_50triplet": float(cda1_50),
        "cda2_50triplet": float(cda2_50) if cda2_50 else None,
        "dbi": float(dbi) if dbi else None,
        "silhouette": float(sil) if sil else None,
        "num_samples": len(dataset),
        "per_seed": cda_results,
        "per_class_cda1": metrics_50.get("per_class_cda1", {}),
        "per_class_cda2": metrics_50.get("per_class_cda2", {}),
    }
    
    print(f"\nSummary: CDA-1={results['cda1_mean']:.4f}±{results['cda1_std']:.4f}")
    if results['cda2_mean']:
        print(f"         CDA-2={results['cda2_mean']:.4f}±{results['cda2_std']:.4f}")
    
    return results


def extract_embeddings(backbone: nn.Module, head: nn.Module, 
                       dataset: PICDDualStreamDataset) -> Dict[int, torch.Tensor]:
    """Extract embeddings for all samples in dataset."""
    backbone.eval()
    head.eval()
    
    embeddings = {}
    
    for idx in tqdm(range(len(dataset)), desc="PICD embeddings"):
        sample = dataset[idx]
        inputs = sample["inputs"].unsqueeze(0).to(device)
        base_u = sample["baseline_gvf_u"].unsqueeze(0).to(device)
        base_v = sample["baseline_gvf_v"].unsqueeze(0).to(device)
        sal_u = sample["saliency_gvf_u"].unsqueeze(0).to(device)
        sal_v = sample["saliency_gvf_v"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = backbone(inputs)
            _, fused, _ = head(features, base_u, base_v, sal_u, sal_v)
            embeddings[idx] = fused.squeeze(0).cpu()
    
    return embeddings


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, config: VFCNetConfig):
    """Load trained VFCNet model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
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
    
    # Load backbone weights if available
    if checkpoint.get("backbone"):
        backbone.load_state_dict(checkpoint["backbone"])
        print("  Loaded backbone weights")
    
    # Build head
    head = build_vfcnet(config).to(device)
    head.load_state_dict(checkpoint["head"])
    print("  Loaded head weights")
    
    # Get config from checkpoint
    ckpt_config = checkpoint.get("config", {})
    mu = ckpt_config.get("mu", config.gvf.mu)
    beta = ckpt_config.get("beta", config.gvf.beta)
    iterations = ckpt_config.get("iterations", config.gvf.iterations)
    
    print(f"  Config: mu={mu}, beta={beta}, iterations={iterations}")
    
    return backbone, head, {"mu": mu, "beta": beta, "iterations": iterations}


def main():
    parser = argparse.ArgumentParser(description="Evaluate VFCNet on PICD")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--picd_cache", type=str, default=None, help="PICD cache directory")
    parser.add_argument("--picd_root", type=str, default=None, help="PICD root directory")
    args = parser.parse_args()
    
    config = VFCNetConfig()
    
    # Load model
    backbone, head, gvf_config = load_model(args.checkpoint, config)
    
    # Setup paths
    picd_cache = Path(args.picd_cache) if args.picd_cache else Path("/home/armin.dadras/PICD/cache_saliency_gvf")
    picd_root = Path(args.picd_root) if args.picd_root else config.paths.picd_root
    
    print(f"\nPICD cache: {picd_cache}")
    print(f"PICD root: {picd_root}")
    
    # Build dataset
    dataset = PICDDualStreamDataset(
        cache_dir=picd_cache,
        rgb_root=picd_root,
        gvf_size=56,
        mu=gvf_config["mu"],
        beta=gvf_config["beta"],
        iterations=gvf_config["iterations"],
        gvf_mode="intensity",
    )
    
    # Get semantic labels
    semantic_labels = get_semantic_labels(dataset)
    
    # Extract embeddings
    embeddings = extract_embeddings(backbone, head, dataset)
    
    # Compute all metrics
    results = compute_all_metrics(embeddings, dataset, semantic_labels)
    
    # Add model info
    results["checkpoint"] = args.checkpoint
    results["gvf_config"] = gvf_config
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).parent / "eval_results.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print per-class results
    print("\n" + "="*70)
    print("PER-CLASS CDA-1 (50 triplets)")
    print("="*70)
    for cls, score in sorted(results.get("per_class_cda1", {}).items()):
        print(f"  {cls}: {score:.4f}")
    
    if results.get("per_class_cda2"):
        print("\n" + "="*70)
        print("PER-CLASS CDA-2 (50 triplets)")
        print("="*70)
        for cls, score in sorted(results.get("per_class_cda2", {}).items()):
            print(f"  {cls}: {score:.4f}")


if __name__ == "__main__":
    main()
