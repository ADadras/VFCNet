"""Configuration for VFCNet training and evaluation.

This file contains all hyperparameters used in the paper:
"Vector Flow Composition Network for Photographic Image Composition Classification"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class PathConfig:
    """File paths for data and models.
    
    IMPORTANT: Update these paths before running training or evaluation.
    """
    
    # Dataset paths (REQUIRED - update these)
    kupcp_root: Path = Path("/path/to/kupcp")  # KU-PCP dataset root
    picd_root: Path = Path("/path/to/PICD")    # PICD dataset root
    
    # Cache paths for pre-computed saliency maps
    cache_root: Path = Path("/path/to/cache/saliency_gvf")  # KUPCP saliency cache
    picd_cache_root: Path = Path("/path/to/PICD/cache_saliency_gvf")  # PICD saliency cache
    
    # DINOv3 model paths (REQUIRED - update these)
    dino_repo: str = "/path/to/models/dinov3"  # DINOv3 repo (cloned from facebookresearch/dinov2)
    dino_model: str = "dinov3_vitb16"
    dino_weights: str = "/path/to/models/dinov3_vitb16_pretrain.pth"  # Pre-trained weights
    
    # Optional paths
    centerbias_path: Optional[Path] = None  # MIT saliency center bias (optional)
    
    # Output paths
    output_dir: Path = Path("./output")
    checkpoint_dir: Path = Path("./output/checkpoints")


@dataclass
class GVFConfig:
    """Gradient Vector Flow computation parameters.
    
    Best settings from ablation study (Table 4 in paper):
    - mu=0.15, iterations=10 achieves best CDA-2=0.629
    - Intensity gradients outperform Sobel and Canny edges
    """
    
    # Diffusion parameters
    mu: float = 0.15  # Regularization weight (controls smoothness vs edge fidelity)
    iterations: int = 10  # Number of diffusion iterations (10 > 30 > 50)
    
    # Saliency force parameters
    beta: float = 0.1  # Saliency gradient force strength
    
    # Edge source: "intensity" | "sobel" | "canny"
    edge_source: str = "intensity"
    
    # Output resolution
    gvf_size: int = 56  # GVF spatial resolution (56x56 is optimal)


@dataclass
class ModelConfig:
    """VFCNet model architecture parameters.
    
    Best settings from ablation study (Table 5 in paper):
    - All differential features (div, curl, mag) contribute significantly
    - Removing divergence hurts most (-8.6%), then magnitude (-7.3%), then curl (-4.1%)
    """
    
    # DINOv3 backbone
    embed_dim: int = 768  # ViT-B/16 embedding dimension
    backbone_trainable: bool = True  # Unfreeze backbone (+4.8% CDA-2)
    
    # Multi-level feature extraction (from DINOv2 blocks)
    low_level_blocks: tuple = (2, 3)  # Early blocks: edges, textures
    mid_level_blocks: tuple = (5, 6)  # Mid blocks: parts, shapes
    high_level_blocks: tuple = (10, 11)  # Late blocks: semantics
    high_level_dropout: float = 0.3  # Dropout on semantic features during training
    
    # GVF feature extraction
    vf_dim: int = 128  # Vector field feature dimension
    gvf_scales: tuple = (1, 2, 4)  # Multi-scale pyramid
    use_divergence: bool = True  # Include divergence features
    use_curl: bool = True  # Include curl/rotation features
    use_magnitude: bool = True  # Include magnitude features
    pooling: str = "avg"  # "avg" or "max" pooling
    
    # Dual-stream attention (for baseline + saliency-enhanced GVF)
    num_attention_heads: int = 4
    
    # Classifier head
    hidden_dim: int = 512
    dropout: float = 0.3
    num_classes: int = 9  # KU-PCP composition classes


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimization
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    
    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 10
    
    # Loss function: "bce" | "focal"
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # Data augmentation
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "horizontal_flip": True,
        "color_jitter": False,
        "gaussian_noise_std": 0.0,
    })
    
    # Mixed precision training
    use_amp: bool = True
    
    # DataLoader
    num_workers: int = 4


@dataclass
class VFCNetConfig:
    """Complete VFCNet configuration."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    gvf: GVFConfig = field(default_factory=GVFConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "VFCNetConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if "paths" in data:
            for key, value in data["paths"].items():
                if hasattr(config.paths, key):
                    if key.endswith("_root") or key.endswith("_dir") or key.endswith("_path"):
                        setattr(config.paths, key, Path(value) if value else None)
                    else:
                        setattr(config.paths, key, value)
        
        if "gvf" in data:
            for key, value in data["gvf"].items():
                if hasattr(config.gvf, key):
                    setattr(config.gvf, key, value)
        
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    if isinstance(value, list):
                        setattr(config.model, key, tuple(value))
                    else:
                        setattr(config.model, key, value)
        
        if "training" in data:
            for key, value in data["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        def to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        with open(path, "w") as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)


# Composition class names (9 classes in KU-PCP)
COMPOSITION_CLASSES = [
    "Rule of Thirds",
    "Vertical",
    "Horizontal",
    "Diagonal",
    "Curved",
    "Triangle",
    "Center",
    "Symmetric",
    "Pattern",
]

# ImageNet normalization (used for DINOv3 input)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
