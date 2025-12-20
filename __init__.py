"""VFCNet: Vector Flow Composition Network.

A deep learning framework for photographic image composition classification
using Gradient Vector Flow (GVF) features.
"""

from .config import (
    VFCNetConfig,
    PathConfig,
    GVFConfig,
    ModelConfig,
    TrainingConfig,
    COMPOSITION_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from .model import (
    VFCNet,
    VFCNetSingleStream,
    DINOv3MultiLevelWrapper,
    MultiScaleGVFExtractor,
    DualStreamGVFAttention,
    build_vfcnet,
)

from .gvf import (
    compute_gvf,
    compute_intensity_gradient_gvf,
    compute_sobel_gvf,
    compute_canny_gvf,
    compute_saliency_enhanced_gvf,
    image_to_gvf,
    visualize_gvf,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "VFCNetConfig",
    "PathConfig",
    "GVFConfig",
    "ModelConfig",
    "TrainingConfig",
    "COMPOSITION_CLASSES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Model
    "VFCNet",
    "VFCNetSingleStream",
    "DINOv3MultiLevelWrapper",
    "MultiScaleGVFExtractor",
    "DualStreamGVFAttention",
    "build_vfcnet",
    # GVF
    "compute_gvf",
    "compute_intensity_gradient_gvf",
    "compute_sobel_gvf",
    "compute_canny_gvf",
    "compute_saliency_enhanced_gvf",
    "image_to_gvf",
    "visualize_gvf",
]
