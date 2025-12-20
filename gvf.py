"""Gradient Vector Flow (GVF) computation utilities.

This module implements GVF computation as described in the VFCNet paper,
including the saliency-enhanced variant that incorporates visual attention.

GVF extends traditional gradient fields by diffusing gradients into homogeneous
regions, providing smooth flow that captures compositional structure.

References:
- Xu & Prince (1998): "Snakes, Shapes, and Gradient Vector Flow"
- VFCNet paper: "Vector Flow Composition Network for Image Composition"
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image


def compute_laplacian(field: np.ndarray) -> np.ndarray:
    """Compute Laplacian using finite differences.
    
    Args:
        field: 2D array (H, W)
        
    Returns:
        Laplacian of the field
    """
    return cv2.Laplacian(field.astype(np.float32), cv2.CV_32F)


def compute_gvf(
    fx: np.ndarray,
    fy: np.ndarray,
    mu: float = 0.15,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gradient Vector Flow from edge gradient field.
    
    GVF is computed by solving an iterative diffusion equation that propagates
    gradient information into homogeneous regions while preserving edges.
    
    The update equations are:
        u_{t+1} = u_t + μ∇²u - (u - fx)(fx² + fy²)
        v_{t+1} = v_t + μ∇²v - (v - fy)(fx² + fy²)
    
    Args:
        fx: Horizontal gradient component (H, W)
        fy: Vertical gradient component (H, W)
        mu: Regularization weight (controls smoothness vs edge fidelity)
            Higher mu = smoother flow, lower mu = sharper edges
            Recommended: 0.15 (from ablation study)
        iterations: Number of diffusion iterations
            Recommended: 10 (from ablation study - fewer is better)
    
    Returns:
        Tuple of (u, v) GVF components, normalized to [-1, 1]
    """
    fx = fx.astype(np.float32)
    fy = fy.astype(np.float32)
    
    # Initialize with gradient field
    u = fx.copy()
    v = fy.copy()
    
    # Squared gradient magnitude (edge strength)
    sq_mag = fx**2 + fy**2
    
    # Iterative diffusion
    for _ in range(iterations):
        lap_u = compute_laplacian(u)
        lap_v = compute_laplacian(v)
        
        # GVF update equation
        u = u + mu * lap_u - sq_mag * (u - fx)
        v = v + mu * lap_v - sq_mag * (v - fy)
    
    # Normalize to [-1, 1]
    u = u / (np.abs(u).max() + 1e-6)
    v = v / (np.abs(v).max() + 1e-6)
    
    return u, v


def compute_intensity_gradient_gvf(
    image: np.ndarray,
    mu: float = 0.15,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GVF from image intensity gradients.
    
    This is the recommended edge source for composition analysis,
    as it provides smooth gradients that capture tonal variations.
    
    Args:
        image: Grayscale image (H, W) in range [0, 1] or [0, 255]
        mu: GVF regularization weight
        iterations: Number of diffusion iterations
        
    Returns:
        Tuple of (u, v) GVF components
    """
    # Ensure float in [0, 1]
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # Compute gradients
    gy, gx = np.gradient(image)
    
    return compute_gvf(gx, gy, mu=mu, iterations=iterations)


def compute_sobel_gvf(
    image: np.ndarray,
    mu: float = 0.15,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GVF from Sobel edge gradients.
    
    Args:
        image: Grayscale image (H, W)
        mu: GVF regularization weight
        iterations: Number of diffusion iterations
        
    Returns:
        Tuple of (u, v) GVF components
    """
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Normalize gradients
    sobelx = sobelx / (np.abs(sobelx).max() + 1e-6)
    sobely = sobely / (np.abs(sobely).max() + 1e-6)
    
    return compute_gvf(sobelx, sobely, mu=mu, iterations=iterations)


def compute_canny_gvf(
    image: np.ndarray,
    mu: float = 0.15,
    iterations: int = 10,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GVF from Canny edge detection.
    
    Note: Canny edges perform worse than intensity gradients for composition
    (CDA-2 = 0.521 vs 0.578) because binary edges lose tonal information.
    
    Args:
        image: Grayscale image (H, W)
        mu: GVF regularization weight
        iterations: Number of diffusion iterations
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
        
    Returns:
        Tuple of (u, v) GVF components
    """
    # Ensure uint8 for Canny
    if image.max() <= 1:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    edges = cv2.Canny(image_uint8, low_threshold, high_threshold)
    edges_float = edges.astype(np.float32) / 255.0
    
    # Compute gradients of edge map
    gy, gx = np.gradient(edges_float)
    gx = gx / (np.abs(gx).max() + 1e-6)
    gy = gy / (np.abs(gy).max() + 1e-6)
    
    return compute_gvf(gx, gy, mu=mu, iterations=iterations)


def compute_saliency_enhanced_gvf(
    image: np.ndarray,
    saliency: np.ndarray,
    mu: float = 0.15,
    beta: float = 0.1,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GVF with saliency gradient as additional force term.
    
    This extends standard GVF by adding saliency gradients as a force that
    biases the flow toward salient regions. The modified update equations are:
    
        u_{t+1} = u_t + μ∇²u - (u - fx)(fx² + fy²) + β∇S_x
        v_{t+1} = v_t + μ∇²v - (v - fy)(fx² + fy²) + β∇S_y
    
    where S is the saliency map and β controls the saliency force strength.
    
    Args:
        image: Grayscale image (H, W)
        saliency: Saliency map (H, W), same size as image, normalized [0, 1]
        mu: GVF regularization weight
        beta: Saliency gradient force strength
        iterations: Number of diffusion iterations
        
    Returns:
        Tuple of (u, v) saliency-enhanced GVF components
    """
    # Ensure float
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    saliency = saliency.astype(np.float32)
    if saliency.max() > 1:
        saliency = saliency / saliency.max()
    
    # Resize saliency to match image if needed
    if saliency.shape != image.shape:
        saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))
    
    # Compute image gradients
    gy, gx = np.gradient(image)
    sq_mag = gx**2 + gy**2
    
    # Compute saliency gradients
    sal_gy, sal_gx = np.gradient(saliency)
    
    # Normalize saliency gradients to match edge gradient magnitude
    # This is important - without normalization, beta=0.1 has minimal effect
    edge_mag = np.sqrt(np.mean(gx**2 + gy**2))
    sal_mag = np.sqrt(np.mean(sal_gx**2 + sal_gy**2)) + 1e-6
    sal_gx = sal_gx * (edge_mag / sal_mag)
    sal_gy = sal_gy * (edge_mag / sal_mag)
    
    # Initialize with gradient field
    u = gx.copy()
    v = gy.copy()
    
    # Iterative diffusion with saliency force
    for _ in range(iterations):
        lap_u = compute_laplacian(u)
        lap_v = compute_laplacian(v)
        
        # Modified GVF update with saliency force
        u = u + mu * lap_u - sq_mag * (u - gx) + beta * sal_gx
        v = v + mu * lap_v - sq_mag * (v - gy) + beta * sal_gy
    
    # Normalize
    u = u / (np.abs(u).max() + 1e-6)
    v = v / (np.abs(v).max() + 1e-6)
    
    return u, v


def image_to_gvf(
    image: Image.Image,
    saliency: Optional[np.ndarray] = None,
    gvf_size: int = 56,
    mu: float = 0.15,
    beta: float = 0.1,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert PIL image to GVF components.
    
    Computes both baseline (edge-only) and saliency-enhanced GVF.
    
    Args:
        image: PIL Image (RGB)
        saliency: Optional saliency map. If None, only baseline GVF is computed.
        gvf_size: Output GVF resolution
        mu: GVF regularization weight
        beta: Saliency force strength
        iterations: Number of diffusion iterations
        
    Returns:
        Tuple of (baseline_u, baseline_v, saliency_u, saliency_v)
        If saliency is None, saliency_u and saliency_v equal baseline
    """
    # Convert to grayscale
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Compute baseline GVF
    baseline_u, baseline_v = compute_intensity_gradient_gvf(gray, mu=mu, iterations=iterations)
    
    # Compute saliency-enhanced GVF
    if saliency is not None:
        saliency_u, saliency_v = compute_saliency_enhanced_gvf(
            gray, saliency, mu=mu, beta=beta, iterations=iterations
        )
    else:
        saliency_u, saliency_v = baseline_u.copy(), baseline_v.copy()
    
    # Resize to target size
    baseline_u = cv2.resize(baseline_u, (gvf_size, gvf_size))
    baseline_v = cv2.resize(baseline_v, (gvf_size, gvf_size))
    saliency_u = cv2.resize(saliency_u, (gvf_size, gvf_size))
    saliency_v = cv2.resize(saliency_v, (gvf_size, gvf_size))
    
    return baseline_u, baseline_v, saliency_u, saliency_v


def visualize_gvf(
    u: np.ndarray,
    v: np.ndarray,
    skip: int = 4,
    scale: float = 1.0,
) -> np.ndarray:
    """Create visualization of GVF as vector field overlaid on magnitude.
    
    Args:
        u: Horizontal GVF component
        v: Vertical GVF component
        skip: Subsample factor for arrows
        scale: Arrow scale factor
        
    Returns:
        RGB visualization image
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    H, W = u.shape
    magnitude = np.sqrt(u**2 + v**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Show magnitude as background
    ax.imshow(magnitude, cmap='gray', origin='upper')
    
    # Create meshgrid for arrows
    y, x = np.mgrid[0:H:skip, 0:W:skip]
    
    # Subsample vectors
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    
    # Plot arrows (note: quiver uses (x, y) convention, v is vertical)
    ax.quiver(x, y, u_sub, -v_sub, color='cyan', scale=scale * 20, width=0.003)
    
    ax.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img
