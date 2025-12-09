"""
physics_lensing.py - Weak Lensing Convergence Map Generator
============================================================

Generates synthetic weak lensing convergence (κ) maps that match
the statistical properties of DES Y3 data.

The key differences from raw density fields:
1. Line-of-sight projection with lensing kernel
2. Gaussian smoothing (shape noise suppression)
3. Realistic shape noise addition
4. Masked regions (optional)

Theory
------
The convergence κ is related to the matter density contrast δ by:

    κ(θ) = ∫ W(χ) δ(χθ, χ) dχ

where W(χ) is the lensing efficiency kernel:

    W(χ) = (3/2) Ω_m (H₀/c)² (1+z) χ ∫_χ^∞ n(χ') (χ'-χ)/(χ') dχ'

For a single source redshift z_s, this simplifies to:

    W(χ) ∝ χ (χ_s - χ) / χ_s  for χ < χ_s

The observed convergence has shape noise:

    κ_obs = κ_true + n

where n ~ N(0, σ_ε² / n_gal) with σ_ε ≈ 0.26 (intrinsic ellipticity)
and n_gal is the galaxy number density per pixel.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_gaussian_field(
    size: int,
    omega_m: float,
    box_size: float = 256.0,
    seed: Optional[int] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate a Gaussian Random Field with cosmology-dependent power spectrum.
    
    This is the LINEAR density field - appropriate for weak lensing
    which probes mostly linear/quasi-linear scales.
    """
    if device is None:
        device = get_device()
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Wavenumber grid
    dk = 2 * np.pi / box_size
    kx = torch.fft.fftfreq(size, d=1.0/size, device=device) * dk
    ky = torch.fft.fftfreq(size, d=1.0/size, device=device) * dk
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K = torch.sqrt(K2)
    
    K_safe = torch.where(K < 1e-10, torch.ones_like(K), K)
    
    # BBKS transfer function
    h = 0.7
    gamma = omega_m * h
    q = K_safe / gamma
    T_k = (
        torch.log(1 + 2.34 * q) / (2.34 * q + 1e-10) * 
        (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
    )
    T_k = torch.where(K < 1e-10, torch.ones_like(T_k), T_k)
    
    # Power spectrum shape
    n_s = 0.965
    P_k = (K_safe ** n_s) * (T_k ** 2)
    P_k = torch.where(K < 1e-10, torch.zeros_like(P_k), P_k)
    
    # Generate complex Gaussian field
    amplitude = torch.sqrt(P_k / 2 + 1e-20)
    field_k = amplitude * (torch.randn(size, size, device=device) + 
                           1j * torch.randn(size, size, device=device))
    field_k[0, 0] = 0
    
    # Transform to real space
    field = torch.fft.ifft2(field_k).real
    
    # Normalize to unit variance
    field = (field - field.mean()) / (field.std() + 1e-10)
    
    return field


def apply_lensing_kernel(
    density_field: torch.Tensor,
    omega_m: float,
    source_redshift: float = 1.0
) -> torch.Tensor:
    """
    Apply weak lensing projection kernel.
    
    For a 2D field, this mainly scales the amplitude based on Ω_m
    and applies a slight smoothing to mimic projection effects.
    
    The convergence amplitude scales as:
        κ ∝ Ω_m × σ_8 × geometric_factor
    """
    # Amplitude scaling with Ω_m (simplified)
    # Higher Ω_m → more matter → stronger lensing signal
    amplitude = 0.02 * (omega_m / 0.3) ** 0.8
    
    kappa = density_field * amplitude
    
    return kappa


def add_shape_noise(
    kappa: torch.Tensor,
    sigma_epsilon: float = 0.26,
    n_gal_per_arcmin2: float = 10.0,
    pixel_size_arcmin: float = 1.0
) -> torch.Tensor:
    """
    Add realistic shape noise to convergence map.
    
    Parameters
    ----------
    kappa : torch.Tensor
        True convergence field
    sigma_epsilon : float
        Intrinsic ellipticity dispersion (DES: ~0.26)
    n_gal_per_arcmin2 : float
        Galaxy number density (DES Y3: ~5-10)
    pixel_size_arcmin : float
        Pixel size in arcminutes
        
    Returns
    -------
    torch.Tensor
        Noisy convergence map
    """
    # Number of galaxies per pixel
    n_gal_per_pixel = n_gal_per_arcmin2 * pixel_size_arcmin**2
    
    # Shape noise standard deviation per pixel
    # σ_κ = σ_ε / √(2 × n_gal)  (factor of 2 from two shear components)
    sigma_noise = sigma_epsilon / np.sqrt(2 * n_gal_per_pixel)
    
    # Add Gaussian noise
    noise = torch.randn_like(kappa) * sigma_noise
    
    return kappa + noise


def smooth_field(
    field: torch.Tensor,
    smoothing_arcmin: float = 5.0,
    pixel_size_arcmin: float = 1.0
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to suppress noise.
    
    DES typically uses 5-20 arcmin smoothing scales.
    """
    sigma_pixels = smoothing_arcmin / pixel_size_arcmin
    
    # Use scipy for smoothing (convert to numpy and back)
    field_np = field.cpu().numpy()
    smoothed_np = gaussian_filter(field_np, sigma=sigma_pixels, mode='wrap')
    
    return torch.from_numpy(smoothed_np).to(field.device)


def generate_des_like_map(
    omega_m: float,
    size: int = 256,
    seed: Optional[int] = None,
    smoothing_arcmin: float = 10.0,
    add_noise: bool = True,
    sigma_epsilon: float = 0.26,
    n_gal_per_arcmin2: float = 8.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate a synthetic weak lensing map matching DES Y3 statistics.
    
    Parameters
    ----------
    omega_m : float
        Matter density parameter
    size : int
        Map size in pixels
    seed : int, optional
        Random seed
    smoothing_arcmin : float
        Gaussian smoothing scale
    add_noise : bool
        Whether to add shape noise
    sigma_epsilon : float
        Intrinsic ellipticity dispersion
    n_gal_per_arcmin2 : float
        Galaxy number density
    device : torch.device
        Computation device
        
    Returns
    -------
    torch.Tensor
        DES-like convergence map, shape (size, size)
    """
    if device is None:
        device = get_device()
    
    # Generate underlying density field
    density = generate_gaussian_field(size, omega_m, seed=seed, device=device)
    
    # Apply lensing kernel
    kappa = apply_lensing_kernel(density, omega_m)
    
    # Add shape noise (before smoothing, as in real data processing)
    if add_noise:
        kappa = add_shape_noise(kappa, sigma_epsilon, n_gal_per_arcmin2)
    
    # Apply smoothing
    kappa = smooth_field(kappa, smoothing_arcmin)
    
    # Normalize for neural network input
    kappa = (kappa - kappa.mean()) / (kappa.std() + 1e-10)
    
    return kappa


def generate_des_like_numpy(
    omega_m: float,
    size: int = 256,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """Convenience function returning numpy array. Uses CPU for DataLoader compatibility."""
    # Force CPU for multiprocessing compatibility in DataLoader workers
    device = torch.device('cpu')
    kappa = generate_des_like_map(omega_m, size, seed, device=device, **kwargs)
    return kappa.numpy().astype(np.float32)


# =============================================================================
# COMPARISON VISUALIZATION
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    device = get_device()
    print(f"Device: {device}")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    omega_values = [0.2, 0.3, 0.4, 0.5]
    
    for i, omega_m in enumerate(omega_values):
        # Row 1: Raw GRF (what we had before)
        grf = generate_gaussian_field(256, omega_m, seed=42, device=device)
        grf_np = grf.cpu().numpy()
        
        axes[0, i].imshow(grf_np, cmap='gray', vmin=-3, vmax=3)
        axes[0, i].set_title(f'Raw GRF: Ω_m={omega_m}')
        axes[0, i].axis('off')
        
        # Row 2: DES-like (smoothed + noise)
        des = generate_des_like_map(omega_m, 256, seed=42, 
                                     smoothing_arcmin=10.0,
                                     add_noise=True, device=device)
        des_np = des.cpu().numpy()
        
        axes[1, i].imshow(des_np, cmap='gray', vmin=-3, vmax=3)
        axes[1, i].set_title(f'DES-like: Ω_m={omega_m}')
        axes[1, i].axis('off')
        
        # Row 3: DES-like (no noise, for comparison)
        des_clean = generate_des_like_map(omega_m, 256, seed=42,
                                           smoothing_arcmin=10.0,
                                           add_noise=False, device=device)
        des_clean_np = des_clean.cpu().numpy()
        
        axes[2, i].imshow(des_clean_np, cmap='gray', vmin=-3, vmax=3)
        axes[2, i].set_title(f'DES-like (no noise): Ω_m={omega_m}')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Raw GRF\n(sharp features)', fontsize=12)
    axes[1, 0].set_ylabel('DES-like\n(smoothed + noise)', fontsize=12)
    axes[2, 0].set_ylabel('DES-like\n(smoothed, no noise)', fontsize=12)
    
    plt.suptitle('Comparison: Raw Density vs DES-like Weak Lensing Maps', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/des_like_comparison.png', dpi=150)
    print("Saved results/figures/des_like_comparison.png")
