"""
physics_lpt.py - GPU-Accelerated 2LPT Density Field Generator
==============================================================

Implements Second-Order Lagrangian Perturbation Theory (2LPT) for generating
realistic dark matter density fields that match N-body simulation statistics.

Why 2LPT instead of GRF?
------------------------
- GRF (Gaussian Random Field) is linear theory - valid only at early times
- Real density fields have non-Gaussian features: filaments, voids, halos
- 2LPT captures the leading non-linear corrections
- Result: Synthetic data that looks like Quijote, reducing domain gap

Theory
------
In Lagrangian perturbation theory, we track particle displacements from
initial (Lagrangian) positions q to final (Eulerian) positions x:

    x(q, t) = q + Ψ(q, t)

where Ψ is the displacement field. Expanding to second order:

    Ψ = Ψ^(1) + Ψ^(2) + ...

First order (Zel'dovich approximation):
    Ψ^(1) = -D₁(t) ∇φ

Second order:
    Ψ^(2) = D₂(t) ∇φ^(2)

where φ^(2) satisfies:
    ∇²φ^(2) = Σᵢⱼ (φ,ᵢᵢ φ,ⱼⱼ - φ,ᵢⱼ²)

References:
- Scoccimarro (1998): Transients from initial conditions
- Crocce et al. (2006): 2LPT initial conditions for simulations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def growth_factor_ratio(omega_m: float) -> float:
    """
    Compute D₂/D₁² ratio for 2LPT.
    
    In Einstein-de Sitter (Ω_m = 1): D₂/D₁² = -3/7
    For general cosmology, this is a weak function of Ω_m.
    """
    # Approximation valid for ΛCDM
    omega_lambda = 1 - omega_m
    f = omega_m**0.6  # Growth rate approximation
    return -3/7 * omega_m**(-1/143)


def generate_linear_potential(
    size: int,
    omega_m: float,
    box_size: float = 256.0,
    seed: Optional[int] = None,
    device: torch.device = None,
    sigma8: float = 0.8
) -> torch.Tensor:
    """
    Generate the linear gravitational potential φ in Fourier space.
    
    The potential is related to the density by Poisson's equation:
        ∇²φ = δ  (in appropriate units)
    
    So in Fourier space:
        φ_k = -δ_k / k²
    
    Parameters
    ----------
    size : int
        Grid size (size × size)
    omega_m : float
        Matter density parameter
    box_size : float
        Physical box size in Mpc/h
    seed : int, optional
        Random seed
    device : torch.device
        Computation device
    sigma8 : float
        RMS density fluctuation at 8 Mpc/h scale
        
    Returns
    -------
    torch.Tensor
        Complex potential in Fourier space, shape (size, size)
    """
    if device is None:
        device = get_device()
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Wavenumber grid (in units of h/Mpc)
    # k = 2π n / L where n is the mode number
    dk = 2 * np.pi / box_size
    kx = torch.fft.fftfreq(size, d=1.0/size, device=device) * dk
    ky = torch.fft.fftfreq(size, d=1.0/size, device=device) * dk
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K = torch.sqrt(K2)
    
    # Avoid division by zero
    K2_safe = torch.where(K2 < 1e-10, torch.ones_like(K2), K2)
    K_safe = torch.where(K < 1e-10, torch.ones_like(K), K)
    
    # BBKS transfer function (k in h/Mpc)
    h = 0.7
    gamma = omega_m * h  # Shape parameter Γ = Ω_m h
    q = K_safe / gamma
    T_k = (
        torch.log(1 + 2.34 * q) / (2.34 * q + 1e-10) * 
        (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
    )
    T_k = torch.where(K < 1e-10, torch.ones_like(T_k), T_k)
    
    # Power spectrum shape: P(k) ∝ k^n_s * T²(k)
    n_s = 0.965
    P_k_shape = (K_safe ** n_s) * (T_k ** 2)
    P_k_shape = torch.where(K < 1e-10, torch.zeros_like(P_k_shape), P_k_shape)
    
    # Normalize to σ₈ using the variance integral
    # σ² = (1/2π²) ∫ P(k) W²(kR) k² dk where W is top-hat window
    # For simplicity, we normalize the field to unit variance and scale by σ₈
    
    # Generate Gaussian random field δ_k with unit variance
    amplitude = torch.sqrt(P_k_shape / 2 + 1e-20)
    delta_k = amplitude * (torch.randn(size, size, device=device) + 
                           1j * torch.randn(size, size, device=device))
    delta_k[0, 0] = 0  # Zero mean
    
    # Transform to real space, normalize, then back
    delta_real = torch.fft.ifft2(delta_k).real
    delta_real = delta_real / (delta_real.std() + 1e-10)  # Unit variance
    delta_real = delta_real * sigma8  # Scale to σ₈
    delta_k = torch.fft.fft2(delta_real)
    
    # Potential: φ_k = -δ_k / k²
    phi_k = -delta_k / K2_safe
    phi_k[0, 0] = 0
    
    return phi_k


def compute_displacement_1lpt(
    phi_k: torch.Tensor,
    box_size: float = 256.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first-order (Zel'dovich) displacement field.
    
    Ψ^(1) = -∇φ
    
    In Fourier space:
        Ψ^(1)_k = -i k φ_k
    
    Parameters
    ----------
    phi_k : torch.Tensor
        Potential in Fourier space
    box_size : float
        Physical box size
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (Ψx, Ψy) displacement fields in real space
    """
    size = phi_k.shape[0]
    device = phi_k.device
    
    # Wavenumber grid
    kx = torch.fft.fftfreq(size, d=box_size/size, device=device) * 2 * np.pi
    ky = torch.fft.fftfreq(size, d=box_size/size, device=device) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    
    # Gradient in Fourier space: ∇ → ik
    psi_x_k = -1j * KX * phi_k
    psi_y_k = -1j * KY * phi_k
    
    # Transform to real space
    psi_x = torch.fft.ifft2(psi_x_k).real
    psi_y = torch.fft.ifft2(psi_y_k).real
    
    return psi_x, psi_y


def compute_displacement_2lpt(
    phi_k: torch.Tensor,
    box_size: float = 256.0,
    omega_m: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute second-order (2LPT) displacement field.
    
    The 2LPT potential φ^(2) satisfies:
        ∇²φ^(2) = Σᵢⱼ (φ,ᵢᵢ φ,ⱼⱼ - φ,ᵢⱼ²)
    
    In 2D:
        ∇²φ^(2) = φ,xx φ,yy - φ,xy²
    
    Parameters
    ----------
    phi_k : torch.Tensor
        Linear potential in Fourier space
    box_size : float
        Physical box size
    omega_m : float
        Matter density parameter
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (Ψx^(2), Ψy^(2)) second-order displacement fields
    """
    size = phi_k.shape[0]
    device = phi_k.device
    
    # Wavenumber grid
    kx = torch.fft.fftfreq(size, d=box_size/size, device=device) * 2 * np.pi
    ky = torch.fft.fftfreq(size, d=box_size/size, device=device) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2_safe = torch.where(K2 < 1e-10, torch.ones_like(K2), K2)
    
    # Compute second derivatives of φ in Fourier space
    # φ,xx → -kx² φ_k, φ,yy → -ky² φ_k, φ,xy → -kx ky φ_k
    phi_xx_k = -KX**2 * phi_k
    phi_yy_k = -KY**2 * phi_k
    phi_xy_k = -KX * KY * phi_k
    
    # Transform to real space
    phi_xx = torch.fft.ifft2(phi_xx_k).real
    phi_yy = torch.fft.ifft2(phi_yy_k).real
    phi_xy = torch.fft.ifft2(phi_xy_k).real
    
    # Source term for 2LPT potential
    source = phi_xx * phi_yy - phi_xy**2
    
    # Solve Poisson equation: ∇²φ^(2) = source
    source_k = torch.fft.fft2(source)
    phi2_k = source_k / K2_safe
    phi2_k[0, 0] = 0
    
    # Compute gradient of φ^(2)
    psi2_x_k = -1j * KX * phi2_k
    psi2_y_k = -1j * KY * phi2_k
    
    psi2_x = torch.fft.ifft2(psi2_x_k).real
    psi2_y = torch.fft.ifft2(psi2_y_k).real
    
    # Apply D₂/D₁² factor
    ratio = growth_factor_ratio(omega_m)
    psi2_x *= ratio
    psi2_y *= ratio
    
    return psi2_x, psi2_y


def cic_deposit(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    size: int,
    box_size: float = 256.0
) -> torch.Tensor:
    """
    Cloud-in-Cell (CIC) mass deposition onto a grid.
    
    CIC distributes each particle's mass to the 4 nearest grid points
    using bilinear interpolation weights.
    
    Parameters
    ----------
    positions_x, positions_y : torch.Tensor
        Particle positions, shape (size, size)
    size : int
        Grid size
    box_size : float
        Physical box size
        
    Returns
    -------
    torch.Tensor
        Density field on grid, shape (size, size)
    """
    device = positions_x.device
    
    # Normalize positions to grid coordinates [0, size)
    cell_size = box_size / size
    x_grid = positions_x / cell_size
    y_grid = positions_y / cell_size
    
    # Wrap to periodic box
    x_grid = x_grid % size
    y_grid = y_grid % size
    
    # Integer and fractional parts
    ix = x_grid.long()
    iy = y_grid.long()
    dx = x_grid - ix.float()
    dy = y_grid - iy.float()
    
    # Weights for 4 corners
    w00 = (1 - dx) * (1 - dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1 - dy)
    w11 = dx * dy
    
    # Initialize density grid
    density = torch.zeros(size, size, device=device)
    
    # Deposit to 4 corners (with periodic wrapping)
    ix0 = ix % size
    ix1 = (ix + 1) % size
    iy0 = iy % size
    iy1 = (iy + 1) % size
    
    # Flatten for scatter_add
    density.view(-1).scatter_add_(0, (ix0 * size + iy0).view(-1), w00.view(-1))
    density.view(-1).scatter_add_(0, (ix0 * size + iy1).view(-1), w01.view(-1))
    density.view(-1).scatter_add_(0, (ix1 * size + iy0).view(-1), w10.view(-1))
    density.view(-1).scatter_add_(0, (ix1 * size + iy1).view(-1), w11.view(-1))
    
    return density


def generate_2lpt_field(
    omega_m: float,
    size: int = 256,
    box_size: float = 256.0,
    growth_factor: float = 1.0,
    seed: Optional[int] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate a 2D dark matter density field using 2LPT.
    
    This produces density fields with realistic non-Gaussian features
    (filaments, voids, proto-halos) that match N-body simulations.
    
    Parameters
    ----------
    omega_m : float
        Matter density parameter Ω_m ∈ [0.1, 0.5]
    size : int
        Grid resolution (size × size)
    box_size : float
        Physical box size in Mpc/h
    growth_factor : float
        Linear growth factor D₁ (controls non-linearity)
        D₁ = 1.0 is roughly z=0, D₁ = 0.5 is roughly z=1
    seed : int, optional
        Random seed for reproducibility
    device : torch.device, optional
        Computation device (default: best available)
        
    Returns
    -------
    torch.Tensor
        Density contrast field δ = ρ/ρ̄ - 1, shape (size, size)
    """
    if device is None:
        device = get_device()
    
    # Generate linear potential
    phi_k = generate_linear_potential(size, omega_m, box_size, seed, device)
    
    # Compute displacement fields
    psi1_x, psi1_y = compute_displacement_1lpt(phi_k, box_size)
    psi2_x, psi2_y = compute_displacement_2lpt(phi_k, box_size, omega_m)
    
    # Total displacement: Ψ = D₁ Ψ^(1) + D₁² Ψ^(2)
    psi_x = growth_factor * psi1_x + growth_factor**2 * psi2_x
    psi_y = growth_factor * psi1_y + growth_factor**2 * psi2_y
    
    # Initial (Lagrangian) particle positions on regular grid
    cell_size = box_size / size
    q_x = torch.arange(size, device=device).float() * cell_size
    q_y = torch.arange(size, device=device).float() * cell_size
    QX, QY = torch.meshgrid(q_x, q_y, indexing='ij')
    
    # Final (Eulerian) positions: x = q + Ψ
    x_final = QX + psi_x
    y_final = QY + psi_y
    
    # Wrap to periodic box
    x_final = x_final % box_size
    y_final = y_final % box_size
    
    # Deposit particles onto grid using CIC
    density = cic_deposit(x_final, y_final, size, box_size)
    
    # Convert to density contrast: δ = ρ/ρ̄ - 1
    mean_density = density.mean()
    delta = (density - mean_density) / (mean_density + 1e-10)
    
    # Normalize for neural network input
    delta = (delta - delta.mean()) / (delta.std() + 1e-10)
    
    return delta


def generate_batch_2lpt(
    omega_m_values: torch.Tensor,
    size: int = 256,
    box_size: float = 256.0,
    growth_factor: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate a batch of 2LPT density fields (for fast training).
    
    Parameters
    ----------
    omega_m_values : torch.Tensor
        Batch of Ω_m values, shape (batch_size,)
    size : int
        Grid resolution
    box_size : float
        Physical box size
    growth_factor : float
        Linear growth factor
    device : torch.device
        Computation device
        
    Returns
    -------
    torch.Tensor
        Batch of density fields, shape (batch_size, 1, size, size)
    """
    if device is None:
        device = get_device()
    
    batch_size = len(omega_m_values)
    fields = torch.zeros(batch_size, 1, size, size, device=device)
    
    for i, omega_m in enumerate(omega_m_values):
        field = generate_2lpt_field(
            omega_m.item(), size, box_size, growth_factor, 
            seed=None, device=device
        )
        fields[i, 0] = field
    
    return fields


# =============================================================================
# NUMPY INTERFACE (for dataset generation)
# =============================================================================

def generate_2lpt_numpy(
    omega_m: float,
    size: int = 256,
    box_size: float = 256.0,
    growth_factor: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate 2LPT field and return as numpy array.
    
    Convenience function for dataset generation scripts.
    Uses CPU for DataLoader worker compatibility.
    """
    # Force CPU for multiprocessing compatibility in DataLoader workers
    device = torch.device('cpu')
    field = generate_2lpt_field(omega_m, size, box_size, growth_factor, seed, device)
    return field.numpy().astype(np.float32)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Benchmark
    print("\nBenchmarking 2LPT generation...")
    start = time.time()
    for _ in range(10):
        field = generate_2lpt_field(0.3, size=256, device=device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.time() - start
    print(f"10 fields in {elapsed:.2f}s ({elapsed/10*1000:.1f}ms per field)")
    
    # Visual comparison: GRF vs 2LPT
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    omega_values = [0.2, 0.3, 0.4, 0.5]
    
    for i, omega_m in enumerate(omega_values):
        # 2LPT field
        field_2lpt = generate_2lpt_field(omega_m, size=256, seed=42, device=device)
        field_2lpt = field_2lpt.cpu().numpy()
        
        # Normalize for display
        vmin, vmax = np.percentile(field_2lpt, [1, 99])
        
        axes[0, i].imshow(field_2lpt, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'2LPT: Ω_m = {omega_m}')
        axes[0, i].axis('off')
        
        # Power spectrum
        ps = np.abs(np.fft.fft2(field_2lpt))**2
        ps_radial = np.mean(ps, axis=0)[:128]
        axes[1, i].semilogy(ps_radial)
        axes[1, i].set_xlabel('k')
        axes[1, i].set_ylabel('P(k)')
        axes[1, i].set_title('Power Spectrum')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('2LPT Density Fields - GPU Accelerated', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/2lpt_fields.png', dpi=150)
    print("\nSaved results/figures/2lpt_fields.png")
