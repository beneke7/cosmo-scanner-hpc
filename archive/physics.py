"""
physics.py - The Universe Simulator
====================================

This module generates synthetic dark matter density fields as Gaussian Random 
Fields (GRFs). These fields serve as training data for our neural network.

THEORETICAL BACKGROUND
----------------------
In cosmology, the matter density field δ(x) = (ρ(x) - ρ̄)/ρ̄ is assumed to be
a Gaussian Random Field in the early universe. This means:
1. The field is completely characterized by its two-point correlation function
2. All higher-order correlations vanish (or factorize into products of 2-point)
3. In Fourier space, different k-modes are independent

The Power Spectrum P(k) is the Fourier transform of the two-point correlation:
    ⟨δ(k)δ*(k')⟩ = (2π)³ δ_D(k-k') P(k)

For the primordial spectrum with scale-invariant initial conditions:
    P(k) ∝ k^{n_s} T²(k)

where:
- n_s ≈ 0.965 is the scalar spectral index (from inflation)
- T(k) is the transfer function encoding sub-horizon physics

BBKS TRANSFER FUNCTION
----------------------
The Bardeen-Bond-Kaiser-Szalay (1986) transfer function is an analytic fit:
    T(q) = ln(1 + 2.34q) / (2.34q) × [1 + 3.89q + (16.1q)² + (5.46q)³ + (6.71q)⁴]^{-1/4}

where q = k / (Ω_m h² Mpc⁻¹) is the normalized wavenumber.

The key physics: Ω_m determines the equality scale k_eq where radiation and 
matter energy densities are equal. This sets the "turnover" in P(k).
"""

import numpy as np
from scipy import fft
from typing import Tuple, Optional


def bbks_transfer_function(k: np.ndarray, omega_m: float, h: float = 0.7) -> np.ndarray:
    """
    Compute the BBKS transfer function T(k).
    
    The transfer function encodes how primordial fluctuations are processed
    by sub-horizon physics (radiation pressure, neutrino free-streaming, etc).
    It acts like a low-pass filter, suppressing power on small scales.
    
    Parameters
    ----------
    k : np.ndarray
        Wavenumber array in units of h/Mpc.
    omega_m : float
        Matter density parameter Ω_m (dimensionless).
    h : float
        Hubble parameter H_0 = 100h km/s/Mpc (default: 0.7).
        
    Returns
    -------
    np.ndarray
        Transfer function values T(k).
        
    Notes
    -----
    The shape parameter Γ = Ω_m h controls the turnover scale.
    Physical interpretation: higher Ω_m → earlier matter-radiation equality
    → larger k_eq → more power on small scales relative to large.
    """
    # Shape parameter (encodes epoch of matter-radiation equality)
    gamma = omega_m * h
    
    # Normalized wavenumber
    q = k / gamma
    
    # Avoid division by zero at k=0
    q = np.where(q < 1e-10, 1e-10, q)
    
    # BBKS fitting formula (Eq. 7 from Bardeen et al. 1986)
    # This is essentially a transfer "Green's function" from initial conditions
    T = (
        np.log(1 + 2.34 * q) / (2.34 * q) * 
        (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
    )
    
    return T


def primordial_power_spectrum(
    k: np.ndarray, 
    omega_m: float, 
    A_s: float = 2.1e-9,
    n_s: float = 0.965,
    h: float = 0.7
) -> np.ndarray:
    """
    Compute the matter power spectrum P(k).
    
    The power spectrum is the "energy" at each Fourier mode. It determines
    the variance of density fluctuations at each scale.
    
    P(k) = A_s × k^{n_s} × T²(k)
    
    Parameters
    ----------
    k : np.ndarray
        Wavenumber array in units of h/Mpc.
    omega_m : float
        Matter density parameter Ω_m.
    A_s : float
        Scalar amplitude (primordial fluctuation amplitude).
    n_s : float
        Scalar spectral index (n_s=1 is Harrison-Zel'dovich scale-invariant).
    h : float
        Hubble parameter.
        
    Returns
    -------
    np.ndarray
        Power spectrum P(k).
        
    Notes
    -----
    The dependence on Ω_m enters through T(k). This is the signal our
    neural network must learn to extract: given a realization of δ(x),
    infer the underlying Ω_m that generated the statistics.
    """
    T_k = bbks_transfer_function(k, omega_m, h)
    
    # Primordial power law × transfer function squared
    # The T² appears because P(k) ∝ |δ_k|² and δ_k ∝ T(k) × primordial
    P_k = A_s * (k ** n_s) * (T_k ** 2)
    
    return P_k


def generate_universe(
    omega_m: float,
    size: int = 64,
    box_size: float = 100.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a 2D dark matter density field as a Gaussian Random Field.
    
    ALGORITHM
    ---------
    1. Construct k-space grid with proper FFT frequencies
    2. Compute P(k) at each k-mode
    3. Draw amplitudes from Rayleigh distribution: |δ_k| ~ √(P(k)/2) × χ_2
    4. Draw phases uniformly: φ_k ~ U[0, 2π)
    5. Construct complex field: δ_k = |δ_k| × exp(iφ_k)
    6. Apply IFFT with proper normalization to get δ(x)
    
    The random phases encode Gaussianity: independence of Fourier modes
    means δ(x) is Gaussian by the Central Limit Theorem.
    
    Parameters
    ----------
    omega_m : float
        Matter density parameter Ω_m ∈ [0.1, 0.5] typically.
    size : int
        Grid resolution (size × size pixels).
    box_size : float
        Physical box size in Mpc/h.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Real-space density field δ(x) with shape (size, size).
        
    Notes
    -----
    IFFT NORMALIZATION:
    scipy.fft.ifft2 uses the convention: 
        f(x) = (1/N) Σ_k F(k) exp(2πi k·x/N)
    
    Our δ_k has units such that ⟨|δ_k|²⟩ = P(k). After IFFT, we get a 
    dimensionless field δ(x). The factor of √(N²) from discrete → continuous
    is absorbed into our amplitude scaling.
    
    PHYSICAL INTERPRETATION:
    Think of this as solving a stochastic PDE:
        ∇²φ = 4πG ρ̄ δ
    with random initial conditions drawn from the primordial power spectrum.
    Each realization is one possible universe consistent with inflation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Construct frequency grid (in units of h/Mpc)
    # fftfreq gives frequencies in cycles per sample; multiply by 2π/L for k
    kx = fft.fftfreq(size, d=box_size/size) * 2 * np.pi
    ky = fft.fftfreq(size, d=box_size/size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Magnitude of wavevector
    K = np.sqrt(KX**2 + KY**2)
    
    # Compute power spectrum (handle k=0 separately)
    K_safe = np.where(K < 1e-10, 1e-10, K)
    P_k = primordial_power_spectrum(K_safe, omega_m)
    P_k[0, 0] = 0  # Zero mean (no DC component)
    
    # Generate complex Gaussian field in k-space
    # Real and imaginary parts are independent N(0, P(k)/2)
    # This is equivalent to Rayleigh amplitude + uniform phase
    amplitude = np.sqrt(P_k / 2)
    delta_k = amplitude * (np.random.randn(size, size) + 1j * np.random.randn(size, size))
    
    # Transform to real space
    # The result should be real (up to numerical precision) because P(-k) = P(k)
    # and we're using the proper Hermitian symmetry implicitly
    delta_x = fft.ifft2(delta_k).real
    
    # Normalize to have unit variance (makes training easier)
    # This is a form of "field renormalization" - we're rescaling to canonical units
    delta_x = (delta_x - delta_x.mean()) / (delta_x.std() + 1e-10)
    
    return delta_x.astype(np.float32)


def compute_power_spectrum_2d(field: np.ndarray, box_size: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the power spectrum from a 2D field realization.
    
    This is the "inverse" operation to generation: given δ(x), estimate P(k).
    Useful for validation and visualization.
    
    Parameters
    ----------
    field : np.ndarray
        Real-space field with shape (size, size).
    box_size : float
        Physical box size in Mpc/h.
        
    Returns
    -------
    k_bins : np.ndarray
        Wavenumber bin centers.
    P_k : np.ndarray
        Azimuthally averaged power spectrum.
    """
    size = field.shape[0]
    
    # FFT and compute power
    delta_k = fft.fft2(field)
    power_2d = np.abs(delta_k)**2
    
    # Construct k-grid
    kx = fft.fftfreq(size, d=box_size/size) * 2 * np.pi
    ky = fft.fftfreq(size, d=box_size/size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    
    # Azimuthal averaging in k-shells
    k_max = np.max(K)
    n_bins = size // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
    
    P_k = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_edges[i]) & (K < k_edges[i+1])
        if np.sum(mask) > 0:
            P_k[i] = np.mean(power_2d[mask])
    
    return k_bins, P_k


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, omega_m in enumerate([0.2, 0.3, 0.4]):
        field = generate_universe(omega_m, size=128, seed=42)
        axes[i].imshow(field, cmap='viridis', origin='lower')
        axes[i].set_title(f'Ωm = {omega_m}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_universes.png', dpi=150)
    print("Saved test_universes.png")
