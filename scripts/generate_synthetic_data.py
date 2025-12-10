#!/usr/bin/env python3
"""
Synthetic DES-like Data Generator (v5.3)
=========================================

Pre-generates synthetic weak lensing maps for training.
Uses a Gaussian Random Field with red power spectrum, smoothing,
masks, and grainy texture to match real DES data.

Usage:
    python scripts/generate_synthetic_data.py --num_samples 100000
    python scripts/generate_synthetic_data.py --preview_only

Output:
    - data/synthetic/images/*.npy (256x256 float32 arrays)
    - data/synthetic/metadata.csv (omega_m labels)
    - data/synthetic/sample_preview.png (gets overwritten each run)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# TUNABLE PARAMETERS - Adjust these to match real DES appearance
# =============================================================================

# Power spectrum
SPECTRAL_SLOPE = -3.0           # Red spectrum slope (more negative = more large-scale)
SLOPE_OMEGA_DEPENDENCE = 0.5    # How much slope changes with Omega_m

# Smoothing
SMOOTHING_SIGMA = 3.0           # Gaussian smoothing in pixels

# Masks (star/galaxy holes)
ADD_MASKS = True
MIN_MASKS = 3                   # Minimum number of masks per image
MAX_MASKS = 8                   # Maximum number of masks per image  
MIN_MASK_RADIUS = 4             # Minimum mask radius in pixels
MAX_MASK_RADIUS = 12            # Maximum mask radius in pixels
MASK_VALUE = 0.5                # Gray value for masks (0-1 range)

# Grain texture (shape noise)
ADD_GRAIN = True
GRAIN_NOISE_STD = 0.02          # Grain amplitude (in normalized 0-1 range)
GRAIN_SMOOTH = 0.7              # Slight smoothing of grain texture

# Output normalization
OUTPUT_MEAN = 0.5               # Target mean of output images
OUTPUT_STD = 0.15               # Target std of output images

# Omega_m range (evenly distributed)
OMEGA_M_MIN = 0.0
OMEGA_M_MAX = 1.0

# =============================================================================
# GENERATOR FUNCTION
# =============================================================================

def generate_des_synthetic(
    omega_m: float,
    size: int = 256,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a synthetic DES-like weak lensing map.
    
    Uses parameters defined at the top of this file.
    
    Parameters
    ----------
    omega_m : float
        Matter density parameter (0 - 1)
    size : int
        Map size in pixels
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Synthetic map, shape (size, size), normalized to [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Wavenumber grid
    kx = np.fft.fftfreq(size) * size
    ky = np.fft.fftfreq(size) * size
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1  # Avoid division by zero
    
    # Power spectrum: P(k) ~ k^n with Omega_m dependence
    # Higher Omega_m -> less negative slope -> more small-scale power
    n_spectral = SPECTRAL_SLOPE + SLOPE_OMEGA_DEPENDENCE * (omega_m - 0.5)
    P_k = K ** n_spectral
    P_k[0, 0] = 0  # Zero mean
    
    # Suppress high frequencies
    P_k *= np.exp(-(K / 30)**2)
    
    # Generate Gaussian random field
    amplitude = np.sqrt(P_k + 1e-10)
    field_k = amplitude * (np.random.randn(size, size) + 1j * np.random.randn(size, size))
    field_k[0, 0] = 0
    
    # Inverse FFT
    field = np.fft.ifft2(field_k).real
    
    # Apply Gaussian smoothing
    if SMOOTHING_SIGMA > 0:
        field = gaussian_filter(field, sigma=SMOOTHING_SIGMA, mode='wrap')
    
    # Normalize to target mean and std
    field = (field - field.mean()) / (field.std() + 1e-10)
    field = field * OUTPUT_STD + OUTPUT_MEAN
    
    # Add grainy texture (shape noise)
    if ADD_GRAIN and GRAIN_NOISE_STD > 0:
        grain = np.random.randn(size, size) * GRAIN_NOISE_STD
        if GRAIN_SMOOTH > 0:
            grain = gaussian_filter(grain, sigma=GRAIN_SMOOTH, mode='wrap')
        field = field + grain
    
    # Add circular masks
    if ADD_MASKS:
        y, x = np.ogrid[:size, :size]
        n_masks = np.random.randint(MIN_MASKS, MAX_MASKS + 1)
        for _ in range(n_masks):
            cx = np.random.randint(MAX_MASK_RADIUS + 5, size - MAX_MASK_RADIUS - 5)
            cy = np.random.randint(MAX_MASK_RADIUS + 5, size - MAX_MASK_RADIUS - 5)
            r = np.random.randint(MIN_MASK_RADIUS, MAX_MASK_RADIUS + 1)
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            field[mask] = MASK_VALUE
    
    # Clip to valid range
    field = np.clip(field, 0, 1)
    
    return field.astype(np.float32)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_dataset(
    output_dir: Path,
    num_samples: int,
    size: int = 256,
    show_progress: bool = True
):
    """
    Generate a full dataset of synthetic maps.
    
    Parameters
    ----------
    output_dir : Path
        Output directory
    num_samples : int
        Number of samples to generate
    size : int
        Image size in pixels
    show_progress : bool
        Show progress bar
    """
    # Create directories
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate evenly distributed Omega_m values
    omega_m_values = np.linspace(OMEGA_M_MIN, OMEGA_M_MAX, num_samples)
    np.random.shuffle(omega_m_values)  # Shuffle for training
    
    # Generate samples
    metadata = []
    iterator = tqdm(range(num_samples), desc='Generating') if show_progress else range(num_samples)
    
    for i in iterator:
        omega_m = omega_m_values[i]
        field = generate_des_synthetic(omega_m, size=size, seed=i)
        
        # Save as numpy array
        filename = f'sample_{i:06d}.npy'
        np.save(images_dir / filename, field)
        
        metadata.append({
            'filename': filename,
            'omega_m': omega_m,
            'seed': i
        })
    
    # Save metadata
    import pandas as pd
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f'Generated {num_samples} samples in {output_dir}')
    print(f'Omega_m range: [{OMEGA_M_MIN}, {OMEGA_M_MAX}]')
    
    return df


def generate_preview(output_dir: Path, num_preview: int = 8):
    """
    Generate a preview image showing sample outputs.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    omega_values = np.linspace(OMEGA_M_MIN, OMEGA_M_MAX, num_preview)
    
    for i, (ax, omega_m) in enumerate(zip(axes.flat, omega_values)):
        field = generate_des_synthetic(omega_m, size=256, seed=1000 + i)
        ax.imshow(field, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Ω_m = {omega_m:.3f}', fontsize=11)
        ax.axis('off')
    
    plt.suptitle(
        f'Synthetic DES v5.3 Preview\n'
        f'slope={SPECTRAL_SLOPE}, smooth={SMOOTHING_SIGMA}, '
        f'grain={GRAIN_NOISE_STD}, masks={MIN_MASKS}-{MAX_MASKS}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    
    preview_path = output_dir / 'sample_preview.png'
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Preview saved to {preview_path}')
    return preview_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic DES-like data for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--num_samples', type=int, default=100000,
        help='Number of samples to generate (default: 100000)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/synthetic',
        help='Output directory (default: data/synthetic)'
    )
    parser.add_argument(
        '--size', type=int, default=256,
        help='Image size in pixels (default: 256)'
    )
    parser.add_argument(
        '--preview_only', action='store_true',
        help='Only generate preview image, no full dataset'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Print current parameters
    print('=' * 60)
    print('Synthetic DES Data Generator v5.3')
    print('=' * 60)
    print(f'Parameters:')
    print(f'  Spectral slope: {SPECTRAL_SLOPE}')
    print(f'  Slope Omega dependence: {SLOPE_OMEGA_DEPENDENCE}')
    print(f'  Smoothing sigma: {SMOOTHING_SIGMA}')
    print(f'  Masks: {MIN_MASKS}-{MAX_MASKS} (radius {MIN_MASK_RADIUS}-{MAX_MASK_RADIUS})')
    print(f'  Grain: std={GRAIN_NOISE_STD}, smooth={GRAIN_SMOOTH}')
    print(f'  Omega_m range: [{OMEGA_M_MIN}, {OMEGA_M_MAX}]')
    print('=' * 60)
    
    # Generate preview
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_preview(output_dir)
    
    if args.preview_only:
        print('Preview only mode - skipping full dataset generation')
        return
    
    # Generate full dataset
    generate_dataset(output_dir, args.num_samples, args.size)
    
    print('Done!')


if __name__ == '__main__':
    main()
