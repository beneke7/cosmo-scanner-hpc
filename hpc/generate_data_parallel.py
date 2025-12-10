#!/usr/bin/env python3
"""
generate_data_parallel.py - Parallel Data Generation for HPC
=============================================================

Generates 1M+ synthetic cosmological maps using all available CPU cores.
Optimized for 128-core systems with multiprocessing.

Usage:
    python hpc/generate_data_parallel.py --num_samples 100000 --num_workers 60
    python hpc/generate_data_parallel.py --preview  # Quick preview

Performance:
    - 60 workers: ~100K samples in ~5-10 minutes
    - Each sample: 256x256 JPG ~10-20KB
    - 100K samples: ~1-2GB disk space

Author: Cosmo Scanner HPC
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# GENERATION PARAMETERS (tuned for DES-like appearance)
# =============================================================================
#
# Physics: Uses BBKS transfer function with proper Ω_m dependence (v6.1)
# The shape parameter Γ = Ω_m × h encodes cosmology in the power spectrum.
# This creates realistic clustering patterns that the CNN can learn from.
#

PARAMS = {
    # Cosmology (BBKS transfer function)
    'hubble_h': 0.7,          # Hubble parameter h
    'spectral_index': 0.965,  # Primordial spectral index n_s
    'box_size': 256.0,        # Simulation box size
    
    # Smoothing
    'smoothing_sigma': 3.0,
    
    # Masks (star/galaxy holes)
    'add_masks': True,
    'min_masks': 3,
    'max_masks': 8,
    'min_mask_radius': 4,
    'max_mask_radius': 12,
    'mask_value': 0.5,
    
    # Grain texture (shape noise)
    'add_grain': True,
    'grain_noise_std': 0.02,
    'grain_smooth': 0.7,
    
    # Output normalization
    'output_mean': 0.5,
    'output_std': 0.15,
    
    # Omega_m range
    'omega_m_min': 0.1,
    'omega_m_max': 0.5,
}


# =============================================================================
# GENERATION FUNCTION (single sample)
# =============================================================================

def generate_single_sample(args):
    """
    Generate a single synthetic sample using PROPER PHYSICS.
    
    Uses the BBKS transfer function for realistic omega_m dependence.
    
    Args:
        args: tuple of (index, omega_m, size, output_dir)
    
    Returns:
        dict with filename and omega_m
    """
    from scipy.ndimage import gaussian_filter
    
    idx, omega_m, size, output_dir = args
    
    # Set seed for reproducibility
    np.random.seed(idx)
    
    # ==========================================================================
    # PROPER COSMOLOGICAL POWER SPECTRUM (BBKS transfer function)
    # ==========================================================================
    box_size = 256.0
    dk = 2 * np.pi / box_size
    
    kx = np.fft.fftfreq(size, d=1.0/size) * dk
    ky = np.fft.fftfreq(size, d=1.0/size) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    K_safe = np.where(K < 1e-10, 1.0, K)
    
    # BBKS transfer function - THIS IS WHERE OMEGA_M MATTERS!
    h = 0.7  # Hubble parameter
    gamma = omega_m * h  # Shape parameter depends on omega_m
    q = K_safe / (gamma + 1e-10)
    
    T_k = (
        np.log(1 + 2.34 * q) / (2.34 * q + 1e-10) * 
        (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
    )
    T_k = np.where(K < 1e-10, 1.0, T_k)
    
    # Primordial power spectrum with spectral index
    n_s = 0.965
    P_k = (K_safe ** n_s) * (T_k ** 2)
    P_k = np.where(K < 1e-10, 0.0, P_k)
    
    # ==========================================================================
    # Generate Gaussian random field
    # ==========================================================================
    amplitude = np.sqrt(P_k / 2 + 1e-20)
    field_k = amplitude * (np.random.randn(size, size) + 1j * np.random.randn(size, size))
    field_k[0, 0] = 0
    
    # Inverse FFT
    field = np.fft.ifft2(field_k).real
    
    # Smoothing
    if PARAMS['smoothing_sigma'] > 0:
        field = gaussian_filter(field, sigma=PARAMS['smoothing_sigma'], mode='wrap')
    
    # Normalize to [0, 1] range for image
    field = (field - field.mean()) / (field.std() + 1e-10)
    field = field * PARAMS['output_std'] + PARAMS['output_mean']
    
    # Add grain texture (shape noise)
    if PARAMS['add_grain'] and PARAMS['grain_noise_std'] > 0:
        grain = np.random.randn(size, size) * PARAMS['grain_noise_std']
        if PARAMS['grain_smooth'] > 0:
            grain = gaussian_filter(grain, sigma=PARAMS['grain_smooth'], mode='wrap')
        field = field + grain
    
    # Add masks (star/galaxy holes)
    if PARAMS['add_masks']:
        y, x = np.ogrid[:size, :size]
        n_masks = np.random.randint(PARAMS['min_masks'], PARAMS['max_masks'] + 1)
        for _ in range(n_masks):
            cx = np.random.randint(PARAMS['max_mask_radius'] + 5, size - PARAMS['max_mask_radius'] - 5)
            cy = np.random.randint(PARAMS['max_mask_radius'] + 5, size - PARAMS['max_mask_radius'] - 5)
            r = np.random.randint(PARAMS['min_mask_radius'], PARAMS['max_mask_radius'] + 1)
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            field[mask] = PARAMS['mask_value']
    
    # Clip to valid range and convert to uint8 for JPG
    field = np.clip(field, 0, 1)
    field_uint8 = (field * 255).astype(np.uint8)
    
    # Save as JPG
    from PIL import Image
    filename = f'sample_{idx:07d}.jpg'
    img = Image.fromarray(field_uint8, mode='L')
    img.save(output_dir / filename, quality=95)
    
    return {'filename': filename, 'omega_m': omega_m, 'seed': idx}


# =============================================================================
# PARALLEL GENERATION
# =============================================================================

def generate_dataset_parallel(
    output_dir: Path,
    num_samples: int,
    size: int = 256,
    num_workers: int = None,
    chunk_size: int = 1000
):
    """
    Generate dataset using parallel processing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        size: Image size
        num_workers: Number of parallel workers (default: CPU count - 8)
        chunk_size: Samples per progress update
    """
    import pandas as pd
    
    # Setup
    if num_workers is None:
        num_workers = max(1, cpu_count() - 8)  # Leave some cores for system
    
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate omega_m values (uniform distribution)
    omega_m_values = np.linspace(
        PARAMS['omega_m_min'], 
        PARAMS['omega_m_max'], 
        num_samples
    )
    np.random.shuffle(omega_m_values)
    
    # Prepare arguments
    args_list = [
        (i, omega_m_values[i], size, images_dir)
        for i in range(num_samples)
    ]
    
    print(f"Generating {num_samples:,} samples using {num_workers} workers...")
    print(f"Output: {output_dir}")
    print(f"Estimated disk usage: {num_samples * size * size * 4 / 1e9:.1f} GB")
    print()
    
    # Parallel generation with progress
    start_time = time.time()
    results = []
    
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_single_sample, args_list, chunksize=100)):
            results.append(result)
            
            # Progress update
            if (i + 1) % chunk_size == 0 or (i + 1) == num_samples:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_samples - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>8,}/{num_samples:,}] "
                      f"{100*(i+1)/num_samples:5.1f}% | "
                      f"{rate:.0f} samples/sec | "
                      f"ETA: {eta/60:.1f} min")
    
    # Save metadata
    df = pd.DataFrame(results)
    df = df.sort_values('filename').reset_index(drop=True)
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    total_time = time.time() - start_time
    print()
    print(f"Done! Generated {num_samples:,} samples in {total_time/60:.1f} minutes")
    print(f"Average rate: {num_samples/total_time:.0f} samples/sec")
    print(f"Metadata saved to: {output_dir / 'metadata.csv'}")
    
    return df


def generate_preview(output_dir: Path, num_preview: int = 8):
    """Generate preview images from existing JPGs."""
    import matplotlib.pyplot as plt
    from PIL import Image
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Read metadata to get omega_m values
    metadata_path = output_dir / 'metadata.csv'
    if metadata_path.exists():
        import pandas as pd
        df = pd.read_csv(metadata_path)
        # Sample evenly across omega_m range
        df_sorted = df.sort_values('omega_m')
        indices = np.linspace(0, len(df)-1, num_preview, dtype=int)
        samples = df_sorted.iloc[indices]
    else:
        # Fallback: just use first 8 images
        samples = [{'filename': f'sample_{i:07d}.jpg', 'omega_m': 0.3} for i in range(num_preview)]
    
    for i, (ax, (_, row)) in enumerate(zip(axes.flat, samples.iterrows() if hasattr(samples, 'iterrows') else enumerate(samples))):
        img_path = output_dir / 'images' / row['filename']
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(np.array(img), cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'Ω_m = {row["omega_m"]:.3f}', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
        ax.axis('off')
    
    plt.suptitle('Synthetic Data Preview (100K Dataset)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    preview_path = output_dir / 'preview.png'
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Preview saved to: {preview_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parallel synthetic data generation for HPC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_samples', type=int, default=100_000,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                        help='Output directory')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 8)')
    parser.add_argument('--preview', action='store_true',
                        help='Only generate preview, no full dataset')
    
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print system info
    print("=" * 70)
    print("COSMO-SCANNER HPC: Parallel Data Generation")
    print("=" * 70)
    print(f"CPU cores available: {cpu_count()}")
    print(f"Workers to use: {args.num_workers or max(1, cpu_count() - 8)}")
    print(f"Samples: {args.num_samples:,}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()
    
    if args.preview:
        print("Preview mode - generating sample images only")
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        generate_preview(output_dir / 'images', num_preview=8)
        return
    
    # Generate full dataset
    generate_dataset_parallel(
        output_dir,
        args.num_samples,
        args.size,
        args.num_workers
    )
    
    # Generate preview
    generate_preview(output_dir / 'images')
    
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
