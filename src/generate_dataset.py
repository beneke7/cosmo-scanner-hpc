"""
generate_dataset.py - Pre-generate Training Dataset
====================================================

Generates dark matter density fields and saves them as JPG images.
This approach trades disk space for training speed - no on-the-fly generation.

Usage:
    python -m src.generate_dataset
"""

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from .physics import generate_universe

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
IMAGE_SIZE = 256
NUM_SAMPLES = 100000
OMEGA_M_MIN = 0.0
OMEGA_M_MAX = 1.0
OMEGA_M_DECIMALS = 5

# Parallel processing config - optimized for 60 cores, 40GB RAM
NUM_WORKERS = 60  # Use all available cores
CHUNK_SIZE = 100  # Process in chunks to manage memory

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================


def normalize_to_uint8(field: np.ndarray) -> np.ndarray:
    """
    Normalize field values to 0-255 range for image saving.
    
    Uses min-max normalization to preserve the full dynamic range.
    
    Parameters
    ----------
    field : np.ndarray
        2D density field with arbitrary value range.
        
    Returns
    -------
    np.ndarray
        Field normalized to uint8 (0-255).
    """
    # Clip extreme values (beyond 4 sigma)
    field_clipped = np.clip(field, -4, 4)
    
    # Normalize to [0, 1]
    field_norm = (field_clipped - field_clipped.min()) / (field_clipped.max() - field_clipped.min() + 1e-10)
    
    # Scale to [0, 255]
    field_uint8 = (field_norm * 255).astype(np.uint8)
    
    return field_uint8


def generate_single_sample(args):
    """
    Generate a single sample (for parallel processing).
    
    Parameters
    ----------
    args : tuple
        (idx, omega_m, image_size, data_dir)
    
    Returns
    -------
    dict
        Metadata entry with filename and omega_m
    """
    idx, omega_m, image_size, data_dir = args
    
    # Generate the density field
    field = generate_universe(omega_m=omega_m, size=image_size)
    
    # Convert to uint8 for image saving
    field_uint8 = normalize_to_uint8(field)
    
    # Create PIL image
    img = Image.fromarray(field_uint8)
    
    # Simple filename with index
    filename = f"{idx:05d}.jpg"
    filepath = os.path.join(data_dir, filename)
    
    # Save as JPEG
    img.save(filepath, quality=95)
    
    return {'filename': filename, 'omega_m': float(omega_m)}


def generate_dataset():
    """
    Generate the full dataset of dark matter density fields in parallel.
    
    - Generates NUM_SAMPLES images with random Omega_m values (5 decimals)
    - Uses multiprocessing to parallelize across CPU cores
    - Saves images as: data/{index:05d}.jpg
    - Creates metadata.csv with columns: filename, omega_m
    """
    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Determine number of workers
    num_workers = min(NUM_WORKERS, cpu_count())
    
    print("=" * 60)
    print("PARALLEL DATASET GENERATION")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}/")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Omega_m range: [{OMEGA_M_MIN}, {OMEGA_M_MAX}]")
    print(f"Omega_m precision: {OMEGA_M_DECIMALS} decimals")
    print(f"Parallel workers: {num_workers} (of {cpu_count()} available)")
    print(f"Chunk size: {CHUNK_SIZE}")
    print("=" * 60)
    
    # Generate random Omega_m values
    np.random.seed(42)  # For reproducibility
    omega_m_values = np.random.uniform(OMEGA_M_MIN, OMEGA_M_MAX, NUM_SAMPLES)
    omega_m_values = np.round(omega_m_values, OMEGA_M_DECIMALS)
    
    # Clamp minimum to avoid omega_m = 0 (physics breaks)
    omega_m_values = np.maximum(omega_m_values, 10 ** (-OMEGA_M_DECIMALS))
    
    # Prepare arguments for parallel processing
    args_list = [(idx, omega_m, IMAGE_SIZE, DATA_DIR) 
                 for idx, omega_m in enumerate(omega_m_values)]
    
    # Generate samples in parallel
    metadata = []
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better memory efficiency with progress bar
        for result in tqdm(pool.imap_unordered(generate_single_sample, args_list, 
                                                chunksize=CHUNK_SIZE),
                          total=NUM_SAMPLES,
                          desc="Generating"):
            metadata.append(result)
    
    # Sort metadata by filename to maintain order
    metadata.sort(key=lambda x: x['filename'])
    
    # Write metadata CSV
    with open(METADATA_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'omega_m'])
        writer.writeheader()
        writer.writerows(metadata)
    
    print("\n" + "=" * 60)
    print(f"✓ Generated {NUM_SAMPLES} images")
    print(f"✓ Images saved to {DATA_DIR}/")
    print(f"✓ Metadata saved to {METADATA_FILE}")
    print("=" * 60)
    
    # Print sample entries
    print("\nSample metadata entries:")
    for entry in metadata[:5]:
        print(f"  {entry['filename']} -> omega_m = {entry['omega_m']:.5f}")
    print("  ...")


if __name__ == "__main__":
    generate_dataset()
