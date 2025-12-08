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

from .physics import generate_universe

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
IMAGE_SIZE = 128
NUM_SAMPLES = 10000
OMEGA_M_MIN = 0.0
OMEGA_M_MAX = 1.0
OMEGA_M_DECIMALS = 5

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


def generate_dataset():
    """
    Generate the full dataset of dark matter density fields.
    
    - Generates NUM_SAMPLES images with random Omega_m values (5 decimals)
    - Saves images as: data/{index:05d}.jpg
    - Creates metadata.csv with columns: filename, omega_m
    """
    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 60)
    print("DATASET GENERATION")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}/")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Omega_m range: [{OMEGA_M_MIN}, {OMEGA_M_MAX}]")
    print(f"Omega_m precision: {OMEGA_M_DECIMALS} decimals")
    print("=" * 60)
    
    # Generate random Omega_m values
    omega_m_values = np.random.uniform(OMEGA_M_MIN, OMEGA_M_MAX, NUM_SAMPLES)
    omega_m_values = np.round(omega_m_values, OMEGA_M_DECIMALS)
    
    # Clamp minimum to avoid omega_m = 0 (physics breaks)
    omega_m_values = np.maximum(omega_m_values, 10 ** (-OMEGA_M_DECIMALS))
    
    # Open metadata file
    metadata = []
    
    # Generate samples
    for idx, omega_m in enumerate(tqdm(omega_m_values, desc="Generating")):
        # Generate the density field
        field = generate_universe(omega_m=omega_m, size=IMAGE_SIZE)
        
        # Convert to uint8 for image saving
        field_uint8 = normalize_to_uint8(field)
        
        # Create PIL image
        img = Image.fromarray(field_uint8)
        
        # Simple filename with index
        filename = f"{idx:05d}.jpg"
        filepath = os.path.join(DATA_DIR, filename)
        
        # Save as JPEG
        img.save(filepath, quality=95)
        
        # Record metadata
        metadata.append({'filename': filename, 'omega_m': omega_m})
    
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
