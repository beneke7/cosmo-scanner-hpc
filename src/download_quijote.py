"""
download_quijote.py - Download Quijote simulation 2D density fields
====================================================================

Quijote provides simulations with known Ω_m values, perfect for validation.

Usage:
    python src/download_quijote.py
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path("data/real/quijote")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quijote 2D density field URLs (publicly available samples)
# Full dataset requires Globus transfer - these are sample files
QUIJOTE_SAMPLES = {
    # The Quijote team provides data via Globus, but we can get sample projections
    # from their public repository
    "readme": "https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/README.md",
}

# For full data access, use Globus:
GLOBUS_INFO = """
=== QUIJOTE FULL DATA ACCESS ===

The full Quijote dataset is available via Globus:

1. Install Globus Connect Personal:
   https://www.globus.org/globus-connect-personal

2. Quijote Globus endpoint:
   Endpoint: "Quijote simulations"
   Path: /Snapshots/fiducial/

3. Download 2D density projections:
   - Path: /2D_maps/density/
   - Files: density_field_z=0.00_*.npy
   - Each file: 256x256 or 512x512 projection

4. Cosmologies available:
   - fiducial: Ω_m = 0.3175 (15,000 realizations)
   - Om_p: Ω_m = 0.3275 (500 realizations)
   - Om_m: Ω_m = 0.3075 (500 realizations)

Documentation: https://quijote-simulations.readthedocs.io/
"""


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def create_sample_quijote_data():
    """
    Create synthetic Quijote-like data for testing the pipeline.
    
    This simulates what real Quijote data would look like.
    For actual Quijote data, use Globus transfer.
    """
    import numpy as np
    from PIL import Image
    
    print("Creating sample Quijote-like density fields...")
    
    # Quijote cosmologies
    cosmologies = {
        "fiducial": 0.3175,
        "Om_p": 0.3275,
        "Om_m": 0.3075,
        "Om_high": 0.40,
        "Om_low": 0.20,
    }
    
    samples_per_cosmo = 10  # Just for testing
    
    for cosmo_name, omega_m in cosmologies.items():
        cosmo_dir = OUTPUT_DIR / cosmo_name
        cosmo_dir.mkdir(exist_ok=True)
        
        for i in range(samples_per_cosmo):
            # Generate density field (simplified model)
            np.random.seed(i + hash(cosmo_name) % 10000)
            
            # Power spectrum amplitude scales with Ω_m
            amplitude = omega_m * 2
            
            # Generate Gaussian random field
            k = np.fft.fftfreq(256, d=1.0)
            kx, ky = np.meshgrid(k, k)
            k_mag = np.sqrt(kx**2 + ky**2)
            k_mag[0, 0] = 1  # Avoid division by zero
            
            # Power spectrum P(k) ∝ k^n with n depending on Ω_m
            n_s = 0.96  # Spectral index
            power = amplitude * k_mag ** (n_s - 4)
            power[0, 0] = 0
            
            # Generate random phases
            phases = np.random.uniform(0, 2 * np.pi, (256, 256))
            
            # Create field in Fourier space
            fft_field = np.sqrt(power) * np.exp(1j * phases)
            
            # Transform to real space
            field = np.fft.ifft2(fft_field).real
            
            # Apply non-linear transformation (log-normal)
            field = np.exp(field - field.mean())
            
            # Normalize to [0, 1]
            field = (field - field.min()) / (field.max() - field.min())
            
            # Save as image
            img = Image.fromarray((field * 255).astype(np.uint8))
            img.save(cosmo_dir / f"density_{i:04d}.jpg", quality=95)
        
        print(f"  {cosmo_name}: {samples_per_cosmo} samples (Ω_m = {omega_m})")
    
    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.csv"
    with open(metadata_path, 'w') as f:
        f.write("filename,omega_m,cosmology\n")
        for cosmo_name, omega_m in cosmologies.items():
            for i in range(samples_per_cosmo):
                f.write(f"{cosmo_name}/density_{i:04d}.jpg,{omega_m},{cosmo_name}\n")
    
    print(f"Metadata saved to: {metadata_path}")


def main():
    print("=" * 60)
    print("QUIJOTE SIMULATION DATA")
    print("=" * 60)
    
    print(GLOBUS_INFO)
    
    print("\nFor now, creating sample Quijote-like data for pipeline testing...")
    create_sample_quijote_data()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Install Globus Connect Personal")
    print("2. Download real Quijote 2D projections")
    print("3. Run src/preprocess_real.py to convert to model format")
    print("=" * 60)


if __name__ == "__main__":
    main()
