"""
analyze_real.py - Analyze and compare real vs synthetic data
============================================================

Visualizes differences between training data and real observations.

Usage:
    python src/analyze_real.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import from train package
import importlib.util
spec = importlib.util.spec_from_file_location("config", os.path.join(PROJECT_ROOT, "train", "config.py"))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

compute_power_spectrum = config_module.compute_power_spectrum
N_POWER_BINS = config_module.N_POWER_BINS

# Directories
SYNTHETIC_DIR = Path("data/images")
QUIJOTE_DIR = Path("data/real/quijote")
DES_DIR = Path("data/real/des")
OUTPUT_DIR = Path("data/real")


def load_random_samples(directory: Path, n_samples: int = 5) -> list:
    """Load random image samples from a directory."""
    images = list(directory.glob("**/*.jpg")) + list(directory.glob("**/*.png"))
    images = [f for f in images if "mask" not in f.name]
    
    if len(images) == 0:
        return []
    
    np.random.seed(42)
    selected = np.random.choice(images, min(n_samples, len(images)), replace=False)
    
    samples = []
    for path in selected:
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        samples.append((path.name, arr))
    
    return samples


def compute_statistics(field: np.ndarray) -> dict:
    """Compute field statistics."""
    return {
        'mean': np.mean(field),
        'std': np.std(field),
        'min': np.min(field),
        'max': np.max(field),
        'skewness': float(np.mean(((field - np.mean(field)) / np.std(field)) ** 3)),
        'kurtosis': float(np.mean(((field - np.mean(field)) / np.std(field)) ** 4) - 3),
    }


def plot_comparison(synthetic: list, quijote: list, des: list, output_path: Path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    datasets = [
        ("Synthetic (Training)", synthetic),
        ("Quijote (Simulation)", quijote),
        ("DES (Real)", des),
    ]
    
    # Row 0-2: Sample images
    for row, (name, samples) in enumerate(datasets):
        if not samples:
            for col in range(5):
                axes[row, col].text(0.5, 0.5, "No data", ha='center', va='center')
                axes[row, col].axis('off')
            axes[row, 0].set_ylabel(name, fontsize=12)
            continue
            
        for col, (fname, field) in enumerate(samples[:5]):
            axes[row, col].imshow(field, cmap='viridis', vmin=0, vmax=1)
            axes[row, col].set_title(fname[:15], fontsize=8)
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(name, fontsize=12)
    
    # Row 3: Power spectra comparison
    for col in range(5):
        axes[3, col].axis('off')
    
    # Create power spectrum comparison in the middle
    ax_ps = fig.add_axes([0.15, 0.05, 0.7, 0.18])
    
    colors = {'Synthetic': 'blue', 'Quijote': 'green', 'DES': 'red'}
    
    for name, samples in [("Synthetic", synthetic), ("Quijote", quijote), ("DES", des)]:
        if not samples:
            continue
        
        # Average power spectrum
        ps_list = [compute_power_spectrum(s[1], N_POWER_BINS) for s in samples]
        ps_mean = np.mean(ps_list, axis=0)
        ps_std = np.std(ps_list, axis=0)
        
        k = np.arange(N_POWER_BINS)
        ax_ps.plot(k, ps_mean, label=name, color=colors[name], linewidth=2)
        ax_ps.fill_between(k, ps_mean - ps_std, ps_mean + ps_std, alpha=0.2, color=colors[name])
    
    ax_ps.set_xlabel('k (bin)', fontsize=12)
    ax_ps.set_ylabel('P(k) (normalized)', fontsize=12)
    ax_ps.set_title('Power Spectrum Comparison', fontsize=14)
    ax_ps.legend()
    ax_ps.grid(True, alpha=0.3)
    
    plt.suptitle('Data Comparison: Synthetic vs Simulations vs Real', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.22, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_path}")


def print_statistics(name: str, samples: list):
    """Print statistics for a dataset."""
    if not samples:
        print(f"\n{name}: No data available")
        return
    
    print(f"\n{name}:")
    print("-" * 40)
    
    all_stats = [compute_statistics(s[1]) for s in samples]
    
    for key in ['mean', 'std', 'skewness', 'kurtosis']:
        values = [s[key] for s in all_stats]
        print(f"  {key:12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def main():
    print("=" * 60)
    print("DATA ANALYSIS: Synthetic vs Real")
    print("=" * 60)
    
    # Load samples
    print("\nLoading samples...")
    synthetic = load_random_samples(SYNTHETIC_DIR, n_samples=5)
    quijote = load_random_samples(QUIJOTE_DIR, n_samples=5)
    des = load_random_samples(DES_DIR, n_samples=5)
    
    print(f"  Synthetic: {len(synthetic)} samples")
    print(f"  Quijote:   {len(quijote)} samples")
    print(f"  DES:       {len(des)} samples")
    
    # Print statistics
    print_statistics("Synthetic (Training Data)", synthetic)
    print_statistics("Quijote (Simulations)", quijote)
    print_statistics("DES (Real Observations)", des)
    
    # Create comparison plot
    print("\nCreating comparison visualization...")
    plot_comparison(synthetic, quijote, des, OUTPUT_DIR / "data_comparison.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
