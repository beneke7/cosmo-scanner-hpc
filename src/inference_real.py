"""
inference_real.py - Run inference on real/simulation data
==========================================================

Tests trained models on Quijote simulations and DES-like data.

Usage:
    python src/inference_real.py --model hybrid_v1
    python src/inference_real.py --model test5 --dataset quijote
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import config
import importlib.util
spec = importlib.util.spec_from_file_location("config", os.path.join(PROJECT_ROOT, "train", "config.py"))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def load_model(model_name: str, device: torch.device):
    """Load a trained model."""
    model_path = Path(PROJECT_ROOT) / "train" / "models" / f"{model_name}_best.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Determine model type from name
    if "hybrid" in model_name.lower():
        model = config.CosmoNetHybrid().to(device)
        is_hybrid = True
    else:
        model = config.CosmoNet().to(device)
        is_hybrid = False
    
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model: {model_name} ({'hybrid' if is_hybrid else 'CNN-only'})")
    return model, is_hybrid


def load_dataset(dataset_name: str) -> tuple:
    """Load a real dataset with metadata."""
    if dataset_name == "quijote":
        data_dir = Path(PROJECT_ROOT) / "data" / "real" / "quijote"
    elif dataset_name == "des":
        data_dir = Path(PROJECT_ROOT) / "data" / "real" / "des"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    metadata_path = data_dir / "metadata.csv"
    
    samples = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column names
            omega_m = float(row.get('omega_m', row.get('omega_m_expected', 0.315)))
            samples.append({
                'filename': row['filename'],
                'omega_m': omega_m,
                'path': data_dir / row['filename']
            })
    
    print(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples, data_dir


def run_inference(model, samples: list, device: torch.device, is_hybrid: bool):
    """Run inference on all samples."""
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sample in samples:
            # Load image
            img = Image.open(sample['path']).convert('L')
            field = np.array(img, dtype=np.float32) / 255.0
            
            # Prepare input
            field_tensor = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).to(device)
            
            if is_hybrid:
                # Compute power spectrum
                ps = config.compute_power_spectrum(field, config.N_POWER_BINS)
                ps_tensor = torch.from_numpy(ps).unsqueeze(0).to(device)
                output = model(field_tensor, ps_tensor)
            else:
                output = model(field_tensor)
            
            pred = output.cpu().numpy().item()
            predictions.append(pred)
            targets.append(sample['omega_m'])
    
    return np.array(predictions), np.array(targets)


def plot_results(predictions: np.ndarray, targets: np.ndarray, 
                 dataset_name: str, model_name: str, output_path: Path):
    """Create visualization of inference results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Predictions vs Targets
    ax1 = axes[0]
    unique_targets = np.unique(targets)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_targets)))
    
    for i, target in enumerate(unique_targets):
        mask = targets == target
        ax1.scatter([target] * np.sum(mask), predictions[mask], 
                   alpha=0.6, c=[colors[i]], label=f'Ω_m={target:.3f}', s=50)
    
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('True Ω_m', fontsize=12)
    ax1.set_ylabel('Predicted Ω_m', fontsize=12)
    ax1.set_title(f'{dataset_name.upper()}: Predictions vs True', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, 0.6)
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot by cosmology
    ax2 = axes[1]
    data_by_target = [predictions[targets == t] for t in unique_targets]
    bp = ax2.boxplot(data_by_target, labels=[f'{t:.3f}' for t in unique_targets])
    ax2.axhline(y=0.315, color='r', linestyle='--', label='Planck Ω_m')
    ax2.set_xlabel('True Ω_m', fontsize=12)
    ax2.set_ylabel('Predicted Ω_m', fontsize=12)
    ax2.set_title('Prediction Distribution by Cosmology', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error analysis
    ax3 = axes[2]
    errors = predictions - targets
    ax3.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(errors):.3f}')
    ax3.set_xlabel('Prediction Error', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model: {model_name} | Dataset: {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def print_results(predictions: np.ndarray, targets: np.ndarray, dataset_name: str):
    """Print summary statistics."""
    errors = predictions - targets
    
    print("\n" + "=" * 60)
    print(f"INFERENCE RESULTS: {dataset_name.upper()}")
    print("=" * 60)
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  MSE:  {np.mean(errors**2):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.6f}")
    print(f"  MAE:  {np.mean(np.abs(errors)):.6f}")
    print(f"  Bias: {np.mean(errors):+.6f}")
    
    # Per-cosmology results
    print(f"\nPer-Cosmology Results:")
    print(f"{'True Ω_m':<12} {'Mean Pred':<12} {'Std':<10} {'Error':<10}")
    print("-" * 44)
    
    for target in np.unique(targets):
        mask = targets == target
        pred_mean = np.mean(predictions[mask])
        pred_std = np.std(predictions[mask])
        error = pred_mean - target
        print(f"{target:<12.4f} {pred_mean:<12.4f} {pred_std:<10.4f} {error:<+10.4f}")
    
    # Comparison to Planck
    print(f"\n{'='*60}")
    print("COMPARISON TO PLANCK (Ω_m = 0.315 ± 0.007)")
    print("=" * 60)
    overall_pred = np.mean(predictions)
    overall_std = np.std(predictions)
    print(f"Model prediction: {overall_pred:.4f} ± {overall_std:.4f}")
    print(f"Planck value:     0.3153 ± 0.0073")
    print(f"Difference:       {overall_pred - 0.3153:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on real data")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., hybrid_v1, test5)")
    parser.add_argument("--dataset", type=str, default="both", 
                       choices=["quijote", "des", "both"], help="Dataset to test on")
    args = parser.parse_args()
    
    device = config.get_device()
    print(f"Device: {device}")
    
    # Load model
    model, is_hybrid = load_model(args.model, device)
    
    # Run on selected datasets
    datasets = ["quijote", "des"] if args.dataset == "both" else [args.dataset]
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Testing on: {dataset_name.upper()}")
        print("=" * 60)
        
        try:
            samples, data_dir = load_dataset(dataset_name)
            predictions, targets = run_inference(model, samples, device, is_hybrid)
            
            # Print results
            print_results(predictions, targets, dataset_name)
            
            # Save plot
            output_path = Path(PROJECT_ROOT) / f"inference_{dataset_name}_{args.model}.png"
            plot_results(predictions, targets, dataset_name, args.model, output_path)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")


if __name__ == "__main__":
    main()
