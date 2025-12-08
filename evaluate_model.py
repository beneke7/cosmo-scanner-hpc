"""
evaluate_model.py - Detailed evaluation of trained model
=========================================================

Shows prediction vs ground truth for test set samples.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.config import (
    CosmoNet, create_dataloaders, get_device, SEED, MODEL_DIR
)


def load_model(model_path: str, device: torch.device) -> CosmoNet:
    """Load trained model from checkpoint."""
    model = CosmoNet().to(device)
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_on_test_set(model: CosmoNet, test_loader, device: torch.device):
    """Run inference on entire test set and collect predictions."""
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for fields, targets in test_loader:
            fields = fields.to(device)
            predictions = model(fields)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    return np.array(all_predictions), np.array(all_targets)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray):
    """Compute various error metrics."""
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    metrics = {
        'mse': np.mean(errors**2),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(abs_errors),
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        'std_error': np.std(errors),
        'mean_bias': np.mean(errors),  # Systematic over/under prediction
        'r2': 1 - np.sum(errors**2) / np.sum((targets - np.mean(targets))**2),
    }
    
    # Percentile errors
    metrics['p50_error'] = np.percentile(abs_errors, 50)
    metrics['p90_error'] = np.percentile(abs_errors, 90)
    metrics['p99_error'] = np.percentile(abs_errors, 99)
    
    return metrics


def plot_results(predictions: np.ndarray, targets: np.ndarray, save_path: str):
    """Create visualization of predictions vs targets."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot: Predicted vs True
    ax1 = axes[0, 0]
    ax1.scatter(targets, predictions, alpha=0.1, s=1)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('True Ω_m', fontsize=12)
    ax1.set_ylabel('Predicted Ω_m', fontsize=12)
    ax1.set_title('Predicted vs True Ω_m', fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    errors = predictions - targets
    ax2.hist(errors, bins=100, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax2.set_xlabel('Prediction Error (Predicted - True)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error vs True value (check for systematic bias)
    ax3 = axes[1, 0]
    ax3.scatter(targets, errors, alpha=0.1, s=1)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('True Ω_m', fontsize=12)
    ax3.set_ylabel('Prediction Error', fontsize=12)
    ax3.set_title('Error vs True Value (Bias Check)', fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. Absolute error vs True value
    ax4 = axes[1, 1]
    abs_errors = np.abs(errors)
    
    # Bin by true value and compute mean absolute error per bin
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(targets, bins) - 1
    bin_mae = [np.mean(abs_errors[bin_indices == i]) if np.sum(bin_indices == i) > 0 else 0 
               for i in range(len(bins)-1)]
    
    ax4.bar(bin_centers, bin_mae, width=0.045, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('True Ω_m', fontsize=12)
    ax4.set_ylabel('Mean Absolute Error', fontsize=12)
    ax4.set_title('MAE by Ω_m Range', fontsize=14)
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {save_path}")


def print_sample_predictions(predictions: np.ndarray, targets: np.ndarray, n_samples: int = 20):
    """Print a table of sample predictions."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (random subset)")
    print("=" * 60)
    print(f"{'Index':<8} {'True Ω_m':<12} {'Predicted':<12} {'Error':<12} {'% Error':<10}")
    print("-" * 60)
    
    indices = np.random.choice(len(predictions), n_samples, replace=False)
    indices = sorted(indices)
    
    for idx in indices:
        true_val = targets[idx]
        pred_val = predictions[idx]
        error = pred_val - true_val
        pct_error = (error / true_val) * 100 if true_val != 0 else 0
        print(f"{idx:<8} {true_val:<12.5f} {pred_val:<12.5f} {error:<+12.5f} {pct_error:<+10.1f}%")


def main():
    # Setup
    device = get_device()
    print(f"Device: {device}")
    
    # Find best model
    model_path = os.path.join(MODEL_DIR, "test5_best.pt")
    if not os.path.exists(model_path):
        # Try to find any model
        models = list(Path(MODEL_DIR).glob("*_best.pt"))
        if models:
            model_path = str(models[-1])
            print(f"Using model: {model_path}")
        else:
            print("No trained model found!")
            return
    
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = create_dataloaders(seed=SEED)
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Run evaluation
    print("Running inference on test set...")
    predictions, targets = evaluate_on_test_set(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    print(f"MSE:           {metrics['mse']:.6f}")
    print(f"RMSE:          {metrics['rmse']:.6f}")
    print(f"MAE:           {metrics['mae']:.6f}")
    print(f"R² Score:      {metrics['r2']:.6f}")
    print(f"Mean Bias:     {metrics['mean_bias']:+.6f}")
    print(f"Std of Error:  {metrics['std_error']:.6f}")
    print("-" * 60)
    print(f"Min Error:     {metrics['min_error']:.6f}")
    print(f"Median Error:  {metrics['p50_error']:.6f}")
    print(f"90th %ile:     {metrics['p90_error']:.6f}")
    print(f"99th %ile:     {metrics['p99_error']:.6f}")
    print(f"Max Error:     {metrics['max_error']:.6f}")
    
    # Print sample predictions
    print_sample_predictions(predictions, targets)
    
    # Create plots
    plot_path = "evaluation_results.png"
    plot_results(predictions, targets, plot_path)
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"• Average prediction error: ±{metrics['rmse']:.3f} in Ω_m")
    print(f"• This is {metrics['rmse']*100:.1f}% of the [0,1] range")
    print(f"• 90% of predictions are within ±{metrics['p90_error']:.3f}")
    print(f"• Model explains {metrics['r2']*100:.1f}% of variance")
    if abs(metrics['mean_bias']) > 0.01:
        direction = "over" if metrics['mean_bias'] > 0 else "under"
        print(f"• ⚠️  Systematic {direction}-prediction bias: {metrics['mean_bias']:+.4f}")
    else:
        print(f"• ✓ No significant systematic bias")


if __name__ == "__main__":
    main()
