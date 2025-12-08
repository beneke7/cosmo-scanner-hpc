"""
Main entry point for cosmo-scanner-hpc.

Usage:
    python -m src [OPTIONS]
    
Run `python -m src --help` for available options.
"""

import argparse
import sys

from .train import train


def main():
    parser = argparse.ArgumentParser(
        description="cosmo-scanner-hpc: Deep Learning for Cosmological Parameter Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=1e-3,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Weight decay (L2 regularization strength)"
    )
    
    # Data configuration
    parser.add_argument(
        "--train-samples", type=int, default=10000,
        help="Number of training samples per epoch"
    )
    parser.add_argument(
        "--val-samples", type=int, default=1000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--field-size", type=int, default=64,
        help="Spatial resolution of generated fields"
    )
    parser.add_argument(
        "--omega-m-min", type=float, default=0.1,
        help="Minimum Omega_m value"
    )
    parser.add_argument(
        "--omega-m-max", type=float, default=0.5,
        help="Maximum Omega_m value"
    )
    
    # Device configuration
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force specific compute device"
    )
    
    # Checkpointing
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="Directory for model checkpoints"
    )
    
    # Weights & Biases
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-wandb", dest="wandb", action="store_false",
        help="Disable Weights & Biases logging"
    )
    parser.set_defaults(wandb=True)
    parser.add_argument(
        "--wandb-project", type=str, default="cosmo-scanner-hpc",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Run training
    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        num_workers=args.num_workers,
        field_size=args.field_size,
        omega_m_range=(args.omega_m_min, args.omega_m_max),
        device=args.device,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
