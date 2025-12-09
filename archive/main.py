"""
main.py - Training Script
=========================

Usage:
    python train/main.py --run_name experiment_1
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import wandb
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import (
    # Paths
    LOG_DIR, MODEL_DIR,
    # Hyperparameters
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    LR_START_FACTOR, LR_END_FACTOR, LR_TOTAL_ITERS, SEED,
    GRADIENT_ACCUMULATION_STEPS, USE_AMP,
    # Model
    CosmoNet, count_parameters,
    # Data
    create_dataloaders,
    # Device
    get_device,
    # Wandb
    WANDB_PROJECT
)

# =============================================================================
# LOGGING SETUP
# =============================================================================


def get_next_run_number() -> int:
    """Get the next available run number."""
    os.makedirs(LOG_DIR, exist_ok=True)
    existing = [d for d in os.listdir(LOG_DIR) if d.startswith("log_")]
    if not existing:
        return 1
    numbers = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    return max(numbers) + 1 if numbers else 1


def setup_logging(run_name: str) -> str:
    """Setup logging to file and console."""
    run_num = get_next_run_number()
    log_dir = os.path.join(LOG_DIR, f"log_{run_num}")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{run_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_dir

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_accum_steps=1):
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (fields, targets) in enumerate(loader):
        fields = fields.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(fields)
                loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(fields)
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, scaler=None):
    """Validate the model with optional mixed precision."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for fields, targets in loader:
            fields = fields.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(fields)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(fields)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

# =============================================================================
# MAIN
# =============================================================================


def main(run_name: str):
    """Main training function."""
    # Setup
    log_dir = setup_logging(run_name)
    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logging.info(f"Run: {run_name}")
    logging.info(f"Device: {device}")
    logging.info(f"Log dir: {log_dir}")
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "seed": SEED
        }
    )
    
    # Data
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(seed=SEED)
    logging.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Model
    model = CosmoNet().to(device)
    num_params = count_parameters(model)
    logging.info(f"Model parameters: {num_params:,}")
    wandb.config.update({"num_parameters": num_params})
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = LinearLR(
        optimizer,
        start_factor=LR_START_FACTOR,
        end_factor=LR_END_FACTOR,
        total_iters=LR_TOTAL_ITERS
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    if scaler:
        logging.info("Using Automatic Mixed Precision (AMP)")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, f"{run_name}_best.pt")
    
    logging.info("Starting training...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                 scaler=scaler, grad_accum_steps=GRADIENT_ACCUMULATION_STEPS)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, scaler=scaler)
        
        # Scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log
        logging.info(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, best_model_path)
            logging.info(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    # Test evaluation
    logging.info("Evaluating on test set...")
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, test_loader, criterion, device, scaler=scaler)
    logging.info(f"Test loss: {test_loss:.6f}")
    wandb.log({"test_loss": test_loss})
    
    # Final summary
    logging.info("=" * 40)
    logging.info(f"Best val loss: {best_val_loss:.6f}")
    logging.info(f"Test loss: {test_loss:.6f}")
    logging.info(f"Model saved: {best_model_path}")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CosmoNet")
    parser.add_argument("--run_name", type=str, required=True, help="Name for this run")
    args = parser.parse_args()
    
    # Set environment variable for Mac OpenMP issue
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    main(args.run_name)
