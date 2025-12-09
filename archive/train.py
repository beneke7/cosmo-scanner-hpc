"""
train.py - The Optimization Loop
=================================

This module implements the training procedure for CosmoNet.

THE PHYSICS OF OPTIMIZATION
---------------------------
Training a neural network is a high-dimensional optimization problem:
    θ* = argmin_θ L(θ)

where L(θ) is the loss function (a scalar "potential") and θ ∈ ℝ^N are the
network parameters.

THE LOSS FUNCTION AS A POTENTIAL ENERGY SURFACE
-----------------------------------------------
Think of L(θ) as a potential energy landscape in N-dimensional parameter space.
Our goal is to find a minimum (ideally the global minimum, but local minima
often work well in practice due to the geometry of neural network loss surfaces).

GRADIENT DESCENT AS DYNAMICS
----------------------------
The gradient ∇_θ L points "uphill" (direction of steepest ascent).
Gradient descent follows the negative gradient:
    θ_{t+1} = θ_t - η ∇_θ L(θ_t)

This is analogous to a particle rolling down a potential:
    dx/dt = -∇V(x)

(overdamped dynamics, like a particle in molasses)

ADAM OPTIMIZER: MOMENTUM + ADAPTIVE LEARNING RATE
-------------------------------------------------
Adam adds two key modifications:
1. MOMENTUM: Keep a running average of gradients (like a heavy ball with inertia)
   - Helps escape shallow local minima
   - Smooths out noisy gradient estimates
   
2. ADAPTIVE LEARNING RATE: Scale step size by running variance of gradients
   - Parameters with consistently large gradients get smaller steps
   - Parameters with small gradients get larger steps

WEIGHT DECAY (L2 REGULARIZATION)
--------------------------------
AdamW adds a term λ||θ||² to the loss, which pulls parameters toward zero.
This is like a spring force F = -kx that prevents weights from growing too large.
Physically: it's a "soft" constraint keeping the system near the origin.
"""

import os
import logging
from typing import Optional, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Logging will be local only.")

from .model import CosmoNet
from .dataset import create_dataloaders
from .utils import get_device, setup_logging, count_parameters


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """
    Execute one training epoch.
    
    One epoch = one pass through the entire training dataset.
    
    THE TRAINING LOOP PHYSICS
    -------------------------
    For each batch:
    1. Forward pass: compute f_θ(x) for all x in batch
    2. Compute loss: L = mean((f_θ(x) - y)²)  [MSE]
    3. Backward pass: compute ∇_θ L via automatic differentiation
    4. Optimizer step: update θ → θ - η * (modified gradient)
    5. Zero gradients: reset for next batch
    
    The batch gradient is a Monte Carlo estimate of the true gradient
    (which would require summing over ALL training data).
    
    Parameters
    ----------
    model : nn.Module
        The neural network.
    train_loader : DataLoader
        Training data loader.
    optimizer : Optimizer
        The optimization algorithm.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch number (for logging).
        
    Returns
    -------
    float
        Average loss over the epoch.
    """
    model.train()  # Set to training mode (enables dropout, batch norm training mode)
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for fields, targets in progress_bar:
        # Move data to compute device
        fields = fields.to(device)
        targets = targets.to(device)
        
        # FORWARD PASS
        # Compute predictions f_θ(x)
        predictions = model(fields)
        
        # COMPUTE LOSS
        # L = (1/B) Σ_i (f_θ(x_i) - y_i)²
        # This is the MSE: the "potential energy" we're minimizing
        loss = criterion(predictions, targets)
        
        # BACKWARD PASS
        # Compute gradients ∇_θ L via backpropagation (chain rule)
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients (fills .grad attributes)
        
        # OPTIMIZER STEP
        # Update parameters: θ → θ - η * AdamW(∇_θ L)
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient computation for efficiency
def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """
    Evaluate model on validation set.
    
    No gradients computed - just forward passes for evaluation.
    
    Parameters
    ----------
    model : nn.Module
        The neural network.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch number.
        
    Returns
    -------
    float
        Average validation loss.
    """
    model.eval()  # Set to evaluation mode (disables dropout, uses running stats for batch norm)
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for fields, targets in progress_bar:
        fields = fields.to(device)
        targets = targets.to(device)
        
        predictions = model(fields)
        loss = criterion(predictions, targets)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_samples: int = 10000,
    val_samples: int = 1000,
    num_workers: int = 4,
    field_size: int = 64,
    omega_m_range: tuple = (0.1, 0.5),
    device: Optional[str] = None,
    save_dir: str = "checkpoints",
    use_wandb: bool = True,
    wandb_project: str = "cosmo-scanner-hpc",
    wandb_run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full training procedure.
    
    THE TRAINING TRAJECTORY
    -----------------------
    We iterate through epochs, and within each epoch, through batches.
    At each step, we move through parameter space along the negative gradient.
    
    The loss curve L(t) shows our "descent" down the potential energy surface.
    A good training run shows:
    - Rapid initial decrease (finding the basin of attraction)
    - Gradual decrease (fine-tuning within the basin)
    - Convergence to a plateau (reached a minimum)
    
    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Initial learning rate η.
    weight_decay : float
        L2 regularization strength λ (spring constant).
    train_samples : int
        Training samples per epoch.
    val_samples : int
        Validation samples.
    num_workers : int
        DataLoader workers.
    field_size : int
        Spatial resolution.
    omega_m_range : tuple
        (min, max) for Ω_m.
    device : str, optional
        Force specific device.
    save_dir : str
        Directory for checkpoints.
    use_wandb : bool
        Enable Weights & Biases logging.
    wandb_project : str
        W&B project name.
    wandb_run_name : str, optional
        W&B run name.
        
    Returns
    -------
    dict
        Training history and final model path.
    """
    setup_logging()
    
    # Initialize device
    device = get_device(device)
    logging.info(f"Using device: {device}")
    
    # Create model
    model = CosmoNet(input_size=field_size).to(device)
    num_params = count_parameters(model)
    logging.info(f"Model parameters: {num_params:,}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        omega_m_range=omega_m_range,
        field_size=field_size
    )
    logging.info(f"Training batches per epoch: {len(train_loader)}")
    logging.info(f"Validation batches: {len(val_loader)}")
    
    # Loss function: Mean Squared Error
    # L = (1/B) Σ (pred - target)²
    # This is the "potential energy" we minimize
    criterion = nn.MSELoss()
    
    # Optimizer: AdamW
    # Gradient descent with momentum + adaptive learning rates + weight decay
    # weight_decay acts like a spring force pulling weights toward zero
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler: Cosine annealing
    # Slowly reduce learning rate following a cosine curve
    # Like simulated annealing: start exploring, then fine-tune
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "field_size": field_size,
                "omega_m_range": omega_m_range,
                "num_parameters": num_params
            }
        )
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # =========================================================================
    # THE MAIN TRAINING LOOP
    # Each epoch is one complete pass through the training data
    # Think of this as evolving through "time" in parameter space
    # =========================================================================
    logging.info("Starting training...")
    logging.info("=" * 60)
    
    for epoch in range(1, epochs + 1):
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        
        # Logging
        logging.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # W&B logging
        # The loss curve is our "energy minimization trajectory"
        # Watch how L(θ) decreases as we descend the potential surface
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'field_size': field_size,
                    'omega_m_range': omega_m_range
                }
            }, best_model_path)
            logging.info(f"  → New best model saved (val_loss: {val_loss:.6f})")
    
    logging.info("=" * 60)
    logging.info(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    logging.info(f"Best model saved to: {best_model_path}")
    
    # Final checkpoint
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'history': history
    }, final_path)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "final_model_path": final_path
    }


if __name__ == "__main__":
    # Quick training run for testing
    result = train(
        epochs=5,
        batch_size=32,
        train_samples=1000,
        val_samples=200,
        num_workers=0,  # Use 0 for debugging
        use_wandb=False  # Disable for quick test
    )
    
    print("\nTraining history:")
    for i, (tl, vl) in enumerate(zip(result["history"]["train_loss"], 
                                      result["history"]["val_loss"]), 1):
        print(f"  Epoch {i}: train={tl:.6f}, val={vl:.6f}")
