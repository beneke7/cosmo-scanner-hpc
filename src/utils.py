"""
utils.py - Device Handling and Logging Utilities
=================================================

NOTE for Mac users: If you encounter OpenMP errors, set:
    export KMP_DUPLICATE_LIB_OK=TRUE
before running. This is a known issue with conflicting OpenMP libraries.

This module provides device-agnostic compute primitives. In the language of 
theoretical physics, think of this as defining the "computational manifold" 
on which our tensors live. Just as a field φ(x) must be defined on some 
spacetime manifold M, our tensors must be allocated on a specific compute device.

The hierarchy of devices mirrors the hierarchy of approximation methods:
- CPU: The "exact" but slow baseline (like numerical integration)
- CUDA: GPU acceleration (like Monte Carlo sampling)
- MPS: Apple Silicon acceleration (Metal Performance Shaders)
"""

import logging
import sys
from typing import Optional

import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Automatically detect and return the optimal compute device.
    
    The device selection follows a cascade:
    1. If `preferred` is specified and available, use it
    2. Otherwise: CUDA > MPS > CPU
    
    This is analogous to selecting the optimal coordinate system for a problem:
    we pick the one that makes computation most efficient.
    
    Parameters
    ----------
    preferred : str, optional
        Explicitly request a device ('cuda', 'mps', 'cpu').
        
    Returns
    -------
    torch.device
        The selected compute device.
        
    Examples
    --------
    >>> device = get_device()  # Auto-detect
    >>> device = get_device('cuda')  # Force CUDA
    """
    if preferred is not None:
        preferred = preferred.lower()
        if preferred == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif preferred == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif preferred == 'cpu':
            return torch.device('cpu')
        else:
            logging.warning(f"Requested device '{preferred}' not available, falling back to auto-detect")
    
    # Auto-detection cascade
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU (consider using GPU for faster training)")
    
    return device


def setup_logging(level: int = logging.INFO, logfile: Optional[str] = None) -> None:
    """
    Configure the logging system.
    
    In experimental physics, we keep detailed lab notebooks. In computational
    physics, logging serves the same purpose: a traceable record of every
    experiment (training run).
    
    Parameters
    ----------
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    logfile : str, optional
        Path to save logs to disk. If None, logs only to stdout.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile))
    
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing config
    )


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    This is the dimensionality of our parameter space θ ∈ ℝ^N.
    The optimization problem lives on this N-dimensional manifold.
    
    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
        
    Returns
    -------
    int
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_device_info() -> None:
    """Print detailed information about available compute devices."""
    print("=" * 60)
    print("COMPUTE DEVICE INFORMATION")
    print("=" * 60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    
    # CUDA info
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    
    # MPS info (Apple Silicon)
    print(f"\nMPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("  Apple Metal Performance Shaders enabled")
    
    print("=" * 60)


if __name__ == "__main__":
    # Quick diagnostic when run directly
    print_device_info()
    device = get_device()
    print(f"\nSelected device: {device}")
