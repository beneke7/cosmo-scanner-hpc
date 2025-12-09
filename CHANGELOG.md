# Changelog

All notable changes to the Cosmo Scanner HPC project.

---

## [v0.5.0] - 2024-12-09

**How to train:** `./run.sh --epochs 50` (DES-like data) or `./run.sh --data_type 2lpt` (N-body)

### Problem Statement
From v0.4.0 analysis:
- **Problem A (Instability)**: Pre-training loss curve shows violent spikes (10⁻² → 10⁻¹)
- **Problem B (Precision Saturation)**: Box plots show overlap between Ω_m = 0.307, 0.318, 0.328

### Solutions Implemented

#### Task 1: GPU-Accelerated 2LPT Physics (`src/physics_lpt.py`)
Replaces simple GRF with Second-Order Lagrangian Perturbation Theory:
- **Zel'dovich approximation** (1LPT): Linear displacement Ψ⁽¹⁾ = -∇φ
- **2LPT correction**: Non-linear displacement Ψ⁽²⁾ from φ,ᵢᵢφ,ⱼⱼ - φ,ᵢⱼ²
- **CIC deposition**: Cloud-in-Cell interpolation for realistic density fields

#### Task 2: Hybrid Architecture (`src/model_hybrid.py`)
Combines CNN with Power Spectrum for scale-dependent information:
- **Differentiable power spectrum**: `compute_power_spectrum_fast()`
- **Channel attention**: SE-Net style squeeze-and-excitation
- **Parameters**: 11.4M

#### Task 3: DES-like Weak Lensing Generator (`src/physics_lensing.py`)
Generates synthetic convergence (κ) maps matching DES Y3 statistics:
- **Gaussian smoothing**: Configurable smoothing scale (default 10 arcmin)
- **Shape noise**: Realistic σ_ε = 0.26, n_gal = 8/arcmin²

#### Task 4: RTX Optimizations (`train/train.py`)
- CosineAnnealingWarmRestarts scheduler
- channels_last memory format, TF32 precision, AMP

### Directory Refactor
- `study/` - Learning materials (notebooks, docs)
- `archive/` - Old/unused code
- `train/` - Minimal: `config.py` + `train.py`
- `run.sh` - Single entry point

### Usage
```bash
# Default: DES-like weak lensing
./run.sh --epochs 50

# N-body like (for Quijote)
./run.sh --data_type 2lpt --epochs 50

# Custom configuration
./run.sh --data_type des --smoothing 5.0 --batch_size 32
```

### Files
| File | Description |
|------|-------------|
| `src/physics_lpt.py` | GPU-accelerated 2LPT generator (for N-body) |
| `src/physics_lensing.py` | DES-like weak lensing generator |
| `src/model_hybrid.py` | Hybrid CNN + Power Spectrum model |
| `train/train.py` | Training pipeline |
| `run.sh` | Training launcher script |

### Data Type Selection Guide
| Target Data | Use `--data_type` | Why |
|-------------|-------------------|-----|
| DES Y3 | `des` | Smoothed κ maps with shape noise |
| Quijote | `2lpt` | N-body like density fields |
| CAMELS | `2lpt` | Hydrodynamic sim projections |
| Custom | `disk` | Pre-generated images |

---

## [v0.4.0] - 2024-12-09

### Added

#### Advanced Architectures (`train/models_advanced.py`)
- **CosmoResNet** - ResNet-18 style with skip connections (11.3M params)
- **CosmoAttentionNet** - ResNet + CBAM attention modules (11.4M params)
- **CosmoNetV2Hybrid** - Multi-scale ResNet + Power Spectrum + Attention fusion (11.7M params)

#### Improved Physics (`src/physics_v2.py`)
- **Eisenstein-Hu transfer function** - More accurate than BBKS
- **σ_8 normalization** - Matches real simulation amplitudes
- **Log-normal transformation** - Realistic non-Gaussian tails
- **Redshift-dependent growth factor**
- **Quijote-like generation** - Matches Quijote simulation statistics

#### Fine-tuning Pipeline (`train/main_finetune.py`)
Two-stage training:
1. **Pre-train** on synthetic data (100k samples)
2. **Fine-tune** on Quijote simulations (50 samples)

Fine-tuning techniques:
- **Layer freezing** - Freeze early conv layers
- **Gradual unfreezing** - Unfreeze after N epochs
- **Discriminative LR** - Lower LR for early layers, higher for head
- **Stronger regularization** - More dropout, weight decay

### Usage
```bash
# Stage 1: Pre-train
python train/main_finetune.py --stage pretrain --run_name resnet_v1 --model resnet

# Stage 2: Fine-tune
python train/main_finetune.py --stage finetune --run_name resnet_v1_ft --pretrained resnet_v1
```

### How Fine-Tuning Works
```
Pre-training (synthetic):     Fine-tuning (Quijote):
┌─────────────────────┐       ┌─────────────────────┐
│ Conv layers         │ ──►   │ Conv layers         │ ← Frozen or slow LR
│ (general features)  │       │ (keep features)     │
├─────────────────────┤       ├─────────────────────┤
│ FC layers           │ ──►   │ FC layers           │ ← Higher LR (adapt)
│ (Ω_m mapping)       │       │ (re-learn for real) │
└─────────────────────┘       └─────────────────────┘
```

### Training Results

#### Pre-training (resnet_v1)
- **Dataset**: 100k synthetic samples
- **Best val loss**: 0.00177 (epoch 8)
- **RMSE on synthetic test**: 0.038

#### Evaluation on Quijote Cosmologies
| True Ω_m | Predicted | Error |
|----------|-----------|-------|
| 0.200 | 0.214 ± 0.018 | +0.014 |
| 0.308 | 0.327 ± 0.028 | +0.020 |
| 0.318 | 0.348 ± 0.020 | +0.030 |
| 0.328 | 0.348 ± 0.024 | +0.020 |
| 0.400 | 0.421 ± 0.025 | +0.021 |

**Overall RMSE: 0.032** ✓

#### Fine-tuning Results
- Fine-tuning on 50 Quijote samples **did not improve** results
- Pre-trained model generalizes better than fine-tuned model
- **Lesson**: Fine-tuning requires more target domain data (hundreds+)

### Figures
All training figures organized in `results/figures/`:
- `resnet_training_results.png` - Training curves and evaluation plots
- `data_comparison.png` - Synthetic vs real data comparison
- `evaluation_results.png` - Model evaluation on test set

---

## [v0.3.0] - 2024-12-09

### Added
- **Hybrid CNN + Power Spectrum Model** (`CosmoNetHybrid`)
  - **Branch 1 (CNN)**: 5 conv blocks extracting spatial features from images
  - **Branch 2 (MLP)**: Processes 32-bin radially-averaged power spectrum
  - **Fusion layer**: Concatenates features (512 + 64 = 576 dims) → final prediction
  
- **Power spectrum computation** (`compute_power_spectrum()`)
  - 2D FFT → radial averaging → log-scale → normalization
  - 32 bins capturing scale-dependent clustering information
  - This is what traditional cosmological analyses use!

- **New training script** (`train/main_hybrid.py`)
  - Handles 3-tuple data: (image, power_spectrum, target)
  - Cosine annealing LR scheduler for better convergence
  - Full wandb integration

- **New run script** (`run_training_hybrid.sh`)

### Configuration
```python
USE_POWER_SPECTRUM = True
N_POWER_BINS = 32
POWER_SPECTRUM_MLP = [128, 64]
```

### Architecture
```
Input Image (256×256) ──► CNN ──► 512-dim features ─┐
                                                    ├──► Fusion (576→256→64→1) ──► Ω_m
Input Power Spectrum (32) ──► MLP ──► 64-dim features ─┘
```

### Goal
- Achieve RMSE < 0.01 to approach Planck-level precision (±0.007)

### Results (hybrid_v1)
- **Test set (synthetic)**: RMSE = 0.071 (MSE = 0.005)
- **Quijote simulations**: RMSE = 0.227 ❌ (severe underestimation)
- **DES-like data**: RMSE = 0.268 ❌ (severe underestimation)

### Key Finding: Domain Gap
The model predicts Ω_m ≈ 0.07-0.10 for all inputs, regardless of true value.
This indicates the model learned features specific to our synthetic data generation,
not generalizable cosmological features.

### Improvement Strategies (v0.4.0 planned)
1. **Train on diverse simulations** - Use Quijote/CAMELS with varied cosmologies
2. **Domain adaptation** - Add domain-invariant feature learning
3. **Better physics** - Improve synthetic data generation to match real simulations
4. **Transfer learning** - Pre-train on large simulation suites

---

## [v0.2.0] - 2024-12-09

### Added
- **Noise augmentation** for real data robustness
  - Gaussian noise (σ = 0.02-0.10) simulating observational noise
  - Random Gaussian smoothing (σ = 0.5-2.0) simulating PSF effects
  - 30% probability of smoothing per sample
  - Configurable via `ADD_NOISE`, `NOISE_STD_MIN`, `NOISE_STD_MAX`, `SMOOTHING_PROB`, `SMOOTHING_SIGMA_MAX`

- **Data augmentation** (physics-preserving)
  - Random horizontal/vertical flips
  - Random 90° rotations
  - Applied only during training

- **Evaluation script** (`evaluate_model.py`)
  - Detailed metrics: MSE, RMSE, MAE, R², bias analysis
  - Visualization: scatter plots, error distributions, bias checks
  - Per-range error analysis

### Changed
- Increased `NUM_EPOCHS` to 50 (from 15) for noise-augmented training
- Increased `PATIENCE` to 10 (from 5)

---

## [v0.1.0] - 2024-12-08

### Added
- **GPU-optimized training** for RTX 5090 (32GB VRAM)
  - Batch size: 128 (optimized for memory)
  - Mixed Precision Training (AMP) enabled
  - 16 DataLoader workers with prefetching
  - Persistent workers for faster epoch transitions

- **Parallel data generation** (`src/generate_dataset.py`)
  - Multiprocessing with 60 CPU cores
  - Chunked processing for memory efficiency
  - ~50-60x speedup vs sequential

- **CNN Architecture** (`CosmoNet`)
  - 5 convolutional blocks with BatchNorm, ReLU, Dropout2d
  - Global Average Pooling
  - 2 FC layers with Dropout
  - Input: 256×256 grayscale images
  - Output: Ω_m prediction

- **Training infrastructure**
  - WandB integration for experiment tracking
  - Early stopping with patience
  - Linear learning rate scheduler
  - Model checkpointing (best validation loss)

### Configuration
```python
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
CONV_DROPOUT = 0.1
FC_DROPOUT = 0.4
IMAGE_SIZE = 256
```

### Results (test5)
- **R² Score**: 0.970
- **RMSE**: 0.050 (±5% in Ω_m)
- **MAE**: 0.041
- **No systematic bias**

---

## [v0.0.1] - 2024-12-08

### Initial Setup
- Project structure created
- Basic CNN model for Ω_m regression
- Synthetic data generation using cosmological simulations
- 100,000 training samples (256×256 density fields)

---

## Roadmap

### Short-term
- [ ] Implement hybrid CNN + Power Spectrum model (v0.3.0)
- [ ] Test on real DES/HSC weak lensing data
- [ ] Achieve RMSE < 0.01

### Long-term
- [ ] 3D density field support
- [ ] Multi-parameter estimation (Ω_m, σ_8, h, etc.)
- [ ] Uncertainty quantification (Bayesian NN or ensemble)
- [ ] Domain adaptation for real observations

---

## References

### Experimental Ω_m Measurements
| Method | Ω_m | Uncertainty | Paper |
|--------|-----|-------------|-------|
| Planck 2018 | 0.3153 | ±0.0073 | arXiv:1807.06209 |
| DES Y3 | 0.339 | ±0.031 | arXiv:2105.13549 |
| KiDS-1000 | 0.305 | ±0.010 | arXiv:2007.15632 |
