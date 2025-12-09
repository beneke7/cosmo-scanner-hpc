# Data Documentation

## Directory Structure

```
data/
├── images/              # Synthetic training data (100k samples)
├── metadata.csv         # Labels for synthetic data
└── real/
    ├── quijote/         # Quijote simulations (known Ω_m)
    ├── des/             # DES Y3 weak lensing maps
    └── processed/       # Preprocessed 256×256 patches
```

---

## 1. Synthetic Training Data

**Location**: `data/images/` + `data/metadata.csv`

| Property | Value |
|----------|-------|
| Samples | 100,000 |
| Size | 256×256 grayscale |
| Format | JPG |
| Ω_m range | [0.1, 1.0] uniform |
| Generation | `src/generate_dataset.py` |

---

## 2. Quijote Simulations

**Source**: https://quijote-simulations.readthedocs.io/

### Why Quijote?
- Simulations with **known Ω_m** values (0.1, 0.2, 0.3, 0.4, 0.5)
- Clean format, similar to our training data
- Perfect for model validation before real observations

### Available Data
| Cosmology | Ω_m | Realizations |
|-----------|-----|--------------|
| fiducial | 0.3175 | 15,000 |
| Om_p | 0.3275 | 500 |
| Om_m | 0.3075 | 500 |
| Ob_p/m | varies | 500 each |

### Download
```bash
# 2D density field projections (smaller, recommended)
wget https://quijote-simulations.readthedocs.io/en/latest/_downloads/density_field_2D.tar.gz
```

### Format
- HDF5 or binary files
- 256³ or 512³ density cubes
- Can project to 2D for our model

---

## 3. DES Y3 Weak Lensing Maps

**Source**: https://des.ncsa.illinois.edu/releases/y3a2

### Why DES?
- **Real observations** of the universe
- Known Ω_m ≈ 0.339 ± 0.031 (DES measurement)
- True test of model generalization

### Data Products
| Product | Description | Size |
|---------|-------------|------|
| Convergence maps (κ) | Mass reconstruction | ~1 GB |
| Shear catalogs | Galaxy shape measurements | ~10 GB |
| Redshift bins | Tomographic slices | 4 bins |

### Download
```bash
# Mass maps (FITS format)
wget https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/mass_maps/
```

### Key Differences from Synthetic Data
| Aspect | Synthetic | DES Real |
|--------|-----------|----------|
| Noise | None/controlled | Shape noise σ_ε ≈ 0.26 |
| Resolution | 256×256 uniform | Variable, ~arcmin |
| Coverage | Full field | Masked regions |
| Signal | Direct density | Reconstructed κ |

---

## 4. Preprocessing Pipeline

### For Quijote
1. Load 3D density cube
2. Project along one axis (sum or mean)
3. Normalize to [0, 1]
4. Save as 256×256 PNG/JPG

### For DES
1. Load FITS convergence map
2. Extract 256×256 patches (avoiding masks)
3. Normalize: `(κ - κ_min) / (κ_max - κ_min)`
4. Match noise level to training augmentation
5. Save as PNG/JPG

---

## 5. Scripts

| Script | Purpose |
|--------|---------|
| `src/download_quijote.py` | Download Quijote data |
| `src/download_des.py` | Download DES maps |
| `src/preprocess_real.py` | Convert to model format |
| `src/analyze_real.py` | Visualize and compare |

---

## 6. Expected Ω_m Values

| Source | Ω_m | Uncertainty | Notes |
|--------|-----|-------------|-------|
| Planck 2018 | 0.3153 | ±0.0073 | CMB, gold standard |
| DES Y3 | 0.339 | ±0.031 | Weak lensing |
| Our model (synthetic test) | varies | ±0.032 | ✓ Works well |
| Our model (Quijote, same physics) | varies | ±0.032 | ✓ Generalizes |

---

## 7. Inference Results (v0.4.0)

### ✓ Success: Same Physics Generation

When Quijote data is generated with the **same physics** as training data:

| Model | Dataset | True Ω_m | Predicted | Error |
|-------|---------|----------|-----------|-------|
| resnet_v1 | Quijote (0.200) | 0.200 | 0.214 ± 0.018 | +0.014 |
| resnet_v1 | Quijote (0.308) | 0.308 | 0.327 ± 0.028 | +0.020 |
| resnet_v1 | Quijote (0.318) | 0.318 | 0.348 ± 0.020 | +0.030 |
| resnet_v1 | Quijote (0.328) | 0.328 | 0.348 ± 0.024 | +0.020 |
| resnet_v1 | Quijote (0.400) | 0.400 | 0.421 ± 0.025 | +0.021 |

**Overall RMSE: 0.032** ✓

### ❌ Failure: Different Physics Generation

When Quijote data is generated with **different physics** (physics_v2.py):

| Model | Dataset | True Ω_m | Predicted | Error |
|-------|---------|----------|-----------|-------|
| hybrid_v1 | Quijote (0.3175) | 0.3175 | 0.094 | -0.224 |
| test5 | Quijote (0.3175) | 0.3175 | 0.048 | -0.270 |

### Key Insight: Domain Gap

The model's performance depends critically on **data generation consistency**:
- ✓ Same `generate_universe()` → RMSE = 0.032
- ❌ Different physics → RMSE > 0.30

### Lessons Learned

1. **Physics consistency is critical** - Models learn the specific statistical properties of training data
2. **Fine-tuning needs data** - 50 samples insufficient; pre-trained model generalizes better
3. **Validation requires care** - Always use same generation pipeline for validation data

---

## 8. File Organization

```
results/
└── figures/
    ├── resnet_training_results.png   # Training curves & evaluation
    ├── data_comparison.png           # Synthetic vs real comparison
    ├── evaluation_results.png        # Test set evaluation
    ├── dataset_preview.png           # Sample images
    ├── noise_augmentation_examples.png
    ├── inference_quijote_*.png       # Quijote inference results
    └── inference_des_*.png           # DES inference results
```
