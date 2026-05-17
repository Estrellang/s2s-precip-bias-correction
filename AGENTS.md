# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

S2S precipitation forecast bias correction system using Conditional Variational Autoencoder (CVAE). The model corrects ECMWF S2S forecasts using ERA5 reanalysis data as ground truth, focusing on China region summer precipitation (June-August).

## Complete Workflow

The project follows a sequential pipeline:

1. **API Setup** → 2. **Data Download** → 3. **Preprocessing** → 4. **Data Alignment** → 5. **Training** → 6. **Evaluation** → 7. **Visualization**

### 1. API Configuration (First Time Only)

```bash
python setup_ecmwf_key.py  # Configure ECMWF API for S2S data
python setup_cds_key.py    # Configure CDS API for ERA5 data
```

### 2. Data Download

```bash
# Single year download
python download_s2s.py     # Downloads S2S forecast data
python download_era5.py    # Downloads ERA5 reanalysis data

# Multi-year batch download
python batch_download.py   # Downloads multiple years at once
```

### 3. Data Preprocessing

```bash
python preprocess.py       # Crops to China region, filters summer months, interpolates grids
```

### 4. Data Alignment

```bash
python align_data.py       # Pairs S2S forecasts with corresponding ERA5 observations
```

### 5. Model Training

```bash
python train.py            # Trains CVAE model, saves best_model.pt
```

### 6. Model Evaluation

```bash
python evaluate.py         # Computes RMSE, correlation, bias metrics
```

### 7. Visualization

```bash
python visualize.py        # Generates spatial maps, scatter plots, time series
```

## Architecture

### Data Flow

```
S2S Forecast (GRIB) ──┐
                      ├─> preprocess.py ─> align_data.py ─> [X_s2s.npy, Y_era5.npy]
ERA5 Reanalysis (NC) ─┘                                              │
                                                                      ▼
                                                              train.py (CVAE)
                                                                      │
                                                                      ▼
                                                            [best_model.pt, norm_params.npy]
                                                                      │
                                                                      ▼
                                                              evaluate.py / visualize.py
```

### CVAE Model Structure

- **Encoder**: Takes concatenated [S2S forecast + ERA5 observation] → latent space (mu, logvar)
- **Decoder**: Takes [latent variable + S2S forecast] → bias-corrected precipitation
- **Loss**: MSE reconstruction loss + 0.001 × KL divergence
- **Inference**: Samples latent variable from N(0,1), averages 10 predictions

### Key Parameters

- **Region**: China (15-55°N, 70-140°E)
- **Season**: Summer (June-August)
- **S2S**: ECMWF, Monday/Thursday forecasts, steps 0-336h (14 days)
- **Grid**: Interpolated to ERA5 resolution (33×57 grid points)
- **Model**: latent_dim=64, batch_size=32, lr=1e-3
- **Data Split**: 70% train / 10% validation / 20% test

## Data Directories

- `data/s2s_raw/` - Raw S2S GRIB files
- `data/era5_obs/` - Raw ERA5 NetCDF files
- `data/processed/` - Processed NetCDF files, aligned numpy arrays, trained model
- `figures/` - Generated plots

## Dependencies

Install with: `pip install -r requirements.txt`

Key packages: `ecmwf-api-client`, `cdsapi`, `xarray`, `torch`, `cfgrib`

## Notes

- S2S data downloads are slow; test with 1-2 years first
- Each year requires ~5-10GB disk space
- `preprocess.py` converts precipitation units from m to mm/day
- `align_data.py` matches S2S forecast valid times with ERA5 observation dates
- Model uses standardization (saved in `norm_params.npy`) - must be applied during inference
