# =============================================================================
# preprocess.py  (UPDATED - FIXES DATA LEAKAGE)
# Purpose: Load raw CSV → exclude anomalous windows → normalize → window → save
# Phase:   2 of 7 (Data Preprocessing)
# Input:   data/synthetic/synthetic_temperature.csv
#          data/synthetic/anomaly_indices.npy
# Output:  data/processed/X_train.npy
#          data/processed/X_test.npy
#          models/scaler.pkl
# =============================================================================

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =============================================================================
# SECTION 1: PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_CSV       = PROJECT_ROOT / "data" / "synthetic" / "synthetic_temperature.csv"
ANOMALY_PATH    = PROJECT_ROOT / "data" / "synthetic" / "anomaly_indices.npy"
OUTPUT_DIR      = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
X_TRAIN_PATH    = OUTPUT_DIR / "X_train.npy"
X_TEST_PATH     = OUTPUT_DIR / "X_test.npy"
SCALER_PATH     = MODELS_DIR / "scaler.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

WINDOW_SIZE = 50
TRAIN_SPLIT = 0.8

# =============================================================================
# SECTION 3: LOAD CSV AND ANOMALY INDICES
# =============================================================================

print("=" * 60)
print("PHASE 2: DATA PREPROCESSING (with anomaly exclusion)")
print("=" * 60)

if not INPUT_CSV.exists():
    print(f"ERROR: Input file not found: {INPUT_CSV}")
    sys.exit(1)

if not ANOMALY_PATH.exists():
    print(f"ERROR: Anomaly indices not found: {ANOMALY_PATH}")
    print("       Re-run generate_synthetic_data.py first.")
    sys.exit(1)

print(f"\n[1/6] Loading CSV and anomaly indices...")
df = pd.read_csv(INPUT_CSV)
anomaly_indices = np.load(ANOMALY_PATH)

print(f"      Rows loaded:       {len(df)}")
print(f"      Anomalies marked:  {len(anomaly_indices)}")
print(f"      Anomaly positions: {anomaly_indices[:5]}...")

# =============================================================================
# SECTION 4: EXTRACT TEMPERATURE VALUES
# =============================================================================

print("\n[2/6] Extracting temperature values...")
temperatures = df['temperature'].values
print(f"      Extracted array shape: {temperatures.shape}")

# =============================================================================
# SECTION 5: NORMALIZE THE DATA
# Why: Fit scaler ONLY on normal data (not anomalies)
# =============================================================================

print("\n[3/6] Normalizing data (scaler fit on normal points only)...")

# Create a boolean mask: True = normal, False = anomaly
normal_mask = np.ones(len(temperatures), dtype=bool)
normal_mask[anomaly_indices] = False

# Fit scaler ONLY on normal data
scaler = StandardScaler()
scaler.fit(temperatures[normal_mask].reshape(-1, 1))

# Transform ALL data using the scaler fitted on normal data
temperatures_normalized = scaler.transform(
    temperatures.reshape(-1, 1)
).flatten()

print(f"      Mean (from normal data): {scaler.mean_[0]:.4f}")
print(f"      Std  (from normal data): {scaler.scale_[0]:.4f}")
print(f"      Normalized mean: {temperatures_normalized[normal_mask].mean():.4f}")
print(f"      Normalized std:  {temperatures_normalized[normal_mask].std():.4f}")

# =============================================================================
# SECTION 6: CREATE SLIDING WINDOWS
# =============================================================================

print(f"\n[4/6] Creating sliding windows (size={WINDOW_SIZE})...")

windows = []
window_is_anomalous = []   # Track which windows contain anomalies

for i in range(len(temperatures_normalized) - WINDOW_SIZE + 1):
    window = temperatures_normalized[i : i + WINDOW_SIZE]
    windows.append(window)

    # Check if any anomaly index falls inside this window's range
    # Window covers samples [i, i + WINDOW_SIZE)
    window_start = i
    window_end   = i + WINDOW_SIZE
    contains_anomaly = np.any(
        (anomaly_indices >= window_start) & (anomaly_indices < window_end)
    )
    window_is_anomalous.append(contains_anomaly)

windows = np.array(windows)
window_is_anomalous = np.array(window_is_anomalous)

n_total     = len(windows)
n_anomalous = window_is_anomalous.sum()
n_clean     = n_total - n_anomalous

print(f"      Total windows:     {n_total}")
print(f"      Clean windows:     {n_clean}")
print(f"      Anomalous windows: {n_anomalous}")

# =============================================================================
# SECTION 7: SPLIT — TRAINING ONLY ON CLEAN WINDOWS
# Why: Autoencoder must only see normal data during training
# =============================================================================

print(f"\n[5/6] Splitting clean windows into train/test...")

# Keep only clean windows for the final dataset
clean_windows = windows[~window_is_anomalous]
anomalous_windows = windows[window_is_anomalous]

# Split clean windows into train/test
split_index = int(TRAIN_SPLIT * len(clean_windows))
X_train = clean_windows[:split_index]
X_test  = clean_windows[split_index:]

print(f"      X_train shape (clean): {X_train.shape}")
print(f"      X_test  shape (clean): {X_test.shape}")
print(f"      Anomalous windows saved separately: {anomalous_windows.shape}")

# =============================================================================
# SECTION 8: SAVE
# =============================================================================

print("\n[6/6] Writing processed files...")

np.save(X_TRAIN_PATH, X_train)
np.save(X_TEST_PATH, X_test)
joblib.dump(scaler, SCALER_PATH)

# Save anomalous windows for Phase 4 evaluation
ANOMALOUS_PATH = OUTPUT_DIR / "X_anomalous.npy"
np.save(ANOMALOUS_PATH, anomalous_windows)

print(f"      Saved: {X_TRAIN_PATH.relative_to(PROJECT_ROOT)}")
print(f"      Saved: {X_TEST_PATH.relative_to(PROJECT_ROOT)}")
print(f"      Saved: {ANOMALOUS_PATH.relative_to(PROJECT_ROOT)}")
print(f"      Saved: {SCALER_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 9: SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE (data leakage fixed)")
print("=" * 60)
print(f"  Input:               {len(df)} raw samples")
print(f"  Total windows:       {n_total}")
print(f"  Clean windows:       {n_clean}  (used for training)")
print(f"  Anomalous windows:   {n_anomalous}  (held out for evaluation)")
print(f"  Training windows:    {X_train.shape[0]}")
print(f"  Testing windows:     {X_test.shape[0]}")
print("\nNext: Re-run train_autoencoder.py (Phase 3)")
print("=" * 60)