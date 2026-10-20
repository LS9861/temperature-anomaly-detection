# preprocess.py (Real Data Version)
import sys
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# If real data exists, use it. Otherwise, use synthetic data.
REAL_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "real_temperature.csv"
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_temperature.csv"

if REAL_DATA_PATH.exists():
    INPUT_CSV = REAL_DATA_PATH
else:
    INPUT_CSV = SYNTHETIC_DATA_PATH
#INPUT_CSV = PROJECT_ROOT / "data" / "synthetic" / "real_temperature.csv"
#INPUT_CSV = PROJECT_ROOT / "data" / "synthetic" / "synthetic_temperature.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
X_TRAIN_PATH = OUTPUT_DIR / "X_train.npy"
X_TEST_PATH = OUTPUT_DIR / "X_test.npy"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50
TRAIN_SPLIT = 0.8

print("=" * 60)
print("PHASE 2: DATA PREPROCESSING (REAL DATA)")
print("=" * 60)

if not INPUT_CSV.exists():
    print(f"ERROR: Input file not found: {INPUT_CSV}")
    sys.exit(1)

print(f"\n[1/4] Loading CSV: {INPUT_CSV.name}")
df = pd.read_csv(INPUT_CSV)

# Handle the 'timestamp_ms' column from the ESP32
temperatures = df['temperature'].values

print(f"\n[2/4] Normalizing data...")
scaler = StandardScaler()
temperatures_normalized = scaler.fit_transform(temperatures.reshape(-1, 1)).flatten()

print(f"\n[3/4] Creating sliding windows (size={WINDOW_SIZE})...")
windows = []
for i in range(len(temperatures_normalized) - WINDOW_SIZE + 1):
    windows.append(temperatures_normalized[i : i + WINDOW_SIZE])
windows = np.array(windows)
print(f"      Windows created: {windows.shape[0]}")

print(f"\n[4/4] Splitting into train/test ({int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)})...")
split_index = int(TRAIN_SPLIT * len(windows))
X_train = windows[:split_index]
X_test = windows[split_index:]

np.save(X_TRAIN_PATH, X_train)
np.save(X_TEST_PATH, X_test)
joblib.dump(scaler, SCALER_PATH)

print(f"\nPHASE 2 COMPLETE")
print(f"  Train shape: {X_train.shape}")
print(f"  Test shape:  {X_test.shape}")
print(f"  Scaler saved.")