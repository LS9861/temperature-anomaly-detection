# =============================================================================
# generate_synthetic_data.py
# Purpose: Create fake temperature data to test the AI pipeline
#          WITHOUT needing real hardware yet.
# =============================================================================

# --- Import required libraries ---
import pandas as pd                          # For creating and saving CSV files
import numpy as np                           # For numerical operations and random numbers
from datetime import datetime, timedelta     # For creating timestamps
from pathlib import Path                     # For handling file paths safely

# =============================================================================
# SECTION 1: PATH SETUP
# Why: So the script works no matter where you run it from
# =============================================================================

# __file__ = the full path of THIS script file
# .resolve() = converts to absolute path (resolves any shortcuts)
# .parent = the folder containing the script (scripts/)
# .parent.parent = the folder containing scripts/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the full path to data/synthetic/ folder
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

# Create the folder if it doesn't exist
# parents=True  = create parent folders if missing
# exist_ok=True = don't error if folder already exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 2: DATA GENERATION PARAMETERS
# Why: Define how much data to create and what it looks like
# =============================================================================

# Set random seed so results are reproducible
# Every time you run, you get the same "random" numbers
np.random.seed(42)

hours = 24                    # Simulate 24 hours of data
samples_per_hour = 60         # 1 sample per minute (60 minutes = 1 hour)
total_samples = hours * samples_per_hour   # 24 * 60 = 1440 samples

# =============================================================================
# SECTION 3: CREATE TIMESTAMPS
# Why: Every data point needs a time label
# =============================================================================

# Create a list of timestamps starting from NOW
# Each timestamp is 1 minute apart
# Example: [2026-09-13 10:00:00, 2026-09-13 10:01:00, ...]
timestamps = [datetime.now() + timedelta(minutes=i) for i in range(total_samples)]

# =============================================================================
# SECTION 4: GENERATE TEMPERATURE VALUES
# Why: Simulate realistic temperature behavior
# =============================================================================

# Base temperature (like room temperature)
base_temp = 22.0              # 22°C average

# Daily cycle: temperature goes up and down like a sine wave
# np.linspace(0, 2*np.pi, total_samples) = angles from 0 to 360°
# np.sin(...) = smooth wave between -1 and +1
# * 3.0 = amplitude of 3°C (so temp varies ±3°C)
daily_cycle = 3.0 * np.sin(np.linspace(0, 2 * np.pi, total_samples))

# Random noise: real sensors are never perfectly smooth
# np.random.normal(0, 0.3, total_samples) = random values
#   0 = mean (centered on zero)
#   0.3 = standard deviation (small fluctuations)
#   total_samples = how many random values to generate
noise = np.random.normal(0, 0.3, total_samples)

# Combine everything: base + daily cycle + noise
temperatures = base_temp + daily_cycle + noise

# =============================================================================
# SECTION 5: INJECT ANOMALIES
# Why: We need "abnormal" data so the AI can learn to detect it
# =============================================================================

# Pick 10 random indices where we will inject anomalies
# np.random.choice(total_samples, 10, replace=False)
#   total_samples = choose from all 1440 samples
#   10 = pick 10 of them
#   replace=False = don't pick the same index twice
anomaly_indices = np.random.choice(total_samples, 10, replace=False)

# Add a random spike (5°C to 10°C) at each anomaly location
# np.random.uniform(5, 10, 10) = 10 random values between 5 and 10
temperatures[anomaly_indices] += np.random.uniform(5, 10, 10)

# =============================================================================
# SECTION 6: CREATE DATAFRAME AND SAVE
# Why: Convert to table format and save as CSV
# =============================================================================

# Create a pandas DataFrame (like an Excel table)
# Two columns: timestamp and temperature
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperatures
})

output_path = OUTPUT_DIR / "synthetic_temperature.csv"
df.to_csv(output_path, index=False)

# --- ADD THIS LINE ---
# Save anomaly indices so Phase 2 can exclude them from training
anomaly_path = OUTPUT_DIR / "anomaly_indices.npy"
np.save(anomaly_path, anomaly_indices)
# --- END ---

print(f"Generated {total_samples} samples with {len(anomaly_indices)} anomalies")
print(f"Saved to: {output_path}")
print(f"Saved anomaly indices to: {anomaly_path}")   # <-- ADD THIS TOO