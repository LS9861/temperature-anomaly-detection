# =============================================================================
# quantize_tflite.py  (quantize from existing .tflite)
# Purpose: Quantize FP32 TFLite to INT8 using calibration data
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TFLITE_PATH   = PROJECT_ROOT / "models" / "lstm_autoencoder.tflite"
INT8_PATH     = PROJECT_ROOT / "models" / "lstm_autoencoder_int8.tflite"
CSV_PATH      = PROJECT_ROOT / "data" / "synthetic" / "real_temperature_24h.csv"
SCALER_PATH   = PROJECT_ROOT / "models" / "scaler.pkl"

WINDOW_SIZE = 50
INPUT_SIZE  = 1
NUM_CALIBRATION_SAMPLES = 100

print("=" * 60)
print("PHASE 6.2: QUANTIZE TFLITE TO INT8")
print("=" * 60)

if not TFLITE_PATH.exists():
    print(f"ERROR: TFLite file not found: {TFLITE_PATH}")
    sys.exit(1)

if not CSV_PATH.exists():
    print(f"ERROR: Calibration data not found: {CSV_PATH}")
    sys.exit(1)

print(f"\n[1/5] Inputs found")
print(f"      TFLite:  {TFLITE_PATH.name} ({TFLITE_PATH.stat().st_size / 1024:.1f} KB)")
print(f"      CSV:     {CSV_PATH.name}")

# --- Load scaler and prepare calibration data ---
print(f"\n[2/5] Preparing calibration data...")

scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(CSV_PATH)
temperatures = df['temperature'].values
temperatures_normalized = scaler.transform(temperatures.reshape(-1, 1)).flatten()

windows = []
for i in range(len(temperatures_normalized) - WINDOW_SIZE + 1):
    windows.append(temperatures_normalized[i : i + WINDOW_SIZE])
windows = np.array(windows, dtype=np.float32)
windows = windows.reshape(-1, WINDOW_SIZE, INPUT_SIZE)
calibration_windows = windows[:NUM_CALIBRATION_SAMPLES]

print(f"      Total windows available: {len(windows)}")
print(f"      Using for calibration:   {len(calibration_windows)}")
print(f"      Window shape:            {calibration_windows.shape}")

# --- Quantize directly from the .tflite file ---
print(f"\n[3/5] Quantizing model to INT8...")

import tensorflow as tf

# Load the FP32 TFLite model as a TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()

# Get the input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a representative dataset generator for INT8 calibration
def representative_dataset_gen():
    for i in range(len(calibration_windows)):
        yield [calibration_windows[i:i+1]]

# Use the TFLiteConverter's from_concrete_functions method
# We need to wrap the TFLite interpreter as a concrete function.
# This is a workaround for not having the SavedModel.
# We will use the TFLite file's model content directly.
# Unfortunately, this is not directly supported by the Python API.
# The easiest workaround is to re-export the model as a SavedModel
# from the PyTorch model, but that failed earlier.

# Instead, we will use the ai-edge-torch approach (but it is Linux-only).
# So we must accept that full INT8 quantization from .tflite is not
# officially supported in TensorFlow 2.x.

# As a fallback, we will use the `tflite_support` package which can
# quantize a TFLite model directly. Let's install it.
try:
    from tflite_support import flatbuffers
    from tflite_support import metadata as _metadata
    from tflite_support import model_analyzer
    print("      tflite_support is available")
except ImportError:
    print("      ERROR: tflite_support is not installed.")
    print("      Run: pip install tflite-support")
    sys.exit(1)

# The tflite_support API is complex. Instead, we will provide
# a simpler solution: use the `tflite_model_maker` library.
# Install it with: pip install tflite-model-maker
# But this is also very heavy.

# Given the constraints, the most reliable way is to
# go back to the PyTorch model and create a SavedModel.
# We already have the PyTorch model. Let's export it to SavedModel
# using TensorFlow's `tf.Module` wrapper. This is the cleanest way.

print("      Falling back to PyTorch -> SavedModel -> TFLite INT8...")

# --- Load PyTorch model ---
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoder_out, _ = self.encoder_lstm(x)
        last_hidden = encoder_out[:, -1, :]
        latent = self.encoder_fc(last_hidden)
        decoded = self.decoder_fc(latent)
        decoded = decoded.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
        decoder_out, _ = self.decoder_lstm(decoded)
        reconstruction = self.output_fc(decoder_out)
        return reconstruction

MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
config = checkpoint['config']
pytorch_model = LSTMAutoencoder(
    input_size=config['input_size'], hidden_size=config['hidden_size'],
    latent_size=config['latent_size'], num_layers=config['num_layers'],
    dropout=config['dropout']
)
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

# --- Wrap PyTorch model as a tf.Module ---
# This is non-trivial. We need to trace the model using `tf.function`.
# But the PyTorch and TensorFlow APIs are different.
# The easiest way is to use `ai-edge-torch` but it is Linux-only.

print("      ERROR: Cannot proceed with quantization on Windows without ai-edge-torch.")
print("      Please use one of these alternatives:")
print("        1. Use Linux (WSL) with ai-edge-torch.")
print("        2. Use Google Colab to convert the model to INT8 TFLite.")
print("        3. Use the FP32 TFLite model on ESP32 with TFLM, if RAM allows.")
sys.exit(1)