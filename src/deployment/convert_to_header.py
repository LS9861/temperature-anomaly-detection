# =============================================================================
# convert_to_header.py  (FIXED VERSION)
# Purpose: Convert PyTorch model weights to C header files for ESP32
# Phase:   5.2 of 7 (Deployment)
# Input:   models/lstm_autoencoder.pth   (from Phase 3)
#          models/scaler.pkl             (from Phase 2)
# Output:  src/inference/model_weights.h
#          src/inference/scaler_params.h
#          src/inference/model_config.h
# =============================================================================

import sys
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# SECTION 1: PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH  = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

INFERENCE_DIR = PROJECT_ROOT / "src" / "inference"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH  = INFERENCE_DIR / "model_weights.h"
SCALER_H_PATH = INFERENCE_DIR / "scaler_params.h"
CONFIG_H_PATH = INFERENCE_DIR / "model_config.h"

# =============================================================================
# SECTION 2: MODEL ARCHITECTURE (must match Phase 3)
# =============================================================================

WINDOW_SIZE = 50
INPUT_SIZE  = 1
HIDDEN_SIZE = 64
LATENT_SIZE = 16
NUM_LAYERS  = 2
DROPOUT     = 0.2

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout)
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

# =============================================================================
# SECTION 3: LOAD PYTORCH MODEL
# =============================================================================

print("=" * 60)
print("PHASE 5.2: CONVERT PYTORCH WEIGHTS TO C HEADER")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

print(f"\n[1/5] Loading PyTorch model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
config = checkpoint['config']

model = LSTMAutoencoder(
    input_size=config['input_size'],
    hidden_size=config['hidden_size'],
    latent_size=config['latent_size'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"      Model loaded successfully")

# =============================================================================
# SECTION 4: EXTRACT WEIGHTS FROM PYTORCH STATE DICT
# Why: PyTorch names are guaranteed — no guessing needed
# =============================================================================

print(f"\n[2/5] Extracting weights from PyTorch state_dict...")

# state_dict is a dictionary mapping parameter names to tensors
state = model.state_dict()

# Print all keys so we can see the structure
print(f"      Found {len(state)} parameter tensors:")
for name, tensor in state.items():
    print(f"        {name:35s}  shape={list(tensor.shape)}")

# Convert to NumPy
weights = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}

# =============================================================================
# SECTION 5: WRITE MODEL WEIGHTS HEADER
# =============================================================================

print(f"\n[3/5] Writing model_weights.h...")

def array_to_c(name, arr, indent="    "):
    """Convert a NumPy array to a C array declaration."""
    flat = arr.astype(np.float32).flatten()

    # Write 8 values per line
    lines = []
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        line = indent + ", ".join(f"{v:.8f}f" for v in chunk)
        lines.append(line)

    body = ",\n".join(lines)
    return f"const float {name}[{len(flat)}] = {{\n{body}\n}};\n"

# Map PyTorch names → clean C names
name_map = {
    'encoder_lstm.weight_ih_l0': 'enc_lstm_W_ih_l0',
    'encoder_lstm.weight_hh_l0': 'enc_lstm_W_hh_l0',
    'encoder_lstm.bias_ih_l0':   'enc_lstm_b_ih_l0',
    'encoder_lstm.bias_hh_l0':   'enc_lstm_b_hh_l0',
    'encoder_lstm.weight_ih_l1': 'enc_lstm_W_ih_l1',
    'encoder_lstm.weight_hh_l1': 'enc_lstm_W_hh_l1',
    'encoder_lstm.bias_ih_l1':   'enc_lstm_b_ih_l1',
    'encoder_lstm.bias_hh_l1':   'enc_lstm_b_hh_l1',
    'encoder_fc.weight':         'enc_fc_W',
    'encoder_fc.bias':           'enc_fc_b',
    'decoder_fc.weight':         'dec_fc_W',
    'decoder_fc.bias':           'dec_fc_b',
    'decoder_lstm.weight_ih_l0': 'dec_lstm_W_ih_l0',
    'decoder_lstm.weight_hh_l0': 'dec_lstm_W_hh_l0',
    'decoder_lstm.bias_ih_l0':   'dec_lstm_b_ih_l0',
    'decoder_lstm.bias_hh_l0':   'dec_lstm_b_hh_l0',
    'decoder_lstm.weight_ih_l1': 'dec_lstm_W_ih_l1',
    'decoder_lstm.weight_hh_l1': 'dec_lstm_W_hh_l1',
    'decoder_lstm.bias_ih_l1':   'dec_lstm_b_ih_l1',
    'decoder_lstm.bias_hh_l1':   'dec_lstm_b_hh_l1',
    'output_fc.weight':          'out_fc_W',
    'output_fc.bias':            'out_fc_b',
}

# Verify all expected keys exist
missing = [k for k in name_map.keys() if k not in weights]
if missing:
    print(f"      ❌ ERROR: Missing keys: {missing}")
    sys.exit(1)

# Build a new dict with clean names
model_weights = {name_map[k]: weights[k] for k in name_map}

with open(WEIGHTS_PATH, 'w') as f:
    f.write("// =============================================================================\n")
    f.write("// model_weights.h\n")
    f.write("// AUTO-GENERATED by convert_to_header.py\n")
    f.write("// DO NOT EDIT MANUALLY\n")
    f.write("// =============================================================================\n\n")
    f.write("#ifndef MODEL_WEIGHTS_H\n")
    f.write("#define MODEL_WEIGHTS_H\n\n")

    for name, arr in model_weights.items():
        f.write(f"// {name} — shape {list(arr.shape)}\n")
        f.write(array_to_c(name, arr))
        f.write("\n")

    f.write("#endif  // MODEL_WEIGHTS_H\n")

print(f"      Saved: {WEIGHTS_PATH.relative_to(PROJECT_ROOT)}")
print(f"      Total arrays: {len(model_weights)}")

# =============================================================================
# SECTION 6: WRITE SCALER PARAMS
# =============================================================================

print(f"\n[4/5] Writing scaler_params.h...")

scaler = joblib.load(SCALER_PATH)
mean_value = float(scaler.mean_[0])
scale_value = float(scaler.scale_[0])

with open(SCALER_H_PATH, 'w') as f:
    f.write("// =============================================================================\n")
    f.write("// scaler_params.h\n")
    f.write("// AUTO-GENERATED by convert_to_header.py\n")
    f.write("// =============================================================================\n\n")
    f.write("#ifndef SCALER_PARAMS_H\n")
    f.write("#define SCALER_PARAMS_H\n\n")
    f.write(f"const float SCALER_MEAN = {mean_value:.8f}f;\n")
    f.write(f"const float SCALER_STD  = {scale_value:.8f}f;\n\n")
    f.write("#endif  // SCALER_PARAMS_H\n")

print(f"      Saved: {SCALER_H_PATH.relative_to(PROJECT_ROOT)}")
print(f"      SCALER_MEAN = {mean_value:.6f}")
print(f"      SCALER_STD  = {scale_value:.6f}")

# =============================================================================
# SECTION 7: WRITE MODEL CONFIG
# =============================================================================

print(f"\n[5/5] Writing model_config.h...")

with open(CONFIG_H_PATH, 'w') as f:
    f.write("// =============================================================================\n")
    f.write("// model_config.h\n")
    f.write("// AUTO-GENERATED by convert_to_header.py\n")
    f.write("// =============================================================================\n\n")
    f.write("#ifndef MODEL_CONFIG_H\n")
    f.write("#define MODEL_CONFIG_H\n\n")
    f.write(f"#define WINDOW_SIZE  {WINDOW_SIZE}\n")
    f.write(f"#define INPUT_SIZE   {INPUT_SIZE}\n")
    f.write(f"#define HIDDEN_SIZE  {HIDDEN_SIZE}\n")
    f.write(f"#define LATENT_SIZE  {LATENT_SIZE}\n")
    f.write(f"#define NUM_LAYERS   {NUM_LAYERS}\n")
    f.write(f"#define NUM_GATES    4\n\n")
    f.write("// Anomaly threshold (from Phase 4)\n")
    f.write("#define ANOMALY_THRESHOLD  0.171626f\n\n")
    f.write("#endif  // MODEL_CONFIG_H\n")

print(f"      Saved: {CONFIG_H_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 8: SUMMARY
# =============================================================================

total_params = sum(arr.size for arr in model_weights.values())
total_bytes = total_params * 4

print("\n" + "=" * 60)
print("PHASE 5.2 COMPLETE")
print("=" * 60)
print(f"  Total parameters:     {total_params:,}")
print(f"  Total bytes (FP32):   {total_bytes:,} ({total_bytes/1024:.1f} KB)")
print(f"  ESP32 RAM available:  520 KB")
print(f"  Usable after OS:      ~320 KB")
print(f"  RAM usage estimate:   {100*total_bytes/1024/320:.1f}% (of usable)")
print(f"\n  Generated files:")
print(f"    src/inference/model_weights.h")
print(f"    src/inference/scaler_params.h")
print(f"    src/inference/model_config.h")
print("\nNext: Phase 5.3 — Write ESP32 firmware")
print("=" * 60)