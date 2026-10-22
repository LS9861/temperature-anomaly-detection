# =============================================================================
# quantize_to_int8.py
# Purpose: Quantize FP32 model weights to INT8 for ESP32 deployment
# Phase:   5.2.5 of 7 (Deployment)
# Input:   models/lstm_autoencoder.pth   (from Phase 3)
# Output:  src/inference/model_weights_int8.h
#          src/inference/quant_params.h
#          src/inference/model_config.h  (regenerated)
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# SECTION 1: PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH  = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
INFERENCE_DIR = PROJECT_ROOT / "src" / "inference"/"src"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_INT8_PATH = INFERENCE_DIR / "model_weights_int8.h"
QUANT_PARAMS_PATH = INFERENCE_DIR / "quant_params.h"
CONFIG_H_PATH     = INFERENCE_DIR / "model_config.h"

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
# SECTION 3: LOAD MODEL
# =============================================================================

print("=" * 60)
print("PHASE 5.2.5: INT8 QUANTIZATION")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

print(f"\n[1/6] Loading PyTorch model...")
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
# SECTION 4: QUANTIZATION FUNCTIONS
# Why: Convert FP32 → INT8 using symmetric quantization
# =============================================================================

print(f"\n[2/6] Defining quantization scheme...")

def quantize_symmetric(tensor):
    """
    Symmetric INT8 quantization.

    Formula:
        scale = max(abs(tensor)) / 127
        int8_value = round(float_value / scale)
        dequantized = int8_value * scale

    Range: [-127, +127]

    Symmetric means zero-point = 0. This is simpler and works
    well for weights (which are roughly centered around zero).
    """
    # Find the maximum absolute value
    max_abs = np.max(np.abs(tensor))

    # Avoid division by zero for empty/all-zero tensors
    if max_abs == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0

    # Compute scale
    scale = max_abs / 127.0

    # Quantize: divide by scale, round to nearest integer, clip to [-127, 127]
    int8_values = np.round(tensor / scale).clip(-127, 127).astype(np.int8)

    return int8_values, scale

def dequantize(int8_values, scale):
    """Convert INT8 back to FP32 for verification."""
    return int8_values.astype(np.float32) * scale

# =============================================================================
# SECTION 5: QUANTIZE ALL WEIGHTS
# =============================================================================

print(f"\n[3/6] Quantizing weights...")

state = model.state_dict()

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

quantized = {}    # name → int8 array
scales = {}       # name → scale (float32)

print(f"      Layer                                    Max Reconstruction Error")
print(f"      " + "-" * 72)

for pytorch_name, c_name in name_map.items():
    fp32_tensor = state[pytorch_name].detach().cpu().numpy()

    # Quantize
    int8_arr, scale = quantize_symmetric(fp32_tensor)

    # Verify: dequantize and compare
    reconstructed = dequantize(int8_arr, scale)
    max_error = np.max(np.abs(fp32_tensor - reconstructed))
    max_error_rel = max_error / (np.max(np.abs(fp32_tensor)) + 1e-9)

    quantized[c_name] = int8_arr
    scales[c_name] = scale

    print(f"      {c_name:35s}    {max_error:.2e}  ({max_error_rel*100:.2f}%)")

# =============================================================================
# SECTION 6: WRITE INT8 WEIGHTS HEADER
# =============================================================================

print(f"\n[4/6] Writing model_weights_int8.h...")

def array_to_c_int8(name, arr, indent="    "):
    """Convert a NumPy int8 array to a C array declaration."""
    flat = arr.flatten()

    lines = []
    for i in range(0, len(flat), 16):
        chunk = flat[i:i+16]
        line = indent + ", ".join(f"{int(v):4d}" for v in chunk)
        lines.append(line)

    body = ",\n".join(lines)
    return f"const int8_t {name}[{len(flat)}] = {{\n{body}\n}};\n"

with open(WEIGHTS_INT8_PATH, 'w') as f:
    f.write("// =============================================================================\n")
    f.write("// model_weights_int8.h\n")
    f.write("// AUTO-GENERATED by quantize_to_int8.py\n")
    f.write("// DO NOT EDIT MANUALLY\n")
    f.write("// =============================================================================\n\n")
    f.write("#ifndef MODEL_WEIGHTS_INT8_H\n")
    f.write("#define MODEL_WEIGHTS_INT8_H\n\n")
    f.write("#include <stdint.h>\n\n")

    for name, arr in quantized.items():
        f.write(f"// {name} — shape {list(arr.shape)}\n")
        f.write(array_to_c_int8(name, arr))
        f.write("\n")

    f.write("#endif  // MODEL_WEIGHTS_INT8_H\n")

print(f"      Saved: {WEIGHTS_INT8_PATH.relative_to(PROJECT_ROOT)}")
print(f"      Total arrays: {len(quantized)}")

# =============================================================================
# SECTION 7: WRITE QUANT PARAMS HEADER
# =============================================================================

print(f"\n[5/6] Writing quant_params.h...")

with open(QUANT_PARAMS_PATH, 'w') as f:
    f.write("// =============================================================================\n")
    f.write("// quant_params.h\n")
    f.write("// AUTO-GENERATED by quantize_to_int8.py\n")
    f.write("// DO NOT EDIT MANUALLY\n")
    f.write("// =============================================================================\n\n")
    f.write("#ifndef QUANT_PARAMS_H\n")
    f.write("#define QUANT_PARAMS_H\n\n")

    f.write("// Symmetric INT8 quantization: float_value = int8_value * scale\n\n")

    for name, scale in scales.items():
        f.write(f"const float {name}_scale = {scale:.10e}f;\n")

    f.write("\n#endif  // QUANT_PARAMS_H\n")

print(f"      Saved: {QUANT_PARAMS_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 8: REGENERATE MODEL CONFIG
# =============================================================================

with open(CONFIG_H_PATH, 'w') as f:
    f.write("// =============================================================================\n")
    f.write("// model_config.h\n")
    f.write("// AUTO-GENERATED by quantize_to_int8.py\n")
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
# SECTION 9: SUMMARY
# =============================================================================

total_params = sum(arr.size for arr in quantized.values())
int8_bytes = total_params * 1                    # int8 = 1 byte
fp32_bytes = total_params * 4                    # for comparison

print("\n" + "=" * 60)
print("PHASE 5.2.5 COMPLETE")
print("=" * 60)
print(f"  Total parameters:        {total_params:,}")
print(f"  FP32 size:               {fp32_bytes/1024:.1f} KB")
print(f"  INT8 size:               {int8_bytes/1024:.1f} KB")
print(f"  Reduction:               {fp32_bytes/int8_bytes:.1f}x smaller")
print(f"  ESP32 usable RAM:        ~320 KB")
print(f"  Usage after quantize:    {100*int8_bytes/1024/320:.1f}% of usable")
print(f"\n  Generated files:")
print(f"    src/inference/src/model_weights_int8.h")
print(f"    src/inference/src/quant_params.h")
print(f"    src/inference/src/model_config.h")
print("\nNext: Phase 5.3 — Write ESP32 firmware")
print("=" * 60)