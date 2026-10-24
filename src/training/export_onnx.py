# =============================================================================
# export_onnx.py  (1D-CNN Version)
# Purpose: Convert trained 1D-CNN model to ONNX format
# Phase:   5.1 of 7 (Deployment Preparation)
# Input:   models/lstm_autoencoder.pth  (from Phase 3)
# Output:  models/lstm_autoencoder.onnx
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
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
ONNX_PATH  = PROJECT_ROOT / "models" / "lstm_autoencoder.onnx"

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

WINDOW_SIZE = 50
INPUT_SIZE  = 1
DEVICE      = torch.device("cpu")

# =============================================================================
# SECTION 3: 1D-CNN ARCHITECTURE (must match Phase 3)
# =============================================================================

class CNNAutoencoder(nn.Module):
    def __init__(self, input_size=1, window_size=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=input_size, kernel_size=2, stride=2),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if decoded.shape[2] < WINDOW_SIZE:
            pad_size = WINDOW_SIZE - decoded.shape[2]
            decoded = nn.functional.pad(decoded, (0, pad_size))
        return decoded

# =============================================================================
# SECTION 4: LOAD TRAINED MODEL
# =============================================================================

print("=" * 60)
print("PHASE 5.1: EXPORT 1D-CNN MODEL TO ONNX")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

print(f"\n[1/4] Loading trained PyTorch model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint['config']
print(f"      Model config: {config}")

model = CNNAutoencoder(
    input_size=config['input_size'],
    window_size=config['window_size']
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"      Model loaded successfully")
print(f"      Training loss (from checkpoint): {checkpoint['final_train_loss']:.6f}")

# =============================================================================
# SECTION 5: DUMMY INPUT FOR ONNX
# =============================================================================

print(f"\n[2/4] Creating dummy input for ONNX export...")

# Shape: (batch, channels, timesteps) for Conv1d
dummy_input = torch.randn(1, INPUT_SIZE, WINDOW_SIZE).to(DEVICE)
print(f"      Dummy input shape: {dummy_input.shape}")

# =============================================================================
# SECTION 6: EXPORT TO ONNX
# =============================================================================

print(f"\n[3/4] Exporting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"      ONNX file saved: {ONNX_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 7: VERIFY
# =============================================================================

print(f"\n[4/4] Verifying ONNX export...")

import onnxruntime as ort
print(f"      onnxruntime version: {ort.__version__}")

with torch.no_grad():
    pytorch_output = model(dummy_input).cpu().numpy()

ort_session = ort.InferenceSession(str(ONNX_PATH))
input_name = ort_session.get_inputs()[0].name

onnx_output = ort_session.run(
    None,
    {input_name: dummy_input.cpu().numpy()}
)[0]

max_diff = np.max(np.abs(pytorch_output - onnx_output))

print(f"      PyTorch output shape: {pytorch_output.shape}")
print(f"      ONNX output shape:    {onnx_output.shape}")
print(f"      Max difference:       {max_diff:.10f}")

if max_diff < 1e-5:
    print(f"      ✅ VERIFICATION PASSED — ONNX matches PyTorch")
else:
    print(f"      ❌ VERIFICATION FAILED")
    sys.exit(1)

# =============================================================================
# SECTION 8: SUMMARY
# =============================================================================

onnx_size_kb = ONNX_PATH.stat().st_size / 1024
pytorch_size_kb = MODEL_PATH.stat().st_size / 1024

print("\n" + "=" * 60)
print("PHASE 5.1 COMPLETE")
print("=" * 60)
print(f"  PyTorch model: {pytorch_size_kb:.1f} KB")
print(f"  ONNX model:    {onnx_size_kb:.1f} KB")
print(f"  Verification:  PASSED (max diff = {max_diff:.2e})")
print(f"\n  Output: models/lstm_autoencoder.onnx")
print("=" * 60)