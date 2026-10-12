# =============================================================================
# export_onnx.py
# Purpose: Convert trained PyTorch model to ONNX format
# Phase:   5.1 of 7 (Deployment Preparation)
# Input:   models/lstm_autoencoder.pth  (from Phase 3)
# Output:  models/lstm_autoencoder.onnx (for C header conversion)
# =============================================================================

# --- Standard library imports ---
import sys                                  # For exiting with error codes
from pathlib import Path                    # For safe file paths

# --- Third-party imports ---
import numpy as np                          # For array operations
import torch                                # PyTorch core
import torch.nn as nn                       # Neural network layers

# =============================================================================
# SECTION 1: PATH SETUP
# Why: Make the script work no matter where you run it from
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input: trained PyTorch model
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"

# Output: ONNX model
ONNX_PATH  = PROJECT_ROOT / "models" / "lstm_autoencoder.onnx"

# =============================================================================
# SECTION 2: CONFIGURATION
# Why: Must match the values used during training
# =============================================================================

WINDOW_SIZE = 50
INPUT_SIZE  = 1
DEVICE      = torch.device("cpu")   # ONNX export works best on CPU

# =============================================================================
# SECTION 3: DEFINE THE SAME MODEL ARCHITECTURE
# Why: PyTorch needs the class definition to load the saved weights
# =============================================================================

class LSTMAutoencoder(nn.Module):
    """
    Identical architecture to Phase 3.
    Must match exactly, otherwise loading weights will fail.
    """

    def __init__(self, input_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()

        # --- ENCODER ---
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # --- DECODER ---
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.output_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch, 50, 1)

        # --- ENCODE ---
        encoder_out, (hidden, cell) = self.encoder_lstm(x)
        last_hidden = encoder_out[:, -1, :]
        latent = self.encoder_fc(last_hidden)

        # --- DECODE ---
        decoded = self.decoder_fc(latent)
        decoded = decoded.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
        decoder_out, _ = self.decoder_lstm(decoded)
        reconstruction = self.output_fc(decoder_out)

        return reconstruction

# =============================================================================
# SECTION 4: LOAD TRAINED MODEL
# Why: We need the weights learned during Phase 3
# =============================================================================

print("=" * 60)
print("PHASE 5.1: EXPORT PYTORCH MODEL TO ONNX")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    print("       Run train_autoencoder.py first (Phase 3).")
    sys.exit(1)

print(f"\n[1/4] Loading trained PyTorch model...")
print(f"      Path: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

# Load the checkpoint dictionary
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Extract configuration saved during training
config = checkpoint['config']
print(f"      Model config: {config}")

# Rebuild the model with the same architecture
model = LSTMAutoencoder(
    input_size=config['input_size'],
    hidden_size=config['hidden_size'],
    latent_size=config['latent_size'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
).to(DEVICE)

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])

# CRITICAL: Set to evaluation mode
# This disables dropout. Without this, ONNX export will include
# random dropout, and the exported model will give wrong results.
model.eval()

print(f"      Model loaded successfully")
print(f"      Training loss (from checkpoint): {checkpoint['final_train_loss']:.6f}")

# =============================================================================
# SECTION 5: CREATE DUMMY INPUT FOR ONNX EXPORT
# Why: ONNX needs to trace the model with a sample input
# =============================================================================

print(f"\n[2/4] Creating dummy input for ONNX export...")

# ONNX export works by running the model once with a sample input
# and recording all operations. This is called "tracing".
#
# Shape: (batch_size=1, window_size=50, input_size=1)
# We use batch_size=1 because ESP32 will process one window at a time
dummy_input = torch.randn(1, WINDOW_SIZE, INPUT_SIZE).to(DEVICE)

print(f"      Dummy input shape: {dummy_input.shape}")

# =============================================================================
# SECTION 6: EXPORT TO ONNX
# Why: Convert PyTorch model to universal ONNX format
# =============================================================================

print(f"\n[3/4] Exporting to ONNX...")

# torch.onnx.export() does the conversion
torch.onnx.export(
    model,                          # The PyTorch model
    dummy_input,                    # Sample input for tracing
    str(ONNX_PATH),                 # Output file path
    export_params=True,             # Include trained weights in the file
    opset_version=13,               # ONNX operator set version (13 is stable)
    do_constant_folding=True,       # Optimize constant expressions
    input_names=['input'],          # Name for the input tensor
    output_names=['output'],        # Name for the output tensor
    dynamic_axes={                  # Allow variable batch size
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"      ONNX file saved: {ONNX_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 7: VERIFY THE EXPORT
# Why: Confirm ONNX model gives same output as PyTorch model
# =============================================================================

print(f"\n[4/4] Verifying ONNX export...")

# Import ONNX runtime for verification
try:
    import onnxruntime as ort
    print(f"      onnxruntime version: {ort.__version__}")
except ImportError:
    print(f"      WARNING: onnxruntime not installed.")
    print(f"      Install with: pip install onnxruntime")
    print(f"      Skipping verification.")
    sys.exit(0)

# --- Run PyTorch model ---
with torch.no_grad():
    pytorch_output = model(dummy_input).cpu().numpy()

# --- Run ONNX model ---
# Create an ONNX runtime session
ort_session = ort.InferenceSession(str(ONNX_PATH))

# Get the input name from the ONNX model
input_name = ort_session.get_inputs()[0].name

# Run inference
onnx_output = ort_session.run(
    None,                                    # Output names (None = all)
    {input_name: dummy_input.cpu().numpy()}  # Input dict
)[0]

# --- Compare outputs ---
# They should be nearly identical (within floating point precision)
max_diff = np.max(np.abs(pytorch_output - onnx_output))

print(f"      PyTorch output shape: {pytorch_output.shape}")
print(f"      ONNX output shape:    {onnx_output.shape}")
print(f"      Max difference:       {max_diff:.10f}")

if max_diff < 1e-5:
    print(f"      ✅ VERIFICATION PASSED — ONNX matches PyTorch")
else:
    print(f"      ❌ VERIFICATION FAILED — outputs differ significantly")
    sys.exit(1)

# =============================================================================
# SECTION 8: SUMMARY
# =============================================================================

# Get file size
onnx_size_kb = ONNX_PATH.stat().st_size / 1024
pytorch_size_kb = MODEL_PATH.stat().st_size / 1024

print("\n" + "=" * 60)
print("PHASE 5.1 COMPLETE")
print("=" * 60)
print(f"  PyTorch model: {pytorch_size_kb:.1f} KB")
print(f"  ONNX model:    {onnx_size_kb:.1f} KB")
print(f"  Verification:  PASSED (max diff = {max_diff:.2e})")
print(f"\n  Output: models/lstm_autoencoder.onnx")
print("\nNext: Phase 5.2 — Convert ONNX to C header")
print("=" * 60)