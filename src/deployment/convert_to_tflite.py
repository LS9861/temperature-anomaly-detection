# =============================================================================
# convert_to_tflite.py  (TinyNeuralNetwork version)
# Purpose: Convert PyTorch LSTM model directly to TFLite
# =============================================================================

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
TFLITE_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.tflite"

# --- Model architecture (must match Phase 3) ---
WINDOW_SIZE = 50
INPUT_SIZE = 1
HIDDEN_SIZE = 64
LATENT_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.2

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

print("=" * 60)
print("PHASE 6.1: PYTORCH -> TFLITE (TinyNeuralNetwork)")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

# Load checkpoint
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

print(f"\n[1/2] PyTorch model loaded: {MODEL_PATH.name}")

# Create a sample input for tracing
dummy_input = torch.randn(1, WINDOW_SIZE, INPUT_SIZE)

# Convert using TinyNeuralNetwork
from tinynn.converter import TFLiteConverter

print(f"\n[2/2] Converting to TFLite...")

# The TFLiteConverter needs the model, a dummy input, and the output path
converter = TFLiteConverter(
    model,
    dummy_input,
    str(TFLITE_PATH),
    nchw_transpose=False,   # Our data is already in the correct layout
    unroll_rnn=True,        # Unrolls the LSTM, which is required for TFLite
)
converter.convert()

print(f"      TFLite saved: {TFLITE_PATH.name} ({TFLITE_PATH.stat().st_size / 1024:.1f} KB)")
print("\n" + "=" * 60)
print("PHASE 6.1 COMPLETE")
print("=" * 60)