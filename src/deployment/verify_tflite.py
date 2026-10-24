# =============================================================================
# verify_tflite.py
# Purpose: Test if the TFLite model produces the same output as PyTorch
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
TFLITE_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder.tflite"

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
print("PHASE 6.1: VERIFY TFLITE MODEL")
print("=" * 60)

# --- 1. Load PyTorch Model ---
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
config = checkpoint['config']
pytorch_model = LSTMAutoencoder(
    input_size=config['input_size'], hidden_size=config['hidden_size'],
    latent_size=config['latent_size'], num_layers=config['num_layers'],
    dropout=config['dropout']
)
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

# --- 2. Create a random test input ---
dummy_input = np.random.randn(1, WINDOW_SIZE, INPUT_SIZE).astype(np.float32)

# --- 3. Run PyTorch Inference ---
with torch.no_grad():
    torch_output = pytorch_model(torch.tensor(dummy_input)).numpy()

# --- 4. Run TFLite Inference ---
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

# --- 5. Compare ---
difference = np.abs(torch_output - tflite_output)
max_diff = np.max(difference)

print(f"\nPyTorch output shape: {torch_output.shape}")
print(f"TFLite output shape:  {tflite_output.shape}")
print(f"Max difference:       {max_diff:.6f}")

if max_diff < 1e-4:
    print("\n✅ VERIFICATION PASSED: TFLite matches PyTorch!")
else:
    print("\n❌ VERIFICATION FAILED: Outputs are different.")