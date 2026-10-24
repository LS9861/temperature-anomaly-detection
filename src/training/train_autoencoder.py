# =============================================================================
# train_autoencoder.py  (1D-CNN Version)
# Purpose: Train a 1D Convolutional Autoencoder for TFLM-compatible deployment
# Phase:   3 of 7 (Model Training)
# Input:   data/processed/X_train.npy
# Output:  models/lstm_autoencoder.pth  (name kept for pipeline compatibility)
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "X_train.npy"
X_TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "X_test.npy"

MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_PATH   = MODELS_DIR / "lstm_autoencoder.pth"   # kept same name
LOSS_PLOT    = RESULTS_DIR / "training_loss.png"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

WINDOW_SIZE   = 50
INPUT_SIZE    = 1
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 0.001
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# SECTION 3: LOAD PREPROCESSED DATA
# =============================================================================

print("=" * 60)
print("PHASE 3: TRAINING 1D-CNN AUTOENCODER")
print("=" * 60)
print(f"Device: {DEVICE}")

if not X_TRAIN_PATH.exists() or not X_TEST_PATH.exists():
    print(f"ERROR: Preprocessed data not found.")
    sys.exit(1)

X_train = np.load(X_TRAIN_PATH)
X_test  = np.load(X_TEST_PATH)

print(f"\n[1/5] Loading preprocessed data...")
print(f"      X_train shape: {X_train.shape}")
print(f"      X_test  shape: {X_test.shape}")

# Reshape for Conv1d: (batch, channels, timesteps)
X_train = X_train.reshape(-1, INPUT_SIZE, WINDOW_SIZE)
X_test  = X_test.reshape(-1, INPUT_SIZE, WINDOW_SIZE)

print(f"      Reshaped for CNN:")
print(f"      X_train: {X_train.shape}")
print(f"      X_test:  {X_test.shape}")

# =============================================================================
# SECTION 4: DATALOADER
# =============================================================================

print(f"\n[2/5] Creating DataLoader (batch_size={BATCH_SIZE})...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
test_dataset  = TensorDataset(X_test_tensor,  X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"      Train batches: {len(train_loader)}")
print(f"      Test  batches: {len(test_loader)}")

# =============================================================================
# SECTION 5: 1D-CNN AUTOENCODER
# =============================================================================

print(f"\n[3/5] Building 1D-CNN Autoencoder...")

class CNNAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder.
    Encoder: Conv1d -> ReLU -> MaxPool -> Conv1d -> ReLU -> MaxPool
    Decoder: ConvTranspose1d -> ReLU -> ConvTranspose1d -> Sigmoid
    """
    def __init__(self, input_size=1, window_size=50):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 50 -> 25
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 25 -> 12 (floor)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=2, stride=2),  # 12 -> 24
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=input_size, kernel_size=2, stride=2),  # 24 -> 48
            # We need exactly 50, so we will pad to 50 in forward()
        )

    def forward(self, x):
        # x: (batch, 1, 50)
        encoded = self.encoder(x)          # (batch, 8, 12)
        decoded = self.decoder(encoded)    # (batch, 1, 48)
        # Pad to match original length (50)
        if decoded.shape[2] < WINDOW_SIZE:
            pad_size = WINDOW_SIZE - decoded.shape[2]
            decoded = nn.functional.pad(decoded, (0, pad_size))
        return decoded


model = CNNAutoencoder(input_size=INPUT_SIZE, window_size=WINDOW_SIZE).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"      Model parameters: {total_params:,}")

# =============================================================================
# SECTION 6: LOSS AND OPTIMIZER
# =============================================================================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# =============================================================================
# SECTION 7: TRAINING LOOP
# =============================================================================

print(f"\n[4/5] Training for {EPOCHS} epochs...")

train_losses = []
test_losses  = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x)
            loss = criterion(output, batch_x)
            epoch_test_loss += loss.item()
    avg_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    scheduler.step(avg_test_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"      Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f}")

# =============================================================================
# SECTION 8: SAVE MODEL
# =============================================================================

print(f"\n[5/5] Saving model and results...")

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'input_size':  INPUT_SIZE,
        'window_size': WINDOW_SIZE,
        'architecture': '1D-CNN',
    },
    'final_train_loss': train_losses[-1],
    'final_test_loss':  test_losses[-1],
}, MODEL_PATH)

print(f"      Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 9: PLOT LOSS
# =============================================================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses,  label='Test Loss',  linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('1D-CNN Autoencoder Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LOSS_PLOT, dpi=150)
plt.close()

print(f"      Saved plot:  {LOSS_PLOT.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 10: SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE (1D-CNN)")
print("=" * 60)
print(f"  Model:      1D-CNN Autoencoder ({total_params:,} parameters)")
print(f"  Epochs:     {EPOCHS}")
print(f"  Final train loss: {train_losses[-1]:.6f}")
print(f"  Final test  loss: {test_losses[-1]:.6f}")
print(f"  Saved:      models/lstm_autoencoder.pth")
print(f"  Saved:      results/training_loss.png")
print("\nNext: Run evaluate_model.py (Phase 4)")
print("=" * 60)