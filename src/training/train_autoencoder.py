# =============================================================================
# train_autoencoder.py
# Purpose: Train an LSTM Autoencoder to learn "normal" temperature patterns
# Phase:   3 of 7 (Model Training)
# Input:   data/processed/X_train.npy  (from Phase 2)
#          data/processed/X_test.npy   (from Phase 2)
# Output:  models/lstm_autoencoder.pth (trained PyTorch model)
#          results/training_loss.png   (loss curve for README)
# =============================================================================

# --- Standard library imports ---
import sys                                  # For exiting with error codes
from pathlib import Path                    # For safe file paths

# --- Third-party imports ---
import numpy as np                          # For array operations
import torch                                # PyTorch core
import torch.nn as nn                       # Neural network layers
from torch.utils.data import DataLoader, TensorDataset   # For batching data
import matplotlib.pyplot as plt             # For plotting loss curve

# =============================================================================
# SECTION 1: PATH SETUP
# Why: Make the script work no matter where you run it from
# =============================================================================

# __file__ = full path of THIS script
# .resolve() = absolute path
# .parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input paths (created in Phase 2)
X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "X_train.npy"
X_TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "X_test.npy"

# Output paths
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_PATH   = MODELS_DIR / "lstm_autoencoder.pth"
LOSS_PLOT    = RESULTS_DIR / "training_loss.png"

# Create output folders if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 2: CONFIGURATION PARAMETERS
# Why: All hyperparameters in one place — easy to tune later
# =============================================================================

WINDOW_SIZE   = 50      # Must match preprocess.py (50 timesteps per window)
INPUT_SIZE    = 1       # 1 = temperature only. Change to 3 for multi-signal
HIDDEN_SIZE   = 64      # How many features the LSTM remembers internally
LATENT_SIZE   = 16       # Bottleneck size — smaller = more compression
NUM_LAYERS    = 2       # Stacked LSTM layers (2 is a good default)
DROPOUT       = 0.2     # Randomly drop 20% of neurons to prevent overfitting
BATCH_SIZE    = 32      # How many windows per training step
EPOCHS        = 50      # How many times to see the full dataset
LEARNING_RATE = 0.001   # How fast the model updates weights
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# SECTION 3: LOAD PREPROCESSED DATA
# Why: Phase 2 already normalized and windowed the data
# =============================================================================

print("=" * 60)
print("PHASE 3: TRAINING LSTM AUTOENCODER")
print("=" * 60)
print(f"Device: {DEVICE}")

# Check that Phase 2 outputs exist
if not X_TRAIN_PATH.exists() or not X_TEST_PATH.exists():
    print(f"ERROR: Preprocessed data not found.")
    print(f"       Run preprocess.py first (Phase 2).")
    sys.exit(1)

print(f"\n[1/5] Loading preprocessed data...")

# Load NumPy arrays
# X_train shape: (1112, 50)
# X_test  shape: (279, 50)
X_train = np.load(X_TRAIN_PATH)
X_test  = np.load(X_TEST_PATH)

print(f"      X_train shape: {X_train.shape}")
print(f"      X_test  shape: {X_test.shape}")

# =============================================================================
# SECTION 4: RESHAPE FOR LSTM
# Why: LSTM expects 3D input: (batch, timesteps, features)
# =============================================================================

# Current shape: (samples, 50)  — 2D
# LSTM needs:   (samples, 50, 1) — 3D (add feature dimension)
X_train = X_train.reshape(-1, WINDOW_SIZE, INPUT_SIZE)
X_test  = X_test.reshape(-1, WINDOW_SIZE, INPUT_SIZE)

print(f"      Reshaped for LSTM:")
print(f"      X_train: {X_train.shape}")
print(f"      X_test:  {X_test.shape}")

# =============================================================================
# SECTION 5: CREATE PYTORCH DATALOADER
# Why: DataLoader handles batching and shuffling automatically
# =============================================================================

print(f"\n[2/5] Creating DataLoader (batch_size={BATCH_SIZE})...")

# Convert NumPy arrays to PyTorch tensors
# Autoencoder: input = target (the model learns to reconstruct its input)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)

# Wrap tensors in a Dataset
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
test_dataset  = TensorDataset(X_test_tensor,  X_test_tensor)

# DataLoader: shuffles and batches the data during training
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"      Train batches: {len(train_loader)}")
print(f"      Test  batches: {len(test_loader)}")

# =============================================================================
# SECTION 6: DEFINE LSTM AUTOENCODER
# Why: Autoencoder learns to compress then reconstruct normal patterns
# =============================================================================

print(f"\n[3/5] Building LSTM Autoencoder...")

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder architecture:

    ENCODER:  input (50, 1) → LSTM → hidden state (latent)
    DECODER:  latent → repeat → LSTM → output (50, 1)

    The model learns to reconstruct NORMAL data.
    Anomalies produce HIGH reconstruction error.
    """

    def __init__(self, input_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()

        # --- ENCODER ---
        # LSTM reads the sequence and compresses it into a hidden state
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,       # 1 (temperature)
            hidden_size=hidden_size,     # 32
            num_layers=num_layers,       # 2
            batch_first=True,            # input shape: (batch, seq, feature)
            dropout=dropout              # regularization
        )

        # Compress hidden state to latent vector
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # --- DECODER ---
        # Expand latent vector back to hidden size
        self.decoder_fc = nn.Linear(latent_size, hidden_size)

        # LSTM reconstructs the sequence from the latent vector
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Final layer: map hidden state back to original feature size
        self.output_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch, 50, 1)

        # --- ENCODE ---
        # encoder_out: (batch, 50, 32)
        # (hidden, cell): internal LSTM states
        encoder_out, (hidden, cell) = self.encoder_lstm(x)

        # Take only the LAST timestep's hidden state
        # Shape: (batch, 32)
        last_hidden = encoder_out[:, -1, :]

        # Compress to latent vector
        # Shape: (batch, 8)
        latent = self.encoder_fc(last_hidden)

        # --- DECODE ---
        # Expand latent back to hidden size
        # Shape: (batch, 32)
        decoded = self.decoder_fc(latent)

        # Repeat the vector 50 times (one per timestep)
        # Shape: (batch, 50, 32)
        decoded = decoded.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)

        # Decoder LSTM reconstructs the sequence
        # Shape: (batch, 50, 32)
        decoder_out, _ = self.decoder_lstm(decoded)

        # Map back to original feature size (1)
        # Shape: (batch, 50, 1)
        reconstruction = self.output_fc(decoder_out)

        return reconstruction

# Create the model and move to GPU/CPU
model = LSTMAutoencoder(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    latent_size=LATENT_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"      Model parameters: {total_params:,}")
print(f"      Architecture:\n{model}")

# =============================================================================
# SECTION 7: DEFINE LOSS AND OPTIMIZER
# Why: Loss measures error; optimizer updates weights
# =============================================================================

# MSE = Mean Squared Error — standard for reconstruction tasks
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- ADD THESE 3 LINES ---
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# --- END ---


# Adam = adaptive learning rate optimizer — works well out of the box
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================================
# SECTION 8: TRAINING LOOP
# Why: This is where the model actually learns
# =============================================================================

print(f"\n[4/5] Training for {EPOCHS} epochs...")

train_losses = []   # Store loss per epoch for plotting
test_losses  = []

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()                    # Enable dropout
    epoch_train_loss = 0.0

    for batch_x, batch_y in train_loader:
        # Move batch to GPU/CPU
        batch_x = batch_x.to(DEVICE)

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass: reconstruct the input
        output = model(batch_x)

        # Compute loss: how different is output from input?
        loss = criterion(output, batch_x)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss for this epoch
        epoch_train_loss += loss.item()

    # Average training loss for this epoch
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- EVALUATION PHASE ---
    model.eval()                     # Disable dropout
    epoch_test_loss = 0.0

    with torch.no_grad():            # No gradient computation = faster
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x)
            loss = criterion(output, batch_x)
            epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

     # --- ADD THIS LINE ---
    scheduler.step(avg_test_loss)
    # --- END ---
    

    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"      Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f}")

# =============================================================================
# SECTION 9: SAVE THE TRAINED MODEL
# Why: Phase 4 (evaluation) and Phase 5 (deployment) need this
# =============================================================================

print(f"\n[5/5] Saving model and results...")

# Save model weights AND architecture info
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'input_size':  INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'latent_size': LATENT_SIZE,
        'num_layers':  NUM_LAYERS,
        'dropout':     DROPOUT,
        'window_size': WINDOW_SIZE,
    },
    'final_train_loss': train_losses[-1],
    'final_test_loss':  test_losses[-1],
}, MODEL_PATH)

print(f"      Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 10: PLOT LOSS CURVE
# Why: Visual proof the model learned (loss should decrease)
# =============================================================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses,  label='Test Loss',  linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Autoencoder Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LOSS_PLOT, dpi=150)
plt.close()

print(f"      Saved plot:  {LOSS_PLOT.relative_to(PROJECT_ROOT)}")

# =============================================================================
# SECTION 11: SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print(f"  Model:       LSTM Autoencoder ({total_params:,} parameters)")
print(f"  Epochs:      {EPOCHS}")
print(f"  Final train loss: {train_losses[-1]:.6f}")
print(f"  Final test  loss: {test_losses[-1]:.6f}")
print(f"  Saved:       models/lstm_autoencoder.pth")
print(f"  Saved:       results/training_loss.png")
print("\nNext: Run evaluate_model.py (Phase 4)")
print("=" * 60)