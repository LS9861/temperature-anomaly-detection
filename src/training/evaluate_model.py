# =============================================================================
# evaluate_model.py
# Purpose: Test if the trained autoencoder can detect anomalies
# Phase:   4 of 7 (Model Evaluation)
# Input:   models/lstm_autoencoder.pth   (from Phase 3)
#          data/processed/X_test.npy      (clean windows)
#          data/processed/X_anomalous.npy (anomalous windows)
# Output:  results/anomaly_detection.png
#          results/error_distribution.png
#          results/confusion_matrix.png
#          results/evaluation_metrics.txt
# =============================================================================

# --- Standard library imports ---
import sys                                  # For exiting with error codes
from pathlib import Path                    # For safe file paths

# --- Third-party imports ---
import numpy as np                          # For array operations
import torch                                # PyTorch core
import torch.nn as nn                       # Neural network layers
import matplotlib.pyplot as plt             # For plotting
from sklearn.metrics import (               # For evaluation metrics
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# =============================================================================
# SECTION 1: PATH SETUP
# Why: Make the script work no matter where you run it from
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input paths
MODEL_PATH       = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
X_TEST_PATH      = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
X_ANOMALOUS_PATH = PROJECT_ROOT / "data" / "processed" / "X_anomalous.npy"

# Output paths
RESULTS_DIR      = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 2: CONFIGURATION
# Why: Must match the values used during training
# =============================================================================

WINDOW_SIZE = 50
INPUT_SIZE  = 1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Decoder
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
        # Encode
        encoder_out, (hidden, cell) = self.encoder_lstm(x)
        last_hidden = encoder_out[:, -1, :]
        latent = self.encoder_fc(last_hidden)

        # Decode
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
print("PHASE 4: MODEL EVALUATION")
print("=" * 60)
print(f"Device: {DEVICE}")

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    print("       Run train_autoencoder.py first (Phase 3).")
    sys.exit(1)

print(f"\n[1/6] Loading trained model...")

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

# Set to evaluation mode (disables dropout)
model.eval()

print(f"      Model loaded successfully")
print(f"      Training loss (from checkpoint): {checkpoint['final_train_loss']:.6f}")
print(f"      Test loss     (from checkpoint): {checkpoint['final_test_loss']:.6f}")

# =============================================================================
# SECTION 5: LOAD TEST DATA
# Why: Clean windows (should have low error) + anomalous windows (high error)
# =============================================================================

print(f"\n[2/6] Loading test data...")

if not X_TEST_PATH.exists() or not X_ANOMALOUS_PATH.exists():
    print(f"ERROR: Test data not found.")
    print("       Run preprocess.py first (Phase 2).")
    sys.exit(1)

# Clean windows (normal data the model should reconstruct well)
X_test = np.load(X_TEST_PATH)
X_test = X_test.reshape(-1, WINDOW_SIZE, INPUT_SIZE)

# Anomalous windows (the model should fail to reconstruct these)
X_anomalous = np.load(X_ANOMALOUS_PATH)
X_anomalous = X_anomalous.reshape(-1, WINDOW_SIZE, INPUT_SIZE)

print(f"      Clean windows:     {X_test.shape}")
print(f"      Anomalous windows: {X_anomalous.shape}")

# =============================================================================
# SECTION 6: COMPUTE RECONSTRUCTION ERROR
# Why: Anomalies = high error. Clean data = low error.
# =============================================================================

print(f"\n[3/6] Computing reconstruction errors...")

def compute_reconstruction_errors(model, data, device):
    """
    For each window, compute the MSE between input and reconstruction.
    Returns a 1D array of errors (one per window).
    """
    # Convert NumPy array to PyTorch tensor
    tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # No gradients needed for evaluation
    with torch.no_grad():
        # Forward pass: reconstruct all windows at once
        reconstructions = model(tensor)

    # Compute per-window MSE
    # Shape after subtraction: (N, 50, 1)
    # Mean over timesteps and features: (N,)
    errors = torch.mean((tensor - reconstructions) ** 2, dim=(1, 2))

    # Move back to CPU and convert to NumPy
    return errors.cpu().numpy()

# Compute errors for both sets
errors_clean     = compute_reconstruction_errors(model, X_test, DEVICE)
errors_anomalous = compute_reconstruction_errors(model, X_anomalous, DEVICE)

print(f"      Clean errors:     mean={errors_clean.mean():.6f}, "
      f"std={errors_clean.std():.6f}, max={errors_clean.max():.6f}")
print(f"      Anomalous errors: mean={errors_anomalous.mean():.6f}, "
      f"std={errors_anomalous.std():.6f}, max={errors_anomalous.max():.6f}")

# =============================================================================
# SECTION 7: DETERMINE THRESHOLD
# Why: Need a cutoff value: above = anomaly, below = normal
# =============================================================================

print(f"\n[4/6] Determining anomaly threshold...")

# Common approach: threshold = mean + k * std of clean errors
# k = 3 means 99.7% of normal data falls below this threshold
k = 3.0
threshold = errors_clean.mean() + k * errors_clean.std()

print(f"      Threshold method: mean + {k} * std (of clean errors)")
print(f"      Threshold value:  {threshold:.6f}")

# Classify based on threshold
# Label 0 = clean, Label 1 = anomaly
predictions_clean     = (errors_clean > threshold).astype(int)
predictions_anomalous = (errors_anomalous > threshold).astype(int)

# True labels: all clean should be 0, all anomalous should be 1
y_true  = np.concatenate([np.zeros(len(errors_clean)), np.ones(len(errors_anomalous))])
y_pred  = np.concatenate([predictions_clean, predictions_anomalous])
y_score = np.concatenate([errors_clean, errors_anomalous])

# =============================================================================
# SECTION 8: COMPUTE METRICS
# Why: Numbers to prove the model works
# =============================================================================

print(f"\n[5/6] Computing metrics...")

# Confusion matrix
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Derived metrics
accuracy    = (tp + tn) / (tp + tn + fp + fn)
precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n      Confusion Matrix:")
print(f"      ┌─────────────────┬─────────────────┐")
print(f"      │ TN = {tn:4d}       │ FP = {fp:4d}       │")
print(f"      ├─────────────────┼─────────────────┤")
print(f"      │ FN = {fn:4d}       │ TP = {tp:4d}       │")
print(f"      └─────────────────┴─────────────────┘")

print(f"\n      Accuracy:    {accuracy:.4f}")
print(f"      Precision:   {precision:.4f}   (of flagged anomalies, how many were real?)")
print(f"      Recall:      {recall:.4f}   (of real anomalies, how many did we catch?)")
print(f"      Specificity: {specificity:.4f}   (of clean windows, how many did we correctly ignore?)")
print(f"      F1 Score:    {f1:.4f}")

# =============================================================================
# SECTION 9: PLOT 1 — ERROR DISTRIBUTION
# Why: Visual proof that clean and anomalous errors are separable
# =============================================================================

print(f"\n[6/6] Generating plots...")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(errors_clean, bins=50, alpha=0.7, label='Clean (normal)', color='green')
ax.hist(errors_anomalous, bins=50, alpha=0.7, label='Anomalous', color='red')
ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
           label=f'Threshold = {threshold:.4f}')
ax.set_xlabel('Reconstruction Error (MSE)')
ax.set_ylabel('Number of Windows')
ax.set_title('Distribution of Reconstruction Errors')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "error_distribution.png", dpi=150)
plt.close()
print(f"      Saved: results/error_distribution.png")

# =============================================================================
# SECTION 10: PLOT 2 — ROC CURVE
# Why: Shows tradeoff between true positive rate and false positive rate
# =============================================================================

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', linewidth=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Anomaly Detection')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
plt.close()
print(f"      Saved: results/roc_curve.png")

# =============================================================================
# SECTION 11: PLOT 3 — CONFUSION MATRIX
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Normal', 'Predicted Anomaly'])
ax.set_yticklabels(['Actual Normal', 'Actual Anomaly'])
ax.set_title('Confusion Matrix')

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=14)

plt.colorbar(im)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print(f"      Saved: results/confusion_matrix.png")

# =============================================================================
# SECTION 12: SAVE METRICS TO TEXT FILE
# Why: Easy reference for README and CV
# =============================================================================

metrics_path = RESULTS_DIR / "evaluation_metrics.txt"
with open(metrics_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("PHASE 4: EVALUATION METRICS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Threshold method:   mean + {k} * std (of clean errors)\n")
    f.write(f"Threshold value:    {threshold:.6f}\n\n")
    f.write(f"Clean windows:      {len(errors_clean)}\n")
    f.write(f"Anomalous windows:  {len(errors_anomalous)}\n\n")
    f.write(f"Clean error (mean):      {errors_clean.mean():.6f}\n")
    f.write(f"Clean error (std):       {errors_clean.std():.6f}\n")
    f.write(f"Anomalous error (mean):  {errors_anomalous.mean():.6f}\n")
    f.write(f"Anomalous error (std):   {errors_anomalous.std():.6f}\n\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"  TN = {tn}, FP = {fp}\n")
    f.write(f"  FN = {fn}, TP = {tp}\n\n")
    f.write(f"Accuracy:    {accuracy:.4f}\n")
    f.write(f"Precision:   {precision:.4f}\n")
    f.write(f"Recall:      {recall:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"F1 Score:    {f1:.4f}\n")
    f.write(f"ROC AUC:     {roc_auc:.4f}\n")

print(f"      Saved: results/evaluation_metrics.txt")

# =============================================================================
# SECTION 13: SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
print(f"  Threshold:   {threshold:.6f}")
print(f"  Accuracy:    {accuracy:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print(f"  F1 Score:    {f1:.4f}")
print(f"  ROC AUC:     {roc_auc:.4f}")
print(f"\n  Plots saved in results/")
print(f"  Metrics saved in results/evaluation_metrics.txt")
print("\nNext: Deploy to ESP32 (Phase 5)")
print("=" * 60)