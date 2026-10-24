# =============================================================================
# evaluate_model.py  (1D-CNN Version)
# Purpose: Evaluate the trained 1D-CNN Autoencoder
# Phase:   4 of 7 (Model Evaluation)
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH       = PROJECT_ROOT / "models" / "lstm_autoencoder.pth"
X_TEST_PATH      = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
X_ANOMALOUS_PATH = PROJECT_ROOT / "data" / "processed" / "X_anomalous.npy"
RESULTS_DIR      = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50
INPUT_SIZE  = 1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1D-CNN Autoencoder (must match training)
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
# LOAD MODEL
# =============================================================================
print("=" * 60)
print("PHASE 4: MODEL EVALUATION (1D-CNN)")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint['config']
print(f"      Model config: {config}")

model = CNNAutoencoder(input_size=config['input_size'], window_size=config['window_size']).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"      Model loaded successfully")
print(f"      Training loss (checkpoint): {checkpoint['final_train_loss']:.6f}")
print(f"      Test loss (checkpoint):     {checkpoint['final_test_loss']:.6f}")

# =============================================================================
# LOAD TEST DATA
# =============================================================================
X_test = np.load(X_TEST_PATH)
X_test = X_test.reshape(-1, INPUT_SIZE, WINDOW_SIZE)

X_anomalous = np.load(X_ANOMALOUS_PATH)
X_anomalous = X_anomalous.reshape(-1, INPUT_SIZE, WINDOW_SIZE)

print(f"\n      Clean windows:     {X_test.shape}")
print(f"      Anomalous windows: {X_anomalous.shape}")

# =============================================================================
# COMPUTE RECONSTRUCTION ERROR
# =============================================================================
def compute_errors(model, data):
    tensor = torch.tensor(data, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        reconstructions = model(tensor)
    errors = torch.mean((tensor - reconstructions) ** 2, dim=(1, 2))
    return errors.cpu().numpy()

errors_clean     = compute_errors(model, X_test)
errors_anomalous = compute_errors(model, X_anomalous)

print(f"\n      Clean errors:     mean={errors_clean.mean():.6f}, "
      f"std={errors_clean.std():.6f}, max={errors_clean.max():.6f}")
print(f"      Anomalous errors: mean={errors_anomalous.mean():.6f}, "
      f"std={errors_anomalous.std():.6f}, max={errors_anomalous.max():.6f}")

# =============================================================================
# THRESHOLD
# =============================================================================
k = 3.0
threshold = errors_clean.mean() + k * errors_clean.std()
print(f"\n      Threshold (mean + {k}*std): {threshold:.6f}")

pred_clean     = (errors_clean > threshold).astype(int)
pred_anomalous = (errors_anomalous > threshold).astype(int)

y_true  = np.concatenate([np.zeros(len(errors_clean)), np.ones(len(errors_anomalous))])
y_pred  = np.concatenate([pred_clean, pred_anomalous])
y_score = np.concatenate([errors_clean, errors_anomalous])

# =============================================================================
# METRICS
# =============================================================================
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy    = (tp + tn) / (tp + tn + fp + fn)
precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n      Confusion Matrix:")
print(f"      TN={tn} FP={fp}")
print(f"      FN={fn} TP={tp}")
print(f"\n      Accuracy:    {accuracy:.4f}")
print(f"      Precision:   {precision:.4f}")
print(f"      Recall:      {recall:.4f}")
print(f"      Specificity: {specificity:.4f}")
print(f"      F1 Score:    {f1:.4f}")

# ROC AUC
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
print(f"      ROC AUC:     {roc_auc:.4f}")

# =============================================================================
# PLOTS (simplified)
# =============================================================================
plt.figure(figsize=(10, 5))
plt.hist(errors_clean, bins=50, alpha=0.7, label='Clean', color='green')
plt.hist(errors_anomalous, bins=50, alpha=0.7, label='Anomalous', color='red')
plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.title('1D-CNN Reconstruction Error Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "error_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — 1D-CNN Anomaly Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
plt.close()

# Save metrics
with open(RESULTS_DIR / "evaluation_metrics.txt", 'w') as f:
    f.write(f"Architecture: 1D-CNN\n")
    f.write(f"Threshold:    {threshold:.6f}\n")
    f.write(f"Accuracy:     {accuracy:.4f}\n")
    f.write(f"Precision:    {precision:.4f}\n")
    f.write(f"Recall:       {recall:.4f}\n")
    f.write(f"F1 Score:     {f1:.4f}\n")
    f.write(f"ROC AUC:      {roc_auc:.4f}\n")

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE (1D-CNN)")
print("=" * 60)
print(f"  Threshold: {threshold:.6f}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC AUC:   {roc_auc:.4f}")
print("=" * 60)