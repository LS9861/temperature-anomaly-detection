# Temperature Anomaly Detection (Edge AI on ESP32)

End-to-end TinyML pipeline: train an LSTM Autoencoder on temperature data,
then deploy to ESP32 for real-time anomaly detection.

## Pipeline

1. **Data Generation** — Synthetic temperature data with injected anomalies
2. **Preprocessing** — Normalization + sliding windows + train/test split
3. **Training** — LSTM Autoencoder (PyTorch)
4. **Evaluation** — Reconstruction error thresholding
5. **Deployment** — ONNX → C header → ESP32 firmware

## Results

- Accuracy: 91.85%
- Precision: 100% (zero false alarms)
- Recall: 88.25%
- ROC AUC: 0.9966

## Project Structure
temperature-anomaly-detection/
├── data/ # Raw and processed data (gitignored)
├── models/ # Trained models (gitignored)
├── results/ # Plots and metrics (gitignored)
├── scripts/ # Data generation utilities
├── src/ # Source code
│ ├── training/ # Python training scripts
│ └── inference/ # ESP32 firmware
└── docs/ # Documentation

Setup
// bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python src/training/preprocess.py
python src/training/train_autoencoder.py
python src/training/evaluate_model.py

Hardware

    ESP32 (or similar)

    DS18B20 temperature sensor
	
	Author

Lokendra Samba
