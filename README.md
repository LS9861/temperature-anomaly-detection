<img width="832" height="522" alt="LogAndHWSetup" src="https://github.com/user-attachments/assets/f8a38678-25d5-4884-b356-4e759cc18249" />
# Temperature Anomaly Detection (Edge AI on ESP32-C3)

End-to-end TinyML pipeline: train a 1D Convolutional Autoencoder on real temperature data, quantize to INT8, and deploy to an ESP32-C3 for real-time anomaly detection.

![CI Pipeline](https://github.com/LS9861/temperature-anomaly-detection/actions/workflows/ci.yml/badge.svg)

![Project Screenshot](https://github.com/user-attachments/assets/0839da14-a219-4770-a17c-e17f5d8d11b2)

## Pipeline

1. **Data Collection** — 24 hours of real temperature data from a DS18B20 sensor
2. **Preprocessing** — Normalization + sliding windows (50 samples) + train/test split
3. **Training** — 1D Convolutional Autoencoder (PyTorch, 761 parameters)
4. **Evaluation** — Reconstruction error thresholding (Threshold = 0.131)
5. **Quantization** — INT8 TFLite via TinyNeuralNetwork
6. **Deployment** — TensorFlow Lite Micro (TFLM) on ESP32-C3

## CI/CD Pipeline

The project includes a full CI/CD pipeline using GitHub Actions:

- **CI:** Every push builds the Python pipeline and compiles the ESP32 firmware.
- **CD:** Every version tag (e.g., `v3.0`) automatically creates a GitHub Release with the ready-to-flash `merged_firmware.bin`.

## Results

- **Precision:** 100% (zero false alarms)
- **Recall:** 64% (catches major anomalies)
- **ROC AUC:** 0.9578
- **Model Size:** 21.4 KB (INT8)
- **RAM Usage:** ~32 KB tensor arena

## Model Evolution

The initial prototype used an LSTM Autoencoder. Due to TFLM op compatibility constraints (unsupported `TILE` and `GATHER` ops), the model was re-architected as a 1D Convolutional Autoencoder. This is the standard architecture for Edge AI time-series anomaly detection.

## Project Structure
temperature-anomaly-detection/
├── data/ # Raw and processed data (gitignored)
├── models/ # Trained models (gitignored)
├── results/ # Plots and metrics (gitignored)
├── scripts/ # Data generation utilities
├── src/
│ ├── training/ # Python training scripts
│ ├── deployment/ # ONNX export, TFLite quantization
│ └── inference/ # ESP32-C3 firmware
└── docs/ # Documentation


## Setup

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python src/training/preprocess.py
python src/training/train_autoencoder.py
python src/training/evaluate_model.py

For TFLite quantization, run the Colab script provided in src/deployment/.

Hardware
Component	Connection
ESP32-C3 SuperMini	Main microcontroller
DS18B20 Temperature Sensor	GPIO4 (Data), 3.3V (VCC), GND
4.7kΩ Resistor	Pull-up between VCC and Data
Green LED	GPIO2 (Normal)
Red LED	GPIO3 (Anomaly)

ESP32-C3 SuperMini          DS18B20
──────────────────          ───────
3.3V ──┬─────────────────── RED (VCC)
       │
       │  4.7kΩ
       │
GPIO4 ─┴─────────────────── YELLOW (DATA)
GND ─────────────────────── BLACK (GND)

Tech Stack
Layer	Technology
Training	PyTorch, scikit-learn, NumPy
Quantization	TinyNeuralNetwork, TensorFlow
Firmware	C++, Arduino, TensorFlow Lite Micro
CI/CD	GitHub Actions, PlatformIO
Hardware	ESP32-C3, DS18B20
Status

    ☑ Data collection (24h real data)
    ☑ 1D-CNN training
    ☑ INT8 quantization
    ☑ TFLM deployment on ESP32-C3
    ☑ Real-time anomaly detection with LED alerts
    ☑ CI/CD pipeline

Author

Lokendra Samba
