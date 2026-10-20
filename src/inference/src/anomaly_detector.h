// =============================================================================
// anomaly_detector.h
// Purpose: Header file for the C++ LSTM Autoencoder inference engine
// =============================================================================

#ifndef ANOMALY_DETECTOR_H
#define ANOMALY_DETECTOR_H

#include <Arduino.h>

// Initialize the AI engine (sets up buffers)
void init_ai_engine();

// Add a new temperature reading to the 50-sample rolling buffer
void add_temperature_sample(float temp_celsius);

// Returns true if the buffer is full (ready for inference)
bool is_buffer_full();

// Run the AI model on the current 50 samples. Returns the reconstruction error.
float run_inference();

#endif // ANOMALY_DETECTOR_H