// =============================================================================
// main.cpp  (Phase 5.3.3 — AI Inference)
// =============================================================================

#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "anomaly_detector.h"
#include "model_config.h"

// --- Pin definitions ---
#define ONE_WIRE_BUS  4
#define LED_GREEN     2
#define LED_RED       3

// --- OneWire setup ---
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// --- Timing ---
unsigned long lastRead = 0;
const unsigned long READ_INTERVAL_MS = 1000;

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n========================================");
    Serial.println("Edge AI Anomaly Detection — ESP32-C3");
    Serial.println("========================================");

    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, LOW);

    sensors.begin();
    int deviceCount = sensors.getDeviceCount();
    Serial.printf("Found %d DS18B20 device(s)\n", deviceCount);

    if (deviceCount == 0) {
        Serial.println("ERROR: No DS18B20 detected!");
        while (1) {
            digitalWrite(LED_RED, !digitalRead(LED_RED));
            delay(200);
        }
    }

    sensors.setResolution(12);
    
    // Initialize the AI engine
    init_ai_engine();
    Serial.println("AI Engine Initialized.");
    Serial.printf("Waiting for %d samples to fill buffer...\n", WINDOW_SIZE);
}

void loop() {
    unsigned long now = millis();

    if (now - lastRead >= READ_INTERVAL_MS) {
        lastRead = now;

        sensors.requestTemperatures();
        float tempC = sensors.getTempCByIndex(0);

        if (tempC == DEVICE_DISCONNECTED_C) {
            Serial.println("ERROR: Sensor disconnected");
            return;
        }

        // Feed the sensor data into the AI engine
        add_temperature_sample(tempC);

        // If the buffer is full, run inference
        if (is_buffer_full()) {
            float error = run_inference();

            Serial.printf("Temp: %.2f °C | Error: %.6f | Thr: %.6f | ", tempC, error, ANOMALY_THRESHOLD);

            if (error > ANOMALY_THRESHOLD) {
                Serial.println("⚠️ ANOMALY DETECTED ⚠️");
                digitalWrite(LED_RED, HIGH);
                digitalWrite(LED_GREEN, LOW);
            } else {
                Serial.println("NORMAL");
                digitalWrite(LED_GREEN, HIGH);
                digitalWrite(LED_RED, LOW);
            }
        } else {
            Serial.printf("Temp: %.2f °C | Filling buffer...\n", tempC);
        }
    }
}