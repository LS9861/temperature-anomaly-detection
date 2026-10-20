// =============================================================================
// main.cpp  (Phase 5.3.1 — Sensor Test)
// Purpose: Verify DS18B20 works on ESP32-C3 SuperMini
// =============================================================================

#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// --- Pin definitions ---
#define ONE_WIRE_BUS  4    // GPIO4 → DS18B20 data pin
#define LED_GREEN     2    // Optional green LED
#define LED_RED       3    // Optional red LED

// --- OneWire + DallasTemperature setup ---
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// --- Timing ---
unsigned long lastRead = 0;
const unsigned long READ_INTERVAL_MS = 1000;   // Read every 1 second

void setup() {
    Serial.begin(115200);
    delay(1000);   // Wait for serial to be ready

    Serial.println();
    Serial.println("========================================");
    Serial.println("DS18B20 Sensor Test — ESP32-C3 SuperMini");
    Serial.println("========================================");

    // Initialize LEDs
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, LOW);

    // Initialize DS18B20
    sensors.begin();
    int deviceCount = sensors.getDeviceCount();
    Serial.printf("Found %d DS18B20 device(s)\n", deviceCount);

    if (deviceCount == 0) {
        Serial.println("ERROR: No DS18B20 detected!");
        Serial.println("Check wiring: VCC=3.3V, GND=GND, DATA=GPIO4");
        Serial.println("Check 4.7k pull-up between VCC and DATA");
        while (1) {
            digitalWrite(LED_RED, !digitalRead(LED_RED));
            delay(200);
        }
    }

    // Set resolution (12-bit = 0.0625°C precision, ~750ms conversion)
    sensors.setResolution(12);
    Serial.println("Sensor ready. Reading every 1 second...\n");
}

void loop() {
    unsigned long now = millis();

    if (now - lastRead >= READ_INTERVAL_MS) {
        lastRead = now;

        // Request temperature (blocking, ~750ms at 12-bit)
        sensors.requestTemperatures();

        // Read in Celsius
        float tempC = sensors.getTempCByIndex(0);

        // Check for read error
        if (tempC == DEVICE_DISCONNECTED_C) {
            Serial.println("ERROR: Sensor disconnected");
            digitalWrite(LED_RED, HIGH);
            digitalWrite(LED_GREEN, LOW);
        } else {
            Serial.printf("Temperature: %.2f °C\n", tempC);
            digitalWrite(LED_GREEN, HIGH);
            digitalWrite(LED_RED, LOW);
        }
    }
}