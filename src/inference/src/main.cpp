#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- Pin Definitions ---
#define ONE_WIRE_BUS 4
#define LED_GREEN 2
#define LED_RED 3

// --- Model Configuration ---
#define WINDOW_SIZE 50
#define INPUT_SIZE 1

// --- CRITICAL: Values from your 1D-CNN evaluation ---
const float SCALER_MEAN = 19.753339f;
const float SCALER_STD = 0.985661f;

// --- HYSTERESIS THRESHOLDS ---
// Alarm turns ON when error > 3.5
// Alarm turns OFF when error < 3.0
const float THRESHOLD_ON = 3.500000f;
const float THRESHOLD_OFF = 3.000000f;

// --- Sensor Setup ---
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// --- TFLM Globals ---
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Arena size (CNN is tiny, 32KB is plenty)
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Rolling buffer
float input_buffer[WINDOW_SIZE];
int buffer_index = 0;
bool buffer_ready = false;

// Hysteresis state
bool alarm_active = false;

void setup()
{
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n=== 1D-CNN Anomaly Detection (Hysteresis) ===");

    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, LOW);

    sensors.begin();
    sensors.setResolution(12);

    // --- 1. Load Model ---
    Serial.println("Loading model...");
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("ERROR: Model version mismatch!");
        while (1)
            ;
    }

    // --- 2. Register Ops (CNN specific) ---
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddRelu();
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddTranspose();
    resolver.AddTransposeConv();

    // --- 3. Create Interpreter ---
    Serial.println("Creating interpreter...");
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // --- 4. Allocate Tensors ---
    Serial.println("Allocating tensors...");
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("ERROR: AllocateTensors failed!");
        while (1)
            ;
    }
    Serial.println("Tensors allocated OK.");

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.printf("Input: %d bytes, scale: %f, zero_point: %d\n",
                  input->bytes, input->params.scale, input->params.zero_point);
    Serial.printf("Output: %d bytes, scale: %f, zero_point: %d\n",
                  output->bytes, output->params.scale, output->params.zero_point);
    Serial.println("Waiting for buffer to fill...");
}

void loop()
{
    static unsigned long lastRead = 0;
    if (millis() - lastRead < 1000)
        return;
    lastRead = millis();

    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);
    if (tempC == DEVICE_DISCONNECTED_C)
        return;

    // Normalize
    float normalized = (tempC - SCALER_MEAN) / SCALER_STD;

    // Add to buffer
    input_buffer[buffer_index] = normalized;
    buffer_index++;
    if (buffer_index >= WINDOW_SIZE)
    {
        buffer_index = 0;
        buffer_ready = true;
    }

    if (!buffer_ready)
    {
        Serial.printf("RAW: %.2f C | Filling buffer (%d/%d)\n", tempC, buffer_index, WINDOW_SIZE);
        return;
    }

    // --- Quantize input ---
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int32_t q = (int32_t)round(input_buffer[i] / input->params.scale) + input->params.zero_point;
        if (q > 127)
            q = 127;
        if (q < -128)
            q = -128;
        input->data.int8[i] = (int8_t)q;
    }

    // --- Run inference ---
    if (interpreter->Invoke() != kTfLiteOk)
    {
        Serial.println("ERROR: Invoke failed!");
        return;
    }

    // --- Dequantize output & compute MSE ---
    float mse = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        float deq = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        float diff = input_buffer[i] - deq;
        mse += diff * diff;
    }
    mse /= WINDOW_SIZE;

    // --- HYSTERESIS LOGIC ---
    // Turn ON if above high threshold
    if (mse > THRESHOLD_ON)
    {
        alarm_active = true;
    }
    // Turn OFF only if below low threshold
    else if (mse < THRESHOLD_OFF)
    {
        alarm_active = false;
    }
    // Between 3.0 and 3.5: keep the previous state (no chatter)

    // --- Output and LED ---
    Serial.printf("RAW: %.2f C | Error: %.4f | ON: %.1f | OFF: %.1f | ",
                  tempC, mse, THRESHOLD_ON, THRESHOLD_OFF);

    if (alarm_active)
    {
        Serial.println("ANOMALY");
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_GREEN, LOW);
    }
    else
    {
        Serial.println("NORMAL");
        digitalWrite(LED_GREEN, HIGH);
        digitalWrite(LED_RED, LOW);
    }
}