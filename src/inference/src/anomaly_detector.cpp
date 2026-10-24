// =============================================================================
// anomaly_detector.cpp (Final Exact Match Version)
// =============================================================================

#include "anomaly_detector.h"
#include "model_config.h"
#include "model_weights_int8.h"
#include "quant_params.h"
#include "scaler_params.h"
#include <math.h>

static float input_buffer[WINDOW_SIZE];
static int buffer_index = 0;
static bool buffer_ready = false;

static float hidden_state[HIDDEN_SIZE];
static float cell_state[HIDDEN_SIZE];
static float gates[4 * HIDDEN_SIZE];

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float tanh_activation(float x) { return tanhf(x); }
static inline float get_weight(const int8_t *weights, int index, float scale)
{
    return (float)weights[index] * scale;
}

void init_ai_engine()
{
    for (int i = 0; i < WINDOW_SIZE; i++)
        input_buffer[i] = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_state[i] = 0.0f;
        cell_state[i] = 0.0f;
    }
    buffer_index = 0;
    buffer_ready = false;
}

void add_temperature_sample(float temp_celsius) {
    // 1. Normalize
    float normalized = (temp_celsius - SCALER_MEAN) / SCALER_STD;

    // 2. Add to buffer
    input_buffer[buffer_index] = normalized;
    
    // 3. Advance index and wrap around
    buffer_index++;
    if (buffer_index >= WINDOW_SIZE) {
        buffer_index = 0;
        buffer_ready = true;
    }

    // DEBUG: Print the raw input to prove the buffer is updating
    Serial.print("DEBUG: Buffer Index = ");
    Serial.print(buffer_index);
    Serial.print(" | Normalized Val = ");
    Serial.println(normalized);
}

bool is_buffer_full() { return buffer_ready; }

// =============================================================================
// LSTM FORWARD PASS (Corrected)
// =============================================================================
static void lstm_forward(
    float *input_seq, float *out_seq, int input_size, int seq_length,
    const int8_t *W_ih_l0, float W_ih_l0_s, const int8_t *W_hh_l0, float W_hh_l0_s,
    const int8_t *b_ih_l0, float b_ih_l0_s, const int8_t *b_hh_l0, float b_hh_l0_s,
    const int8_t *W_ih_l1, float W_ih_l1_s, const int8_t *W_hh_l1, float W_hh_l1_s,
    const int8_t *b_ih_l1, float b_ih_l1_s, const int8_t *b_hh_l1, float b_hh_l1_s)
{
    static float layer0_out[WINDOW_SIZE * HIDDEN_SIZE];

    // --- LAYER 0 ---
    // PyTorch ALWAYS starts layer 0 with zeros
    for (int h = 0; h < HIDDEN_SIZE; h++)
    {
        hidden_state[h] = 0.0f;
        cell_state[h] = 0.0f;
    }

    for (int t = 0; t < seq_length; t++)
    {
        for (int g = 0; g < 4 * HIDDEN_SIZE; g++)
        {
            float sum = get_weight(b_ih_l0, g, b_ih_l0_s) + get_weight(b_hh_l0, g, b_hh_l0_s);
            for (int i = 0; i < input_size; i++)
            {
                sum += input_seq[t * input_size + i] * get_weight(W_ih_l0, g * input_size + i, W_ih_l0_s);
            }
            for (int h = 0; h < HIDDEN_SIZE; h++)
            {
                sum += hidden_state[h] * get_weight(W_hh_l0, g * HIDDEN_SIZE + h, W_hh_l0_s);
            }
            gates[g] = sum;
        }

        for (int h = 0; h < HIDDEN_SIZE; h++)
        {
            float i_gate = sigmoid(gates[h]);
            float o_gate = sigmoid(gates[HIDDEN_SIZE + h]);
            float f_gate = sigmoid(gates[2 * HIDDEN_SIZE + h]);
            float g_gate = tanh_activation(gates[3 * HIDDEN_SIZE + h]);

            cell_state[h] = f_gate * cell_state[h] + i_gate * g_gate;
            hidden_state[h] = o_gate * tanh_activation(cell_state[h]);
            layer0_out[t * HIDDEN_SIZE + h] = hidden_state[h];
        }
    }

    // --- LAYER 1 ---
    // PyTorch ALWAYS starts layer 1 with zeros (it does NOT inherit layer 0's state)
    for (int h = 0; h < HIDDEN_SIZE; h++)
    {
        hidden_state[h] = 0.0f;
        cell_state[h] = 0.0f;
    }

    for (int t = 0; t < seq_length; t++)
    {
        for (int g = 0; g < 4 * HIDDEN_SIZE; g++)
        {
            float sum = get_weight(b_ih_l1, g, b_ih_l1_s) + get_weight(b_hh_l1, g, b_hh_l1_s);
            // Layer 1 takes input from Layer 0's output
            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                sum += layer0_out[t * HIDDEN_SIZE + i] * get_weight(W_ih_l1, g * HIDDEN_SIZE + i, W_ih_l1_s);
            }
            for (int h = 0; h < HIDDEN_SIZE; h++)
            {
                sum += hidden_state[h] * get_weight(W_hh_l1, g * HIDDEN_SIZE + h, W_hh_l1_s);
            }
            gates[g] = sum;
        }

        for (int h = 0; h < HIDDEN_SIZE; h++)
        {
            float i_gate = sigmoid(gates[h]);
            float o_gate = sigmoid(gates[HIDDEN_SIZE + h]);
            float f_gate = sigmoid(gates[2 * HIDDEN_SIZE + h]);
            float g_gate = tanh_activation(gates[3 * HIDDEN_SIZE + h]);

            cell_state[h] = f_gate * cell_state[h] + i_gate * g_gate;
            hidden_state[h] = o_gate * tanh_activation(cell_state[h]);
            out_seq[t * HIDDEN_SIZE + h] = hidden_state[h];
        }
    }
}

static void linear_forward(float *input, int input_size, float *output, int output_size, const int8_t *W, float W_s, const int8_t *b, float b_s)
{
    for (int o = 0; o < output_size; o++)
    {
        float sum = get_weight(b, o, b_s);
        for (int i = 0; i < input_size; i++)
            sum += input[i] * get_weight(W, o * input_size + i, W_s);
        output[o] = sum;
    }
}

// =============================================================================
// FULL AUTOENCODER INFERENCE (Final)
// =============================================================================
float run_inference()
{
    if (!buffer_ready)
        return 0.0f;
	    // HARD RESET STATES to prevent drift
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_state[i] = 0.0f;
        cell_state[i] = 0.0f;
    }

    static float encoder_out[WINDOW_SIZE * HIDDEN_SIZE];
    static float latent[LATENT_SIZE];
    static float decoder_in[WINDOW_SIZE * HIDDEN_SIZE];
    static float decoder_out[WINDOW_SIZE * HIDDEN_SIZE];
    static float reconstruction[WINDOW_SIZE * INPUT_SIZE];

    // 1. ENCODER LSTM
    lstm_forward(
        input_buffer, encoder_out, INPUT_SIZE, WINDOW_SIZE,
        enc_lstm_W_ih_l0, enc_lstm_W_ih_l0_scale, enc_lstm_W_hh_l0, enc_lstm_W_hh_l0_scale,
        enc_lstm_b_ih_l0, enc_lstm_b_ih_l0_scale, enc_lstm_b_hh_l0, enc_lstm_b_hh_l0_scale,
        enc_lstm_W_ih_l1, enc_lstm_W_ih_l1_scale, enc_lstm_W_hh_l1, enc_lstm_W_hh_l1_scale,
        enc_lstm_b_ih_l1, enc_lstm_b_ih_l1_scale, enc_lstm_b_hh_l1, enc_lstm_b_hh_l1_scale);

    // 2. BOTTLENECK
    float *last_hidden = &encoder_out[(WINDOW_SIZE - 1) * HIDDEN_SIZE];
    linear_forward(last_hidden, HIDDEN_SIZE, latent, LATENT_SIZE,
                   enc_fc_W, enc_fc_W_scale, enc_fc_b, enc_fc_b_scale);

    // 3. DECODER LINEAR
    float decoder_hidden[HIDDEN_SIZE];
    linear_forward(latent, LATENT_SIZE, decoder_hidden, HIDDEN_SIZE,
                   dec_fc_W, dec_fc_W_scale, dec_fc_b, dec_fc_b_scale);

    // 4. PREPARE DECODER INPUT
    for (int t = 0; t < WINDOW_SIZE; t++)
    {
        for (int h = 0; h < HIDDEN_SIZE; h++)
        {
            decoder_in[t * HIDDEN_SIZE + h] = decoder_hidden[h];
        }
    }

    // 5. DECODER LSTM
    lstm_forward(
        decoder_in, decoder_out, HIDDEN_SIZE, WINDOW_SIZE,
        dec_lstm_W_ih_l0, dec_lstm_W_ih_l0_scale, dec_lstm_W_hh_l0, dec_lstm_W_hh_l0_scale,
        dec_lstm_b_ih_l0, dec_lstm_b_ih_l0_scale, dec_lstm_b_hh_l0, dec_lstm_b_hh_l0_scale,
        dec_lstm_W_ih_l1, dec_lstm_W_ih_l1_scale, dec_lstm_W_hh_l1, dec_lstm_W_hh_l1_scale,
        dec_lstm_b_ih_l1, dec_lstm_b_ih_l1_scale, dec_lstm_b_hh_l1, dec_lstm_b_hh_l1_scale);

    // 6. OUTPUT LINEAR
    for (int t = 0; t < WINDOW_SIZE; t++)
    {
        linear_forward(&decoder_out[t * HIDDEN_SIZE], HIDDEN_SIZE,
                       &reconstruction[t], INPUT_SIZE,
                       out_fc_W, out_fc_W_scale, out_fc_b, out_fc_b_scale);
    }

    // 7. COMPUTE MSE
    float mse = 0.0f;
    for (int t = 0; t < WINDOW_SIZE; t++)
    {
        float diff = input_buffer[t] - reconstruction[t];
        mse += diff * diff;
    }
    return mse / WINDOW_SIZE;
}