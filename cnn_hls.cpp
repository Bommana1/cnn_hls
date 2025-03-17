#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// Define parameters
#define IN_SIZE 32  // Input size (e.g., 32x32 image)
#define IN_CHANNELS 3 // RGB
#define OUT_CHANNELS 16 // First Conv Layer Filters
#define KERNEL_SIZE 3
#define FC_IN 256
#define FC_OUT 10

// Define data types
typedef ap_fixed<16, 4> data_t;
typedef ap_uint<8> pixel_t;

// BRAM Storage for weights
data_t conv1_weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
data_t conv1_bias[OUT_CHANNELS];
data_t fc_weights[FC_OUT][FC_IN];
data_t fc_bias[FC_OUT];

// AXI Interface for loading weights
extern "C" {
void load_weights(data_t* weight_mem) {
#pragma HLS INTERFACE m_axi port=weight_mem depth=10000 offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int idx = 0;
    // Load Conv1 weights
    for (int oc = 0; oc < OUT_CHANNELS; oc++)
        for (int ic = 0; ic < IN_CHANNELS; ic++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    conv1_weights[oc][ic][i][j] = weight_mem[idx++];

    // Load Conv1 Bias
    for (int oc = 0; oc < OUT_CHANNELS; oc++)
        conv1_bias[oc] = weight_mem[idx++];

    // Load Fully Connected Layer Weights
    for (int i = 0; i < FC_OUT; i++)
        for (int j = 0; j < FC_IN; j++)
            fc_weights[i][j] = weight_mem[idx++];

    // Load FC Bias
    for (int i = 0; i < FC_OUT; i++)
        fc_bias[i] = weight_mem[idx++];
}
}

// Convolution Layer
void conv2d(data_t input[IN_SIZE][IN_SIZE][IN_CHANNELS],
            data_t output[IN_SIZE][IN_SIZE][OUT_CHANNELS]) {
#pragma HLS PIPELINE
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
        for (int i = 0; i < IN_SIZE - KERNEL_SIZE + 1; i++) {
            for (int j = 0; j < IN_SIZE - KERNEL_SIZE + 1; j++) {
                data_t sum = conv1_bias[oc];
                for (int ic = 0; ic < IN_CHANNELS; ic++) {
                    for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                            sum += input[i + ki][j + kj][ic] * conv1_weights[oc][ic][ki][kj];
                        }
                    }
                }
                output[i][j][oc] = sum > 0 ? sum : 0; // ReLU Activation
            }
        }
    }
}

// Fully Connected Layer
void fully_connected(data_t input[FC_IN], data_t output[FC_OUT]) {
#pragma HLS PIPELINE
    for (int i = 0; i < FC_OUT; i++) {
        data_t sum = fc_bias[i];
        for (int j = 0; j < FC_IN; j++) {
            sum += input[j] * fc_weights[i][j];
        }
        output[i] = sum > 0 ? sum : 0; // ReLU
    }
}

// Top-Level Function with AXI Interface
extern "C" {
void cnn_inference(data_t* input_data, data_t* output_data) {
#pragma HLS INTERFACE m_axi port=input_data depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output_data depth=10 offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t input[IN_SIZE][IN_SIZE][IN_CHANNELS];
    data_t conv_output[IN_SIZE][IN_SIZE][OUT_CHANNELS];
    data_t fc_input[FC_IN];
    data_t fc_output[FC_OUT];

    // Load input data
    int idx = 0;
    for (int i = 0; i < IN_SIZE; i++)
        for (int j = 0; j < IN_SIZE; j++)
            for (int c = 0; c < IN_CHANNELS; c++)
                input[i][j][c] = input_data[idx++];

    // Run convolution
    conv2d(input, conv_output);

    // Flatten for FC input
    idx = 0;
    for (int i = 0; i < IN_SIZE; i++)
        for (int j = 0; j < IN_SIZE; j++)
            for (int c = 0; c < OUT_CHANNELS; c++)
                if (idx < FC_IN) fc_input[idx++] = conv_output[i][j][c];

    // Run Fully Connected Layer
    fully_connected(fc_input, fc_output);

    // Output result
    for (int i = 0; i < FC_OUT; i++)
        output_data[i] = fc_output[i];
}
}
