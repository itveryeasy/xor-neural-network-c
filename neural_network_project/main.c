#include <stdio.h>
#include "neural_network.h"

int main() {
    // Define input and output data
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double outputs[4] = {0, 1, 1, 0}; // XOR

    // Create and initialize the neural network
    NeuralNetwork nn;
    initialize_network(&nn, 2, 1, 0.1);

    // Train the network
    train(&nn, inputs, outputs, 4, 10000);

    // Test the network
    for (int i = 0; i < 4; i++) {
        double prediction = predict(&nn, inputs[i]);
        printf("Input: [%%0.1f, %%0.1f] Prediction: %%0.1f\n",
               inputs[i][0], inputs[i][1], prediction > 0.5 ? 1.0 : 0.0);
    }

    return 0;
}
