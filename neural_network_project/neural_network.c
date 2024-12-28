#include <stdlib.h>
#include "neural_network.h"
#include "utils.h"

void initialize_network(NeuralNetwork *nn, int input_size, int output_size, double learning_rate) {
    for (int i = 0; i < input_size; i++) {
        nn->weights[i] = (double)rand() / RAND_MAX;
    }
    nn->bias = 0.5; // Initialize bias
    nn->learning_rate = learning_rate;
}

void train(NeuralNetwork *nn, double inputs[][2], double outputs[], int num_samples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            double input1 = inputs[i][0];
            double input2 = inputs[i][1];
            double target = outputs[i];

            // Forward pass
            double z = input1 * nn->weights[0] + input2 * nn->weights[1] + nn->bias;
            double prediction = sigmoid(z);

            // Error calculation
            double error = target - prediction;

            // Backpropagation
            double adjustment = error * sigmoid_derivative(prediction);
            nn->weights[0] += adjustment * input1 * nn->learning_rate;
            nn->weights[1] += adjustment * input2 * nn->learning_rate;
            nn->bias += adjustment * nn->learning_rate;
        }
    }
}

double predict(NeuralNetwork *nn, double input[2]) {
    double z = input[0] * nn->weights[0] + input[1] * nn->weights[1] + nn->bias;
    return sigmoid(z);
}
