import os

# Create the directory for the C project
project_dir = 'neural_network_project'
if not os.path.exists(project_dir):
    os.makedirs(project_dir)

# Content for the C files

files = {
    'main.c': '''#include <stdio.h>
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
        printf("Input: [%%0.1f, %%0.1f] Prediction: %%0.1f\\n",
               inputs[i][0], inputs[i][1], prediction > 0.5 ? 1.0 : 0.0);
    }

    return 0;
}
''',

    'neural_network.h': '''#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct {
    double weights[2]; // Weights for 2 inputs
    double bias;       // Bias
    double learning_rate; // Learning rate
} NeuralNetwork;

void initialize_network(NeuralNetwork *nn, int input_size, int output_size, double learning_rate);
void train(NeuralNetwork *nn, double inputs[][2], double outputs[], int num_samples, int epochs);
double predict(NeuralNetwork *nn, double input[2]);

#endif
''',

    'neural_network.c': '''#include <stdlib.h>
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
''',

    'utils.h': '''#ifndef UTILS_H
#define UTILS_H

double sigmoid(double x);
double sigmoid_derivative(double x);

#endif
''',

    'utils.c': '''#include <math.h>
#include "utils.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}
''',

    'Makefile': '''CC = gcc
CFLAGS = -Wall -Wextra -std=c99

OBJ = main.o neural_network.o utils.o

all: neural_network

neural_network: $(OBJ)
	$(CC) $(CFLAGS) -o neural_network $(OBJ)

main.o: main.c neural_network.h
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c neural_network.h utils.h
	$(CC) $(CFLAGS) -c neural_network.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o neural_network
'''
}

# Create and write the files
for filename, content in files.items():
    file_path = os.path.join(project_dir, filename)
    with open(file_path, 'w') as file:
        file.write(content)

print(f"Project files have been generated in the '{project_dir}' directory.")
