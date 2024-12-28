#ifndef NEURAL_NETWORK_H
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
