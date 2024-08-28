#include "neural_network.h"
#include "nn_utils/nn_exception.h"

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backward(Matrix predictions, Matrix target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix error = func->dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backward(error, learning_rate);
	}

	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}

void NeuralNetwork::setCostFunction(CostFunction * func) {
	this->func = func;
}
