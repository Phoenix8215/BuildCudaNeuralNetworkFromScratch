#include <iostream>
#include <time.h>

#include "neural_network.h"
#include "layers/linear_layer.h"
#include "layers/relu_activation.h"
#include "layers/sigmoid_activation.h"
#include "layers/softmax_activation.h"
#include "nn_utils/nn_exception.h"
#include "cost_functions/bce_cost.h"
#include "datasets/mnist_dataset.h"

using namespace std;

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

	srand( time(NULL) );

	size_t batch_size = 10;
    std::string filename = "/home/phoenix/workstation/learnCudaNeuralNetworkV1/mnist_dataset/mnist_train_100.csv";

	// CoordinatesDataset dataset(100, 21);
	MNISTDataset dataset(filename, batch_size);
	BCECost bce_cost("bce");

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(784, 128)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(128, 10)));
	// nn.addLayer(new SoftmaxActivation("softmax"));
	nn.addLayer(new SigmoidActivation("sigmoid"));
	nn.setCostFunction(new BCECost("bce"));
	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;
		// cout << "getNumOfBatches" << dataset.getNumOfBatches() << endl;
		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			// cout << "Y" << Y << endl;
			nn.backward(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(
			Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;
	
	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 0.99f : 0.01f;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
