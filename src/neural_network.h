#pragma once

#include <vector>
#include "layers/nn_layer.h"
#include "cost_functions/bce_cost.h"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	CostFunction *func;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backward(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

	void setCostFunction(CostFunction *func);

};
