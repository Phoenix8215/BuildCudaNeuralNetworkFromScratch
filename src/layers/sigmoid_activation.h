#pragma once

#include "nn_layer.h"

class SigmoidActivation : public NNLayer {
private:
	Matrix A; // 输出

	Matrix Z; // 输入
	Matrix dZ;

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	Matrix& forward(Matrix& Z);
	// dA = dCost/dsigma
	Matrix& backward(Matrix& dA, float learning_rate = 0.01);
};
