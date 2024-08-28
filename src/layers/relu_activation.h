#pragma once

#include "nn_layer.h"

class ReLUActivation : public NNLayer {
private:
	Matrix A; // 输出

	Matrix Z; // 输入
	Matrix dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backward(Matrix& dA, float learning_rate = 0.01);
};
