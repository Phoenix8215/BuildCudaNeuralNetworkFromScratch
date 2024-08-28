#pragma once

#include "nn_layer.h"

class SoftmaxActivation : public NNLayer {
private:
	Matrix A; // 输出
	Matrix Z; // 输入
	Matrix dZ;

public:
	SoftmaxActivation(std::string name);
	~SoftmaxActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backward(Matrix& dA, float learning_rate = 0.01);
};
