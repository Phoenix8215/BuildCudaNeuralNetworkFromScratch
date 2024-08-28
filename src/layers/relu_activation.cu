#include "relu_activation.h"
#include "../nn_utils/nn_exception.h"

__global__ void reluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) A[index] = fmaxf(Z[index], 0.0f);
}

__global__ void reluActivationBackward(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// 使用条件掩码，避免产生warp divergence
	if(index < Z_x_dim * Z_y_dim) dZ[index] = dA[index] * (Z[index] > 0);
}

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	reluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward propagation.");

	return A;
}

Matrix& ReLUActivation::backward(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	reluActivationBackward<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
													      dZ.data_device.get(),
														  Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU back propagation");

	return dZ;
}
