#include "bce_cost.h"
#include "../nn_utils/nn_exception.h"

#include <math.h>
#include <iostream>
#include <assert.h>

__global__ void binaryCrossEntropyCost(float* predictions, float* target,
									   int size, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * logf(predictions[index])
				+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
		atomicAdd(cost, - partial_cost / size);
	}
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY,
								     	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// dY[index] 存储了损失函数对于第 index 个预测值的梯度。
	if (index < size) dY[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
}

BCECost::BCECost(std::string name) {
	this->name = name;
}

BCECost::~BCECost() {}

float BCECost::cost(Matrix predictions, Matrix target) {
	assert(predictions.shape.x == target.shape.x);

	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	// std::cout << predictions.shape.x << std::endl;
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														  target.data_device.get(),
														  predictions.shape.x, cost);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");

	float cost_value = *cost;
	// std::cout << cost_value << std::endl;
	cudaFree(cost);

	return cost_value;
}

Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														   target.data_device.get(),
														   dY.data_device.get(),
														   predictions.shape.x);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");

	return dY;
}
