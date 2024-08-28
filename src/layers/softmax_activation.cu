#include "softmax_activation.h"
#include "../nn_utils/nn_exception.h"
#include <iostream>
#include <float.h>  


__global__ void softmaxActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    extern __shared__ float shared_data[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        shared_data[tid] = Z[index];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
            }
            __syncthreads();
        }

        float max_val = shared_data[0];  
        __syncthreads();

        shared_data[tid] = expf(Z[index] - max_val);
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }

        float sum = shared_data[0];  
        __syncthreads();

        A[index] = expf(Z[index] - max_val) / sum;
    }
}


__global__ void softmaxActivationBackward(float* A, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		float softmax_val = A[index];

		dZ[index] = softmax_val * (1.0f - softmax_val) * dA[index];
	}
}


SoftmaxActivation::SoftmaxActivation(std::string name) {
	this->name = name;
}

SoftmaxActivation::~SoftmaxActivation()
{ }

Matrix& SoftmaxActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y + block_size.x - 1) / block_size.x);

	softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

	return A;
}

Matrix& SoftmaxActivation::backward(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y + block_size.x - 1) / block_size.x);
	softmaxActivationBackward<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

	return dZ;
}
