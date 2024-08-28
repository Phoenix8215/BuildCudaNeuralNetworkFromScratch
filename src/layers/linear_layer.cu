#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "linear_layer.h"
#include "../nn_utils/nn_exception.h"



__global__ void linearLayerForward(float* W, float* A, float* Z, float* b, int W_x_dim, 
                                    int W_y_dim, int A_x_dim, int A_y_dim) {

    constexpr int BM = 32;
    constexpr int BN = 32;
    __shared__ float s_W[BM][BM];
    __shared__ float s_A[BM][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float Z_value = 0.0f;

    for (int bk = 0; bk < (W_x_dim + BM - 1) / BM; ++bk) {
        if (row < W_y_dim && (bk * BM + tx) < W_x_dim) {
            s_W[ty][tx] = W[row * W_x_dim + bk * BM + tx];
        } else {
            s_W[ty][tx] = 0.0f;
        }

        if ((bk * BM + ty) < A_y_dim && col < A_x_dim) {
            s_A[ty][tx] = A[(bk * BM + ty) * A_x_dim + col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BM; ++k) {
            Z_value += s_W[ty][k] * s_A[k][tx];
        }

        __syncthreads();
    }

    if (row < W_y_dim && col < A_x_dim) {
        Z[row * A_x_dim + col] = Z_value + b[row];
    }
}



__global__ void linearLayerBackward(float* W, float* dZ, float* dA,
                                    int W_x_dim, int W_y_dim,
                                    int dZ_x_dim, int dZ_y_dim) {

    constexpr int BM = 32;
    constexpr int BN = 32;
    __shared__ float s_W[BM][BM];
    __shared__ float s_dZ[BM][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float dA_value = 0.0f;

    for (int bk = 0; bk < (W_y_dim + BM - 1) / BM; ++bk) {

        if ((bk * BM + tx) < W_y_dim && row < W_x_dim) {
            s_W[tx][ty] = W[(bk * BM + tx) * W_x_dim + row];
        } else {
            s_W[tx][ty] = 0.0f;
        }

        if ((bk * BM + ty) < dZ_y_dim && col < dZ_x_dim) {
            s_dZ[ty][tx] = dZ[(bk * BM + ty) * dZ_x_dim + col];
        } else {
            s_dZ[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BM; ++k) {
            dA_value += s_W[k][ty] * s_dZ[k][tx];
        }

        __syncthreads();
    }

    if (row < W_x_dim && col < dZ_x_dim) {
        dA[row * dZ_x_dim + col] = dA_value;
    }
}


__global__ void linearLayerUpdateWeights(float* dZ, float* A, float* W,
                                         int dZ_x_dim, int dZ_y_dim,
                                         int A_x_dim, int A_y_dim,
                                         float learning_rate) {
    
    constexpr int BM = 32;
    constexpr int BN = 32;
    __shared__ float s_dZ[BM][BM];
    __shared__ float s_A[BM][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float dW_value = 0.0f;

    for (int bk = 0; bk < (dZ_x_dim + BM - 1) / BM; ++bk) {

        if (row < dZ_y_dim && (bk * BM + tx) < dZ_x_dim) {
            s_dZ[ty][tx] = dZ[row * dZ_x_dim + bk * BM + tx];
        } else {
            s_dZ[ty][tx] = 0.0f;
        }

        if (col < A_y_dim && (bk * BM + ty) < A_x_dim) {
            s_A[ty][tx] = A[col * A_x_dim + bk * BM + ty];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BM; ++k) {
            dW_value += s_dZ[ty][k] * s_A[k][tx];
        }

        __syncthreads();
    }

    if (row < dZ_y_dim && col < A_y_dim) {
        W[row * A_y_dim + col] -= learning_rate * (dW_value / A_x_dim);
    }
}


__global__ void linearLayerUpdateBias(  float* dZ, float* b,
										int dZ_x_dim, int dZ_y_dim,
										int b_x_dim,
										float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
	}
}

LinearLayer::LinearLayer(std::string name, Shape W_shape) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

LinearLayer::~LinearLayer() { }




__global__ void initializeWeightsKernel(float* W, int width, int height, float weights_init_threshold, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_size = width * height;

    curandState state;
    curand_init(seed, idx, 0, &state);

    for (int i = idx; i < total_size; i += stride) {
        float rand_value = curand_normal(&state) * weights_init_threshold;
        W[i] = rand_value;
    }
}

void LinearLayer::initializeWeightsRandomly() {

    int blockSize = 256;
    int numBlocks = (W.shape.x * W.shape.y + blockSize - 1) / blockSize;

    initializeWeightsKernel<<<numBlocks, blockSize>>>(W.data_device.get(), W.shape.x, W.shape.y, weights_init_threshold, time(NULL));

    cudaDeviceSynchronize();
}



__global__ void initializeBiasWithZerosKernel(float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) b[idx] = 0.0f;
}

void LinearLayer::initializeBiasWithZeros() {
    int blockSize = 256;
    int numBlocks = (b.shape.x + blockSize - 1) / blockSize;

    initializeBiasWithZerosKernel<<<numBlocks, blockSize>>>(b.data_device.get(), b.shape.x);

    cudaDeviceSynchronize();
}

Matrix& LinearLayer::forward(Matrix& A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput(A);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");

	return Z;
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
	dim3 block_size(32, 32);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y);
	linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device.get(),
													   A.data_device.get(),
													   Z.data_device.get(),
													   b.data_device.get(),
													   W.shape.x, W.shape.y,
													   A.shape.x, A.shape.y);
}

Matrix& LinearLayer::backward(Matrix& dZ, float learning_rate) {
	dA.allocateMemoryIfNotAllocated(A.shape);

	computeAndStoreBackwardError(dZ);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation.");

	updateBias(dZ, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");

	updateWeights(dZ, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform weights update.");

	return dA;
}

void LinearLayer::computeAndStoreBackwardError(Matrix& dZ) {
	dim3 block_size(32, 32);
	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
						(A.shape.y + block_size.y - 1) / block_size.y);
	linearLayerBackward<<<num_of_blocks, block_size>>>( W.data_device.get(),
														dZ.data_device.get(),
														dA.data_device.get(),
														W.shape.x, W.shape.y,
														dZ.shape.x, dZ.shape.y);
}

void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {
	dim3 block_size(32, 32);
	dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
						(W.shape.y + block_size.y - 1) / block_size.y);
	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
															A.data_device.get(),
															W.data_device.get(),
															dZ.shape.x, dZ.shape.y,
															A.shape.x, A.shape.y,
															learning_rate);
}

void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
														 b.data_device.get(),
														 dZ.shape.x, dZ.shape.y,
														 b.shape.x, learning_rate);
}

int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}
