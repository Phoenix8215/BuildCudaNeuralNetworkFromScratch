#include "matrix.h"
#include "nn_exception.h"

Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{ }

void Matrix::allocateCudaMemory() {
	if (!device_allocated) {
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<float>(device_memory,
											 [&](float* ptr){ cudaFree(ptr); });
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
										   [&](float* ptr){ delete[] ptr; });
		host_allocated = true;
	}
}

void Matrix::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

float& Matrix::operator[](const int index) {
	return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
	return data_host.get()[index];
}


std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    size_t x_dim = matrix.shape.x;
    size_t y_dim = matrix.shape.y;

    os << "Matrix (" << x_dim << " x " << y_dim << "):" << std::endl;

    // 打印矩阵内容，假设数据已经从 device 复制到 host
    for (size_t i = 0; i < x_dim; ++i) {
        for (size_t j = 0; j < y_dim; ++j) {
            os << matrix[i * y_dim + j] << " ";
        }
        os << std::endl;
    }

    return os;
}