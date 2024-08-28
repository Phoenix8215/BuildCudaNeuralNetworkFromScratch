# Build CUDA Neural Network From Scratch

This project is a CUDA-based neural network implementation, developed from scratch with performance optimizations and modifications. The project is inspired by [cuda-neural-network](ttps://github.comh/pwlnk/cuda-neural-network) and has been extended with additional features and optimizations to improve performance. Notably, the input dataset has been changed to MNIST, a popular dataset for handwritten digit recognition.
> 🚀a c++ version is here https://github.com/Phoenix8215/build_neural_network_from_scratch_CPP

## Features

- Fully implemented neural network in CUDA for high-performance computations.
- Optimized CUDA kernels for enhanced performance.
- MNIST dataset integration for training and testing the neural network.
- Customizable network architecture to experiment with different configurations.

## Installation

### Prerequisites

- **CUDA**: Ensure that you have CUDA installed on your system. You can download it from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
- **Make**: Required for building the project.
- **A modern C++ compiler**: GCC or Clang is recommended.

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Phoenix8215/BuildCudaNeuralNetworkFromScratch
   cd BuildCudaNeuralNetworkFromScratch
   ```

2. Build the project using Make:

   ```bash
   make
   ```

3. Run the application:

   ```bash
   ./cuda-conv
   ```

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and make your changes in a separate branch. Then, submit a pull request with a description of your changes.

## Acknowledgments

- Thanks to [pwlnk](https://github.com/pwlnk) for the original [cuda-neural-network](https://github.com/pwlnk/cuda-neural-network) project, which served as the inspiration and foundation for this work.
- Thanks to the developers and contributors of the CUDA toolkit and the MNIST dataset.
