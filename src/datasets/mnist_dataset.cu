#include "mnist_dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

MNISTDataset::MNISTDataset(const std::string& filename, size_t batch_size)
    : batch_size(batch_size) {

    std::vector<std::string> lines = readCSV(filename);
    // std::cout << "line.size: " << lines.size() << std::endl;
    number_of_batches = lines.size() / batch_size;

    for (size_t i = 0; i < number_of_batches; i++) {
        batches.push_back(Matrix(Shape(batch_size, 784))); // 28x28=784
        targets.push_back(Matrix(Shape(batch_size, 10)));  // 10 categories

        batches[i].allocateMemory();
        targets[i].allocateMemory();

        for (size_t k = 0; k < batch_size; k++) {
            size_t line_index = i * batch_size + k;
            std::stringstream ss(lines[line_index]);
            std::string item;
            std::vector<int> all_values;

            while (std::getline(ss, item, ',')) {
                all_values.push_back(std::stoi(item));
            }

            std::vector<float> inputs = scaleAndShiftInputs(std::vector<int>(all_values.begin() + 1, all_values.end()));
            std::vector<float> outputs(10, 0.01f); // 使用 float 类型
            outputs[all_values[0]] = 0.99f;

            // Copy data to the Matrix objects
            for (size_t j = 0; j < 784; j++) {
                batches[i][k * 784 + j] = inputs[j];
            }

            for (size_t j = 0; j < 10; j++) {
                targets[i][k * 10 + j] = outputs[j];
            }
        }

        // Debugging: Print some of the data to ensure it's correct
        // std::cout << "Batch " << i << " first input: " << batches[0] << ", first target: " << targets[0] << std::endl;

        batches[i].copyHostToDevice();
        targets[i].copyHostToDevice();
    }
}

std::vector<std::string> MNISTDataset::readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::vector<float> MNISTDataset::scaleAndShiftInputs(const std::vector<int>& input_values) {
    std::vector<float> inputs;
    std::transform(input_values.begin(), input_values.end(), std::back_inserter(inputs), [](int val) {
        return (val / 255.0f * 0.99f) + 0.01f;
    });
    return inputs;
}

int MNISTDataset::getNumOfBatches() const {
    return number_of_batches;
}

std::vector<Matrix>& MNISTDataset::getBatches() {
    return batches;
}

std::vector<Matrix>& MNISTDataset::getTargets() {
    return targets;
}
