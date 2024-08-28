#pragma once

#include <vector>
#include <string>
#include "../nn_utils/matrix.h"

class MNISTDataset {
public:
    MNISTDataset(const std::string& filename, size_t batch_size);
    
    int getNumOfBatches() const;
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();

private:
    size_t batch_size;
    size_t number_of_batches;
    std::vector<Matrix> batches;
    std::vector<Matrix> targets;

    std::vector<std::string> readCSV(const std::string& filename);
    std::vector<float> scaleAndShiftInputs(const std::vector<int>& input_values);
};

