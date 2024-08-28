#pragma once

#include <string>
#include "../nn_utils/matrix.h"

class CostFunction {
protected:
    std::string name;

public:
    virtual float cost(Matrix predictions, Matrix target) = 0;
    virtual Matrix dCost(Matrix predictions, Matrix target, Matrix dY) = 0;

    const std::string getName() const { return this->name; }
};

