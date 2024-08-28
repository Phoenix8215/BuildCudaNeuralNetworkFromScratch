#pragma once
#include "../nn_utils/matrix.h"
#include "cost_function.h"


class BCECost : public CostFunction {
public:
	BCECost(std::string name);
	~BCECost();
	
	float cost(Matrix predictions, Matrix target) override;
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY) override;

};
