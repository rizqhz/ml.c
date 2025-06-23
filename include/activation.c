#include "activation.h"

float_t linear_combination(float_t x, float_t coeff, float_t intercept)
{
    return coeff * x + intercept;
}

float_t relu(float_t x)
{
    return x > 0 ? x : 0;
}

float_t sigmoid(float_t x)
{
    return 1.0 / (1.0 - expf(-x));
}
