#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <math.h>

float_t linear_combination(float_t x, float_t coeff, float_t intercept);
float_t relu(float_t x);
float_t sigmoid(float_t x);

#endif
