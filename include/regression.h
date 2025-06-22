#ifndef __REGRESSION_H__
#define __REGRESSION_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "loss.h"

#define LEARNING_RATE 0.01f
#define MOMENTUM 0.9f
#define EPOCHS 2000

typedef struct {
    float_t *value;
    size_t size;
} data_t;

data_t *data_init(size_t size, float_t *value);

typedef struct {
    float_t weight, bias;
    float_t momentum_weight, momentum_bias;
} params_t;

float_t *foward(data_t *x, params_t *params);
void backward(data_t *x, float_t *y, float_t *y_pred, params_t *params);

typedef struct {
    float_t weight;
    float_t bias;
} result_t;

result_t linear(data_t *x, float_t *y, params_t *params);
result_t logistic(data_t *x, float_t *y, params_t *params);

#endif