#ifndef __REGRESSION_H__
#define __REGRESSION_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct opts_t {
    float_t learning_rate, momentum;
    size_t epochs;
};

typedef struct opts_t opts_t;

struct params_t {
    float_t intercept;
    float_t coeff;
    struct params_t *momentum;
};

typedef struct params_t params_t;

struct sets_t {
    struct params_t params;
    struct opts_t opts;
    size_t size;
};

typedef struct sets_t sets_t;

float_t *linear_foward(float_t *x, sets_t *sets);
void linear_backward(float_t *x, float_t *y, float_t *y_pred, sets_t *sets);
params_t linear(float_t *x, float_t *y, size_t len, opts_t opts);

float_t *logistic_foward(float_t *x, sets_t *sets);
void logistic_backward(float_t *x, float_t *y, float_t *y_pred, sets_t *sets);
params_t logistic(float_t *x, float_t *y, size_t len, opts_t opts);

#endif