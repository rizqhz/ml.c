#include "regression.h"

data_t *data_init(size_t size, float_t *value)
{
    data_t *data = malloc(sizeof(data_t));
    data->value = value;
    data->size = size;
    return data;
}

float_t *foward(data_t *x, params_t *params)
{
    const size_t n = x->size;
    float *result = calloc(sizeof(float), n);

    for (size_t i = 0; i < n; i++) {
        result[i] = x->value[i] * params->weight + params->bias;
    }

    return result;
}

void backward(data_t *x, float_t *y, float_t *y_pred, params_t *params)
{
    const size_t n = x->size;
    float grad_weight = 0, grad_bias = 0, diff;

    for (size_t i = 0; i < n; i++) {
        diff = y_pred[i] - y[i];
        grad_weight += diff * x->value[i];
        grad_bias += diff;
    }

    grad_weight /= n;
    grad_bias /= n;

    params->momentum_weight = LEARNING_RATE * grad_weight + MOMENTUM * params->momentum_weight;
    params->momentum_bias = LEARNING_RATE * grad_bias + MOMENTUM * params->momentum_bias;

    params->weight -= params->momentum_weight;
    params->bias -= params->momentum_bias;
}

result_t linear(data_t *x, float_t *y, params_t *params)
{
    const size_t n = x->size;
    float_t *y_pred = NULL, err;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        printf("\e[Hepoch: %d\n\n", epoch);
        printf("y = %.2fx + %.2f\n", params->weight, params->bias);

        y_pred = foward(x, params);
        printf("loss: %.4f\n\n", mean_square_error(y, y_pred, n));

        backward(x, y, y_pred, params);
    }

    free(y_pred);

    return (result_t) {
        .weight = params->weight,
        .bias = params->bias,
    };
}

result_t logistic(data_t *x, float_t *y, params_t *params)
{
    // TODO: logistic regression

    return (result_t) {
        .weight = 0,
        .bias = 0,
    };
}
