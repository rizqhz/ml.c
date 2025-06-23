#include "regression.h"
#include "activation.h"
#include "loss.h"

float_t *linear_foward(float_t *x, sets_t *sets)
{
    float_t *result = calloc(sizeof(float_t), sets->size);
    for (size_t i = 0; i < sets->size; i++) {
        result[i] = linear_combination(x[i], sets->params.coeff, sets->params.intercept);
    }
    return result;
}

void linear_backward(float_t *x, float_t *y, float_t *y_pred, sets_t *sets)
{
    const float_t lr = sets->opts.learning_rate;
    const float_t v = sets->opts.momentum;
    float_t grad_coeff = 0, grad_intercept = 0, diff = 0;

    for (size_t i = 0; i < sets->size; i++) {
        diff = y_pred[i] - y[i];
        grad_coeff += diff * x[i];
        grad_intercept += diff;
    }

    grad_coeff /= sets->size;
    grad_intercept /= sets->size;

    params_t *params = &sets->params;
    params_t *momentum = sets->params.momentum;

    momentum->coeff = grad_coeff * lr + momentum->coeff * v;
    momentum->intercept = grad_intercept * lr + momentum->intercept * v;
    params->coeff -= momentum->coeff;
    params->intercept -= momentum->intercept;
}

params_t linear(float_t *x, float_t *y, size_t len, opts_t opts)
{
    sets_t sets = {
        .params = (params_t) {
            .coeff = 0.5f, .intercept = 0.5f,
            .momentum = NULL,
        },
        .opts = opts,
        .size = len,
    };

    sets.params.momentum = malloc(sizeof(params_t));
    sets.params.momentum->coeff = 0;
    sets.params.momentum->intercept = 0;

    for (int epoch = 1; epoch <= sets.opts.epochs; epoch++) {
        printf("\e[Hepoch: %d\n\n", epoch);
        printf("y = %.2fx + %.2f\n", sets.params.coeff, sets.params.intercept);

        float *y_pred = linear_foward(x, &sets);
        printf("loss: %.4f\n\n", mean_square_error(y, y_pred, sets.size));

        linear_backward(x, y, y_pred, &sets);
        free(y_pred);
    }

    free(sets.params.momentum);

    return (params_t) {
        .coeff = sets.params.coeff,
        .intercept = sets.params.intercept,
    };
}

float_t *logistic_foward(float_t *x, sets_t *sets)
{
    float_t *result = calloc(sizeof(float_t), sets->size);
    for (size_t i = 0; i < sets->size; i++) {
        result[i] = sigmoid(linear_combination(x[i], sets->params.coeff, sets->params.intercept));
    }
    return result;
}

void logistic_backward(float_t *x, float_t *y, float_t *y_pred, sets_t *sets)
{
    const float_t lr = sets->opts.learning_rate;
    const float_t v = sets->opts.momentum;
    float_t grad_coeff = 0, grad_intercept = 0, diff = 0;

    for (size_t i = 0; i < sets->size; i++) {
        diff = y_pred[i] - y[i];
        grad_coeff += diff * x[i];
        grad_intercept += diff;
    }

    grad_coeff /= sets->size;
    grad_intercept /= sets->size;

    params_t *params = &sets->params;
    params_t *momentum = sets->params.momentum;

    momentum->coeff = grad_coeff * lr + momentum->coeff * v;
    momentum->intercept = grad_intercept * lr + momentum->intercept * v;
    params->coeff -= momentum->coeff;
    params->intercept -= momentum->intercept;
}

params_t logistic(float_t *x, float_t *y, size_t len, opts_t opts)
{
    sets_t sets = {
        .params = (params_t) {
            .coeff = 0.5f, .intercept = -3.0f,
            .momentum = NULL,
        },
        .opts = opts,
        .size = len,
    };

    sets.params.momentum = malloc(sizeof(params_t));
    sets.params.momentum->coeff = 0;
    sets.params.momentum->intercept = 0;

    for (int epoch = 1; epoch <= sets.opts.epochs; epoch++) {
        printf("\e[Hepoch: %d\n\n", epoch);
        printf("y = %.2fx + %.2f\n", sets.params.coeff, sets.params.intercept);

        float *y_pred = logistic_foward(x, &sets);
        printf("loss: %.4f\n\n", binary_cross_entropy(y, y_pred, sets.size));

        logistic_backward(x, y, y_pred, &sets);
        free(y_pred);
    }

    free(sets.params.momentum);

    return (params_t) {
        .coeff = sets.params.coeff,
        .intercept = sets.params.intercept,
    };
}
