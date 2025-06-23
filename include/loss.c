#include "loss.h"

float_t mean_square_error(float_t *y_train, float_t *y_pred, size_t n)
{
    float_t result = 0;
    for (size_t i = 0; i < n; i++) {
        result += (y_train[i] - y_pred[i]) * (y_train[i] - y_pred[i]);
    }
    return result / n;
}

float_t mean_absolute_error(float_t *y_train, float_t *y_pred, size_t n)
{
    float_t result = 0;
    for (size_t i = 0; i < n; i++) {
        result += fabsf(y_train[i] - y_pred[i]);
    }
    return result / n;
}

float_t binary_cross_entropy(float_t *y_train, float_t *y_pred, size_t n)
{
    float_t result = 0;
    for (size_t i = 0; i < n; i++) {
        result += y_train[i] * logf(y_pred[i]) + (1 - y_train[i]) * logf(1 - y_pred[i]);
    }
    return result * (-1.0 / n);
}
