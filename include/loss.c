#include "loss.h"

float mean_square_error(float *y_train, float *y_pred, size_t n)
{
    float result = 0;
    for (size_t i = 0; i < n; i++) {
        result += (y_train[i] - y_pred[i]) * (y_train[i] - y_pred[i]);
    }
    return result / n;
}

float binary_cross_entropy(float *y_train, float *y_pred, size_t n)
{
    float result = 0;
    for (size_t i = 0; i < n; i++) {
        result += y_train[i] * logf(y_pred[i]) + (1 - y_train[i]) * logf(1 - y_pred[i]);
    }
    return result / n;
}
