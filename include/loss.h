#ifndef __LOSS_H__
#define __LOSS_H__

#include <math.h>

float_t mean_square_error(float_t *y_train, float_t *y_pred, size_t n);
float_t mean_absolute_error(float_t *y_train, float_t *y_pred, size_t n);
float_t binary_cross_entropy(float_t *y_train, float_t *y_pred, size_t n);

#endif
