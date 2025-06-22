#ifndef __LOSS_H__
#define __LOSS_H__

#include <stdint.h>
#include <math.h>

float mean_square_error(float *y_train, float *y_pred, size_t n);
float binary_cross_entropy(float *y_train, float *y_pred, size_t n);

#endif
