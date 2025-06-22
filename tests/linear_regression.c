#include <stdio.h>
#include <time.h>

#include "include/regression.h"

int main(int argc, char **argv)
{
    clock_t t_start, t_end;

    float_t __x[] = {8, 2, 6, 4, 7, 3};
    data_t *x = data_init(6, __x);

    float_t __y[] = {7, 3, 7, 2, 8, 3};
    data_t *y = data_init(6, __y);

    params_t params = {
        .weight = 0.5, .bias = 0.5,
        .momentum_weight = 0, .momentum_bias = 0,
    };

    t_start = clock();
    result_t beta = linear(x, y->value, &params);
    t_end = clock();

    float cpu_time_used = ((float) (t_end - t_start)) / CLOCKS_PER_SEC;
    printf("time: %f seconds\n", cpu_time_used);

    free(x);
    free(y);

    return 0;
}
