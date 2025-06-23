#include <stdio.h>
#include <time.h>

#include "include/regression.h"

int main(int argc, char **argv)
{
    clock_t t_start, t_end;

    float_t x[] = {1, 2, 3, 4, 5};
    float_t y[] = {0, 0, 1, 1, 1};

    t_start = clock();
    params_t beta = logistic(x, y, 5, (opts_t){
        .learning_rate = 0.01f,
        .momentum = 0.9f,
        .epochs = 1000,
    });
    t_end = clock();

    float cpu_time_used = ((float) (t_end - t_start)) / CLOCKS_PER_SEC;
    printf("time: %.2fs\n\n", cpu_time_used);

    return 0;
}
