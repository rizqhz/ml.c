#include <stdio.h>
#include <time.h>

#include "include/regression.h"

int main(int argc, char **argv)
{
    clock_t t_start, t_end;

    float_t x[] = {8, 2, 6, 4, 7, 3};
    float_t y[] = {7, 3, 7, 2, 8, 3};

    t_start = clock();
    params_t beta = linear(x, y, 6, (opts_t){
        .learning_rate = 0.01f,
        .momentum = 0.9f,
        .epochs = 250,
    });
    t_end = clock();

    float cpu_time_used = ((float) (t_end - t_start)) / CLOCKS_PER_SEC;
    printf("time: %.2fs\n\n", cpu_time_used);

    return 0;
}
