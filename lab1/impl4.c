#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
#define NUM_STEPS 100000
double step;

inline double f(double x)
{
    return 4 / (1 + x * x);
}

int main()
{
    int i;
    double pi = 0.0;
    double sum = 0.0;
    double x = 0.0;
    step = 1.0 / (double)NUM_STEPS;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for reduction(+: sum) private(x)
    for (i = 1; i <= NUM_STEPS; i++)
    {
        x = (i - 0.5) * step;
        sum += f(x);
    }
    pi = sum * step;
    printf("%lf\n", pi);
}