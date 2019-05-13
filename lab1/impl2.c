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
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)NUM_STEPS;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel 
    {
        double x;
        int id;
        id = omp_get_thread_num();
        sum[id] = 0;
#pragma omp for 
        for (i = 0; i < NUM_STEPS; i++)
        {
            x = (i + 0.5) * step;
            sum[id] += f(x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    printf("%lf\n", pi);
} 