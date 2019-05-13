#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
#define NUM_STEPS 100000

inline double f(double x)
{
    return 4 / (1 + x * x);
}

int main()
{
    int i;
    double x, sum[NUM_THREADS];
    double pi = 0;
    double step = 1.0 / NUM_STEPS;
    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel private(i)
    {
        double x;
        int id;
        // id = 0 or 1
        id = omp_get_thread_num();
        for (i = id, sum[id] = 0.0; i < NUM_STEPS; i = i + NUM_THREADS)
        {
            x = (i + 0.5) * step;
            sum[id] += f(x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    printf("%lf\n", pi);
}