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
    double pi = 0.0;
    double sum = 0.0;
    double x = 0.0;
    int id;
    step = 1.0 / (double)NUM_STEPS;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(x, sum, id) 
//x,sum,id变量对于每个线程私有
    {
        id = omp_get_thread_num();
        int i;
        for (i = id, sum = 0.0; i < NUM_STEPS; i = i + NUM_THREADS)
        {
            x = (i + 0.5) * step;
            sum += f(x);
        }
#pragma omp critical //指定代码段在同一时刻只能由一个线程进行执行
        pi += sum * step;
    }
    printf("%lf\n", pi);
}