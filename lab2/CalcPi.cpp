#include <mpi.h>
#include <iostream>
using namespace std;

inline double f(double x)
{
    return 4 / (1 + x * x);
}

int main(int argc, char *argv[])
{
    double pi, step, sum, x, StartTime, EndTime;
    int NumProcs, myid;
    long long n = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProcs);

    if (0 == myid)
    {
        cout << "Please enter number of steps" << endl;
        cin >> n;
        StartTime = MPI_Wtime();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    step = 1.0 / (double)n;
    sum = 0.0;
    for (int i = myid; i < n; i = i + NumProcs)
    {
        x = step * ((double)i + 0.5);
        sum = sum + f(x);
    }
    sum = sum * step;                                                    
    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
    if (myid == 0)
    {
        EndTime = MPI_Wtime();
        printf("进程数:%d\n",NumProcs);
        printf("用时:%f\n", EndTime - StartTime);
        printf("%0.15f\n", pi);
    }
    MPI_Finalize();
    return 0;
}
