#include <stdlib.h>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std;

int i, j, k;
const int NUM_PROCS = 4;
const int INT_MAX = 100000;

void PSRS(int *data, int N);

int main(int argc, char *argv[])
{
    int num;
    int NumProcs, myid;
    MPI_Init(&argc, &argv); //MPI初始化
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProcs);

    // 读入数组大小
    if (myid == 0)
    {
        cout << "Please input the num:\n";
        cin >> num;
    }
    MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 处理不整除情况
    int beginpoint = 0;
    beginpoint = NUM_PROCS - num % NUM_PROCS;
    // if (myid == 0)
    // {
    srand(0);
    int *data = new int[num + beginpoint];
    for (int i = 0; i < num; i++)
    {
        data[i] = rand() % 200;
    }
    for (int i = num; i < num + beginpoint; i++)
    {
        data[i] = 0;
    }
    // }
    // MPI_Bcast(data,1,MPI_INT,0,MPI_COMM_WORLD);

    // 输出随机数组结果
    if (myid == 0)
    {
        cout << "Random vector:\n";
        for (int i = 0; i < num; i++)
        {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    PSRS(data, num); //调用PSRS算法进行并行排序

    return 0;
}

int cmp(const void *a, const void *b)
{
    if (*(int *)a < *(int *)b)
        return -1;
    if (*(int *)a > *(int *)b)
        return 1;
    else
        return 0;
}

void Comu(int *partitions, int *partitionSizes, int ProcNum, int myId, int *data)
{
    int *sortedSubList;
    int *recvDisp, *indexes, *partitionEnds, *subListSizes, totalListSize;

    indexes = (int *)malloc(ProcNum * sizeof(int));
    partitionEnds = (int *)malloc(ProcNum * sizeof(int));
    indexes[0] = 0;
    totalListSize = partitionSizes[0];
    for (i = 1; i < ProcNum; i++)
    {
        totalListSize += partitionSizes[i];
        indexes[i] = indexes[i - 1] + partitionSizes[i - 1];
        partitionEnds[i - 1] = indexes[i];
    }
    partitionEnds[ProcNum - 1] = totalListSize;

    sortedSubList = (int *)malloc(totalListSize * sizeof(int));
    subListSizes = (int *)malloc(ProcNum * sizeof(int));
    recvDisp = (int *)malloc(ProcNum * sizeof(int));

    // 归并排序
    for (i = 0; i < totalListSize; i++)
    {
        int lowest = INT_MAX;
        int ind = -1;
        for (j = 0; j < ProcNum; j++)
        {
            if ((indexes[j] < partitionEnds[j]) && (partitions[indexes[j]] < lowest))
            {
                lowest = partitions[indexes[j]];
                ind = j;
            }
        }
        sortedSubList[i] = lowest;
        indexes[ind] += 1;
    }

    // 发送各子列表的大小回根进程中
    MPI_Gather(&totalListSize, 1, MPI_INT, subListSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 计算根进程上的相对于recvbuf的偏移量
    if (myId == 0)
    {
        recvDisp[0] = 0;
        for (i = 1; i < ProcNum; i++)
        {
            recvDisp[i] = subListSizes[i - 1] + recvDisp[i - 1];
        }
    }

    //发送各排好序的子列表回根进程中
    MPI_Gatherv(sortedSubList, totalListSize, MPI_INT, data, subListSizes, recvDisp, MPI_INT, 0, MPI_COMM_WORLD);

    free(partitionEnds);
    free(sortedSubList);
    free(indexes);
    free(subListSizes);
    free(recvDisp);
    return;
}

void PSRS(int *data, int N)
{
    double StartTime = MPI_Wtime();
    int ProcNum, myId, *partitionSizes, *newPartitionSizes;
    int subArraySize, startIndex, endIndex, *sample, *newPartitions;
    int localN = N / NUM_PROCS;
    int step = localN / NUM_PROCS;

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    sample = (int *)malloc(ProcNum * ProcNum * sizeof(int));
    partitionSizes = (int *)malloc(ProcNum * sizeof(int));
    newPartitionSizes = (int *)malloc(ProcNum * sizeof(int));

    for (k = 0; k < ProcNum; k++)
    {
        partitionSizes[k] = 0;
    }

    startIndex = myId * N / ProcNum;
    if (ProcNum == (myId + 1))
    {
        endIndex = N;
    }
    else
    {
        endIndex = (myId + 1) * N / ProcNum;
    }
    subArraySize = endIndex - startIndex;

    MPI_Barrier(MPI_COMM_WORLD);

    // 对子数组进行局部排序
    qsort(data + startIndex, subArraySize, sizeof(data[0]), cmp);

    // 正则采样
    for (i = 0; i < ProcNum; i++)
    {
        sample[myId * NUM_PROCS + i] = *(data + (myId * localN + i * step));
    }

    int *pivot_number = (int *)malloc((ProcNum - 1) * sizeof(sample[0])); //主元
    int index = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    if (myId == 0)
    {
        qsort(sample, ProcNum * ProcNum, sizeof(sample[0]), cmp); //对正则采样的样本进行排序

        // 采样排序后进行主元的选择
        for (i = 0; i < (ProcNum - 1); i++)
        {
            pivot_number[i] = sample[(((i + 1) * ProcNum) + (ProcNum / 2)) - 1];
        }
    }

    //发送广播
    MPI_Bcast(pivot_number, ProcNum - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 主元划分
    for (i = 0; i < subArraySize; i++)
    {
        if (data[startIndex + i] > pivot_number[index])
        {
            index += 1;
        }
        if (index == ProcNum)
        {
            partitionSizes[ProcNum - 1] = subArraySize - i + 1;
            break;
        }
        partitionSizes[index]++; //划分大小自增
    }
    free(pivot_number);

    int totalSize = 0;
    int *sendDisp = (int *)malloc(ProcNum * sizeof(int));
    int *recvDisp = (int *)malloc(ProcNum * sizeof(int));

    // 全局到全局的发送
    MPI_Alltoall(partitionSizes, 1, MPI_INT, newPartitionSizes, 1, MPI_INT, MPI_COMM_WORLD);

    // 计算划分的总大小，并给新划分分配空间
    for (i = 0; i < ProcNum; i++)
    {
        totalSize += newPartitionSizes[i];
    }
    newPartitions = (int *)malloc(totalSize * sizeof(int));

    sendDisp[0] = 0;
    recvDisp[0] = 0;
    for (i = 1; i < ProcNum; i++)
    {
        sendDisp[i] = partitionSizes[i - 1] + sendDisp[i - 1];
        recvDisp[i] = newPartitionSizes[i - 1] + recvDisp[i - 1];
    }

    //发送数据，实现n次点对点通信
    MPI_Alltoallv(&(data[startIndex]), partitionSizes, sendDisp, MPI_INT, newPartitions, newPartitionSizes, recvDisp, MPI_INT, MPI_COMM_WORLD);

    free(sendDisp);
    free(recvDisp);

    Comu(newPartitions, newPartitionSizes, ProcNum, myId, data);

    double EndTime = MPI_Wtime();

    if (myId == 0)
    {
        cout << "Result:\n";
        for (int i = 0; i < N; i++)
        {
            cout << data[i] << " ";
        }
        cout << endl;

        cout << "线程数:" << NUM_PROCS << endl;
        cout << "运行时间：" << EndTime - StartTime << " s" << endl;
    }
    if (ProcNum > 1)
    {
        free(newPartitions);
    }
    free(partitionSizes);
    free(newPartitionSizes);
    free(sample);

    free(data);
    MPI_Finalize();
}
