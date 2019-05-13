# 并行计算Lab1

**陈宇轩 PB16060738**



## 1.用4种不同并行方式的OpenMP实现π值的计算

### 计算Pi的公式

本次实验求π的积分方法选择使用公式$arctan(1)=π/4$以及$(arctan(x))’=1/(1+x^2)$

在求解$arctan(1)$ 时使用矩形法求解：

取$a=0$, $b=1$ 

$\int_a^bf(x) dx = y_0\Delta x+y_1\Delta x+ ...+y_{n-1}\Delta x$

$\Delta x =(b-a)/n$

$y=f(x)$

$y_i=f'(a+i*(b-a)/n) $	$i=0,1,2,...,n$



### 方法1

使用并行域并行化。

共2个线程参加计算，其中线程0进行迭代步0,2,4,...,100000，线程1进行迭代步1,3,5,...,99999。

```c
#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
#define NUM_STEPS 100000

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
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    printf("%lf\n", pi);
}
```



### 方法2

使用共享任务结构并行化。

共2个线程参加计算，其中线程0进行迭代步0~49999，线程1进行迭代步50000～99999。

```c
#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
#define NUM_STEPS 100000
double step;
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
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    printf("%lf\n", pi);
} 
```



### 方法3

使用private子句和critical部分并行化。

共2个线程参加计算，其中线程0进行迭代步0,2,4,...,100000，线程1进行迭代步1,3,5,...,99999。

当被指定为critical的代码段 ，正在被0线程执行时，1线程的执行也到达该代码段，则它将被阻塞直到0线程退出临界区。

```C
#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
#define NUM_STEPS 100000
double step;
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
            sum += 4.0 / (1.0 + x * x);
        }
#pragma omp critical //指定代码段在同一时刻只能由一个线程进行执行
        pi += sum * step;
    }
    printf("%lf\n", pi);
}
```



### 方法4

使用并行规约。

共2个线程参加计算，其中线程0进行迭代步0~49999，线程1进行迭代步50000～99999。

```c
#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
#define NUM_STEPS 100000
double step;
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
        sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
    printf("%lf\n", pi);
}
```



## 2.用OpenMP实现PSRS排序

### 算法分析

**输入**：n个待排序的数据，均匀地分布在P个处理器上

**输出**：分布在各个处理器上，得到全局有序的数据序列

**Begin**

(1) 每个处理器将自己的n/P个数据用串行快速排序(Quicksort)，得到一个排好序的序列；

(2) 每个处理器从排好序的序列中选取第w，2w，3w，…，(P-1)w个共P-1个数据作为代表元素，其中$w=n/P^2$；

(3) 每个处理器将选好的代表元素送到处理器P0中，并将送来的P段有序的数据序列做P路归并，再选择排序后的第P-1，2(P-1)，…，(P-1)(P-1)个共P-1个主元；

(4) 处理器P0将这P-1个主元播送到所有处理器中；

(5) 每个处理器根据上步送来的P-1个主元把自己的n/P个数据分成P段，记 为处理器Pi的第j+1段，其中i=0，…，P-1，j=0，…，P-1；

(6) 每个处理器送它的第i+1段给处理器Pi，从而使得第i个处理器含有所有处理器的第i段数据(i=0，…，P-1)；

(7) 每个处理器再通过P路归并排序将上一步的到的数据排序；从而这n个数据便是有序的。

**End**

### 

## 附录

PSRS算法代码

```c
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

const int MBSIZE = 1;
const int NUM_THREADS = 4;

int **temp;
int **segment; //各处理器按照主元的各自的有序段号
int *sizes;
int *sample;
int *pivot_number;

void PSRS(int *data, int size, int NUM_THREADS);

int main()
{

	// 读入数组大小
	int num;
	cout << "Please input the num:\n";
	cin >> num;

	// 处理不整除情况
	int beginpoint = 0;
	beginpoint = NUM_THREADS - num % NUM_THREADS;

	srand(time(0));
	int *data = new int[num + beginpoint];
	for (int i = 0; i < num; i++)
	{
		data[i] = rand() % 200;
	}
	for (int i = num; i < num + beginpoint; i++)
	{
		data[i] = 0;
	}

	// 输出随机数组结果
	cout << "Random vector:\n";
	for (int i = 0; i < num; i++)
	{
		cout << data[i] << " ";
	}
	cout << endl;
	num += beginpoint;

	// 进行排序
	double startTime = omp_get_wtime();
	PSRS(data, num * MBSIZE, NUM_THREADS);
	double endTime = omp_get_wtime();

	// 检查排序
	cout << "Result:\n";
	for (int i = beginpoint; i < num; i++)
	{
		cout << data[i] << " ";
	}
	cout << endl;

	cout << "排序数组大小:" << num - beginpoint << ",线程数:" << NUM_THREADS << endl;
	cout << "运行时间：" << endTime - startTime << " s" << endl;

	// 释放动态数据

	free(data);

	for (int i = 0; i < NUM_THREADS; i++)
	{
		free(temp[i]);
		free(segment[i]);
	}

	return 0;
}

void PSRS(int *data, int size, int NUM_THREADS)
{
	int localN = size / NUM_THREADS;
	sample = (int *)malloc(sizeof(int) * (NUM_THREADS * NUM_THREADS));
	pivot_number = (int *)malloc(sizeof(int) * (NUM_THREADS - 1));

	// 均匀划分 + 局部排序 + 正则采样
#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		int localLeft = id * localN;
		int localRight = (id + 1) * localN;
		int step = localN / NUM_THREADS;
		sort(data + localLeft, data + localRight);
		for (int i = 0; i < NUM_THREADS; i++)
		{
			sample[id * NUM_THREADS + i] = *(data + (id * localN + i * step));
		}
	}
	//样本排序
	sort(sample, sample + NUM_THREADS * NUM_THREADS);

	//选择主元
	for (int i = 1; i < NUM_THREADS; i++)
	{
		pivot_number[i - 1] = sample[i * NUM_THREADS];
	}
	segment = (int **)malloc(sizeof(int *) * NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		segment[i] = (int *)malloc(sizeof(int) * (NUM_THREADS + 1));
	}

	//主元划分
#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		int localLeft = id * localN;
		int localRight = (id + 1) * localN;
		int count = 0;
		int mleft = localLeft;
		segment[id][count] = 0;
		segment[id][NUM_THREADS] = localN;
		for (; mleft < localRight && count < NUM_THREADS - 1;)
		{
			if (*(data + mleft) <= pivot_number[count])
			{
				mleft += 1;
			}
			else
			{
				count += 1;
				segment[id][count] = mleft - localLeft;
			}
		}

		for (; count < NUM_THREADS - 1; count++)
		{

			segment[id][count + 1] = mleft - localLeft;
		}
	}
	// 释放动态数据
	free(sample);
	free(pivot_number);
	//sizes = (int*)malloc(sizeof(int)*NUM_THREADS);
	sizes = (int *)malloc(sizeof(int) * NUM_THREADS);
	temp = (int **)malloc(sizeof(int *) * NUM_THREADS);
	//全局交换
	// 计算每一段的大小，动态初始化
	for (int i = 0; i < NUM_THREADS; i++)
	{
		sizes[i] = 0;
		for (int j = 0; j < NUM_THREADS; j++)
		{
			sizes[i] += (segment[j][i + 1] - segment[j][i]);
			//cout << sizes[i] << endl;
		}
		temp[i] = (int *)malloc(sizeof(int) * sizes[i]);
		int index = 0;
		for (int j = 0; j < NUM_THREADS; j++)
		{
			for (int k = segment[j][i]; k < segment[j][i + 1]; k++)
			{
				data[localN * j + k];

				temp[i][index] = data[localN * j + k];
				index += 1;
			}
		}
	}
	//归并排序
#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		sort(temp[id], temp[id] + sizes[id]);
	}
	int i = 0;
	for (int j = 0; j < NUM_THREADS; j++)
	{
		for (int k = 0; k < sizes[j]; k++)
		{
			*(data + i) = *(temp[j] + k);
			i++;
		}
	}
	free(sizes);
}
```



