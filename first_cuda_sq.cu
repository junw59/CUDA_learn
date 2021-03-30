#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 1048576

bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}

__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
    int sum = 0;
    int i;
    clock_t start = clock();
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i];
    }

    *result = sum;
    *time = clock() - start;
}

int main()
{
    if(!InitCUDA()) {
        return 0;
    }

    printf("CUDA initialized.\n");

    int data[DATA_SIZE];

    GenerateNumbers(data, DATA_SIZE);
    int* gpudata, *result;
    clock_t* time;
    clock_t start_g, stop_g;
    start_g = clock();
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMalloc((void**) &time, sizeof(clock_t));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<1, 1, 0>>>(gpudata, result, time);

    int sum;
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);
    stop_g = (clock() - start_g);

    printf("sum (GPU): %d time: %f timeg: %f \n", sum, (double)time_used / CLOCKS_PER_SEC, (double) stop_g / CLOCKS_PER_SEC);

    clock_t start, stop;

    start = clock();
    sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }

    stop = clock() - start;
    printf("sum (CPU): %d time: %f \n", sum, (double)stop / CLOCKS_PER_SEC);

    return 0;
}
