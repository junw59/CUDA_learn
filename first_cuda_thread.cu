#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 1048576
#define THREAD_NUM 256

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
    const int tid = threadIdx.x;
    // const int size = DATA_SIZE / THREAD_NUM;
    int sum = 0;
    int i;
    clock_t start;
    if(tid == 0) start = clock();
    // for(i = tid * size; i < (tid + 1) * size; i++) {
    //     sum += num[i] * num[i];
    // }
    for(i = tid; i < DATA_SIZE; i += THREAD_NUM) {
        sum += num[i] * num[i];
    }

    result[tid] = sum;
    if(tid == 0) *time = clock() - start;
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
    cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM);
    cudaMalloc((void**) &time, sizeof(clock_t));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<1, THREAD_NUM, 0>>>(gpudata, result, time);

    int sum[THREAD_NUM];
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);
    stop_g = (clock() - start_g);

    int final_num = 0;
    for(int i = 0; i < THREAD_NUM; i++){
        final_num += sum[i];
    }

    printf("sum (GPU): %d time: %f timeg: %f \n", final_num, (double)time_used / CLOCKS_PER_SEC, (double) stop_g / CLOCKS_PER_SEC);

    clock_t start, stop;

    start = clock();
    final_num = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        final_num += data[i] * data[i];
    }

    stop = clock() - start;
    printf("sum (CPU): %d time: %f \n", final_num, (double)stop / CLOCKS_PER_SEC);

    return 0;
}
