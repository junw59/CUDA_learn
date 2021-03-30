#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 1048576
#define THREAD_NUM 256
#define BLOCK_NUM 32

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
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    int offset = 1;
    if(tid == 0) time[bid] = clock();
    shared[tid] = 0;
    for(i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        shared[tid] += num[i] * num[i];
    }

    __syncthreads();
    offset = THREAD_NUM / 2;
    while(offset > 0){
        if(tid < offset){
            // for(i = 1; i < THREAD_NUM; i++){
            shared[tid] += shared[tid + offset];
            // }
        }
        offset >>= 1;
        // result[bid] = shared[0];
        __syncthreads();
    }

    if(tid == 0){
        result[bid] = shared[0];
        time[bid + BLOCK_NUM] = clock();
    }
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
    cudaMalloc((void**) &result, sizeof(int) * BLOCK_NUM);
    cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int)>>>(gpudata, result, time);

    int sum[BLOCK_NUM];
    clock_t time_used[BLOCK_NUM * 2];
    cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);
    stop_g = (clock() - start_g);

    int final_num = 0;
    for(int i = 0; i < BLOCK_NUM; i++){
        final_num += sum[i];
    }

    clock_t min_start, max_end;
    min_start = time_used[0];
    max_end = time_used[BLOCK_NUM];
    for(int i = 1; i < BLOCK_NUM; i++){
        if(min_start > time_used[i]) min_start = time_used[i];
        if(max_end < time_used[i+BLOCK_NUM]) max_end = time_used[i + BLOCK_NUM];
    }
    printf("sum (GPU): %d time: %f timeg: %f \n", final_num, (double)(max_end - min_start) / CLOCKS_PER_SEC, (double) stop_g / CLOCKS_PER_SEC);

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
