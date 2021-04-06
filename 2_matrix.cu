#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#define NUM_THREADS 256

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

void matgen(float* a, int lda, int n)
{
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            a[i * lda + j] = (float) rand() / RAND_MAX;
        }
    }
}

void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
    int i, j, k;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            double t = 0;
            for(k = 0; k < n; k++) {
                t += a[i * lda + k] * b[k * ldb + j];
            }
            c[i * ldc + j] = t;
        }
    }
}

void compare_mat(const float* a, int lda, const float* b, int ldb, int n)
{
    float max_err = 0;
    float average_err = 0;
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            if(b[i * ldb + j] != 0) {
            float err = fabs((a[i * lda + j] - b[i * ldb + j]) / b[i * ldb + j]);
            if(max_err < err) max_err = err;
            average_err += err;
            }
        }
    }
    printf("Max error: %g Average error: %g\n", max_err, average_err / (n * n));
}

__global__ void matrixCUDA(float* a, float* b, float* c, int n)
{
    int row = threadIdx.x+blockDim.x*blockIdx.x;
    int column = threadIdx.y+blockDim.y*blockIdx.y;
    int i;

    if(row < n && column < n) {
        float t = 0;
        float y = 0;
        for(i = 0; i < n; i++) {
            float r;
            y -= a[row * n + i] * b[i * n + column];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
        c[row * n + column] = t;
    }
}

int main()
{
    float *a, *b, *c, *d;
    int n = 1000;

    if(!InitCUDA()) return 0;

    a = (float*) malloc(sizeof(float) * n * n);
    b = (float*) malloc(sizeof(float) * n * n);
    c = (float*) malloc(sizeof(float) * n * n);
    d = (float*) malloc(sizeof(float) * n * n);

    srand(0);
    matgen(a, n, n);
    matgen(b, n, n);

    float *ac, *bc, *cc;
    clock_t start, time;

    start = clock();
    cudaMalloc((void**) &ac, sizeof(float) * n * n);
    cudaMalloc((void**) &bc, sizeof(float) * n * n);
    cudaMalloc((void**) &cc, sizeof(float) * n * n);

    cudaMemcpy(ac, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    dim3 block(32,32);
    dim3 grid((n-1)/block.x + 1,(n-1)/block.y + 1);
    matrixCUDA<<<grid,block>>>(ac,bc,cc,n);

    cudaMemcpy(c, cc, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    cudaFree(ac);
    cudaFree(bc);
    cudaFree(cc);

    time = clock() - start;

    clock_t startc, timec;
    startc = clock();
    matmult(a, n, b, n, d, n, n);
    timec = clock() - startc;
    compare_mat(c, n, d, n, n);

    printf("GPU time used: %f \n", (double) time / CLOCKS_PER_SEC);
    printf("CPU time used: %f \n", (double) timec / CLOCKS_PER_SEC);

    return 0;
}
