#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__device__ double f(double x) {
    return 1.0 / (1.0 + x * x);
}

__global__ void trapezoidal_kernel(double a, int n, double h, double* partial_sums) {
    int T = gridDim.x * blockDim.x;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    int base = n / T;
    int reszta = n % T;
    int start = t * base + (t < reszta ? t : reszta);
    int count = base + (t < reszta ? 1 : 0);

    double local_sum = 0.0;
    for (int i = 0; i < count; i++) {
        int idx = start + i;
        double x = a + idx * h;
        double x_next = x + h;
        local_sum += (f(x) + f(x_next)) * h / 2.0;
    }
    partial_sums[t] = local_sum;
}

int main() {
    double a = 0.0, b = 1.0;
    int n = 10000;
    double h = (b - a) / n;

    int threadsPerBlock = 256;
    int numThreads = 1024;
    int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    double* d_partial_sums, * h_partial_sums;
    h_partial_sums = (double*)malloc(numThreads * sizeof(double));
    cudaMalloc(&d_partial_sums, numThreads * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    trapezoidal_kernel << <blocks, threadsPerBlock >> > (a, n, h, d_partial_sums);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cuda_time = 0;
    cudaEventElapsedTime(&cuda_time, start, stop);

    cudaMemcpy(h_partial_sums, d_partial_sums, numThreads * sizeof(double), cudaMemcpyDeviceToHost);

    double result = 0.0;
    for (int i = 0; i < numThreads; i++)
        result += h_partial_sums[i];

    printf("CUDA: wynik = %.10f, czas = %.6f s\n", 4.0 * result, cuda_time / 1000.0);

    cudaFree(d_partial_sums);
    free(h_partial_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
