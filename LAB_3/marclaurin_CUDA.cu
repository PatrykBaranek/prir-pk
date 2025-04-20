#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void sin_maclaurin_kernel(double x, int N, double* terms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= N) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        double num = pow(x, 2 * i + 1);
        double denom = 1.0;
        for (int j = 2; j <= 2 * i + 1; j++)
            denom *= j;
        terms[i] = sign * num / denom;
    }
}

int main() {
    double x;
    int N;
    printf("Podaj x (w radianach): ");
    scanf("%lf", &x);
    printf("Podaj liczbe wyrazow szeregu N: ");
    scanf("%d", &N);

    double* d_terms, * h_terms;
    h_terms = (double*)malloc((N + 1) * sizeof(double));
    cudaMalloc(&d_terms, (N + 1) * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock) / threadsPerBlock + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sin_maclaurin_kernel << <blocks, threadsPerBlock >> > (x, N, d_terms);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cuda_time = 0;
    cudaEventElapsedTime(&cuda_time, start, stop);

    cudaMemcpy(h_terms, d_terms, (N + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i <= N; i++)
        sum += h_terms[i];

    double exact = sin(x);
    double abs_error = fabs(sum - exact);

    printf("CUDA:\n");
    printf("sin(x) z CUDA:              %.15f\n", sum);
    printf("sin(x) z biblioteki math.h: %.15f\n", exact);
    printf("Blad bezwzgledny:           %.15e\n", abs_error);
    printf("Czas wykonania:             %.6f s\n", cuda_time / 1000.0);

    cudaFree(d_terms);
    free(h_terms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
