#include <stdio.h>
#include <omp.h>

#define ITER 150000

int main() {
    int nr_w = omp_get_max_threads();
    int i;
    double start, stop;
    double suma = 0.0;
    double time_static_default, time_static_chunk, time_dynamic_default, time_dynamic_chunk;

    omp_set_num_threads(nr_w);

    // 1. schedule(static) - domyślny rozmiar porcji
    suma = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static) reduction(+:suma)
    for(int i = 0; i < ITER; i++){
        suma += i * 0.0001;
    }
    stop = omp_get_wtime();
    time_static_default = stop - start;
    printf("Static default: czas = %lf sekund, suma = %lf\n", time_static_default, suma);

    // 2. schedule(static, 3)
    suma = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static, 3) reduction(+:suma)
    for(int i = 0; i < ITER; i++){
        suma += i * 0.0001;
    }
    stop = omp_get_wtime();
    time_static_chunk = stop - start;
    printf("Static, chunk=3: czas = %lf sekund, suma = %lf\n", time_static_chunk, suma);

    // 3. schedule(dynamic) - domyślny rozmiar porcji
    suma = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic) reduction(+:suma)
    for(int i = 0; i < ITER; i++){
        suma += i * 0.0001;
    }
    stop = omp_get_wtime();
    time_dynamic_default = stop - start;
    printf("Dynamic default: czas = %lf sekund, suma = %lf\n", time_dynamic_default, suma);

    // 4. schedule(dynamic, 3)
    suma = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 3) reduction(+:suma)
    for(int i = 0; i < ITER; i++){
        suma += i * 0.0001;
    }
    stop = omp_get_wtime();
    time_dynamic_chunk = stop - start;
    printf("Dynamic, chunk=3: czas = %lf sekund, suma = %lf\n", time_dynamic_chunk, suma);


    return 0;
}
