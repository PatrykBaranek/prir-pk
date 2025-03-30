#include <stdio.h>
#include <omp.h>

#define ITER 150000

int main() {
    int nr_w = omp_get_max_threads();
    int i;
    double start, stop;
    double suma = 0.0;

    omp_set_num_threads(nr_w);
    
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic) reduction(+:suma)
    for(i = 0; i < ITER; i++){
        suma += i * 0.0001; // Przykładowa operacja
    }
    stop = omp_get_wtime();

    printf("Czas wykonania (dynamic): %lf sekund\n", stop - start);
    printf("Suma = %lf\n", suma); // suma kontrolna dla wyników aby sprawdzić poprawność podczas testów

    return 0;
}
