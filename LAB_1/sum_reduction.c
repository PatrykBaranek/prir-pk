#include <stdio.h>
#include <omp.h>

#define ITER 5000

int main() {
  int i;
  double suma = 0.0;
  double liczba = 5.0;

  omp_set_num_threads(omp_get_max_threads());

  #pragma omp parallel for reduction(+:suma)
  for(i = 0; i < ITER; i++) {
    suma += liczba * liczba;
  }

  printf("Wersja z reduction: Suma kwadratów = %lf\n", suma);
  return 0;
}
