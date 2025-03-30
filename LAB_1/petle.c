#include <stdio.h>
#include <omp.h>

#define ITER 15

int main() {
  int nr_w = omp_get_max_threads();  // lub ustalona liczba, np. 4
  int i;

  printf("== Petla 1: schedule(static, 3) ==\n");
  omp_set_num_threads(nr_w);
  #pragma omp parallel for schedule(static, 3)
  for(i = 0; i < ITER; i++) {
    printf("Watek %d, indeks %d\n", omp_get_thread_num(), i);
  }
  printf("\n");

  printf("== Petla 2: schedule(static) (domyslny rozmiar porcji) ==\n");
  omp_set_num_threads(nr_w);
  #pragma omp parallel for schedule(static)
  for(i = 0; i < ITER; i++) {
    printf("Watek %d, indeks %d\n", omp_get_thread_num(), i);
  }
  printf("\n");

  printf("== Petla 3: schedule(dynamic, 3) ==\n");
  omp_set_num_threads(nr_w);
  #pragma omp parallel for schedule(dynamic, 3)
  for(i = 0; i < ITER; i++) {
    printf("Wątek %d, indeks %d\n", omp_get_thread_num(), i);
  }
  printf("\n");

  printf("== Petla 4: schedule(dynamic) (domyslny rozmiar porcji) ==\n");
  omp_set_num_threads(nr_w);
  #pragma omp parallel for schedule(dynamic)
  for(i = 0; i < ITER; i++) {
    printf("Watek %d, indeks %d\n", omp_get_thread_num(), i);
  }
  printf("\n");

  return 0;
}
