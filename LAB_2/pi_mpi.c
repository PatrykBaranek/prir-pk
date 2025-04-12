/*
 * pi_mpi.c
 *
 * Program obliczający przybliżenie liczby π wykorzystując szereg Leibniza:
 *    π = 4 * (1 - 1/3 + 1/5 - 1/7 + ... )
 *
 * Proces 0 odczytuje liczbę składników N, a następnie dzieli obliczenia
 * między wszystkie procesy (z uwzględnieniem przypadku N niepodzielnego przez
 * liczbę procesów). Każdy proces oblicza swoją sumę częściową, która jest
 * redukowana do procesu 0.
 *
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int rank, size;
  long long n; // liczba składników szeregu
  long long local_start, local_end, local_n;
  long long i;
  double local_sum = 0.0, global_sum = 0.0;
  double start_time, end_time;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    printf("Podaj liczbę składników szeregu: ");
    fflush(stdout);
    scanf("%lld", &n);
  }

  MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  /* Obliczenie zakresu iteracji dla każdego procesu.
   * Jeśli n nie dzieli się równomiernie przez size, to:
   *   - dla procesów o randze < remainder przypada (quotient+1) składników,
   *   - pozostałym procesom przypada quotient składników.
   */
  long long quotient = n / size;
  long long remainder = n % size;

  if (rank < remainder)
  {
    local_n = quotient + 1;
    local_start = rank * local_n;
  }
  else
  {
    local_n = quotient;
    local_start = rank * local_n + remainder;
  }
  local_end = local_start + local_n;

  start_time = MPI_Wtime();

  for (i = local_start; i < local_end; i++)
  {
    /* Obliczamy składnik szeregu: (-1)^i / (2*i+1) */
    double term = ((i % 2) ? -1.0 : 1.0) / (2 * i + 1);
    local_sum += term;
  }

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  end_time = MPI_Wtime();

  if (rank == 0)
  {
    double pi = 4.0 * global_sum;
    printf("Przybliżenie liczby π = %.15f\n", pi);
    printf("Czas obliczeń: %.6f sekundy\n", end_time - start_time);
  }

  MPI_Finalize();
  return 0;
}
