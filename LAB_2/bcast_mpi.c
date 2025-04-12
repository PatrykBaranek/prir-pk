/*
 * bcast_mpi.c
 *
 * Program wykorzystujący MPI_Bcast do rozpropagowania liczby do wszystkich procesów.
 * Proces 0 odczytuje liczbę, a następnie przesyła ją do pozostałych procesów.
 * Każdy proces wypisuje swój identyfikator oraz uzyskaną wartość.
 * Program kończy działanie, gdy otrzymana liczba jest ujemna.
 *
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int rank, size;
  int num;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  while (1)
  {
    if (rank == 0)
    {
      printf("Podaj liczbę:\n");
      fflush(stdout);
      scanf("%d", &num);
    }

    /* Rozpropagowanie liczby do wszystkich procesów */
    MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (num < 0)
    {
      break;
    }

    printf("Proces %d otrzymał %d\n", rank, num);
  }

  MPI_Finalize();
  return 0;
}
