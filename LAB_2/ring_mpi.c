/*
 * ring_mpi.c
 *
 * Program przesyłający liczbę w konwencji pierścienia.
 * Proces 0 odczytuje liczbę od użytkownika i wysyła ją do kolejnych procesów,
 * aż komunikat wróci do procesu 0. Program kończy działanie, gdy odczytana liczba
 * jest ujemna.
 *
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int rank, size;
  int num;
  MPI_Status status;

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

      /* Jeśli wprowadzona liczba jest ujemna,
       * wysyłamy ją, aby zakończyć propagację,
       * a potem przerywamy pętlę.
       */
      MPI_Send(&num, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      if (num < 0)
      {
        break;
      }

      /* Proces 0 oczekuje komunikatu od ostatniego procesu */
      MPI_Recv(&num, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);
      printf("Proces 0 dostał %d od procesu %d\n", num, size - 1);
    }
    else
    {
      /* Procesy o randze > 0 odbierają komunikat od poprzednika */
      MPI_Recv(&num, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);

      if (num < 0)
      {
        /* Jeżeli otrzymana liczba jest ujemna, przekazujemy ją dalej
         * (dla ostatniego procesu wysyłka do 0), a następnie kończymy działanie.
         */
        int dest = (rank == size - 1) ? 0 : rank + 1;
        MPI_Send(&num, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        break;
      }

      printf("Proces %d dostał %d od procesu %d\n", rank, num, rank - 1);
      int dest = (rank == size - 1) ? 0 : rank + 1;
      MPI_Send(&num, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}
