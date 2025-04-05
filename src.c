#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.c"

int main(int agc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double stime1 , stime2 , stime3 , stime4;
    double etime1 , etime2 , etime3 , etime4;
    // file read and data distribution
    stime1 = MPI_Wtime();

    etime1 = MPI_Wtime();

    // main code
    stime2 = MPI_Wtime();

    etime2 = MPI_Wtime();


    // output
    stime3 = MPI_Wtime();

    etime3 = MPI_Wtime();

    
    // finalize
    stime4 = MPI_Wtime();

    etime4 = MPI_Wtime();
    MPI_Finalize();
    return 0;
}