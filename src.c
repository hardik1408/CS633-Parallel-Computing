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
    int px, py, pz;
    int nx, ny, nz;
    int nc;
    int input_file_size = argv[1]/sizeof(char);
    int output_file_size = argv[1]/sizeof(char);
    char *input_file = (char *)malloc(input_file_size);
    char *output_file = (char *)malloc(output_file_size);
    
    input_file = argv[1];
    px = atoi(argv[2]);
    py = atoi(argv[3]);
    pz = atoi(argv[4]);
    nx = atoi(argv[5]);
    ny = atoi(argv[6]);
    nz = atoi(argv[7]);
    nc = atoi(argv[8]);
    output_file = argv[9];
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