#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double stime1 , stime2 , stime3 , stime4;
    double etime1 , etime2 , etime3 , etime4;

    int px, py, pz;
    int nx, ny, nz;
    int nc;

    px = atoi(argv[2]);
    py = atoi(argv[3]);
    pz = atoi(argv[4]);
    nx = atoi(argv[5]);
    ny = atoi(argv[6]);
    nz = atoi(argv[7]);
    nc = atoi(argv[8]);

    char *input_file = argv[1];
    char *output_file = argv[9];

    // file read and data distribution
    stime1 = MPI_Wtime();

    // Compute process grid coordinates
    int px_rank = rank % px;
    int py_rank = (rank / px) % py;
    int pz_rank = rank / (px * py);

    int local_nx = nx / px;
    int local_ny = ny / py;
    int local_nz = nz / pz;

    int x_offset = px_rank * local_nx;
    int y_offset = py_rank * local_ny;
    int z_offset = pz_rank * local_nz;

    // Allocate 4D array: local_nx x local_ny x local_nz x nc
    float ****local_data = malloc(local_nx * sizeof(float ***));
    for (int i = 0; i < local_nx; i++) {
        local_data[i] = malloc(local_ny * sizeof(float **));
        for (int j = 0; j < local_ny; j++) {
            local_data[i][j] = malloc(local_nz * sizeof(float *));
            for (int k = 0; k < local_nz; k++) {
                local_data[i][j][k] = malloc(nc * sizeof(float));
            }
        }
    }

    FILE *fp = fopen(input_file, "rb");

    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                int gi = x_offset + i;
                int gj = y_offset + j;
                int gk = z_offset + k;

                size_t offset = ((((size_t)gi * ny + gj) * nz + gk) * nc) * sizeof(float);
                fseek(fp, offset, SEEK_SET);
                fread(local_data[i][j][k], sizeof(float), nc, fp);
            }
        }
    }

    fclose(fp);

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
