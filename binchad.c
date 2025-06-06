#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define MAX_FILENAME 100
#define MAX_LINE_LENGTH 100000
#define MAX_LINE_LEN 10000

// Macro to compute index in a 3D array stored in 1D (C order)
#define IDX(x,y,z,nx,ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

// Compute the Cartesian coordinates from the rank without using MPI Cartesian functions.
void get_coords(int rank, int PX, int PY, int PZ, int *coords) {
    coords[0] = rank % PX;
    coords[1] = (rank / PX) % PY;
    coords[2] = rank / (PX * PY);
}

// Compute the rank from the Cartesian coordinates.
int get_rank_from_coords(int x, int y, int z, int PX, int PY, int PZ) {
    if(x < 0 || x >= PX || y < 0 || y >= PY || z < 0 || z >= PZ)
        return -1;
    return x + y * PX + z * PX * PY;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t1 = MPI_Wtime();

    if (argc != 10) {
        if (rank == 0)
            printf("Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
    int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    const char *output_file = argv[9];
    int total_points = NX * NY * NZ;

    if (PX * PY * PZ != size) {
        if (rank == 0)
            printf("Error: PX*PY*PZ (%d) != number of processes (%d)\n", PX * PY * PZ, size);
        MPI_Finalize();
        return 1;
    }

    // Compute Cartesian coordinates manually.
    int coords[3];
    get_coords(rank, PX, PY, PZ, coords);
    int px = coords[0], py = coords[1], pz = coords[2];

    // Determine local dimensions.
    int lx = NX / PX + (px < NX % PX ? 1 : 0);
    int ly = NY / PY + (py < NY % PY ? 1 : 0);
    int lz = NZ / PZ + (pz < NZ % PZ ? 1 : 0);

    // Global offsets.
    int offset_x = (NX / PX) * px + (px < NX % PX ? px : NX % PX);
    int offset_y = (NY / PY) * py + (py < NY % PY ? py : NY % PY);
    int offset_z = (NZ / PZ) * pz + (pz < NZ % PZ ? pz : NZ % PZ);

    // Determine local line indices (global indices).
    int local_line_count = lx * ly * lz;
    int *local_line_indices = (int *)malloc(local_line_count * sizeof(int));
    int idx = 0;
    for (int z = 0; z < lz; z++) {
        for (int y = 0; y < ly; y++) {
            for (int x = 0; x < lx; x++) {
                int global_x = offset_x + x;
                int global_y = offset_y + y;
                int global_z = offset_z + z;
                int global_idx = global_x + global_y * NX + global_z * NX * NY;
                local_line_indices[idx++] = global_idx;
            }
        }
    }

    /*------------------------------------------------------------------
      Updated Reading Section for Binary File Input:
      The input file is now in binary format. Each point in the file
      contains NC floating-point numbers stored as 4-byte (single precision)
      values without delimiters. We use an MPI derived datatype to create a
      noncontiguous view of the file and perform a collective read.
    ------------------------------------------------------------------*/
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    /* Create an indexed datatype for the file view.
       Each point's data occupies NC floats, starting at the offset:
         global_index * NC * sizeof(float)
    */
    MPI_Aint *displacements = (MPI_Aint *)malloc(local_line_count * sizeof(MPI_Aint));
    int *blocklens = (int *)malloc(local_line_count * sizeof(int));
    for (int i = 0; i < local_line_count; i++) {
        displacements[i] = (MPI_Aint)local_line_indices[i] * NC * sizeof(float);
        blocklens[i] = NC;
    }
    MPI_Datatype filetype;
    MPI_Type_create_hindexed(local_line_count, blocklens, displacements, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);

    /* Read the binary data collectively.
       Allocate a buffer to store the data as floats.
    */
    float *local_data = (float *)malloc(local_line_count * NC * sizeof(float));
    MPI_File_read_all(fh, local_data, local_line_count * NC, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    free(displacements);
    free(blocklens);
    free(local_line_indices);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Allocate an extended buffer with a one-cell halo on all sides.
    int ext_nx = lx + 2;
    int ext_ny = ly + 2;
    int ext_nz = lz + 2;
    int ext_size = ext_nx * ext_ny * ext_nz;
    float *ext_data = (float *)malloc(sizeof(float) * ext_size * NC);
    for (int i = 0; i < ext_size * NC; i++) {
        ext_data[i] = 0.0;
    }

    // Copy local_data into the center of ext_data.
    for (int t = 0; t < NC; t++) {
        for (int z = 0; z < lz; z++) {
            for (int y = 0; y < ly; y++) {
                for (int x = 0; x < lx; x++) {
                    int local_idx = IDX(x, y, z, lx, ly);
                    int ext_idx = IDX(x + 1, y + 1, z + 1, ext_nx, ext_ny);
                    ext_data[ext_idx * NC + t] = local_data[local_idx * NC + t];
                }
            }
        }
    }

    // Perform ghost exchange for the full halo.
    for (int t = 0; t < NC; t++) {
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;
                    
                    int nbr_x = px + dx;
                    int nbr_y = py + dy;
                    int nbr_z = pz + dz;
                    int nbr = get_rank_from_coords(nbr_x, nbr_y, nbr_z, PX, PY, PZ);
                    if(nbr < 0)
                        continue;
                    
                    int send_x_start = (dx == -1) ? 1 : (dx == 0 ? 1 : lx);
                    int send_y_start = (dy == -1) ? 1 : (dy == 0 ? 1 : ly);
                    int send_z_start = (dz == -1) ? 1 : (dz == 0 ? 1 : lz);
                    int send_x_end = (dx == -1 || dx == 1) ? send_x_start : lx;
                    int send_y_end = (dy == -1 || dy == 1) ? send_y_start : ly;
                    int send_z_end = (dz == -1 || dz == 1) ? send_z_start : lz;
                    
                    int block_nx = (dx == 0 ? lx : 1);
                    int block_ny = (dy == 0 ? ly : 1);
                    int block_nz = (dz == 0 ? lz : 1);
                    int block_size = block_nx * block_ny * block_nz;
                    
                    float *sendbuf = (float *)malloc(sizeof(float) * block_size * NC);
                    float *recvbuf = (float *)malloc(sizeof(float) * block_size * NC);
                    
                    int idx = 0;
                    for (int z = send_z_start; z <= send_z_end; z++) {
                        for (int y = send_y_start; y <= send_y_end; y++) {
                            for (int x = send_x_start; x <= send_x_end; x++) {
                                int ext_idx = IDX(x, y, z, ext_nx, ext_ny);
                                for (int c = 0; c < NC; c++) {
                                    sendbuf[idx * NC + c] = ext_data[ext_idx * NC + t];
                                }
                                idx++;
                            }
                        }
                    }
                    
                    MPI_Request send_req, recv_req;
                    MPI_Isend(sendbuf, block_size * NC, MPI_FLOAT, nbr, 100,
                             MPI_COMM_WORLD, &send_req);
                    MPI_Irecv(recvbuf, block_size * NC, MPI_FLOAT, nbr, 100,
                             MPI_COMM_WORLD, &recv_req);
                    
                    // Wait for both operations to complete
                    MPI_Request requests[2] = {send_req, recv_req};
                    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
                    
                    int dest_x = (dx == -1) ? 0 : (dx == 0 ? 1 : lx + 1);
                    int dest_y = (dy == -1) ? 0 : (dy == 0 ? 1 : ly + 1);
                    int dest_z = (dz == -1) ? 0 : (dz == 0 ? 1 : lz + 1);
                    
                    idx = 0;
                    for (int z = 0; z < block_nz; z++) {
                        for (int y = 0; y < block_ny; y++) {
                            for (int x = 0; x < block_nx; x++) {
                                int ex = dest_x + x;
                                int ey = dest_y + y;
                                int ez = dest_z + z;
                                int ext_idx = IDX(ex, ey, ez, ext_nx, ext_ny);
                                for (int c = 0; c < NC; c++) {
                                    ext_data[ext_idx * NC + t] = recvbuf[idx * NC + c];
                                }
                                idx++;
                            }
                        }
                    }
                    
                    free(sendbuf);
                    free(recvbuf);
                }
            }
        }
    }

    // Compute local minima and maxima with 6-neighbor check.
    int *local_min_count = (int *)malloc(sizeof(int) * NC);
    int *local_max_count = (int *)malloc(sizeof(int) * NC);
    float *local_min_vals = (float *)malloc(sizeof(float) * NC);
    float *local_max_vals = (float *)malloc(sizeof(float) * NC);
    
    for (int t = 0; t < NC; t++) {
        local_min_count[t] = 0;
        local_max_count[t] = 0;
        local_min_vals[t] = FLT_MAX;
        local_max_vals[t] = -FLT_MAX;
        for (int z = 1; z <= lz; z++) {
            for (int y = 1; y <= ly; y++) {
                for (int x = 1; x <= lx; x++) {
                    int ext_idx = IDX(x, y, z, ext_nx, ext_ny);
                    float val = ext_data[ext_idx * NC + t];
                    
                    int global_x = offset_x + (x - 1);
                    int global_y = offset_y + (y - 1);
                    int global_z = offset_z + (z - 1);
                    
                    if (val < local_min_vals[t])
                        local_min_vals[t] = val;
                    if (val > local_max_vals[t])
                        local_max_vals[t] = val;
                    
                    int isMin = 1, isMax = 1;
                    int d[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
                    for (int i = 0; i < 6; i++) {
                        int dx = d[i][0], dy = d[i][1], dz = d[i][2];
                        int nb_global_x = global_x + dx;
                        int nb_global_y = global_y + dy;
                        int nb_global_z = global_z + dz;
                        if (nb_global_x < 0 || nb_global_x >= NX ||
                            nb_global_y < 0 || nb_global_y >= NY ||
                            nb_global_z < 0 || nb_global_z >= NZ)
                            continue;
                        int nb_ext_x = x + dx;
                        int nb_ext_y = y + dy;
                        int nb_ext_z = z + dz;
                        int nb_ext_idx = IDX(nb_ext_x, nb_ext_y, nb_ext_z, ext_nx, ext_ny);
                        float nb_val = ext_data[nb_ext_idx * NC + t];
                        if (nb_val <= val)
                            isMin = 0;
                        if (nb_val >= val)
                            isMax = 0;
                    }
                    if (isMin)
                        local_min_count[t]++;
                    if (isMax)
                        local_max_count[t]++;
                }
            }
        }
    }
    
    // Reduce global results.
    int *global_min_count = NULL, *global_max_count = NULL;
    float *global_min_vals = NULL, *global_max_vals = NULL;
    if (rank == 0) {
        global_min_count = (int *)calloc(NC, sizeof(int));
        global_max_count = (int *)calloc(NC, sizeof(int));
        global_min_vals = (float *)malloc(sizeof(float) * NC);
        global_max_vals = (float *)malloc(sizeof(float) * NC);
    }
    
    MPI_Reduce(local_min_count, global_min_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_count, global_max_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min_vals, global_min_vals, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_vals, global_max_vals, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double t3 = MPI_Wtime();
    
    if (rank == 0) {
        FILE *fout = fopen(output_file, "w");
        if (!fout) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < NC; i++) {
            fprintf(fout, "(%d,%d) ", global_min_count[i], global_max_count[i]);
            if (i != NC - 1)
                fprintf(fout, ", ");
        }
        fprintf(fout, "\n");
        for (int i = 0; i < NC; i++) {
            fprintf(fout, "(%.6f,%.6f) ", global_min_vals[i], global_max_vals[i]);
            if (i != NC - 1)
                fprintf(fout, ", ");
        }
        fprintf(fout, "\n");
        fprintf(fout, "%.6f %.6f %.6f\n", t2 - t1, t3 - t2, t3 - t1);
        fclose(fout);
        
        free(global_min_count);
        free(global_max_count);
        free(global_min_vals);
        free(global_max_vals);
    }
    
    free(local_data);
    free(ext_data);
    free(local_min_count);
    free(local_max_count);
    free(local_min_vals);
    free(local_max_vals);
    
    MPI_Finalize();
    return 0;
}