#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// Macro to compute index in a 3D array stored in 1D (C order)
#define IDX(x,y,z,nx,ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

// Read all floating‐point numbers from a file into a buffer.
void read_data(float* data, const char* fname, int total_size, int nc) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open input file %s\n", fname);
        exit(1);
    }
    for (int i = 0; i < total_size * nc; i++) {
        if (fscanf(f, "%f", &data[i]) != 1) {
            fprintf(stderr, "Error reading data from file\n");
            exit(1);
        }
    }
    fclose(f);
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

    const char* input_file = argv[1];
    int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
    int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    const char* output_file = argv[9];
    int total_points = NX * NY * NZ;

    if(PX*PY*PZ != size) {
        if(rank==0)
            printf("Error: PX*PY*PZ (%d) != number of processes (%d)\n", PX*PY*PZ, size);
        MPI_Finalize();
        return 1;
    }

    // Create a Cartesian communicator
    int dims[3] = {PX, PY, PZ};
    int periods[3] = {0,0,0}; // no periodic boundaries
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    
    // Get our Cartesian coordinates.
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    int px = coords[0], py = coords[1], pz = coords[2];

    // Determine the local dimensions.
    int lx = NX / PX + (px < NX % PX ? 1 : 0);
    int ly = NY / PY + (py < NY % PY ? 1 : 0);
    int lz = NZ / PZ + (pz < NZ % PZ ? 1 : 0);

    // Global offsets (for later use when computing global indices)
    int offset_x = (NX / PX) * px + (px < NX % PX ? px : NX % PX);
    int offset_y = (NY / PY) * py + (py < NY % PY ? py : NY % PY);
    int offset_z = (NZ / PZ) * pz + (pz < NZ % PZ ? pz : NZ % PZ);

    if(rank==0)
        printf("Problem Parameters: NX=%d, NY=%d, NZ=%d, NC=%d, PX=%d, PY=%d, PZ=%d\n",
               NX, NY, NZ, NC, PX, PY, PZ);

    // Rank 0 reads full data.
    float *full_data = NULL;
    if(rank == 0) {
        full_data = (float*)malloc(sizeof(float) * total_points * NC);
        read_data(full_data, input_file, total_points, NC);
    }
    
    // Allocate local_data buffer for this process.
    int local_size = lx * ly * lz;
    float *local_data = (float*)malloc(sizeof(float) * local_size * NC);

    // Manual distribution of data.
    for (int t = 0; t < NC; t++) {
        if(rank == 0) {
            for (int p = 0; p < size; p++) {
                int pcoords[3];
                MPI_Cart_coords(cart_comm, p, 3, pcoords);
                int p_px = pcoords[0], p_py = pcoords[1], p_pz = pcoords[2];
                int p_lx = NX / PX + (p_px < NX % PX ? 1 : 0);
                int p_ly = NY / PY + (p_py < NY % PY ? 1 : 0);
                int p_lz = NZ / PZ + (p_pz < NZ % PZ ? 1 : 0);
                int p_offset_x = (NX / PX) * p_px + (p_px < NX % PX ? p_px : NX % PX);
                int p_offset_y = (NY / PY) * p_py + (p_py < NY % PY ? p_py : NY % PY);
                int p_offset_z = (NZ / PZ) * p_pz + (p_pz < NZ % PZ ? p_pz : NZ % PZ);
                
                for (int z = 0; z < p_lz; z++) {
                    for (int y = 0; y < p_ly; y++) {
                        for (int x = 0; x < p_lx; x++) {
                            int global_idx = IDX(p_offset_x + x, p_offset_y + y, p_offset_z + z, NX, NY);
                            int local_idx = IDX(x, y, z, p_lx, p_ly);
                            if(p == 0)
                                local_data[local_idx * NC + t] = full_data[global_idx * NC + t];
                            else
                                MPI_Send(&full_data[global_idx * NC + t], 1, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }
        } else {
            for (int z = 0; z < lz; z++) {
                for (int y = 0; y < ly; y++) {
                    for (int x = 0; x < lx; x++) {
                        int local_idx = IDX(x,y,z,lx,ly);
                        MPI_Recv(&local_data[local_idx * NC + t], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
    }
    if(rank==0)
        free(full_data);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Allocate an extended buffer that includes a one-cell halo on all sides.
    // Dimensions: (lx+2) x (ly+2) x (lz+2)
    int ext_nx = lx + 2;
    int ext_ny = ly + 2;
    int ext_nz = lz + 2;
    int ext_size = ext_nx * ext_ny * ext_nz;
    float *ext_data = (float*)malloc(sizeof(float) * ext_size * NC);
    
    // Initialize ext_data to zero.
    for (int i = 0; i < ext_size * NC; i++) {
        ext_data[i] = 0.0;
    }
    
    // Copy local_data into the center of ext_data.
    // The interior of ext_data spans indices [1..lx] x [1..ly] x [1..lz]
    for (int t = 0; t < NC; t++) {
        for (int z = 0; z < lz; z++) {
            for (int y = 0; y < ly; y++) {
                for (int x = 0; x < lx; x++) {
                    int local_idx = IDX(x, y, z, lx, ly);
                    int ext_idx = IDX(x+1, y+1, z+1, ext_nx, ext_ny);
                    ext_data[ext_idx * NC + t] = local_data[local_idx * NC + t];
                }
            }
        }
    }
    
    /* 
       Perform ghost exchange for the full halo.
       We loop over all 26 neighbor offsets. For each neighbor that exists, we pack the corresponding
       block from the interior of ext_data, exchange it with that neighbor, and then place the received data
       into the appropriate halo region in ext_data.
    */
    for (int t = 0; t < NC; t++) {
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if(dx==0 && dy==0 && dz==0)
                        continue;
                    
                    int nbr_coords[3] = {px+dx, py+dy, pz+dz};
                    if(nbr_coords[0] < 0 || nbr_coords[0] >= dims[0] ||
                       nbr_coords[1] < 0 || nbr_coords[1] >= dims[1] ||
                       nbr_coords[2] < 0 || nbr_coords[2] >= dims[2]) {
                        continue; // at global boundary, no exchange
                    }
                    int nbr;
                    MPI_Cart_rank(cart_comm, nbr_coords, &nbr);
                    
                    // Determine block to send (from the interior region of ext_data)
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
                    
                    float *sendbuf = (float*)malloc(sizeof(float) * block_size * NC);
                    float *recvbuf = (float*)malloc(sizeof(float) * block_size * NC);
                    
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
                    
                    MPI_Sendrecv(sendbuf, block_size*NC, MPI_FLOAT, nbr, 100,
                                 recvbuf, block_size*NC, MPI_FLOAT, nbr, 100,
                                 cart_comm, MPI_STATUS_IGNORE);
                    
                    // Determine where to place received data in ext_data.
                    int dest_x = (dx == -1) ? 0 : (dx == 0 ? 1 : lx+1);
                    int dest_y = (dy == -1) ? 0 : (dy == 0 ? 1 : ly+1);
                    int dest_z = (dz == -1) ? 0 : (dz == 0 ? 1 : lz+1);
                    
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
    
    /*
       Compute local minima and maxima (with 6-neighbor check) for all interior cells
       and now also include cells at the boundary of the local subdomain.
       For each cell in the interior of ext_data (indices 1..lx, 1..ly, 1..lz),
       we compute its global coordinates and then check each of the 6 neighbors.
       For each neighbor, if the neighbor’s global coordinate is outside the overall global domain,
       that neighbor is skipped.
    */
    int *local_min_count = (int*)malloc(sizeof(int) * NC);
    int *local_max_count = (int*)malloc(sizeof(int) * NC);
    float *local_min_vals = (float*)malloc(sizeof(float) * NC);
    float *local_max_vals = (float*)malloc(sizeof(float) * NC);
    
    for (int t = 0; t < NC; t++) {
        local_min_count[t] = 0;
        local_max_count[t] = 0;
        local_min_vals[t] = FLT_MAX;
        local_max_vals[t] = -FLT_MAX;
        // Loop over all cells in our local subdomain (stored in ext_data at indices 1..lx, 1..ly, 1..lz)
        for (int z = 1; z <= lz; z++) {
            for (int y = 1; y <= ly; y++) {
                for (int x = 1; x <= lx; x++) {
                    int ext_idx = IDX(x, y, z, ext_nx, ext_ny);
                    float val = ext_data[ext_idx * NC + t];
                    
                    // Compute the global coordinates of this cell.
                    int global_x = offset_x + (x - 1);
                    int global_y = offset_y + (y - 1);
                    int global_z = offset_z + (z - 1);
                    
                    // Update our local extreme values.
                    if(val < local_min_vals[t]) local_min_vals[t] = val;
                    if(val > local_max_vals[t]) local_max_vals[t] = val;
                    
                    int isMin = 1, isMax = 1;
                    // Define the 6 neighbor offsets.
                    int d[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
                    for (int i = 0; i < 6; i++) {
                        int dx = d[i][0], dy = d[i][1], dz = d[i][2];
                        int nb_global_x = global_x + dx;
                        int nb_global_y = global_y + dy;
                        int nb_global_z = global_z + dz;
                        // Check if neighbor exists in the global domain.
                        if(nb_global_x < 0 || nb_global_x >= NX ||
                           nb_global_y < 0 || nb_global_y >= NY ||
                           nb_global_z < 0 || nb_global_z >= NZ)
                        {
                            continue; // Skip comparison if neighbor is out-of-bound.
                        }
                        // Compute the neighbor's index in the extended array.
                        // Since interior cell (global) corresponds to ext_data index = local index + 1:
                        int nb_ext_x = x + dx;
                        int nb_ext_y = y + dy;
                        int nb_ext_z = z + dz;
                        int nb_ext_idx = IDX(nb_ext_x, nb_ext_y, nb_ext_z, ext_nx, ext_ny);
                        float nb_val = ext_data[nb_ext_idx * NC + t];
                        if(nb_val < val) isMin = 0;
                        if(nb_val > val) isMax = 0;
                    }
                    if(isMin) local_min_count[t]++;
                    if(isMax) local_max_count[t]++;
                }
            }
        }
    }
    
    // Reduce global results.
    int *global_min_count = NULL, *global_max_count = NULL;
    float *global_min_vals = NULL, *global_max_vals = NULL;
    if(rank == 0) {
        global_min_count = (int*)calloc(NC, sizeof(int));
        global_max_count = (int*)calloc(NC, sizeof(int));
        global_min_vals = (float*)malloc(sizeof(float) * NC);
        global_max_vals = (float*)malloc(sizeof(float) * NC);
    }
    
    MPI_Reduce(local_min_count, global_min_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_count, global_max_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min_vals, global_min_vals, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_vals, global_max_vals, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double t3 = MPI_Wtime();
    
    if(rank==0) {
        FILE *fout = fopen(output_file, "w");
        if(!fout) {
            printf("Error: Could not open output file %s\n", output_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for(int i = 0; i<NC; i++)
        {
            fprintf(fout, "(%d,%d) ", global_min_count[i], global_max_count[i]);
            if(i!=NC-1)
                fprintf(fout, ", ");
        }
        fprintf(fout, "\n");
        
        for(int i = 0; i<NC; i++)
        {
            fprintf(fout, "(%.6f,%.6f) ", global_min_vals[i], global_max_vals[i]);
            if(i!=NC-1)
                fprintf(fout, ", ");
        }
        fprintf(fout, "\n");
        fprintf(fout, "%.6f %.6f %.6f\n", t2-t1, t3-t2, t3-t1);
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
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
