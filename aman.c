#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define MAX_LINE_LEN 10000
// Macro to compute index in a 3D array stored in 1D (C order)
#define IDX(x,y,z,nx,ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 10) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s input_file PX PY PZ NX NY NZ NC output_file\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    // Parse arguments.
    char *input_file = argv[1];
    int PX = atoi(argv[2]);
    int PY = atoi(argv[3]);
    int PZ = atoi(argv[4]);
    int NX = atoi(argv[5]);
    int NY = atoi(argv[6]);
    int NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    char *output_file = argv[9];
    
    // Create Cartesian communicator.
    int dims[3] = {PX, PY, PZ};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    
    int P = PX * PY * PZ;
    if (P != size) {
        if (rank == 0)
            fprintf(stderr, "Error: PX*PY*PZ must equal number of MPI processes\n");
        MPI_Finalize();
        return 1;
    }
    
    double t1 = MPI_Wtime();
    
    // Global number of points.
    int total_points = NX * NY * NZ;
    
    // Determine local dimensions using load-balancing in each direction.
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    int px = coords[0], py = coords[1], pz = coords[2];
    
    int lx = NX / PX + (px < (NX % PX) ? 1 : 0);
    int ly = NY / PY + (py < (NY % PY) ? 1 : 0);
    int lz = NZ / PZ + (pz < (NZ % PZ) ? 1 : 0);
    
    // Global offsets (for mapping local cells to global indices)
    int offset_x = (NX / PX) * px + (px < (NX % PX) ? px : (NX % PX));
    int offset_y = (NY / PY) * py + (py < (NY % PY) ? py : (NY % PY));
    int offset_z = (NZ / PZ) * pz + (pz < (NZ % PZ) ? pz : (NZ % PZ));
    
    // -----------------------
    // Reading and distribution using MPI-IO.
    // Each line in the input file corresponds to one global data point.
    // Process 0 computes the offset and length of each line.
    int *line_offsets = NULL;
    int *line_lengths = NULL;
    if (rank == 0) {
        line_offsets = (int *)malloc(total_points * sizeof(int));
        line_lengths = (int *)malloc(total_points * sizeof(int));
        FILE *f = fopen(input_file, "r");
        if (!f) {
            fprintf(stderr, "Process 0: Cannot open input file %s\n", input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int pos = 0;
        char buf[MAX_LINE_LEN];
        for (int i = 0; i < total_points; i++) {
            line_offsets[i] = pos;
            if (fgets(buf, MAX_LINE_LEN, f)) {
                int len = strlen(buf);
                pos += len;
                line_lengths[i] = len;
            } else {
                fprintf(stderr, "Process 0: Unexpected EOF at line %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(f);
    }
    // Broadcast total_points to all processes.
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate on non-root processes.
    if (rank != 0) {
        line_offsets = (int *)malloc(total_points * sizeof(int));
        line_lengths = (int *)malloc(total_points * sizeof(int));
    }
    MPI_Bcast(line_offsets, total_points, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(line_lengths, total_points, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Each process computes which global indices belong to its local subdomain.
    // Here we assume the input file ordering corresponds to row-major order:
    // global index = x + y*NX + z*NX*NY.
    int local_count = lx * ly * lz;
    int *local_line_indices = (int *)malloc(local_count * sizeof(int));
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
    
    // Each process opens the file using MPI I/O and reads its assigned lines.
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    
    // Allocate a raw buffer to hold each line. We use MAX_LINE_LEN per line.
    char *raw_buf = (char *)malloc(local_count * MAX_LINE_LEN);
    // Buffer to store parsed floating-point values.
    float *local_data = (float *)malloc(local_count * NC * sizeof(float));
    
    for (int i = 0; i < local_count; i++) {
        int gi = local_line_indices[i];
        // Read the exact number of bytes for this line.
        MPI_File_read_at(fh, line_offsets[gi], raw_buf + i * MAX_LINE_LEN,
                         line_lengths[gi], MPI_CHAR, MPI_STATUS_IGNORE);
        // Ensure null termination.
        raw_buf[i * MAX_LINE_LEN + line_lengths[gi]] = '\0';
    }
    MPI_File_close(&fh);
    free(local_line_indices);
    free(line_offsets);
    free(line_lengths);
    
    // Parse each line to extract NC floats.
    for (int i = 0; i < local_count; i++) {
        char *ptr = raw_buf + i * MAX_LINE_LEN;
        for (int j = 0; j < NC; j++) {
            while (*ptr == ' ' || *ptr == '\t')
                ptr++;
            if (sscanf(ptr, "%f", &local_data[i * NC + j]) != 1) {
                fprintf(stderr, "Process %d: Failed to parse float %d in line %d\n", rank, j, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n')
                ptr++;
        }
    }
    free(raw_buf);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    
    // -----------------------
    // Now, continue with the rest of the computation.
    // local_data now holds the NC values for each cell in this process’s subdomain.
    
    // Create an extended buffer with a one-cell halo on all sides.
    int ext_nx = lx + 2;
    int ext_ny = ly + 2;
    int ext_nz = lz + 2;
    int ext_size = ext_nx * ext_ny * ext_nz;
    float *ext_data = (float *)malloc(ext_size * NC * sizeof(float));
    // Initialize extended buffer to zero.
    for (int i = 0; i < ext_size * NC; i++) {
        ext_data[i] = 0.0f;
    }
    
    // Copy local_data into the center of ext_data.
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
       Loop over all 26 neighbor offsets. For each neighbor, pack the corresponding
       block from ext_data, exchange with the neighbor, and place received data into the halo.
    */
    for (int t = 0; t < NC; t++) {
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx==0 && dy==0 && dz==0)
                        continue;
                    
                    int nbr_coords[3] = {px+dx, py+dy, pz+dz};
                    if (nbr_coords[0] < 0 || nbr_coords[0] >= dims[0] ||
                        nbr_coords[1] < 0 || nbr_coords[1] >= dims[1] ||
                        nbr_coords[2] < 0 || nbr_coords[2] >= dims[2]) {
                        continue; // Skip if neighbor is outside global domain.
                    }
                    int nbr;
                    MPI_Cart_rank(cart_comm, nbr_coords, &nbr);
                    
                    // Determine block boundaries in ext_data to send.
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
                    
                    float *sendbuf = (float *)malloc(block_size * NC * sizeof(float));
                    float *recvbuf = (float *)malloc(block_size * NC * sizeof(float));
                    
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
    
    // Compute local minima and maxima (with 6-neighbor check) for each channel.
    int *local_min_count = (int *)malloc(NC * sizeof(int));
    int *local_max_count = (int *)malloc(NC * sizeof(int));
    float *local_min_vals = (float *)malloc(NC * sizeof(float));
    float *local_max_vals = (float *)malloc(NC * sizeof(float));
    
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
                    
                    // Global coordinates of the cell.
                    int global_x = offset_x + (x - 1);
                    int global_y = offset_y + (y - 1);
                    int global_z = offset_z + (z - 1);
                    
                    if (val < local_min_vals[t]) local_min_vals[t] = val;
                    if (val > local_max_vals[t]) local_max_vals[t] = val;
                    
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
                        if (nb_val <= val) isMin = 0;
                        if (nb_val >= val) isMax = 0;
                    }
                    if (isMin) local_min_count[t]++;
                    if (isMax) local_max_count[t]++;
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
        global_min_vals = (float *)malloc(NC * sizeof(float));
        global_max_vals = (float *)malloc(NC * sizeof(float));
    }
    
    MPI_Reduce(local_min_count, global_min_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_count, global_max_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min_vals, global_min_vals, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_vals, global_max_vals, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double t3 = MPI_Wtime();
    
    if (rank == 0) {
        FILE *fout = fopen(output_file, "w");
        if (!fout) {
            fprintf(stderr, "Error: Could not open output file %s\n", output_file);
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
