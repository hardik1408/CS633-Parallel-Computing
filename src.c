#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#define MAX_FILENAME 100
#define MAX_LINE_LENGTH 100000
#define MAX_LINE_LEN 10000

// Macro to compute index in a 3D array stored in 1D (C order)
#define compute_index(x,y,z,nx,ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

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

// Function to process interior points (does not depend on halo data)
void process_interior_points(float *ext_data, int ext_nx, int ext_ny, int lx, int ly, int lz, 
                          int NC, int offset_x, int offset_y, int offset_z,
                          int *local_min_count, int *local_max_count, 
                          float *local_min_vals, float *local_max_vals) {
    // Define interior region (exclude outermost layer that depends on halo)
    int start_x = 2;
    int start_y = 2;
    int start_z = 2;
    int end_x = lx;
    int end_y = ly;
    int end_z = lz;
    
    // Define the 6 neighbor directions (face-adjacent neighbors)
    int d[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
    
    // Process interior points
    for (int z = start_z; z < end_z; z++) {
        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {
                int ext_idx = compute_index(x, y, z, ext_nx, ext_ny);
                
                // Global coordinates for this point
                int global_x = offset_x + (x - 1);
                int global_y = offset_y + (y - 1);
                int global_z = offset_z + (z - 1);
                
                for (int t = 0; t < NC; t++) {
                    float val = ext_data[ext_idx * NC + t];
                    
                    // Update local min/max values
                    local_min_vals[t] = fminf(local_min_vals[t], val);
                    local_max_vals[t] = fmaxf(local_max_vals[t], val);
                    
                    // Check if this is a local minimum or maximum
                    int isMin = 1, isMax = 1;
                    
                    for (int i = 0; i < 6; i++) {
                        int dx = d[i][0], dy = d[i][1], dz = d[i][2];
                        
                        // Get the neighbor's value
                        int nb_ext_x = x + dx;
                        int nb_ext_y = y + dy;
                        int nb_ext_z = z + dz;
                        int nb_ext_idx = compute_index(nb_ext_x, nb_ext_y, nb_ext_z, ext_nx, ext_ny);
                        float nb_val = ext_data[nb_ext_idx * NC + t];
                        
                        // Check if this invalidates min/max status
                        if (nb_val <= val)
                            isMin = 0;
                        if (nb_val >= val)
                            isMax = 0;
                    }
                    
                    // Update counts if local min/max
                    if (isMin)
                        local_min_count[t]++;
                    if (isMax)
                        local_max_count[t]++;
                }
            }
        }
    }
}

// Function to process boundary points (depends on halo data)
void process_boundary_points(float *ext_data, int ext_nx, int ext_ny, int lx, int ly, int lz, 
                          int NC, int NX, int NY, int NZ, int offset_x, int offset_y, int offset_z,
                          int *local_min_count, int *local_max_count, 
                          float *local_min_vals, float *local_max_vals) {
    // Define the 6 neighbor directions (face-adjacent neighbors)
    int d[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
    
    // Process all boundary points (outermost layer)
    for (int z = 1; z <= lz; z++) {
        for (int y = 1; y <= ly; y++) {
            for (int x = 1; x <= lx; x++) {
                // Skip interior points which were already processed
                if (x > 1 && x < lx && y > 1 && y < ly && z > 1 && z < lz)
                    continue;
                    
                int ext_idx = compute_index(x, y, z, ext_nx, ext_ny);
                
                // Global coordinates for this point
                int global_x = offset_x + (x - 1);
                int global_y = offset_y + (y - 1);
                int global_z = offset_z + (z - 1);
                
                for (int t = 0; t < NC; t++) {
                    float val = ext_data[ext_idx * NC + t];
                    
                    // Update local min/max values
                    local_min_vals[t] = fminf(local_min_vals[t], val);
                    local_max_vals[t] = fmaxf(local_max_vals[t], val);
                    
                    // Check if this is a local minimum or maximum
                    int isMin = 1, isMax = 1;
                    
                    for (int i = 0; i < 6; i++) {
                        int dx = d[i][0], dy = d[i][1], dz = d[i][2];
                        
                        // Global coordinates of neighbor
                        int nb_global_x = global_x + dx;
                        int nb_global_y = global_y + dy;
                        int nb_global_z = global_z + dz;
                        
                        // Skip neighbors outside the global domain
                        if (nb_global_x < 0 || nb_global_x >= NX ||
                            nb_global_y < 0 || nb_global_y >= NY ||
                            nb_global_z < 0 || nb_global_z >= NZ)
                            continue;
                        
                        // Get the neighbor's value
                        int nb_ext_x = x + dx;
                        int nb_ext_y = y + dy;
                        int nb_ext_z = z + dz;
                        int nb_ext_idx = compute_index(nb_ext_x, nb_ext_y, nb_ext_z, ext_nx, ext_ny);
                        float nb_val = ext_data[nb_ext_idx * NC + t];
                        
                        // Check if this invalidates min/max status
                        if (nb_val <= val)
                            isMin = 0;
                        if (nb_val >= val)
                            isMax = 0;
                    }
                    
                    // Update counts if local min/max
                    if (isMin)
                        local_min_count[t]++;
                    if (isMax)
                        local_max_count[t]++;
                }
            }
        }
    }
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

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

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
    // Initialize with zeros more efficiently
    memset(ext_data, 0, sizeof(float) * ext_size * NC);

    // Copy local_data into the center of ext_data.
    for (int t = 0; t < NC; t++) {
        for (int z = 0; z < lz; z++) {
            for (int y = 0; y < ly; y++) {
                for (int x = 0; x < lx; x++) {
                    int local_idx = compute_index(x, y, z, lx, ly);
                    int ext_idx = compute_index(x + 1, y + 1, z + 1, ext_nx, ext_ny);
                    ext_data[ext_idx * NC + t] = local_data[local_idx * NC + t];
                }
            }
        }
    }

    // Store neighbor ranks for each direction (left, right, down, up, back, front)
    int neighbors[6];
    neighbors[0] = get_rank_from_coords(px-1, py, pz, PX, PY, PZ); // -X
    neighbors[1] = get_rank_from_coords(px+1, py, pz, PX, PY, PZ); // +X
    neighbors[2] = get_rank_from_coords(px, py-1, pz, PX, PY, PZ); // -Y
    neighbors[3] = get_rank_from_coords(px, py+1, pz, PX, PY, PZ); // +Y
    neighbors[4] = get_rank_from_coords(px, py, pz-1, PX, PY, PZ); // -Z
    neighbors[5] = get_rank_from_coords(px, py, pz+1, PX, PY, PZ); // +Z

    // Calculate buffer sizes for each direction
    int bufsize_x = ly * lz * NC;
    int bufsize_y = ext_nx * lz * NC;
    int bufsize_z = ext_nx * ext_ny * NC;

    // Allocate send and receive buffers for all directions
    float *sendbuf[6], *recvbuf[6];
    for (int i = 0; i < 6; i++) {
        int bufsize = (i < 2) ? bufsize_x : ((i < 4) ? bufsize_y : bufsize_z);
        sendbuf[i] = (float*)malloc(bufsize * sizeof(float));
        recvbuf[i] = (float*)malloc(bufsize * sizeof(float));
    }

    MPI_Request reqs[12]; // For non-blocking communication
    int req_idx = 0;

    // Start all communications at once

    // X-direction communication
    for (int dir = 0; dir < 2; dir++) {
        if (neighbors[dir] == -1) continue;
        
        int x = (dir == 0) ? 1 : lx; // Left or right face
        int idx = 0;
        
        // Pack the data
        for (int z = 1; z <= lz; z++) {
            for (int y = 1; y <= ly; y++) {
                for (int t = 0; t < NC; t++) {
                    sendbuf[dir][idx++] = ext_data[compute_index(x, y, z, ext_nx, ext_ny) * NC + t];
                }
            }
        }
        
        // Send and receive
        MPI_Isend(sendbuf[dir], bufsize_x, MPI_FLOAT, neighbors[dir], 0, MPI_COMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(recvbuf[dir], bufsize_x, MPI_FLOAT, neighbors[dir], 0, MPI_COMM_WORLD, &reqs[req_idx++]);
    }

    // Y-direction communication
    for (int dir = 2; dir < 4; dir++) {
        if (neighbors[dir] == -1) continue;
        
        int y = (dir == 2) ? 1 : ly; // Bottom or top face
        int idx = 0;
        
        // Pack the data (including X halos)
        for (int z = 1; z <= lz; z++) {
            for (int x = 0; x <= lx+1; x++) { // Include X halos
                for (int t = 0; t < NC; t++) {
                    sendbuf[dir][idx++] = ext_data[compute_index(x, y, z, ext_nx, ext_ny) * NC + t];
                }
            }
        }
        
        // Send and receive
        MPI_Isend(sendbuf[dir], bufsize_y, MPI_FLOAT, neighbors[dir], 0, MPI_COMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(recvbuf[dir], bufsize_y, MPI_FLOAT, neighbors[dir], 0, MPI_COMM_WORLD, &reqs[req_idx++]);
    }

    // Z-direction communication
    for (int dir = 4; dir < 6; dir++) {
        if (neighbors[dir] == -1) continue;
        
        int z = (dir == 4) ? 1 : lz; // Back or front face
        int idx = 0;
        
        // Pack the data (including X and Y halos)
        for (int y = 0; y <= ly+1; y++) { // Include Y halos
            for (int x = 0; x <= lx+1; x++) { // Include X halos
                for (int t = 0; t < NC; t++) {
                    sendbuf[dir][idx++] = ext_data[compute_index(x, y, z, ext_nx, ext_ny) * NC + t];
                }
            }
        }
        
        // Send and receive
        MPI_Isend(sendbuf[dir], bufsize_z, MPI_FLOAT, neighbors[dir], 0, MPI_COMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(recvbuf[dir], bufsize_z, MPI_FLOAT, neighbors[dir], 0, MPI_COMM_WORLD, &reqs[req_idx++]);
    }

    // Initialize arrays for tracking minima and maxima
    int *local_min_count = (int *)calloc(NC, sizeof(int));
    int *local_max_count = (int *)calloc(NC, sizeof(int));
    float *local_min_vals = (float *)malloc(NC * sizeof(float));
    float *local_max_vals = (float *)malloc(NC * sizeof(float));
    
    for (int t = 0; t < NC; t++) {
        local_min_vals[t] = FLT_MAX;
        local_max_vals[t] = -FLT_MAX;
    }
    
    // OVERLAP: Process interior points while communication is happening
    if (lx > 2 && ly > 2 && lz > 2) {  // Only if there are interior points
        process_interior_points(ext_data, ext_nx, ext_ny, lx, ly, lz, NC, 
                           offset_x, offset_y, offset_z,
                           local_min_count, local_max_count, 
                           local_min_vals, local_max_vals);
    }
    
    // Wait for all communications to complete
    MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);
    
    // Unpack all halos
    
    // Unpack X-direction halos
    for (int dir = 0; dir < 2; dir++) {
        if (neighbors[dir] == -1) continue;
        
        int x = (dir == 0) ? 0 : lx+1; // Left or right halo
        int idx = 0;
        
        for (int z = 1; z <= lz; z++) {
            for (int y = 1; y <= ly; y++) {
                for (int t = 0; t < NC; t++) {
                    ext_data[compute_index(x, y, z, ext_nx, ext_ny) * NC + t] = recvbuf[dir][idx++];
                }
            }
        }
    }

    // Unpack Y-direction halos
    for (int dir = 2; dir < 4; dir++) {
        if (neighbors[dir] == -1) continue;
        
        int y = (dir == 2) ? 0 : ly+1; // Bottom or top halo
        int idx = 0;
        
        for (int z = 1; z <= lz; z++) {
            for (int x = 0; x <= lx+1; x++) { // Include X halos
                for (int t = 0; t < NC; t++) {
                    ext_data[compute_index(x, y, z, ext_nx, ext_ny) * NC + t] = recvbuf[dir][idx++];
                }
            }
        }
    }

    // Unpack Z-direction halos
    for (int dir = 4; dir < 6; dir++) {
        if (neighbors[dir] == -1) continue;
        
        int z = (dir == 4) ? 0 : lz+1; // Back or front halo
        int idx = 0;
        
        for (int y = 0; y <= ly+1; y++) { // Include Y halos
            for (int x = 0; x <= lx+1; x++) { // Include X halos
                for (int t = 0; t < NC; t++) {
                    ext_data[compute_index(x, y, z, ext_nx, ext_ny) * NC + t] = recvbuf[dir][idx++];
                }
            }
        }
    }

    // Now process boundary points that depend on halo data
    process_boundary_points(ext_data, ext_nx, ext_ny, lx, ly, lz, NC, NX, NY, NZ,
                        offset_x, offset_y, offset_z,
                        local_min_count, local_max_count, 
                        local_min_vals, local_max_vals);

    // Free communication buffers
    for (int i = 0; i < 6; i++) {
        free(sendbuf[i]);
        free(recvbuf[i]);
    }
    
    // Reduce results across all processes
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
    
    // Output results
    if (rank == 0) {
        FILE *fout = fopen(output_file, "w");
        if (!fout) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        for (int i = 0; i < NC; i++) {
            fprintf(fout, "(%d,%d)%s", global_min_count[i], global_max_count[i], 
                   (i < NC-1) ? " , " : "\n");
        }
        
        for (int i = 0; i < NC; i++) {
            fprintf(fout, "(%.6f,%.6f)%s", global_min_vals[i], global_max_vals[i], 
                   (i < NC-1) ? " , " : "\n");
        }
        
        fprintf(fout, "%.6f %.6f %.6f\n", t2 - t1, t3 - t2, t3 - t1);
        fclose(fout);
        
        free(global_min_count);
        free(global_max_count);
        free(global_min_vals);
        free(global_max_vals);
    }
    // Clean up
    free(local_data);
    free(ext_data);
    free(local_min_count);
    free(local_max_count);
    free(local_min_vals);
    free(local_max_vals);
    
    MPI_Finalize();
    return 0;
}
