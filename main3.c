#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

// Macro to calculate the index in a 3D array stored in 1D
#define IDX(x, y, z, nx, ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

// Function to read data from input file
void read_data(float* data, const char* fname, int total_size, int nc) {
    FILE* f = fopen(fname, "r");
    if (f == NULL) {
        printf("Error: Could not open input file %s\n", fname);
        exit(1);
    }
    for (int i = 0; i < total_size * nc; ++i) {
        if (fscanf(f, "%f", &data[i]) != 1) {
            printf("Error reading data from file\n");
            exit(1);
        }
    }
    fclose(f);
}

// Calculate neighboring process ranks
int get_neighbor_rank(int px, int py, int pz, int dx, int dy, int dz, int PX, int PY, int PZ) {
    int nx = px + dx;
    int ny = py + dy;
    int nz = pz + dz;
    
    // Check if neighbor exists
    if (nx < 0 || nx >= PX || ny < 0 || ny >= PY || nz < 0 || nz >= PZ) {
        return -1;  // No neighbor
    }
    
    return nz * (PX * PY) + ny * PX + nx;
}

// Function to check if a point is a local minimum in its neighborhood
int is_local_min(float* data, float* ghost_data, int x, int y, int z, 
                int lx, int ly, int lz, int t, int nc,
                int global_x, int global_y, int global_z, int NX, int NY, int NZ) {
    
    float val = data[((z * ly + y) * lx + x) * nc + t];
    
    // Check all 26 neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                int global_nx = global_x + dx;
                int global_ny = global_y + dy;
                int global_nz = global_z + dz;
                
                // Skip if outside global domain
                if (global_nx < 0 || global_nx >= NX || 
                    global_ny < 0 || global_ny >= NY || 
                    global_nz < 0 || global_nz >= NZ) {
                    continue;
                }
                
                float neighbor_val;
                
                // Check if neighbor is within local domain
                if (nx >= 0 && nx < lx && ny >= 0 && ny < ly && nz >= 0 && nz < lz) {
                    neighbor_val = data[((nz * ly + ny) * lx + nx) * nc + t];
                }
                // Otherwise, it's in a ghost cell
                else {
                    // Determine which ghost region
                    int ghost_idx = -1;
                    
                    // Left face (x = -1)
                    if (nx == -1 && ny >= 0 && ny < ly && nz >= 0 && nz < lz) {
                        ghost_idx = 0 * (ly * lz) + nz * ly + ny;
                    }
                    // Right face (x = lx)
                    else if (nx == lx && ny >= 0 && ny < ly && nz >= 0 && nz < lz) {
                        ghost_idx = 1 * (ly * lz) + nz * ly + ny;
                    }
                    // Bottom face (y = -1)
                    else if (nx >= 0 && nx < lx && ny == -1 && nz >= 0 && nz < lz) {
                        ghost_idx = 2 * (lx * lz) + nz * lx + nx;
                    }
                    // Top face (y = ly)
                    else if (nx >= 0 && nx < lx && ny == ly && nz >= 0 && nz < lz) {
                        ghost_idx = 3 * (lx * lz) + nz * lx + nx;
                    }
                    // Front face (z = -1)
                    else if (nx >= 0 && nx < lx && ny >= 0 && ny < ly && nz == -1) {
                        ghost_idx = 4 * (lx * ly) + ny * lx + nx;
                    }
                    // Back face (z = lz)
                    else if (nx >= 0 && nx < lx && ny >= 0 && ny < ly && nz == lz) {
                        ghost_idx = 5 * (lx * ly) + ny * lx + nx;
                    }
                    // Corner or edge case - skip
                    else {
                        continue;
                    }
                    
                    neighbor_val = ghost_data[ghost_idx * nc + t];
                }
                
                // If any neighbor is smaller or equal, not a minimum
                if (neighbor_val <= val) {
                    return 0;
                }
            }
        }
    }
    
    return 1;
}

// Function to check if a point is a local maximum in its neighborhood
int is_local_max(float* data, float* ghost_data, int x, int y, int z, 
                int lx, int ly, int lz, int t, int nc,
                int global_x, int global_y, int global_z, int NX, int NY, int NZ) {
    
    float val = data[((z * ly + y) * lx + x) * nc + t];
    
    // Check all 26 neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                int global_nx = global_x + dx;
                int global_ny = global_y + dy;
                int global_nz = global_z + dz;
                
                // Skip if outside global domain
                if (global_nx < 0 || global_nx >= NX || 
                    global_ny < 0 || global_ny >= NY || 
                    global_nz < 0 || global_nz >= NZ) {
                    continue;
                }
                
                float neighbor_val;
                
                // Check if neighbor is within local domain
                if (nx >= 0 && nx < lx && ny >= 0 && ny < ly && nz >= 0 && nz < lz) {
                    neighbor_val = data[((nz * ly + ny) * lx + nx) * nc + t];
                }
                // Otherwise, it's in a ghost cell
                else {
                    // Determine which ghost region
                    int ghost_idx = -1;
                    
                    // Left face (x = -1)
                    if (nx == -1 && ny >= 0 && ny < ly && nz >= 0 && nz < lz) {
                        ghost_idx = 0 * (ly * lz) + nz * ly + ny;
                    }
                    // Right face (x = lx)
                    else if (nx == lx && ny >= 0 && ny < ly && nz >= 0 && nz < lz) {
                        ghost_idx = 1 * (ly * lz) + nz * ly + ny;
                    }
                    // Bottom face (y = -1)
                    else if (nx >= 0 && nx < lx && ny == -1 && nz >= 0 && nz < lz) {
                        ghost_idx = 2 * (lx * lz) + nz * lx + nx;
                    }
                    // Top face (y = ly)
                    else if (nx >= 0 && nx < lx && ny == ly && nz >= 0 && nz < lz) {
                        ghost_idx = 3 * (lx * lz) + nz * lx + nx;
                    }
                    // Front face (z = -1)
                    else if (nx >= 0 && nx < lx && ny >= 0 && ny < ly && nz == -1) {
                        ghost_idx = 4 * (lx * ly) + ny * lx + nx;
                    }
                    // Back face (z = lz)
                    else if (nx >= 0 && nx < lx && ny >= 0 && ny < ly && nz == lz) {
                        ghost_idx = 5 * (lx * ly) + ny * lx + nx;
                    }
                    // Corner or edge case - skip
                    else {
                        continue;
                    }
                    
                    neighbor_val = ghost_data[ghost_idx * nc + t];
                }
                
                // If any neighbor is larger or equal, not a maximum
                if (neighbor_val >= val) {
                    return 0;
                }
            }
        }
    }
    
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Start timing
    double t1 = MPI_Wtime();

    if (argc != 10) {
        if (rank == 0) printf("Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char* input_file = argv[1];
    int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
    int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    const char* output_file = argv[9];
    int total_points = NX * NY * NZ;

    if (PX * PY * PZ != size) {
        if (rank == 0) printf("Error: PX*PY*PZ (%d) != number of processes (%d)\n", PX * PY * PZ, size);
        MPI_Finalize();
        return 1;
    }

    // Calculate process coordinates in 3D grid
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    // Calculate local domain size for this process
    int lx = NX / PX + (px < NX % PX ? 1 : 0);
    int ly = NY / PY + (py < NY % PY ? 1 : 0);
    int lz = NZ / PZ + (pz < NZ % PZ ? 1 : 0);

    // Calculate offset in global domain
    int offset_x = (NX / PX) * px + (px < NX % PX ? px : NX % PX);
    int offset_y = (NY / PY) * py + (py < NY % PY ? py : NY % PY);
    int offset_z = (NZ / PZ) * pz + (pz < NZ % PZ ? pz : NZ % PZ);

    if (rank == 0) {
        printf("Problem Parameters: NX=%d, NY=%d, NZ=%d, NC=%d, PX=%d, PY=%d, PZ=%d\n", 
               NX, NY, NZ, NC, PX, PY, PZ);
    }

    // Read data (only rank 0)
    float* full_data = NULL;
    if (rank == 0) {
        full_data = (float*)malloc(sizeof(float) * total_points * NC);
        read_data(full_data, input_file, total_points, NC);
    }

    // Allocate memory for local data
    int local_size = lx * ly * lz;
    float* local_data = (float*)malloc(sizeof(float) * local_size * NC);

    // Create MPI datatypes for distributing data
    MPI_Datatype local_block_type, global_block_type;
    int local_sizes[3] = {lx, ly, lz};
    int local_starts[3] = {0, 0, 0};
    int local_subsizes[3] = {lx, ly, lz};
    
    MPI_Type_create_subarray(3, local_sizes, local_subsizes, local_starts, 
                             MPI_ORDER_C, MPI_FLOAT, &local_block_type);
    MPI_Type_commit(&local_block_type);
    
    int global_sizes[3] = {NX, NY, NZ};
    int global_starts[3] = {offset_x, offset_y, offset_z};
    
    MPI_Type_create_subarray(3, global_sizes, local_subsizes, global_starts, 
                             MPI_ORDER_C, MPI_FLOAT, &global_block_type);
    MPI_Type_commit(&global_block_type);

    // Distribute data for each time step
    MPI_Request* reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * NC);
    for (int t = 0; t < NC; t++) {
        // Setup to distribute data from global to local
        if (rank == 0) {
            for (int p = 0; p < size; p++) {
                int p_px = p % PX;
                int p_py = (p / PX) % PY;
                int p_pz = p / (PX * PY);
                
                int p_lx = NX / PX + (p_px < NX % PX ? 1 : 0);
                int p_ly = NY / PY + (p_py < NY % PY ? 1 : 0);
                int p_lz = NZ / PZ + (p_pz < NZ % PZ ? 1 : 0);
                
                int p_offset_x = (NX / PX) * p_px + (p_px < NX % PX ? p_px : NX % PX);
                int p_offset_y = (NY / PY) * p_py + (p_py < NY % PY ? p_py : NY % PY);
                int p_offset_z = (NZ / PZ) * p_pz + (p_pz < NZ % PZ ? p_pz : NZ % PZ);
                
                // Manual distribution (instead of MPI scatter to handle irregular decomposition)
                for (int z = 0; z < p_lz; z++) {
                    for (int y = 0; y < p_ly; y++) {
                        for (int x = 0; x < p_lx; x++) {
                            int global_idx = IDX(p_offset_x + x, p_offset_y + y, p_offset_z + z, NX, NY);
                            int local_idx = IDX(x, y, z, p_lx, p_ly);
                            
                            if (p == 0) {
                                local_data[local_idx * NC + t] = full_data[global_idx * NC + t];
                            } else {
                                MPI_Send(&full_data[global_idx * NC + t], 1, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
            }
        } else {
            // Receive local data
            for (int z = 0; z < lz; z++) {
                for (int y = 0; y < ly; y++) {
                    for (int x = 0; x < lx; x++) {
                        int local_idx = IDX(x, y, z, lx, ly);
                        MPI_Recv(&local_data[local_idx * NC + t], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
    }
    
    // Free full data once distributed
    if (rank == 0) {
        free(full_data);
    }
    
    // Synchronize all processes before timing the main computation
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Identify neighbors
    int neighbors[6];
    neighbors[0] = get_neighbor_rank(px, py, pz, -1, 0, 0, PX, PY, PZ);  // Left
    neighbors[1] = get_neighbor_rank(px, py, pz, 1, 0, 0, PX, PY, PZ);   // Right
    neighbors[2] = get_neighbor_rank(px, py, pz, 0, -1, 0, PX, PY, PZ);  // Bottom
    neighbors[3] = get_neighbor_rank(px, py, pz, 0, 1, 0, PX, PY, PZ);   // Top
    neighbors[4] = get_neighbor_rank(px, py, pz, 0, 0, -1, PX, PY, PZ);  // Front
    neighbors[5] = get_neighbor_rank(px, py, pz, 0, 0, 1, PX, PY, PZ);   // Back

    // Allocate memory for ghost regions - one for each face per time step
    int ghost_sizes[6] = {
        ly * lz,  // Left face
        ly * lz,  // Right face
        lx * lz,  // Bottom face
        lx * lz,  // Top face
        lx * ly,  // Front face
        lx * ly   // Back face
    };
    
    int total_ghost_size = 0;
    for (int i = 0; i < 6; i++) {
        total_ghost_size += ghost_sizes[i];
    }
    
    // Allocate ghost data for all time steps
    float* ghost_data = (float*)malloc(sizeof(float) * total_ghost_size * NC);
    float* send_data = (float*)malloc(sizeof(float) * total_ghost_size * NC);
    
    // Allocate arrays for results
    int* min_counts = (int*)calloc(NC, sizeof(int));
    int* max_counts = (int*)calloc(NC, sizeof(int));
    float* local_min = (float*)malloc(sizeof(float) * NC);
    float* local_max = (float*)malloc(sizeof(float) * NC);
    
    // Initialize local min/max
    for (int t = 0; t < NC; t++) {
        local_min[t] = FLT_MAX;
        local_max[t] = -FLT_MAX;
    }

    // Process each time step
    for (int t = 0; t < NC; t++) {
        // Exchange ghost region data with neighbors
        MPI_Request send_reqs[6], recv_reqs[6];
        int send_offset = 0, recv_offset = 0;
        
        // Prepare data to send (boundary layers)
        // Left face (x = 0)
        if (neighbors[0] != -1) {
            for (int z = 0; z < lz; z++) {
                for (int y = 0; y < ly; y++) {
                    send_data[send_offset++] = local_data[((z * ly + y) * lx + 0) * NC + t];
                }
            }
            MPI_Isend(&send_data[0], ghost_sizes[0], MPI_FLOAT, neighbors[0], 0, MPI_COMM_WORLD, &send_reqs[0]);
            MPI_Irecv(&ghost_data[0], ghost_sizes[0], MPI_FLOAT, neighbors[0], 1, MPI_COMM_WORLD, &recv_reqs[0]);
        }

        // Right face (x = lx-1)
        if (neighbors[1] != -1) {
            for (int z = 0; z < lz; z++) {
                for (int y = 0; y < ly; y++) {
                    send_data[ghost_sizes[0] + send_offset++] = local_data[((z * ly + y) * lx + (lx-1)) * NC + t];
                }
            }
            MPI_Isend(&send_data[ghost_sizes[0]], ghost_sizes[1], MPI_FLOAT, neighbors[1], 1, MPI_COMM_WORLD, &send_reqs[1]);
            MPI_Irecv(&ghost_data[ghost_sizes[0]], ghost_sizes[1], MPI_FLOAT, neighbors[1], 0, MPI_COMM_WORLD, &recv_reqs[1]);
        }

        // Bottom face (y = 0)
        if (neighbors[2] != -1) {
            for (int z = 0; z < lz; z++) {
                for (int x = 0; x < lx; x++) {
                    send_data[ghost_sizes[0] + ghost_sizes[1] + send_offset++] = local_data[((z * ly + 0) * lx + x) * NC + t];
                }
            }
            MPI_Isend(&send_data[ghost_sizes[0] + ghost_sizes[1]], ghost_sizes[2], MPI_FLOAT, 
                     neighbors[2], 2, MPI_COMM_WORLD, &send_reqs[2]);
            MPI_Irecv(&ghost_data[ghost_sizes[0] + ghost_sizes[1]], ghost_sizes[2], MPI_FLOAT, 
                    neighbors[2], 3, MPI_COMM_WORLD, &recv_reqs[2]);
        }

        // Top face (y = ly-1)
        if (neighbors[3] != -1) {
            for (int z = 0; z < lz; z++) {
                for (int x = 0; x < lx; x++) {
                    send_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + send_offset++] = 
                        local_data[((z * ly + (ly-1)) * lx + x) * NC + t];
                }
            }
            MPI_Isend(&send_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2]], ghost_sizes[3], MPI_FLOAT, 
                     neighbors[3], 3, MPI_COMM_WORLD, &send_reqs[3]);
            MPI_Irecv(&ghost_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2]], ghost_sizes[3], MPI_FLOAT, 
                    neighbors[3], 2, MPI_COMM_WORLD, &recv_reqs[3]);
        }

        // Front face (z = 0)
        if (neighbors[4] != -1) {
            for (int y = 0; y < ly; y++) {
                for (int x = 0; x < lx; x++) {
                    send_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + ghost_sizes[3] + send_offset++] = 
                        local_data[((0 * ly + y) * lx + x) * NC + t];
                }
            }
            MPI_Isend(&send_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + ghost_sizes[3]], 
                     ghost_sizes[4], MPI_FLOAT, neighbors[4], 4, MPI_COMM_WORLD, &send_reqs[4]);
            MPI_Irecv(&ghost_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + ghost_sizes[3]], 
                    ghost_sizes[4], MPI_FLOAT, neighbors[4], 5, MPI_COMM_WORLD, &recv_reqs[4]);
        }

        // Back face (z = lz-1)
        if (neighbors[5] != -1) {
            for (int y = 0; y < ly; y++) {
                for (int x = 0; x < lx; x++) {
                    send_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + ghost_sizes[3] + ghost_sizes[4] + send_offset++] = 
                        local_data[(((lz-1) * ly + y) * lx + x) * NC + t];
                }
            }
            MPI_Isend(&send_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + ghost_sizes[3] + ghost_sizes[4]], 
                     ghost_sizes[5], MPI_FLOAT, neighbors[5], 5, MPI_COMM_WORLD, &send_reqs[5]);
            MPI_Irecv(&ghost_data[ghost_sizes[0] + ghost_sizes[1] + ghost_sizes[2] + ghost_sizes[3] + ghost_sizes[4]], 
                    ghost_sizes[5], MPI_FLOAT, neighbors[5], 4, MPI_COMM_WORLD, &recv_reqs[5]);
        }

        // Wait for all exchanges to complete
        for (int i = 0; i < 6; i++) {
            if (neighbors[i] != -1) {
                MPI_Wait(&send_reqs[i], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_reqs[i], MPI_STATUS_IGNORE);
            }
        }

        // Find local minima, maxima, and update global min/max
        int min_count = 0, max_count = 0;
        float gmin = FLT_MAX, gmax = -FLT_MAX;
        
        for (int z = 0; z < lz; z++) {
            for (int y = 0; y < ly; y++) {
                for (int x = 0; x < lx; x++) {
                    int local_idx = (z * ly + y) * lx + x;
                    float val = local_data[local_idx * NC + t];
                    
                    // Update global min/max
                    if (val < gmin) gmin = val;
                    if (val > gmax) gmax = val;
                    
                    // Check if this point is a local minimum or maximum
                    if (is_local_min(local_data, ghost_data, x, y, z, lx, ly, lz, t, NC,
                                   offset_x + x, offset_y + y, offset_z + z, NX, NY, NZ)) {
                        min_count++;
                    }
                    
                    if (is_local_max(local_data, ghost_data, x, y, z, lx, ly, lz, t, NC,
                                   offset_x + x, offset_y + y, offset_z + z, NX, NY, NZ)) {
                        max_count++;
                    }
                }
            }
        }
        
        min_counts[t] = min_count;
        max_counts[t] = max_count;
        local_min[t] = gmin;
        local_max[t] = gmax;
    }

    // Allocate arrays for global results (only rank 0)
    int* global_min_counts = NULL;
    int* global_max_counts = NULL;
    float* global_min_vals = NULL;
    float* global_max_vals = NULL;

    if (rank == 0) {
        global_min_counts = (int*)calloc(NC, sizeof(int));
        global_max_counts = (int*)calloc(NC, sizeof(int));
        global_min_vals = (float*)malloc(sizeof(float) * NC);
        global_max_vals = (float*)malloc(sizeof(float) * NC);
    }

    // Gather results from all processes
    MPI_Reduce(min_counts, global_min_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(max_counts, global_max_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min, global_min_vals, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max, global_max_vals, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    double t3 = MPI_Wtime();

    // Write results to output file (only rank 0)
    if (rank == 0) {
        FILE* fout = fopen(output_file, "w");
        if (fout == NULL) {
            printf("Error: Could not open output file %s\n", output_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Print problem parameters
        fprintf(fout, "Problem Parameters:\n");
        fprintf(fout, "NX: %d, NY: %d, NZ: %d, NC: %d\n", NX, NY, NZ, NC);
        fprintf(fout, "Process Grid: PX: %d, PY: %d, PZ: %d, Total Processes: %d\n\n", PX, PY, PZ, size);
        
        // Print local minima and maxima counts for each time step
        fprintf(fout, "Local Minima and Maxima Counts for Each Time Step:\n");
        fprintf(fout, "Time Step | Local Minima | Local Maxima\n");
        fprintf(fout, "----------------------------------------\n");
        for (int t = 0; t < NC; ++t) {
            fprintf(fout, "%9d | %12d | %12d\n", t, global_min_counts[t], global_max_counts[t]);
        }
        fprintf(fout, "\n");
        
        // Print global min and max for each time step
        fprintf(fout, "Global Minima and Maxima for Each Time Step:\n");
        fprintf(fout, "Time Step | Global Min | Global Max\n");
        fprintf(fout, "----------------------------------------\n");
        for (int t = 0; t < NC; ++t) {
            fprintf(fout, "%9d | %10.6f | %10.6f\n", t, global_min_vals[t], global_max_vals[t]);
        }
        fprintf(fout, "\n");
        
        // Print timing information
        fprintf(fout, "Timing Information:\n");
        fprintf(fout, "Data Distribution Time: %.6f seconds\n", t2 - t1);
        fprintf(fout, "Computation Time: %.6f seconds\n", t3 - t2);
        fprintf(fout, "Total Execution Time: %.6f seconds\n", t3 - t1);
        
        fclose(fout);
        
        // Free memory
        free(global_min_counts);
        free(global_max_counts);
        free(global_min_vals);
        free(global_max_vals);
    }
    
    // Free local memory
    free(local_data);
    free(ghost_data);
    free(send_data);
    free(min_counts);
    free(max_counts);
    free(local_min);
    free(local_max);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
