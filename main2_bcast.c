#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// Macro to calculate the index in a 3D array stored in 1D
#define IDX(x, y, z, nx, ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

// Function to read data from input file
void read_data(float* data, const char* fname, int total_size, int nc) {
    FILE* f = fopen(fname, "r");
    if (f == NULL) {
        printf("Error: Could not open input file %s\n", fname);
        exit(1);
    }
    for (int i = 0; i < total_size * nc; ++i)
        fscanf(f, "%f", &data[i]);
    fclose(f);
}

// Function to check if a point is a local minimum in its neighborhood
int is_local_min(float* volume, int x, int y, int z, int nx, int ny, int nz, int t, int nc) {
    int idx = IDX(x, y, z, nx, ny);
    float val = volume[idx * nc + t];
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx_ = x + dx, ny_ = y + dy, nz_ = z + dz;
                if (nx_ >= 0 && nx_ < nx && ny_ >= 0 && ny_ < ny && nz_ >= 0 && nz_ < nz) {
                    int nidx = IDX(nx_, ny_, nz_, nx, ny);
                    if (volume[nidx * nc + t] <= val) return 0;
                }
            }
        }
    }
    return 1;
}

// Function to check if a point is a local maximum in its neighborhood
int is_local_max(float* volume, int x, int y, int z, int nx, int ny, int nz, int t, int nc) {
    int idx = IDX(x, y, z, nx, ny);
    float val = volume[idx * nc + t];
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx_ = x + dx, ny_ = y + dy, nz_ = z + dz;
                if (nx_ >= 0 && nx_ < nx && ny_ >= 0 && ny_ < ny && nz_ >= 0 && nz_ < nz) {
                    int nidx = IDX(nx_, ny_, nz_, nx, ny);
                    if (volume[nidx * nc + t] >= val) return 0;
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

    printf("Process %d: Starting with parameters NX=%d, NY=%d, NZ=%d, NC=%d, PX=%d, PY=%d, PZ=%d\n", 
           rank, NX, NY, NZ, NC, PX, PY, PZ);

    // Read data (only rank 0)
    float* full_data = NULL;
    if (rank == 0) {
        printf("Process 0: Reading data from file %s\n", input_file);
        full_data = (float*)malloc(sizeof(float) * total_points * NC);
        read_data(full_data, input_file, total_points, NC);
        printf("Process 0: Data read complete\n");
    } else {
        // Allocate memory for full data on other processes
        full_data = (float*)malloc(sizeof(float) * total_points * NC);
    }

    double t2 = MPI_Wtime();

    // Broadcast the full data to all processes
    printf("Process %d: Broadcasting data\n", rank);
    MPI_Bcast(full_data, total_points * NC, MPI_FLOAT, 0, MPI_COMM_WORLD);
    printf("Process %d: Data broadcast complete\n", rank);

    // Calculate process coordinates in 3D grid
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    // Calculate local domain size for this process
    int lx = NX / PX + (px < NX % PX);
    int ly = NY / PY + (py < NY % PY);
    int lz = NZ / PZ + (pz < NZ % PZ);

    // Calculate offset in global domain
    int offset_x = (NX / PX) * px + (px < NX % PX ? px : NX % PX);
    int offset_y = (NY / PY) * py + (py < NY % PY ? py : NY % PY);
    int offset_z = (NZ / PZ) * pz + (pz < NZ % PZ ? pz : NZ % PZ);

    printf("Process %d: Local domain size: %dx%dx%d, Offset: (%d,%d,%d)\n", 
           rank, lx, ly, lz, offset_x, offset_y, offset_z);

    // Allocate memory for local data
    int local_size = lx * ly * lz;
    float* local_data = (float*)malloc(sizeof(float) * local_size * NC);

    // Extract local data from full data
    printf("Process %d: Extracting local data\n", rank);
    for (int z = 0; z < lz; ++z)
        for (int y = 0; y < ly; ++y)
            for (int x = 0; x < lx; ++x)
                for (int t = 0; t < NC; ++t)
                    local_data[(((z * ly + y) * lx + x) * NC) + t] =
                        full_data[(((offset_z + z) * NY + (offset_y + y)) * NX + (offset_x + x)) * NC + t];
    printf("Process %d: Local data extraction complete\n", rank);

    // Free full data as it's no longer needed
    free(full_data);

    // Allocate arrays for results
    int* min_counts = (int*)calloc(NC, sizeof(int));
    int* max_counts = (int*)calloc(NC, sizeof(int));
    float* global_min = (float*)malloc(sizeof(float) * NC);
    float* global_max = (float*)malloc(sizeof(float) * NC);

    // Find local minima, maxima, and global min/max for each time step
    printf("Process %d: Starting computation\n", rank);
    for (int t = 0; t < NC; ++t) {
        float gmin = FLT_MAX, gmax = -FLT_MAX;
        int count_min = 0, count_max = 0;
        for (int z = 0; z < lz; ++z)
            for (int y = 0; y < ly; ++y)
                for (int x = 0; x < lx; ++x) {
                    int idx = IDX(x, y, z, lx, ly);
                    float val = local_data[idx * NC + t];
                    if (is_local_min(local_data, x, y, z, lx, ly, lz, t, NC)) count_min++;
                    if (is_local_max(local_data, x, y, z, lx, ly, lz, t, NC)) count_max++;
                    if (val < gmin) gmin = val;
                    if (val > gmax) gmax = val;
                }
        min_counts[t] = count_min;
        max_counts[t] = count_max;
        global_min[t] = gmin;
        global_max[t] = gmax;
    }
    printf("Process %d: Computation complete\n", rank);

    // Allocate arrays for global results (only rank 0)
    int* total_min = NULL;
    int* total_max = NULL;
    float* min_vals = NULL;
    float* max_vals = NULL;

    if (rank == 0) {
        total_min = (int*)calloc(NC, sizeof(int));
        total_max = (int*)calloc(NC, sizeof(int));
        min_vals = (float*)malloc(sizeof(float) * NC);
        max_vals = (float*)malloc(sizeof(float) * NC);
    }

    // Gather results from all processes
    printf("Process %d: Starting result gathering\n", rank);
    MPI_Reduce(min_counts, total_min, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(max_counts, total_max, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(global_min, min_vals, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(global_max, max_vals, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    printf("Process %d: Result gathering complete\n", rank);

    double t3 = MPI_Wtime();

    // Write results to output file (only rank 0)
    if (rank == 0) {
        printf("Process 0: Writing results to file %s\n", output_file);
        FILE* fout = fopen(output_file, "w");
        
        // Print problem parameters
        fprintf(fout, "Problem Parameters:\n");
        fprintf(fout, "NX: %d, NY: %d, NZ: %d, NC: %d\n", NX, NY, NZ, NC);
        fprintf(fout, "Process Grid: PX: %d, PY: %d, PZ: %d, Total Processes: %d\n\n", PX, PY, PZ, size);
        
        // Print local minima and maxima counts for each time step
        fprintf(fout, "Local Minima and Maxima Counts for Each Time Step:\n");
        fprintf(fout, "Time Step | Local Minima | Local Maxima\n");
        fprintf(fout, "----------------------------------------\n");
        for (int t = 0; t < NC; ++t) {
            fprintf(fout, "%9d | %12d | %12d\n", t, total_min[t], total_max[t]);
        }
        fprintf(fout, "\n");
        
        // Print global min and max for each time step
        fprintf(fout, "Global Minima and Maxima for Each Time Step:\n");
        fprintf(fout, "Time Step | Global Min | Global Max\n");
        fprintf(fout, "----------------------------------------\n");
        for (int t = 0; t < NC; ++t) {
            fprintf(fout, "%9d | %10.3f | %10.3f\n", t, min_vals[t], max_vals[t]);
        }
        fprintf(fout, "\n");
        
        // Print timing information
        fprintf(fout, "Timing Information (in seconds):\n");
        fprintf(fout, "Initialization Time: %.6f\n", t2 - t1);
        fprintf(fout, "Computation Time: %.6f\n", t3 - t2);
        fprintf(fout, "Total Execution Time: %.6f\n", t3 - t1);
        
        fclose(fout);
        printf("Process 0: Results written to file\n");
    }

    // Free allocated memory
    free(local_data);
    free(min_counts);
    free(max_counts);
    free(global_min);
    free(global_max);
    if (rank == 0) {
        free(total_min);
        free(total_max);
        free(min_vals);
        free(max_vals);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
