#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#define MAX_FILENAME 100
#define MAX_LINE_LENGTH 100000  
        
int main(int argc, char *argv[])
{
    if (argc != 10) {
        fprintf(stderr, "Usage: %s <input_file> <pX> <pY> <pZ> <nX> <nY> <nZ> <nC> <output_file>\n", argv[0]);
        return 1;
    }

    double stime,etime,read_time,max_read_time;
    int global_rank, global_size;

    /* Arguments Handling */
    char inFilename[MAX_FILENAME], outFilename[MAX_FILENAME];
    strncpy(inFilename, argv[1], MAX_FILENAME - 1);
    inFilename[MAX_FILENAME - 1] = '\0';
    int pX = atoi(argv[2]);
    int pY = atoi(argv[3]);
    int pZ = atoi(argv[4]);
    int nX = atoi(argv[5]);
    int nY = atoi(argv[6]);
    int nZ = atoi(argv[7]);
    int nC = atoi(argv[8]);
    strncpy(outFilename, argv[9], MAX_FILENAME - 1);  
    outFilename[MAX_FILENAME - 1] = '\0';

    int N = nX * nY * nZ;

    /* MPI Initialize */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    stime = MPI_Wtime();
    // Cartesian topology
    int dims[3] = {pX, pY, pZ}, periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    int coords[3];
    MPI_Cart_coords(cart_comm, global_rank, 3, coords);

    // Local grid sizes and offsets
    int local_nX = nX / pX, local_nY = nY / pY, local_nZ = nZ / pZ;
    int start_X = coords[0] * local_nX;
    int start_Y = coords[1] * local_nY;
    int start_Z = coords[2] * local_nZ;

    // Have process 0 calculate line positions
    long *line_positions = NULL;
    
    if (global_rank == 0) {
        if (global_rank == 0) printf("Process 0: Analyzing file to determine line positions...\n");
        
        line_positions = (long*)malloc((N + 1) * sizeof(long));
        if (!line_positions) {
            fprintf(stderr, "Process 0: Memory allocation failed for line positions\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        FILE *fp = fopen(inFilename, "r");
        if (!fp) {
            fprintf(stderr, "Process 0: Failed to open file '%s'\n", inFilename);
            free(line_positions);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        line_positions[0] = 0; // Start of file
        char line[MAX_LINE_LENGTH];
        for (int i = 0; i < N; i++) {
            if (fgets(line, sizeof(line), fp) == NULL) {
                if (i < N - 1) {
                    fprintf(stderr, "Process 0: Error - reached end of file after reading %d lines, expected %d\n", i, N);
                    free(line_positions);
                    fclose(fp);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                break;
            }
            line_positions[i + 1] = ftell(fp); // Position after reading current line
        }
        
        fclose(fp);
        if (global_rank == 0) printf("Process 0: Line position analysis complete.\n");
    }
    
    // Broadcast line positions to all processes
    if (global_rank != 0) {
        line_positions = (long*)malloc((N + 1) * sizeof(long));
        if (!line_positions) {
            fprintf(stderr, "Process %d: Memory allocation failed for line positions\n", global_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    MPI_Bcast(line_positions, N + 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Allocate memory for local data
    float *local_data = malloc(local_nX * local_nY * local_nZ * nC * sizeof(float));
    if (!local_data) {
        fprintf(stderr, "Process %d: Memory allocation failed for local data\n", global_rank);
        free(line_positions);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create a list of lines that this process needs to read
    int local_line_count = local_nX * local_nY * local_nZ;
    int *local_line_indices = (int*)malloc(local_line_count * sizeof(int));
    if (!local_line_indices) {
        fprintf(stderr, "Process %d: Memory allocation failed for local line indices\n", global_rank);
        free(local_data);
        free(line_positions);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int idx = 0;
    for (int z = start_Z; z < start_Z + local_nZ; z++) {
        for (int y = start_Y; y < start_Y + local_nY; y++) {
            for (int x = start_X; x < start_X + local_nX; x++) {
                int global_idx = x + y * nX + z * nX * nY;
                local_line_indices[idx++] = global_idx;
            }
        }
    }

    // Open the file for parallel reading
    FILE *fp = fopen(inFilename, "r");
    if (!fp) {
        fprintf(stderr, "Process %d: Failed to open file '%s'\n", global_rank, inFilename);
        free(local_line_indices);
        free(local_data);
        free(line_positions);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read assigned lines
    char line[MAX_LINE_LENGTH];
    for (int i = 0; i < local_line_count; i++) {
        int global_line_idx = local_line_indices[i];
        
        // Calculate local (x, y, z) from global index
        int x = global_line_idx % nX;
        int y = (global_line_idx / nX) % nY;
        int z = global_line_idx / (nX * nY);
        
        // Calculate local indices
        int local_x = x - start_X;
        int local_y = y - start_Y;
        int local_z = z - start_Z;
        
        // Seek to the position of this line
        fseek(fp, line_positions[global_line_idx], SEEK_SET);
        
        // Read the line
        if (fgets(line, sizeof(line), fp) == NULL) {
            fprintf(stderr, "Process %d: Error reading line %d\n", global_rank, global_line_idx);
            free(local_line_indices);
            free(local_data);
            free(line_positions);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Parse the line for nC floats using sscanf
        float values[nC];
        char *ptr = line;
        for (int c = 0; c < nC; c++) {
            if (sscanf(ptr, "%f", &values[c]) != 1) {
                fprintf(stderr, "Process %d: Failed to parse float %d on line %d\n", 
                        global_rank, c, global_line_idx);
                break;
            }
            
            // Move to next number in line
            while (*ptr && (*ptr == ' ' || *ptr == '\t' || *ptr == '\n')) ptr++;
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n') ptr++;
            
            // Store in local data array
            int local_idx = c + local_x * nC + local_y * local_nX * nC + local_z * local_nY * local_nX * nC;
            local_data[local_idx] = values[c];
        }
    }

    fclose(fp);
    free(local_line_indices);
    free(line_positions);
    etime = MPI_Wtime();
    read_time = etime - stime;
    MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    // Print local data
    printf("Process %d: Full local data:\n", global_rank);
    for (int k = 0; k < local_nZ; k++) {
        for (int j = 0; j < local_nY; j++) {
            for (int i = 0; i < local_nX; i++) {
                for (int c = 0; c < nC; c++) {
                    int idx = c + i * nC + j * local_nX * nC + k * local_nY * local_nX * nC;
                    printf("  [%d][%d][%d][%d] = %f\n", k, j, i, c, local_data[idx]);
                }
            }
        }
    }

    if (global_rank == 0) {
        printf("time : %.6f\n", max_read_time);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}
