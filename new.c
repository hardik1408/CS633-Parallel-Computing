#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE_LEN 10000
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
    char *input_file = argv[1];
    int PX = atoi(argv[2]);
    int PY = atoi(argv[3]);
    int PZ = atoi(argv[4]);
    int NX = atoi(argv[5]);
    int NY = atoi(argv[6]);
    int NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    char *output_file = argv[9];
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
    double time1 = MPI_Wtime();
    int N = NX * NY * NZ;
    int *line_offsets = NULL;
    int *line_lengths = NULL;
    if (rank == 0) {
        line_offsets = (int *)malloc(N * sizeof(int));
        line_lengths = (int *)malloc(N * sizeof(int));
        FILE *f = fopen(input_file, "r");
        if (!f) {
            fprintf(stderr, "Cannot open input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int pos = 0;
        for (int i = 0; i < N; i++) {
            line_offsets[i] = pos;
            char buf[MAX_LINE_LEN];
            if (fgets(buf, MAX_LINE_LEN, f)) {
                int len = strlen(buf);
                pos += len;
                line_lengths[i] = len;
            } else {
                fprintf(stderr, "Unexpected EOF at line %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(f);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        line_offsets = (int *)malloc(N * sizeof(int));
        line_lengths = (int *)malloc(N * sizeof(int));
    }
    MPI_Bcast(line_offsets, N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(line_lengths, N, MPI_INT, 0, MPI_COMM_WORLD);
    int local_n = N / size;
    int rem = N % size;
    int start = rank * local_n + (rank < rem ? rank : rem);
    int count = local_n + (rank < rem ? 1 : 0);
    char *raw_buf = (char *)malloc(count * MAX_LINE_LEN);
    float *local_data = (float *)malloc(count * NC * sizeof(float));
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    for (int i = 0; i < count; i++) {
        MPI_File_read_at(fh, line_offsets[start + i], raw_buf + i * MAX_LINE_LEN, line_lengths[start + i], MPI_CHAR, MPI_STATUS_IGNORE);
        raw_buf[i * MAX_LINE_LEN + line_lengths[start + i]] = '\0';
    }
    MPI_File_close(&fh);
    for (int i = 0; i < count; i++) {
        char *ptr = raw_buf + i * MAX_LINE_LEN;
        for (int j = 0; j < NC; j++) {
            while (*ptr == ' ' || *ptr == '\t')
                ptr++;
            sscanf(ptr, "%f", &local_data[i * NC + j]);
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n')
                ptr++;
        }
    }
    double time2 = MPI_Wtime();
    double read_time = time2 - time1;
    double main_time = 0.0;
    double total_time = time2 - time1;
    double max_read, max_main, max_total;
    MPI_Reduce(&read_time, &max_read, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("time : %.6f\n", max_read);
    }
    // MPI_Reduce(&main_time, &max_main, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // if (rank == 0) {
    //     FILE *out = fopen(output_file, "w");
    //     fprintf(out, "\n\n%.6lf %.6lf %.6lf\n", max_read, max_main, max_total);
    //     fclose(out);
    // }
    free(line_offsets);
    free(line_lengths);
    free(raw_buf);
    free(local_data);
    MPI_Finalize();
    return 0;
}
