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
    if(argc != 10) {
        if(rank==0)
            fprintf(stderr, "Usage: %s input_file PX PY PZ NX NY NZ NC output_file\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    char *input_file = argv[1];
    int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
    int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]), NC = atoi(argv[8]);
    char *output_file = argv[9];
    int dims[3] = {PX, PY, PZ}, periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    int P = PX * PY * PZ;
    if(P != size) {
        if(rank==0)
            fprintf(stderr, "Error: PX*PY*PZ must equal number of MPI processes\n");
        MPI_Finalize();
        return 1;
    }
    double time1 = MPI_Wtime();
    int N = NX * NY * NZ;
    int *line_offsets = NULL, *line_lengths = NULL;
    if(rank==0) {
        line_offsets = malloc(N * sizeof(int));
        line_lengths = malloc(N * sizeof(int));
        FILE *f = fopen(input_file, "r");
        if(!f) {
            fprintf(stderr, "Cannot open input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int pos = 0;
        for(int i = 0; i < N; i++){
            line_offsets[i] = pos;
            char buf[MAX_LINE_LEN];
            if(fgets(buf, MAX_LINE_LEN, f)) {
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
    if(rank != 0) {
        line_offsets = malloc(N * sizeof(int));
        line_lengths = malloc(N * sizeof(int));
    }
    MPI_Bcast(line_offsets, N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(line_lengths, N, MPI_INT, 0, MPI_COMM_WORLD);
    int my_coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, my_coords);
    int extra, local_NX, start_x;
    extra = NX % PX;
    local_NX = NX / PX + (my_coords[0] < extra ? 1 : 0);
    start_x = (NX / PX) * my_coords[0] + (my_coords[0] < extra ? my_coords[0] : extra);
    int local_NY, start_y;
    extra = NY % PY;
    local_NY = NY / PY + (my_coords[1] < extra ? 1 : 0);
    start_y = (NY / PY) * my_coords[1] + (my_coords[1] < extra ? my_coords[1] : extra);
    int local_NZ, start_z;
    extra = NZ % PZ;
    local_NZ = NZ / PZ + (my_coords[2] < extra ? 1 : 0);
    start_z = (NZ / PZ) * my_coords[2] + (my_coords[2] < extra ? my_coords[2] : extra);
    int num_local = local_NX * local_NY * local_NZ;
    int *local_global_indices = malloc(num_local * sizeof(int));
    int idx = 0;
    for(int z = start_z; z < start_z + local_NZ; z++){
        for(int y = start_y; y < start_y + local_NY; y++){
            for(int x = start_x; x < start_x + local_NX; x++){
                local_global_indices[idx++] = x + y * NX + z * NX * NY;
            }
        }
    }
    int *local_line_lengths = malloc(num_local * sizeof(int));
    MPI_Aint *local_displs = malloc(num_local * sizeof(MPI_Aint));
    int total_bytes = 0;
    for(int i = 0; i < num_local; i++){
        int gi = local_global_indices[i];
        local_line_lengths[i] = line_lengths[gi];
        total_bytes += line_lengths[gi];
    }
    for(int i = 0; i < num_local; i++){
        int gi = local_global_indices[i];
        local_displs[i] = line_offsets[gi];
    }
    char *local_buffer = malloc(total_bytes + 1);
    MPI_Datatype filetype;
    MPI_Type_create_hindexed(num_local, local_line_lengths, local_displs, MPI_CHAR, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File fh;
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_no_locks", "true");
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, info, &fh);
    MPI_Info_free(&info);
    MPI_File_set_view(fh, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, local_buffer, total_bytes, MPI_CHAR, MPI_STATUS_IGNORE);
    local_buffer[total_bytes] = '\0';
    MPI_File_close(&fh);
    float *local_data = malloc(num_local * NC * sizeof(float));
    int offset = 0;
    for(int i = 0; i < num_local; i++){
        int len = local_line_lengths[i];
        char *line = malloc(len + 1);
        memcpy(line, local_buffer + offset, len);
        line[len] = '\0';
        offset += len;
        char *ptr = line;
        for(int j = 0; j < NC; j++){
            while(*ptr == ' ' || *ptr == '\t')
                ptr++;
            sscanf(ptr, "%f", &local_data[i * NC + j]);
            while(*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n')
                ptr++;
        }
        free(line);
    }
    double time2 = MPI_Wtime();
    double read_time = time2 - time1;
    double main_time = 0.0;
    double total_time = time2 - time1;
    double max_read, max_main, max_total;
    MPI_Reduce(&read_time, &max_read, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&main_time, &max_main, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        FILE *out = fopen(output_file, "w");
        fprintf(out, "\n\n%.6lf %.6lf %.6lf\n", max_read, max_main, max_total);
        fclose(out);
    }
    free(local_global_indices);
    free(local_line_lengths);
    free(local_displs);
    free(local_buffer);
    free(local_data);
    free(line_offsets);
    free(line_lengths);
    MPI_Type_free(&filetype);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
