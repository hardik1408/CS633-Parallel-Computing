#include <stdio.h>
#include <string.h>

int main(int agc, char *argv[])
{
    int global_rank, global_size;
    char file_name[100];
    strncpy(file_name,argv[1],sizeof(file_name)-1);
    file_name[sizeof(file_name) - 1] = '\0';
    printf("%s",file_name);

    return 0;
}