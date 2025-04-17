# CS633 Parallel Computing Project
# Group 20 Submission

### Instructions to run the code

A Makefile is provided to compile the code using an MPI-enabled compiler mpicc with optimization flags. The Makefile is defined as follows:

```makefile
CC = mpicc
CFLAGS = -O3 -lm
TARGET = src

all: $(TARGET)

$(TARGET): src.c
    $(CC) $(CFLAGS) -o $@ $<

clean:
    rm -f $(TARGET)
```
To compile the program, run:
```bash
make
```
The program is executed on a cluster managed by SLURM. A SLURM job script (\texttt{job.sh}) is used to specify the execution parameters and submit the job. 
To submit the job, use:
```bash
sbatch job.sh
```

The job script and execution command accept the following parameters:
- num_processes: Number of MPI processes (e.g., 8, 16, 32, 64), specified via \texttt{-np} in the \texttt{mpirun} command within the SLURM script.
- Parameters:
    - input_file: Binary file containing the input dataset (e.g., data_64_64_64_3.txt).
    - PX PY PZ: Dimensions of the process grid (e.g., 2 2 2).
    - NX NY NZ: Dimensions of the global spatial grid (e.g., 64 64 64).
    - NC: Number of time steps (e.g., 3).
    - output_file: File for results, preferably named output_NX_NY_NZ_NC.txt (e.g., output_64_64_64_3.txt).

For the provided test cases, submit the following SLURM job scripts:
```bash
#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:20:00
#SBATCH --partition=standard

echo `date`
mpirun -np 8 ./executable data_64_64_64_3.txt 2 2 2 64 64 64 3 output_64_64_64_3.txt
echo `date`
```
```bash
#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:20:00
#SBATCH --partition=standard

echo `date`
mpirun -np 8 ./executable data_64_64_96_7.txt 2 2 2 64 64 96 7 output_64_64_96_7.txt
echo `date`
```
Submit each script using:
```bash
sbatch job.sh
```
