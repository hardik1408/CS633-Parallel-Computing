#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:20:00         ## wall-clock time limit
#SBATCH --partition=standard    ## can be "standard" or "cpu"


echo `date`
mpirun -np 8 ./src ./data_64_64_64_3.txt 2 2 2 64 64 64 3 ./output_64_64_64_3.txt
echo `date`