# Parallel Computing - 3D Domain Decomposition

This project implements a parallel algorithm for processing time series data in a 3D volume using MPI (Message Passing Interface). The algorithm performs 3D domain decomposition to distribute the workload across multiple processes.

## Problem Description

Given a 3D volume of size NX × NY × NZ with NC time steps, the program finds:
1. Count of local minima for each time step
2. Count of local maxima for each time step
3. Global minimum for each time step
4. Global maximum for each time step

## Implementation Details

The implementation uses a 3D domain decomposition approach where:
- The 3D volume is divided into PX × PY × PZ sub-domains
- Each process handles its own sub-domain
- Process 0 reads the input data and distributes it to all processes
- Each process computes local minima/maxima and global min/max for its sub-domain
- Results are gathered and combined using MPI_Reduce operations

## Compilation

To compile the program:

```bash
mpicc -o main2 main2.c
```

## Usage

The program takes 9 command-line arguments:

```bash
./main2 <input_file> PX PY PZ NX NY NZ NC <output_file>
```

Where:
- `input_file`: Path to the input data file
- `PX`: Number of processes in X-dimension
- `PY`: Number of processes in Y-dimension
- `PZ`: Number of processes in Z-dimension
- `NX`: Number of grid points in X-dimension
- `NY`: Number of grid points in Y-dimension
- `NZ`: Number of grid points in Z-dimension
- `NC`: Number of time steps (columns)
- `output_file`: Path to the output file

## Input File Format

The input file contains time series data for a 3D volume. The data is organized in XYZ order:
- Each row represents a point in the 3D volume
- Each column represents a time step
- The total number of rows is NX × NY × NZ
- The total number of columns is NC

## Output File Format

The output file contains:
1. Problem parameters (dimensions and process grid)
2. Local minima and maxima counts for each time step
3. Global minima and maxima values for each time step
4. Timing information (initialization, computation, total execution time)

## Running Tests

A test script is provided to run the program with different process configurations:

```bash
chmod +x run_test.sh
./run_test.sh
```

This will:
1. Compile the program
2. Create a small test input file
3. Run the program with different process configurations
4. Save the results in the test_data directory

## Constraints

- PX ≥ 1, PY ≥ 1, PZ ≥ 1
- NX ≤ 1024, NY ≤ 1024, NZ ≤ 1024
- NC ≤ 1000
- PX × PY × PZ must equal the total number of processes

## Performance Considerations

- The program uses a 3D domain decomposition to minimize communication overhead
- Each process only processes its own sub-domain
- Communication is minimized by having process 0 distribute the data once at the beginning
- Results are gathered at the end using efficient MPI_Reduce operations
