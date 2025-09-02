# Parallel Extrema Computation with MPI

This project implements a high-performance **MPI-based parallel program** for computing local and global extrema over a **4D time-series dataset** representing 3D volumes. The code was developed and tested on the **Param Rudra cluster**, demonstrating strong scalability and efficient communication patterns.

## Overview

- Designed using **3D domain decomposition** to distribute volumetric data across processes.  
- Implemented **halo exchange** with non-blocking communication for face neighbors, ensuring efficient data sharing.  
- Applied **parallel I/O with MPI datatypes** to eliminate centralized bottlenecks and improve scalability.  
- Optimized computation of local and global minima/maxima with **MPI collective reductions**.  

## Key Optimizations

- **Parallel I/O** using `MPI_File_read_all` and derived subarray datatypes for precise data mapping.  
- **Non-blocking communication** (`MPI_Isend`/`MPI_Irecv`) to overlap computation with halo exchanges.  
- **Reduced memory copies** and compact global reductions for scalability.  

## Results

- Achieved strong scalability on **8–64 MPI processes** across datasets up to `64×64×96×7`.  
- **Compute phase scaled efficiently** due to 3D decomposition and localized operations.  
- **Read time became the main bottleneck** at higher process counts, reflecting parallel file system limits.  
- Optimized version (parallel I/O + non-blocking communication) outperformed naive centralized I/O and synchronous communication.  

## Execution

The program is written in C with MPI.  
To compile and run on Param Rudra (or any MPI-enabled cluster):  

```bash
mpicc -o extrema src.c
sbatch job1.sh
