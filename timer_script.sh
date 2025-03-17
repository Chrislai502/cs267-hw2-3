#!/bin/bash

# MOVE THIS SCRIPT TO THE BUILD DIRECTORY TO RUN

# Define the range of NUM_THREADS you want to test
THREAD_COUNTS=(32 64 128 256 512 1024)

# Number of particles for the simulation
NUM_PARTICLES=10000000

# Loop through different thread counts and run the program
for NUMBER_THREADS in "${THREAD_COUNTS[@]}"; do
    echo "Running with NUMBER_THREADS = $NUMBER_THREADS"

    # Set the NUMBER_THREADS and ENABLE_TIMERS flags, then recompile
    cmake -DCMAKE_CUDA_FLAGS="-DNUMBER_THREADS=$NUMBER_THREADS -DENABLE_TIMERS" ..
    make

    # Run the program with -n flag
    ./gpu -n $NUM_PARTICLES

    # Add a separator for clarity in the output
    echo "---------------------------------------"
done
