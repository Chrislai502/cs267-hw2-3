#!/usr/bin/env bash


# without timers
rm -r ./build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null
make > /dev/null
cd ..

printf "\n=== WITHOUT TIMERS ===\n\n"

for n in 1000 5000 10000 50000 100000 500000 1000000 5000000 10000000; do
    for i in 1 2 3; do
        ./build/gpu -n $n
    done
done


# with timers
rm -r ./build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-DENABLE_TIMERS" .. > /dev/null
make > /dev/null
cd ..

printf "\n=== WITH TIMERS ===\n\n"

for n in 1000 5000 10000 50000 100000 500000 1000000 5000000 10000000; do
    for i in 1 2 3; do
        ./build/gpu -n $n
        # add new line
        echo
    done
done
