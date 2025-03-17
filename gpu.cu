#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <iostream>

#define NUM_THREADS 256
#define MULTIPLIER  16
#define CELL_SIZE   MULTIPLIER* cutoff

// Put any static global variables here that you will use throughout the simulation.
int blks;

typedef struct particle_info_t {
    particle_t* particle;
    int bin_id;
} particle_info_t;

// array of particle infos, ordered by bin id
particle_info_t* cpu_particle_info;
// number of particles per bin
int* cpu_bin_counts;
// starting index for each bin, as a prefix sum of `bin_counts`
int* cpu_prefix_sum;

/**
 * CUDA Kernel to count the number of particles in each bin.
 *
 * Only modifies `bin_counts`, and does not update the `bin_id` in `particle_info`.
 */
__global__ void count_particles_in_bins(particle_info_t* particle_info, int* bin_counts, int num_parts, int num_bins_along_axis) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return; // If the thread exceeds the number of particles, return

    particle_info_t* cur_info = &(particle_info[tid]);
    particle_t* cur_particle = cur_info->particle;

    // Calculate the bin ID based on the particle's current position
    int grid_x = cur_particle->x / CELL_SIZE;
    int grid_y = cur_particle->y / CELL_SIZE;
    int bin_id = grid_y * num_bins_along_axis + grid_x;

    // Updating the bin_counts array
    atomicAdd(&bin_counts[bin_id], 1);
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void move_gpu(particle_info_t* particle_info, int num_parts, double size, int num_bins_along_axis) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_info_t* cur_info = &particle_info[tid];
    particle_t* p = cur_info->particle;
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }

    // update bin ID
    int bin_row = p->y / CELL_SIZE;
    int bin_col = p->x / CELL_SIZE;
    int bin_id = bin_row * num_bins_along_axis + bin_col;
    cur_info->bin_id = bin_id;
}

__global__ void init_particle_info(particle_t* parts, particle_info_t* particle_info, int num_parts, int num_bins_along_axis) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;


    particle_t* cur_particle = &parts[tid];
    int bin_row = cur_particle->y / CELL_SIZE;
    int bin_col = cur_particle->x / CELL_SIZE;
    int bin_id = bin_row * num_bins_along_axis + bin_col;

    // std::cout << "(" << bin_row << ", " << bin_col << ") id " << bin_id << std::endl;

    particle_info[tid] = {cur_particle, bin_id};

}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    // number of particles per thread
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // dimensions for neighbor checks
    int num_bins_along_axis = size / (CELL_SIZE) + 1;
    int num_bins = num_bins_along_axis * num_bins_along_axis;

    // Initialize the Arrays
    cudaMalloc(&cpu_particle_info, num_parts * sizeof(particle_info_t));
    cudaMalloc(&cpu_bin_counts, num_bins * sizeof(int));
    cudaMalloc(&cpu_prefix_sum, num_bins * sizeof(int));

    // std::cout << "done malloc" << std::endl;

    // Initialize the particle_info array
    init_particle_info<<<blks, NUM_THREADS>>>(parts, cpu_particle_info, num_parts, num_bins_along_axis);

    // Count the number of particles in each bin
    cudaMemset(cpu_bin_counts, 0, num_bins * sizeof(int));
    count_particles_in_bins<<<blks, NUM_THREADS>>>(cpu_particle_info, cpu_bin_counts, num_parts, num_bins_along_axis);

    // std::cout << "done init bin counts array" << std::endl;

    // Do a Prefix Sum to get the starting indices prefix_sum
    auto device_bin_counts = thrust::device_pointer_cast(cpu_bin_counts);
    auto device_prefix_sum = thrust::device_pointer_cast(cpu_prefix_sum);
    thrust::exclusive_scan(thrust::device, device_bin_counts, device_bin_counts + num_bins, device_prefix_sum);
    // std::cout << "done init prefix sum" << std::endl;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // dimensions for neighbor checks
    int num_bins_along_axis = size / (CELL_SIZE) + 1;
    int num_bins = num_bins_along_axis * num_bins_along_axis;

    // Compute forces
    // TODO: rewrite to use bins
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts); // one worker per particle
    // std::cout << "done compute forces" << std::endl;

    // Move particles (Modified to also update the bin_ids upon moving)
    move_gpu<<<blks, NUM_THREADS>>>(cpu_particle_info, num_parts, size, num_bins_along_axis);
    // std::cout << "done move particles" << std::endl;

    // Update the bin_count array
    cudaMemset(cpu_bin_counts, 0, num_bins * sizeof(int));
    count_particles_in_bins<<<blks, NUM_THREADS>>>(cpu_particle_info, cpu_bin_counts, num_bins, num_bins_along_axis);
    // std::cout << "done count particles in bins" << std::endl;

    // Calculate the prefix sum of the count array
    auto device_bin_counts = thrust::device_pointer_cast(cpu_bin_counts);
    auto device_prefix_sum = thrust::device_pointer_cast(cpu_prefix_sum);
    thrust::exclusive_scan(thrust::device, device_bin_counts, device_bin_counts + num_bins, device_prefix_sum);
    // std::cout << "done exclusive scan" << std::endl;
}
