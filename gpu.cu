#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS   256
#endif
#define MULTIPLIER    1
#define CELL_SIZE     (MULTIPLIER * cutoff)
#ifdef ENABLE_TIMERS
double compute_force_time = 0;
double move_gpu_time = 0;
double count_particles_in_bins_time = 0;
double exclusive_scan_time = 0;
double sort_time = 0;
double total_compute_time = 0;
#endif

// Put any static global variables here that you will use throughout the simulation.
int blks;

// typedef struct particle_info_t {
//     particle_t* particle;
//     int bin_id;
// } particle_info_t;

struct ParticleSOA {
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* ax;
    double* ay;
    int* bin_ids;
};

ParticleSOA soa;

// number of particles per bin
int* cpu_bin_counts;
// starting index for each bin, as a prefix sum of `bin_counts`
int* cpu_prefix_sum;
// Particle Indices Array
// int* particle_indices; // Sort this
// int* particle_bin_ids; // based on this using thrust



// /**
//  * CUDA Kernel to count the number of particles in each bin.
//  *
//  * Only modifies `bin_counts`, and does not update the `bin_id` in `particle_info`.
//  */
// __global__ void count_particles_in_bins(particle_t * particles, int* bin_counts,
//                                         int num_parts, int num_bins_along_axis) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= num_parts) return; // If the thread exceeds the number of particles, return

//     particle_t* cur_particle = &particles[tid];

//     // Calculate the bin ID based on the particle's current position
//     int grid_x = cur_particle->x / CELL_SIZE;
//     int grid_y = cur_particle->y / CELL_SIZE;
//     int bin_id = grid_y * num_bins_along_axis + grid_x;

//     // Updating the bin_counts array
//     atomicAdd(&bin_counts[bin_id], 1);
// }

/**
 * CUDA Kernel to count the number of particles in each bin. SOA style
 */
__global__ void count_particles_in_bins_soa(ParticleSOA soa, int* bin_counts, int num_parts, int num_bins_along_axis) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return; // If the thread exceeds the number of particles, return

    double x = soa.x[tid];    // Fetch first for memory coalescing
    double y = soa.y[tid]; 
    int grid_x = x / CELL_SIZE;
    int grid_y = y / CELL_SIZE;
    int bin_id = grid_y * num_bins_along_axis + grid_x;

    atomicAdd(&bin_counts[bin_id], 1);
}


/**
 * CUDA Kernel to convert the Array of Structs into struct of arrays
 */
__global__ void aos_to_soa(const particle_t* aos, ParticleSOA soa, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_parts){
        soa.x[tid] = aos[tid].x;
        soa.y[tid] = aos[tid].y;
        soa.vx[tid] = aos[tid].vx;
        soa.vy[tid] = aos[tid].vy;
        soa.ax[tid] = aos[tid].ax;
        soa.ay[tid] = aos[tid].ay;
    }    
}

__global__ void soa_to_aos(ParticleSOA soa, particle_t* aos, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_parts) {
        aos[tid].x  = soa.x[tid];
        aos[tid].y  = soa.y[tid];
        aos[tid].vx = soa.vx[tid];
        aos[tid].vy = soa.vy[tid];
        aos[tid].ax = soa.ax[tid];
        aos[tid].ay = soa.ay[tid];
    }
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

// list of adjacencies
__device__ int ALL_ADJ[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1},
                                {1, -1},  {1, 0},  {1, 1},  {0, 0}};

// /**
//  * Compute forces, using the bins provided
//  */
// __global__ void compute_forces_bins(particle_t* particles, int* particle_indices, int* bin_counts,
//                                     int* prefix_sum, int num_parts, int num_bins_along_axis) {
//     // Get thread (particle) ID
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= num_parts) return;
    
//     int particle_id = particle_indices[tid]; // Fetching it into registers first for better coalescing
//     particle_t* particle = &particles[particle_id]; // TODO: This is random access for now, pending sorting this with SOA
    
//     int cur_bin_r = particle->y / CELL_SIZE; // TODO: Some Hack     int cur_bin_r = __double2int_rz(particle->y / CELL_SIZE);  // Faster integer conversion
//     int cur_bin_c = particle->x / CELL_SIZE;

//     // zero out acceleration
//     particle->ax = particle->ay = 0;

//     // Iterate through all neighboring bins
//     for (int adj_idx = 0; adj_idx < 9; ++adj_idx) {
//         int adj_grid_r = cur_bin_r + ALL_ADJ[adj_idx][0];
//         int adj_grid_c = cur_bin_c + ALL_ADJ[adj_idx][1];

//         // check bounds
//         if (adj_grid_r < 0 || adj_grid_r >= num_bins_along_axis || adj_grid_c < 0 ||
//             adj_grid_c >= num_bins_along_axis) {
//             continue;
//         }

//         // get the index of the adjacent bin
//         int adj_grid_id = adj_grid_r * num_bins_along_axis + adj_grid_c;
//         int bin_offset = prefix_sum[adj_grid_id];
//         int bin_size = bin_counts[adj_grid_id];

//         // if (bin_size == 0) { // Remove for cuda kernel cycle optimization
//         //     // nothing to iterate
//         //     continue;
//         // }
        
//         int end_idx = bin_offset + bin_size; // Put it in registers first

//         // iterate through all particles in the bin to compute forces
//         for (int bin_particle_idx = bin_offset; bin_particle_idx < end_idx; ++bin_particle_idx) {
//             int neighbor_id = particle_indices[bin_particle_idx];
//             apply_force_gpu(*particle, particles[neighbor_id]);
//         }
//     }
// }

__global__ void compute_forces_bins_soa(ParticleSOA soa, int* bin_counts, int* prefix_sum, int num_parts, int num_bins_along_axis) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    
    double x = soa.x[tid]; 
    double y = soa.y[tid];
    int cur_bin_r = y / CELL_SIZE;
    int cur_bin_c = x / CELL_SIZE;

    // Zero out acceleration
    soa.ax[tid] = soa.ay[tid] = 0;

    // Iterate through all neighboring bins
    for (int adj_idx = 0; adj_idx < 9; ++adj_idx) {
        int adj_grid_r = cur_bin_r + ALL_ADJ[adj_idx][0];
        int adj_grid_c = cur_bin_c + ALL_ADJ[adj_idx][1];
        
        // check bounds
        if (adj_grid_r < 0 || adj_grid_r >= num_bins_along_axis || adj_grid_c < 0 ||
            adj_grid_c >= num_bins_along_axis) {
            continue;
        }

        // get the index of the adjacent bin
        int adj_grid_id = adj_grid_r * num_bins_along_axis + adj_grid_c;
        int bin_offset = prefix_sum[adj_grid_id];
        int bin_size = bin_counts[adj_grid_id];

        if (bin_size == 0) { // Remove for cuda kernel cycle optimization
            // nothing to iterate
            continue;
        }

        int end_idx = bin_offset + bin_size; // Put it in registers first

        // iterate through all particles in the bin to compute forces
        for (int bin_particle_idx = bin_offset; bin_particle_idx < end_idx; ++bin_particle_idx) {
            int neighbor_id = bin_particle_idx;// TODO: Check this
            double neigh_x = soa.x[neighbor_id]; // Fetch for memory coalescing
            double neigh_y = soa.y[neighbor_id];
            double dx = neigh_x - x;
            double dy = neigh_y - y;
            double r2 = dx * dx + dy * dy;
            if (r2 > cutoff * cutoff) continue;
                
            r2 = fmax( r2, min_r*min_r );
            double r = sqrt(r2);
        
            //  very simple short-range repulsive force
            double coef = (1 - cutoff / r) / r2 / mass;
            soa.ax[tid] += coef * dx;
            soa.ay[tid] += coef * dy;
        }
    }
}

__global__ void move_gpu(particle_t* particles, int * particle_indices, int* particle_bin_ids, int num_parts, double size,
                         int num_bins_along_axis) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int particle_id = particle_indices[tid]; // Fetching it into registers first for better coalescing (after SOA)
    particle_t* p = &particles[particle_id];

    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //  bounce from walls
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
    particle_bin_ids[tid] = bin_id;
}

__global__ void move_gpu_soa(ParticleSOA soa, int num_parts, double size, int num_bins_along_axis) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    //  slightly simplified Velocity Verlet integration
    // Retrieve current acceleration; these are read-only in this kernel
    double ax = soa.ax[tid];
    double ay = soa.ay[tid];

    // Promote register caching and coalesced access
    double* vx = &soa.vx[tid];
    double* vy = &soa.vy[tid];
    double* x = &soa.x[tid];
    double* y = &soa.y[tid];

    *vx += ax * dt;
    *vy += ay * dt;
    *x += *vx * dt;
    *y += *vy * dt;

    //  bounce from walls
    while (*x < 0 || *x > size) {
        *x = *x < 0 ? -( *x ) : 2 * size - *x;
        *vx = -( *vx );
    }
    while (*y < 0 || *y > size) {
        *y = *y < 0 ? -( *y ) : 2 * size - *y;
        *vy = -( *vy );
    }

    // update bin ID
    int bin_row = *y / CELL_SIZE;
    int bin_col = *x / CELL_SIZE;
    int bin_id = bin_row * num_bins_along_axis + bin_col;
    soa.bin_ids[tid] = bin_id;
}

// __global__ void init_particle_bins(particle_t* parts, int* particle_bin_ids, int num_parts, int num_bins_along_axis){
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= num_parts) return;

//     particle_t* cur_particle = &parts[tid];
//     int bin_row = cur_particle->y / CELL_SIZE;
//     int bin_col = cur_particle->x / CELL_SIZE;
//     int bin_id = bin_row * num_bins_along_axis + bin_col;

//     particle_bin_ids[tid] = bin_id;
// }

__global__ void init_particle_bins_soa(ParticleSOA soa, int num_parts, int num_bins_along_axis){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    double x = soa.x[tid];    // Fetch first for memory coalescing
    double y = soa.y[tid];
    int bin_row = y / CELL_SIZE;
    int bin_col = x / CELL_SIZE;
    int bin_id = bin_row * num_bins_along_axis + bin_col;
    soa.bin_ids[tid] = bin_id; // Coalesced writing
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    // number of particles per thread
    blks = (num_parts + NUMBER_THREADS - 1) / NUMBER_THREADS;

    // dimensions for neighbor checks
    int num_bins_along_axis = size / (CELL_SIZE) + 1;
    int num_bins = num_bins_along_axis * num_bins_along_axis;

    // Initialize the Arrays
    cudaMalloc(&cpu_bin_counts,  num_bins * sizeof(int));
    cudaMalloc(&cpu_prefix_sum,  num_bins * sizeof(int));

    // Initialize SOA
    // cudaMalloc(&soa_gpu, sizeof(ParticleSOA)); // Don't need because SOA is in host
    cudaMalloc(&soa.x,  num_parts * sizeof(double));
    cudaMalloc(&soa.y,  num_parts * sizeof(double));
    cudaMalloc(&soa.vx, num_parts * sizeof(double));
    cudaMalloc(&soa.vy, num_parts * sizeof(double));
    cudaMalloc(&soa.ax, num_parts * sizeof(double));
    cudaMalloc(&soa.ay, num_parts * sizeof(double));
    cudaMalloc(&soa.bin_ids, num_parts * sizeof(int));

    // Packing: Convert Array of Structs to Struct of Arrays
    aos_to_soa<<<blks, NUMBER_THREADS>>>(parts, soa, num_parts);

    // Initialize the Particle Bin IDs array
    init_particle_bins_soa<<<blks, NUMBER_THREADS>>>(soa, num_parts, num_bins_along_axis);

    // std::cout << "done malloc" << std::endl;

    // Count the number of particles in each bin
    cudaMemset(cpu_bin_counts, 0, num_bins * sizeof(int));
    count_particles_in_bins_soa<<<blks, NUMBER_THREADS>>>(soa, cpu_bin_counts, num_parts, num_bins_along_axis);

    // std::cout << "done init bin counts array" << std::endl;

    // Do a Prefix Sum to get the starting indices prefix_sum
    auto device_bin_counts = thrust::device_pointer_cast(cpu_bin_counts);
    auto device_prefix_sum = thrust::device_pointer_cast(cpu_prefix_sum);
    thrust::exclusive_scan(thrust::device, device_bin_counts, device_bin_counts + num_bins, device_prefix_sum);

    // Zip everything together
    thrust::device_ptr<double> d_x(soa.x);
    thrust::device_ptr<double> d_y(soa.y);
    thrust::device_ptr<double> d_vx(soa.vx);
    thrust::device_ptr<double> d_vy(soa.vy);
    thrust::device_ptr<double> d_ax(soa.ax);
    thrust::device_ptr<double> d_ay(soa.ay);
    thrust::device_ptr<int> device_particle_bin_ids(soa.bin_ids);
    
    // std::cout << "done init exclusive scan" << std::endl;

    // TODO: Do we really need the device_particle_bin_ids to be sorted? because it is only used as keys once.
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_x, d_y, d_vx, d_vy, d_ax, d_ay));
    thrust::sort_by_key(thrust::device, device_particle_bin_ids, device_particle_bin_ids + num_parts, zip_begin);
    // std::cout << "done init prefix sum" << std::endl;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    auto total_start = std::chrono::steady_clock::now();
#endif

    // dimensions for neighbor checks
    int num_bins_along_axis = size / (CELL_SIZE) + 1;
    int num_bins = num_bins_along_axis * num_bins_along_axis;

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
#endif

    // Compute forces
    // TODO: rewrite to use bins
    compute_forces_bins_soa<<<blks, NUMBER_THREADS>>>(soa, cpu_bin_counts, cpu_prefix_sum,
                                               num_parts,
                                               num_bins_along_axis); // one worker per particle
    // std::cout << "done compute forces" << std::endl;
#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    compute_force_time += std::chrono::duration<double>(end - start).count();
    start = std::chrono::steady_clock::now();
#endif

    // Move particles (Modified to also update the bin_ids upon moving)
    move_gpu_soa<<<blks, NUMBER_THREADS>>>(soa, num_parts, size, num_bins_along_axis);
    // std::cout << "done move particles" << std::endl;

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    move_gpu_time += std::chrono::duration<double>(end - start).count();
    start = std::chrono::steady_clock::now();
#endif

    // Update the bin_count array
    cudaMemset(cpu_bin_counts, 0, num_bins * sizeof(int));
    count_particles_in_bins_soa<<<blks, NUMBER_THREADS>>>(soa, cpu_bin_counts, num_parts, num_bins_along_axis);
    // std::cout << "done count particles in bins" << std::endl;

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    count_particles_in_bins_time += std::chrono::duration<double>(end - start).count();
    start = std::chrono::steady_clock::now();
#endif

    // Calculate the prefix sum of the count array
    auto device_bin_counts = thrust::device_pointer_cast(cpu_bin_counts);
    auto device_prefix_sum = thrust::device_pointer_cast(cpu_prefix_sum);
    thrust::exclusive_scan(thrust::device, device_bin_counts, device_bin_counts + num_bins, device_prefix_sum);
    // std::cout << "done exclusive scan" << std::endl;

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    exclusive_scan_time += std::chrono::duration<double>(end - start).count();
    start = std::chrono::steady_clock::now();
#endif

    // Zip everything together
    thrust::device_ptr<double> d_x(soa.x);
    thrust::device_ptr<double> d_y(soa.y);
    thrust::device_ptr<double> d_vx(soa.vx);
    thrust::device_ptr<double> d_vy(soa.vy);
    thrust::device_ptr<double> d_ax(soa.ax);
    thrust::device_ptr<double> d_ay(soa.ay);
    thrust::device_ptr<int> device_particle_bin_ids(soa.bin_ids);
    
    // TODO: Do we really need the device_particle_bin_ids to be sorted? because it is only used as keys once.
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_x, d_y, d_vx, d_vy, d_ax, d_ay));

    // Sort the Particle ID Array
    // https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1ga7a399a3801f1684d465f4adbac462982.html
    thrust::sort_by_key(thrust::device, device_particle_bin_ids, device_particle_bin_ids + num_parts, zip_begin);

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    sort_time += std::chrono::duration<double>(end - start).count();

    auto total_end = std::chrono::steady_clock::now();
    total_compute_time += std::chrono::duration<double>(total_end - total_start).count();
#endif

    soa_to_aos<<<blks, NUMBER_THREADS>>>(soa, parts, num_parts);
}
