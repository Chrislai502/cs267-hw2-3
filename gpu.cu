#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

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
struct ParticleSOA {
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* ax;
    double* ay;
    int* bin_ids;
    int* particle_indices;
};

ParticleSOA soa;

// number of particles per bin
int* cpu_bin_counts;
// starting index for each bin, as a prefix sum of `bin_counts`
int* cpu_prefix_sum;

// Particle Indices Array
double *x_temp, *y_temp, *vx_temp, *vy_temp, *ax_temp, *ay_temp;
int *indices_temp, *bin_counters;

/**
 * CUDA Kernel to count the number of particles in each bin. SOA style
 */
__global__ void count_particles_in_bins_soa(ParticleSOA soa, int* bin_counts, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return; // If the thread exceeds the number of particles, return

    // Then in count_particles_in_bins_soa, reuse it
    int bin_id = soa.bin_ids[tid];
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
        // Use reversed mapping: write to original positions in AOS
        int particle_index = soa.particle_indices[tid];
        
        // These reads are coalesced (good)
        double x = soa.x[tid];
        double y = soa.y[tid];
        double vx = soa.vx[tid];
        double vy = soa.vy[tid];
        double ax = soa.ax[tid];
        double ay = soa.ay[tid];
        
        // Write to the appropriate location in AOS (scattered writes)
        aos[particle_index].x = x;
        aos[particle_index].y = y;
        aos[particle_index].vx = vx;
        aos[particle_index].vy = vy;
        aos[particle_index].ax = ax;
        aos[particle_index].ay = ay;
    }
}

__global__ void bin_bucketing_kernel(
    // Source data (current SOA)
    ParticleSOA soa,
    // Destination data (temporary storage)
    double* x_dst, double* y_dst, double* vx_dst, double* vy_dst, 
    double* ax_dst, double* ay_dst, int* indices_dst,
    // Bin information
    int* bin_offsets,   // prefix sum array with starting positions
    int* bin_counters,  // temporary counters for each bin
    int num_parts
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    
    // Get the bin id for this particle
    int bin_id = soa.bin_ids[tid];
    
    // Calculate destination position using atomic add
    // This ensures each thread gets a unique position within its bin
    int bin_pos = atomicAdd(&bin_counters[bin_id], 1);
    int dst_idx = bin_offsets[bin_id] + bin_pos;
    
    // Copy data to the destination arrays - these reads and writes are coalesced
    double x = soa.x[tid];
    double y = soa.y[tid];
    double vx = soa.vx[tid];
    double vy = soa.vy[tid];
    double ax = soa.ax[tid];
    double ay = soa.ay[tid];

    // Write data to the destination arrays
    x_dst[dst_idx] = x;
    y_dst[dst_idx] = y;
    vx_dst[dst_idx] = vx;
    vy_dst[dst_idx] = vy;
    ax_dst[dst_idx] = ax;
    ay_dst[dst_idx] = ay;

    indices_dst[dst_idx] = soa.particle_indices[tid];
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
__device__ int ALL_ADJ[9][2] = {
    {-1, -1}, {-1, 0}, {-1, 1}, 
    {0 , -1}, {0 , 0}, {0 , 1},                            
    {1 , -1}, {1 , 0}, {1 , 1},  };

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

__global__ void move_gpu_soa(ParticleSOA soa, int num_parts, double size, int num_bins_along_axis, int* bin_counts) {
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

    // Count the number of particles in each bin
    atomicAdd(&bin_counts[bin_id], 1);
}

__global__ void init_particle_bins_soa(ParticleSOA soa, int num_parts, int num_bins_along_axis, int* bin_counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    double x = soa.x[tid];    // Fetch first for memory coalescing
    double y = soa.y[tid];
    int bin_row = y / CELL_SIZE;
    int bin_col = x / CELL_SIZE;
    int bin_id = bin_row * num_bins_along_axis + bin_col;
    soa.bin_ids[tid] = bin_id; // Coalesced writing

    // Count the number of particles in each bin
    atomicAdd(&bin_counts[bin_id], 1);
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
    cudaMalloc(&soa.particle_indices, num_parts * sizeof(int));

    // Initilize the particle_indices array
    thrust::sequence(thrust::device, thrust::device_pointer_cast(soa.particle_indices),
                        thrust::device_pointer_cast(soa.particle_indices) + num_parts);

    // Packing: Convert Array of Structs to Struct of Arrays
    aos_to_soa<<<blks, NUMBER_THREADS>>>(parts, soa, num_parts);

    // Initialize the Particle Bin IDs array
    cudaMemset(cpu_bin_counts, 0, num_bins * sizeof(int));
    init_particle_bins_soa<<<blks, NUMBER_THREADS>>>(soa, num_parts, num_bins_along_axis, cpu_bin_counts);

    // std::cout << "done malloc" << std::endl;

    // Count the number of particles in each bin
    // count_particles_in_bins_soa<<<blks, NUMBER_THREADS>>>(soa, cpu_bin_counts, num_parts);

    // std::cout << "done init bin counts array" << std::endl;

    // Do a Prefix Sum to get the starting indices prefix_sum
    auto device_bin_counts = thrust::device_pointer_cast(cpu_bin_counts);
    auto device_prefix_sum = thrust::device_pointer_cast(cpu_prefix_sum);
    thrust::exclusive_scan(thrust::device, device_bin_counts, device_bin_counts + num_bins, device_prefix_sum);

    // Reorganization

    // Initializing temporary storage for reordering
    cudaMalloc(&x_temp, num_parts * sizeof(double));
    cudaMalloc(&y_temp, num_parts * sizeof(double));
    cudaMalloc(&vx_temp, num_parts * sizeof(double));
    cudaMalloc(&vy_temp, num_parts * sizeof(double));
    cudaMalloc(&ax_temp, num_parts * sizeof(double));
    cudaMalloc(&ay_temp, num_parts * sizeof(double));
    cudaMalloc(&indices_temp, num_parts * sizeof(int));
    cudaMalloc(&bin_counters, num_bins * sizeof(int));

    // Reset bin counters to zero
    cudaMemset(bin_counters, 0, num_bins * sizeof(int));
    
    // Perform the bin bucketing
    bin_bucketing_kernel<<<blks, NUMBER_THREADS>>>(
        soa,
        x_temp, y_temp, vx_temp, vy_temp, ax_temp, ay_temp, indices_temp,
        cpu_prefix_sum, bin_counters, num_parts
    );
    
    // Copy the reordered data back to the original arrays
    cudaMemcpy(soa.x, x_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.y, y_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.vx, vx_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.vy, vy_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.ax, ax_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.ay, ay_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.particle_indices, indices_temp, num_parts * sizeof(int), cudaMemcpyDeviceToDevice);
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
    // Also include counting now
    cudaMemset(cpu_bin_counts, 0, num_bins * sizeof(int));
    move_gpu_soa<<<blks, NUMBER_THREADS>>>(soa, num_parts, size, num_bins_along_axis, cpu_bin_counts);
    // std::cout << "done move particles" << std::endl;

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    move_gpu_time += std::chrono::duration<double>(end - start).count();
    start = std::chrono::steady_clock::now();
#endif

    // Update the bin_count array
    // count_particles_in_bins_soa<<<blks, NUMBER_THREADS>>>(soa, cpu_bin_counts, num_parts);
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

    // Reset bin counters to zero
    cudaMemset(bin_counters, 0, num_bins * sizeof(int));
    
    // Perform the bin bucketing
    bin_bucketing_kernel<<<blks, NUMBER_THREADS>>>(
        soa,
        x_temp, y_temp, vx_temp, vy_temp, ax_temp, ay_temp, indices_temp,
        cpu_prefix_sum, bin_counters, num_parts
    );
    
    // Copy the reordered data back to the original arrays
    cudaMemcpy(soa.x, x_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.y, y_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.vx, vx_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.vy, vy_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.ax, ax_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.ay, ay_temp, num_parts * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(soa.particle_indices, indices_temp, num_parts * sizeof(int), cudaMemcpyDeviceToDevice);

#ifdef ENABLE_TIMERS
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    sort_time += std::chrono::duration<double>(end - start).count();

    auto total_end = std::chrono::steady_clock::now();
    total_compute_time += std::chrono::duration<double>(total_end - total_start).count();
#endif

    soa_to_aos<<<blks, NUMBER_THREADS>>>(soa, parts, num_parts);
}
