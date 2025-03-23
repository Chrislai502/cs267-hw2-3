#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>


#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int* bin_counts = nullptr;
int* bin_ids = nullptr;
particle_t* particle_ids = nullptr;
int* keep_track_array = nullptr;

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

// Apply the force from neighbor to particle
inline void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
inline void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
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

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
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
}

__global__ void update_bin_ids(particle_t* parts, int* bin_counts, int num_parts, double size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    double cellsize = cutoff;
    int gridsize = (size / cellsize) + 1;

    int grid_r = parts[tid].y / cellsize;
    int grid_c = parts[tid].x / cellsize;
    atomicAdd(&bin_counts[grid_r * gridsize + grid_c], 1);
}

__global__ void update_particle_ids(particle_t* parts, int* bin_ids, int* keep_track_array, particle_t* particle_ids, int num_parts, double size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    double cellsize = cutoff;
    int gridsize = (size / cellsize) + 1;

    // populate particles_ids sorted by bins
    int grid_r = parts[tid].y / cellsize;
    int grid_c = parts[tid].x / cellsize;
    int particle_idx = bin_ids[grid_r * gridsize + grid_c] + keep_track_array[grid_r * gridsize + grid_c];
    atomicAdd(&keep_track_array[grid_r * gridsize + grid_c], 1);
    particle_ids[particle_idx] = parts[tid];
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    double cellsize = cutoff;
    int gridsize = (size / cellsize) + 1;

    int num_bins = gridsize * gridsize;
    bin_counts = new int[num_bins];
    bin_ids = new int[num_bins];
    cudaMalloc(&bin_ids, num_bins * sizeof(int));
    cudaMalloc(&bin_counts, num_bins * sizeof(int));
    cudaMemset(bin_ids, 0, num_bins * sizeof(int));
    cudaMemset(bin_counts, 0, num_bins * sizeof(int));

    std::cout << "here 1" << std::endl;

    // reset particle_ids and bin_ids
    update_bin_ids<<<blks, NUM_THREADS>>>(parts, bin_counts, num_parts, size);

    auto device_bin_counts = thrust::device_pointer_cast(bin_counts);
    auto device_bin_ids = thrust::device_pointer_cast(bin_ids);

    std::cout << "here 2" << std::endl;

    // bin_ids is now an array of the index of each bin
    thrust::exclusive_scan(thrust::device, device_bin_counts, device_bin_counts + num_bins, device_bin_ids);
    particle_ids = new particle_t[num_parts];
    cudaMalloc(&particle_ids, num_parts * sizeof(int));

    std::cout << "here 3" << std::endl;

    cudaMalloc(&keep_track_array, gridsize * gridsize * sizeof(int));
    update_particle_ids<<<blks, NUM_THREADS>>>(parts, bin_ids, keep_track_array, particle_ids, num_parts, size);
    std::cout << "here 4 " << device_bin_ids[0] << " " << device_bin_ids[1] << std::endl;
}

// list of adjacencies
static int ALL_ADJ[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                            {0, 1},   {1, -1}, {1, 0},  {1, 1}, {0, 0}};

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    // compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    // move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

    std::cout << "here 4.5" << std::endl;

    // convert parts to cpu for testing
    particle_t* cpu_parts = new particle_t[num_parts];
    cudaMemcpy(cpu_parts, parts, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);

    double cellsize = cutoff;
    int gridsize = (size / cellsize) + 1;
    int num_bins = gridsize * gridsize;

    std::cout << "here 4.6" << std::endl;

    for (int b = 0; b < num_bins; ++b) {
        // iterate through particles in bin
        std::cout << "heree " << bin_ids[0] << " " << bin_ids[1] << std::endl;
        for (int par_id = bin_ids[b]; par_id < bin_ids[b] + bin_counts[b]; ++par_id) {
            particle_t* cur_particle = &particle_ids[par_id];
            cur_particle->ax = cur_particle->ay = 0;
            
            int r = cur_particle->y / cellsize;
            int c = cur_particle->x / cellsize;

            std::cout << "here 4.7" << std::endl;

            for (int i = 0; i < 9; ++i) {
                int new_r = r + ALL_ADJ[i][0];
                int new_c = c + ALL_ADJ[i][1];

                std::cout << "here 4.8" << std::endl;
                
                int neigh_bin_idx = new_r * gridsize + new_c;
                if (neigh_bin_idx >= 0 && neigh_bin_idx < num_bins) {
                    int particle_idx = bin_ids[neigh_bin_idx];
                    // neighbors are particles in neighboring cells
                    std::cout << "here 4.9" << std::endl;
                    for (int neigh = particle_idx; neigh < particle_idx + bin_counts[neigh_bin_idx]; ++neigh) {
                        particle_t* neigh_particle = &particle_ids[neigh];
                        std::cout << "here 4.95" << std::endl;
                        apply_force(*cur_particle, *neigh_particle);
                    }
                }
            }            
        }
    }

    std::cout << "here 5" << std::endl;

    // move and adjust grid assignments for each particle
    for (int i = 0; i < num_parts; ++i) {
        particle_t* particle = &particle_ids[i];
        int r = particle->y / cellsize;
        int c = particle->x / cellsize;
        
        // move the particle to its new grid cell
        move(*particle, size);
        // move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

    }
    // reset particle_ids and bin_ids
    cudaMemset(bin_counts, 0, num_bins * sizeof(int));

    std::cout << "here 6" << std::endl;

    update_bin_ids<<<blks, NUM_THREADS>>>(parts, bin_counts, num_parts, size);

    std::cout << "here 7" << std::endl;

    // bin_ids is now an array of the index of each bin
    thrust::exclusive_scan(thrust::host, bin_counts, bin_counts + num_bins, bin_ids);
    particle_ids = new particle_t[num_parts];
    cudaMemset(particle_ids, 0, (num_parts) * sizeof(int));

    std::cout << "here 8" << std::endl;

    cudaMemset(keep_track_array, 0, gridsize * gridsize * sizeof(int));
    update_particle_ids<<<blks, NUM_THREADS>>>(parts, bin_ids, keep_track_array, particle_ids, num_parts, size);

    std::cout << "here 9" << std::endl;

    delete[] cpu_parts;
}
