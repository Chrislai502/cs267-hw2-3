#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
} particle_t;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size);
void simulate_one_step(particle_t* parts, int num_parts, double size);

#endif

#ifdef ENABLE_TIMERS
extern double compute_force_time;
extern double move_particles_and_count_time;
extern double exclusive_scan_time;
extern double bin_bucketing_time;
extern double repacking_time;
extern double total_compute_time;
#endif
