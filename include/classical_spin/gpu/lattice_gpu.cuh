#ifndef LATTICE_GPU_CUH
#define LATTICE_GPU_CUH

/**
 * CUDA-side declarations of the Lattice GPU backend (CUDA translation units
 * only): the device data (GPULatticeData), the Landau-Lifshitz right-hand
 * side functor (GPUODESystem) and the internal integration entry points
 * implemented in src/gpu/lattice_gpu.cu. Host C++ code uses the opaque API
 * in lattice_gpu_api.h.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

#include "classical_spin/gpu/ode/integrator.cuh"

// Device helpers and the element-wise kernels live in gpu_common_helpers.cuh;
// the steppers in gpu/ode/integrator.cuh. (The never-defined LatticeGPU class
// and its raw-pointer data structures that used to be declared here were
// removed: no translation unit implemented or used them.)

namespace gpu {

/**
 * GPU state vector type (thrust device vector)
 */
using GPUState = thrust::device_vector<double>;

/**
 * GPU lattice data structure - holds all interaction data on device
 */
struct GPULatticeData {
    // Spin configuration
    thrust::device_vector<double> spins;
    
    // Fields
    thrust::device_vector<double> field;
    thrust::device_vector<double> onsite;
    
    // Bilinear interactions
    thrust::device_vector<double> bilinear_vals;
    thrust::device_vector<size_t> bilinear_idx;
    thrust::device_vector<size_t> bilinear_counts;
    
    // Trilinear interactions (optional)
    thrust::device_vector<double> trilinear_vals;
    thrust::device_vector<size_t> trilinear_idx;
    thrust::device_vector<size_t> trilinear_counts;
    
    // Field drive pulse parameters
    thrust::device_vector<double> field_drive;
    double pulse_amp = 0.0;
    double pulse_width = 1.0;
    double pulse_freq = 0.0;
    double t_pulse_1 = 0.0;
    double t_pulse_2 = 0.0;
    
    // Lattice dimensions
    size_t lattice_size = 0;
    size_t spin_dim = 0;
    size_t N_atoms = 0;
    size_t max_bilinear = 0;
    size_t max_trilinear = 0;
    
    // Working arrays for integration (pre-allocated to avoid per-step allocation)
    // Basic working arrays
    thrust::device_vector<double> work_1;
    thrust::device_vector<double> work_2;
    thrust::device_vector<double> work_3;
    thrust::device_vector<double> local_field;
    
    // RK stage storage (pre-allocated, owned by the shared gpu::ode module).
    // Persisting the workspace across steps avoids per-step reallocation.
    ode::Workspace rk_ws;
    
    bool initialized = false;
};

/**
 * GPU ODE system functor for Landau-Lifshitz integration
 */
struct GPUODESystem {
    GPULatticeData& data;
    
    GPUODESystem(GPULatticeData& d) : data(d) {}
    
    /**
     * Compute dS/dt = S × H_eff on GPU
     * This is called by the integration routine
     */
    void operator()(const GPUState& x, GPUState& dxdt, double t) const;
};

/**
 * Create GPU lattice data from host arrays
 * Internal function - use the API version (create_gpu_lattice_data returning a handle) for C++ code
 */
GPULatticeData create_gpu_lattice_data_internal(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    const std::vector<double>& flat_field,
    const std::vector<double>& flat_onsite,
    const std::vector<double>& flat_bilinear,
    const std::vector<size_t>& flat_partners,
    const std::vector<size_t>& num_bilinear_per_site
);

/**
 * Set pulse parameters on GPU
 */
void set_gpu_pulse(
    GPULatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2
);

/**
 * Integrate and record (t, state) on the output grid (see integrate_gpu in
 * lattice_gpu_api.h for the methods and the grid).
 */
void integrate_gpu(
    GPUODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method = "dopri5",
    double abs_tol = 1e-8,
    double rel_tol = 1e-8
);

/**
 * One step of size dt (no error control).
 */
void step_gpu(
    GPUODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method = "rk4"
);

/**
 * Compute total energy on GPU
 */
double compute_energy_gpu(const GPULatticeData& data, const GPUState& state);

/**
 * Normalize spins on GPU
 */
void normalize_spins_gpu(GPUState& state, size_t lattice_size, size_t spin_dim, double spin_length);

} // namespace gpu

#endif // LATTICE_GPU_CUH

