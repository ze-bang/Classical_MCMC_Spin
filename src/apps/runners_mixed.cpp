/**
 * runners_mixed.cpp — spin_solver runners for MixedLattice (TmFeO3 SU(2)+SU(3)).
 *
 * Split out of `spin_solver.cpp`; see `src/apps/spin_solver_runners.h`.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/lattice.h"       // Lattice::generate_geometric_temperature_ladder
#include "classical_spin/lattice/mixed_lattice.h"
#include "pt_runner_common.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;

/**
 * Run simulated annealing on MixedLattice (SU(2)+SU(3)).
 */
void run_simulated_annealing_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.simulated_annealing(
            config.T_start,
            config.T_end,
            config.annealing_steps,
            config.gaussian_move,
            config.cooling_rate,
            trial_dir,
            config.save_observables,
            config.T_zero,
            config.n_deterministics,
        config.twist_sweep_count
        );
        

        if (rank == 0) {
            ofstream energy_file(trial_dir + "/final_energy.txt");
            energy_file << "Energy Density: " << lattice.energy_density() << "\n";
            energy_file.close();
            
            cout << "Trial " << trial << " completed. Final energy: " << lattice.energy_density() << endl;
        }
    }
    
    if (rank == 0) {
        cout << "Simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run parallel tempering for mixed lattice
 * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
 */
void run_parallel_tempering_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm) {
    if (rank == 0) {
        cout << "Running parallel tempering on mixed lattice with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }

    comm = pt_runner::replica_comm(comm, size);

    // Ladder: geometric, or tuned with the replica chain itself (nrpt by
    // default); the tuned replicas are kept for the production run.
    const vector<double> temps = pt_runner::prepare_ladder(config, rank, size, [&](const mc::LadderTuningOptions& o) {
        return lattice.tune_temperature_ladder(o, config.overrelaxation_rate, config.gaussian_move,
                                               true, comm);
    });
    MPI_Barrier(comm);
    
    for (int trial = 0; trial < config.num_trials; ++trial) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        if (rank == 0) {
            filesystem::create_directories(trial_dir);
        }
        MPI_Barrier(comm);  // Ensure directory is created before others proceed
        
        if (rank == 0 && config.num_trials > 1) {
            cout << "\n=== Trial " << trial << " / " << config.num_trials << " ===" << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.parallel_tempering(
            temps,
            pt_runner::equilibration_steps(config),
            pt_runner::measurement_steps(config),
            config.overrelaxation_rate,
            config.pt_exchange_frequency,
            config.probe_rate,
            trial_dir,
            config.ranks_to_write,
            config.gaussian_move,
            true,  // use_interleaved
            comm,
            false  // verbose
        );
        
        // T=0 deterministic quench for coldest replica (rank 0)
        if (config.T_zero && rank == 0 && config.n_deterministics > 0) {
            size_t total_sites = lattice.lattice_size_SU2 + lattice.lattice_size_SU3;
            cout << "Rank 0: Performing " << config.n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < config.n_deterministics; ++sweep) {
                lattice.deterministic_sweep();
                if (sweep % 100 == 0 || sweep == config.n_deterministics - 1) {
                    cout << "Deterministic sweep " << sweep << "/" << config.n_deterministics 
                         << ", E/N = " << lattice.total_energy() / total_sites << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << lattice.total_energy() / total_sites << endl;
            // Save the T=0 quenched configuration
            lattice.save_spin_config_to_dir(trial_dir + "/rank_0", "spins_T0_quench");
        }
        MPI_Barrier(comm);
        
        if (rank == 0) {
            cout << "Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(comm);
    
    if (rank == 0) {
        cout << "Parallel tempering completed (" << config.num_trials << " trials)." << endl;
    }
}

// Dynamics configuration shared by every MD-type runner (MD, pump-probe,
// 2DCS); defined below.
static void configure_dynamics(MixedLattice& lattice, const SpinConfig& config, int rank);
static bool apply_trilinear_reference(MixedLattice& lattice, const SpinConfig& config, int rank,
                                      bool requench);

namespace {

// Copy rank 0's spins to every rank.
void broadcast_spins(MixedLattice& lattice, int rank) {
    vector<double> buf = lattice.spins_to_state();
    buf.resize(lattice.spin_state_size());
    MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        MixedLattice::SpinConfigSU2 s2;
        MixedLattice::SpinConfigSU3 s3;
        lattice.state_to_spins(buf, s2, s3);
        lattice.spins_SU2 = std::move(s2);
        lattice.spins_SU3 = std::move(s3);
    }
}

// Initial configuration of a trial. With a loaded configuration every trial
// starts from it (previously only the first trial on each rank did; the
// others silently ran from random spins without annealing); otherwise every
// trial after the first on a rank starts from fresh random spins. Any SU(3)
// reference subtraction of a previous trial is removed before annealing.
struct TrialStart {
    MixedLattice::SpinConfigSU2 spins_SU2;
    MixedLattice::SpinConfigSU3 spins_SU3;
    bool loaded = false;

    TrialStart(const MixedLattice& lattice, const SpinConfig& config)
        : spins_SU2(lattice.spins_SU2), spins_SU3(lattice.spins_SU3),
          loaded(!config.initial_spin_config.empty()) {}

    void apply(MixedLattice& lattice, int trial, int rank) const {
        if (!lattice.mixed_trilinear_reference_SU3().empty()) {
            lattice.set_mixed_trilinear_reference_SU3({});
        }
        if (loaded) {
            lattice.spins_SU2 = spins_SU2;
            lattice.spins_SU3 = spins_SU3;
        } else if (trial != rank) {
            lattice.init_random();
        }
    }
};

}  // namespace

/**
 * Run molecular dynamics for mixed lattice
 */
void run_molecular_dynamics_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running molecular dynamics on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Distribute trials across MPI ranks
    const TrialStart start(lattice, config);
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        start.apply(lattice, trial, rank);
        
        // Equilibrate (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating system..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
            config.twist_sweep_count
            );
        } else if (rank == 0) {
            cout << "Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        apply_trilinear_reference(lattice, config, rank, /*requench=*/true);
        configure_dynamics(lattice, config, rank);
        // Save the initial spin configuration of the time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        
        // Run MD
        if (rank == 0) {
            cout << "Starting MD integration..." << endl;
            cout << "Time range: " << config.md_time_start << " -> " << config.md_time_end << endl;
            cout << "Timestep: " << config.md_timestep << endl;
            cout << "Integration method: " << config.md_integrator << endl;
        }
        
        lattice.molecular_dynamics(
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            trial_dir,
            config.md_save_interval,
            config.md_integrator,
            config.use_gpu,
            // Ingredient XVIII: MD tolerance overrides (default 1e-6 if unset).
            config.md_abs_tol,
            config.md_rel_tol
        );
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/trajectory.h5" << endl;
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Molecular dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run pump-probe experiment for mixed lattice
 */

void run_pump_probe_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running pump-probe simulation on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Prepare per-sublattice pulse directions for SU2
    // Normalize all pump directions
    vector<vector<double>> pump_dirs_norm = config.pump_directions;
    for (auto& dir : pump_dirs_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate pump direction count: must be 1 (broadcast to all) or match N_atoms_SU2
    if (pump_dirs_norm.size() != 1 && pump_dirs_norm.size() != lattice.N_atoms_SU2) {
        if (rank == 0) {
            cerr << "Error: pump_direction must have either 1 direction (broadcast to all SU2 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU2 << " directions (one per SU2 sublattice). "
                 << "Got " << pump_dirs_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction dimension matches lattice spin_dim_SU2
    for (const auto& dir : pump_dirs_norm) {
        if (dir.size() != lattice.spin_dim_SU2) {
            if (rank == 0) {
                cerr << "Error: pump_direction dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU2 (" << lattice.spin_dim_SU2 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Prepare per-sublattice pulse directions for SU3 (Gell-Mann basis)
    double local_pump_amplitude_su3 = config.pump_amplitude_su3;
    double local_pump_width_su3 = config.pump_width_su3;
    double local_pump_frequency_su3 = config.pump_frequency_su3;
    
    // Auto-compute SU3 pulse from physical 3D direction using mu_act projection
    vector<vector<double>> pump_dirs_su3_norm = config.pump_directions_su3;
    if (config.auto_su3_pump) {
        const double mu_2x = config.get_param("mu_2x", 0.0);
        const double mu_2y = config.get_param("mu_2y", 0.0);
        const double mu_2z = config.get_param("mu_2z", 5.264);
        const double mu_5x = config.get_param("mu_5x", 2.3915);
        const double mu_5y = config.get_param("mu_5y", -2.7866);
        const double mu_5z = config.get_param("mu_5z", 0.0);
        const double mu_7x = config.get_param("mu_7x", 0.9128);
        const double mu_7y = config.get_param("mu_7y", 0.4655);
        const double mu_7z = config.get_param("mu_7z", 0.0);
        const double g_ratio = config.get_param("g_ratio_tm", 7.0/12.0);
        
        double mu_act[3][3] = {
            {mu_2x, mu_5x, mu_7x},
            {mu_2y, mu_5y, mu_7y},
            {mu_2z, mu_5z, mu_7z}
        };
        const int active_idx[3] = {1, 4, 6};
        
        // Project each physical 3D pump direction to 8D Gell-Mann space
        // B_a = Σ_α μ_{αa} n̂_α  (sublattice-0 reference, frames handle the rest)
        pump_dirs_su3_norm.clear();
        for (const auto& dir3d : pump_dirs_norm) {
            vector<double> su3_dir(lattice.spin_dim_SU3, 0.0);
            for (int a = 0; a < 3; ++a) {
                for (int al = 0; al < 3; ++al) {
                    su3_dir[active_idx[a]] += mu_act[al][a] * dir3d[al];
                }
            }
            pump_dirs_su3_norm.push_back(su3_dir);
        }
        
        double su3_norm = 0.0;
        for (double v : pump_dirs_su3_norm[0]) su3_norm += v * v;
        su3_norm = sqrt(su3_norm);
        
        // SU3 amplitude = physical_amplitude * g_ratio * |μ^T n̂|
        local_pump_amplitude_su3 = config.pump_amplitude * g_ratio * su3_norm;
        local_pump_width_su3 = config.pump_width;
        local_pump_frequency_su3 = config.pump_frequency;
        
        if (rank == 0) {
            cout << "Auto-computing SU3 pulse from physical direction:" << endl;
            cout << "  g_ratio_tm = " << g_ratio << endl;
            cout << "  |mu^T * n| = " << su3_norm << endl;
            cout << "  SU3 amplitude = " << local_pump_amplitude_su3 
                 << " (Fe amplitude = " << config.pump_amplitude << ")" << endl;
        }
    }
    
    // Normalize the SU3 pump directions.  A single (broadcast) direction is
    // normalized to unit length as before.  A per-sublattice list is normalized
    // by its COMMON maximum norm so that the relative magnitudes (and hence the
    // sublattice pattern purity) supplied by the user are preserved.
    {
        double norm_max = 0.0;
        for (const auto& dir : pump_dirs_su3_norm) {
            double norm = 0.0;
            for (const auto& comp : dir) norm += comp * comp;
            norm_max = std::max(norm_max, sqrt(norm));
        }
        for (auto& dir : pump_dirs_su3_norm) {
            double norm = norm_max;
            if (pump_dirs_su3_norm.size() == 1) {
                norm = 0.0;
                for (const auto& comp : dir) norm += comp * comp;
                norm = sqrt(norm);
            }
            if (norm > 1e-10) {
                for (auto& comp : dir) comp /= norm;
            }
        }
    }
    
    // Validate SU3 pump direction count: must be 1 (broadcast to all) or match N_atoms_SU3
    if (pump_dirs_su3_norm.size() != 1 && pump_dirs_su3_norm.size() != lattice.N_atoms_SU3) {
        if (rank == 0) {
            cerr << "Error: pump_direction_su3 must have either 1 direction (broadcast to all SU3 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU3 << " directions (one per SU3 sublattice). "
                 << "Got " << pump_dirs_su3_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate SU3 pump direction dimension matches lattice spin_dim_SU3
    for (const auto& dir : pump_dirs_su3_norm) {
        if (dir.size() != lattice.spin_dim_SU3) {
            if (rank == 0) {
                cerr << "Error: pump_direction_su3 dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU3 (" << lattice.spin_dim_SU3 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Distribute trials across MPI ranks
    const TrialStart start(lattice, config);
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        start.apply(lattice, trial, rank);
        
        // Equilibrate (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating system..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
            config.twist_sweep_count
            );
        } else if (rank == 0) {
            cout << "Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        apply_trilinear_reference(lattice, config, rank, /*requench=*/true);
        configure_dynamics(lattice, config, rank);
        // Save the initial spin configuration of the time evolution
        lattice.save_spin_config_to_dir(trial_dir, "initial_spins");
        
        // Setup pump field directions
        if (rank == 0) {
            cout << "Setting up pump-probe pulses..." << endl;
        }
        
        // Create field directions for SU2 (Fe) - using per-sublattice directions and general spin_dim
        vector<SpinVector> field_dirs_su2(lattice.lattice_size_SU2);
        vector<SpinVector> field_dirs_su3(lattice.lattice_size_SU3);
        
        for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
            size_t atom = i % lattice.N_atoms_SU2;
            // Use per-sublattice direction if provided, otherwise broadcast first direction to all
            size_t dir_idx = (pump_dirs_norm.size() == 1) ? 0 : atom;
            SpinVector pump_dir_su2(lattice.spin_dim_SU2);
            for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
                pump_dir_su2(d) = pump_dirs_norm[dir_idx][d];
            }
            field_dirs_su2[i] = pump_dir_su2;
        }
        for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
            size_t atom = i % lattice.N_atoms_SU3;
            // Use per-sublattice direction if provided, otherwise broadcast first direction to all
            size_t dir_idx = (pump_dirs_su3_norm.size() == 1) ? 0 : atom;
            SpinVector pump_dir_su3(lattice.spin_dim_SU3);
            for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
                pump_dir_su3(d) = pump_dirs_su3_norm[dir_idx][d];
            }
            field_dirs_su3[i] = pump_dir_su3;
        }
        
        // Run single pulse magnetization dynamics
        if (rank == 0) {
            cout << "Running pump-probe dynamics..." << endl;
        }
        
        auto trajectory = lattice.single_pulse_drive(
            field_dirs_su2, field_dirs_su3, config.pump_time,
            config.pump_amplitude, config.pump_width, config.pump_frequency,
            local_pump_amplitude_su3, local_pump_width_su3, local_pump_frequency_su3,
            config.md_time_start, config.md_time_end, config.md_timestep,
            config.md_integrator, config.use_gpu
        );
        
        // Save the trajectory of this trial (each rank owns its trials; this
        // used to be written by rank 0 only, discarding the other ranks' work).
        {
            ofstream traj_file(trial_dir + "/pump_probe_trajectory.txt");
            traj_file << std::setprecision(12);
            for (const auto& [t, mag_data] : trajectory) {
                traj_file << t << " "
                         << mag_data.first[0].transpose() << " "  // SU2 mag antiferro
                         << mag_data.first[1].transpose() << " "  // SU2 mag local
                         << mag_data.first[2].transpose() << " "  // SU2 mag global
                         << mag_data.second[0].transpose() << " " // SU3 mag antiferro
                         << mag_data.second[1].transpose() << " " // SU3 mag local
                         << mag_data.second[2].transpose() << "\n"; // SU3 mag global
            }
            traj_file.close();
            cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Pump-probe simulation completed (" << config.num_trials << " trials)." << endl;
    }
}   

/**
 * Configure the dynamics of the mixed lattice from the config. Called by every
 * MD-type runner once the initial (annealed / loaded) state of a trial is in
 * place, so MD, pump-probe and 2DCS see the same physics:
 *
 *   alpha_gilbert          SU(2) Gilbert damping (default 0)
 *   alpha_su3              SU(3) Landau-Lifshitz damping (default 0, see mixed_lattice.h)
 *   gamma_su3              uniform SU(3) Bloch relaxation rate (default 0)
 *   gamma_su3_lambda{1..8} per-generator override
 *       dn^a/dt += -Gamma_a (n^a - n^a_eq), with n_eq = the current state, so
 *       pump-induced deviations relax towards the ground state.
 *   thermal_heat, thermal_cap, thermal_cool
 *       thermal reservoir (only with Bloch damping; see mixed_lattice.h)
 *   linear_drive_torque    ablation: SU(2) drive torque about the current state
 * The SU(3) bracket convention (su3_legacy_convention) is part of the unit
 * cell and needs nothing here; see apply_trilinear_reference for
 * tm_trilinear_reference.
 */
/**
 * tm_trilinear_reference = 1: subtract the current SU(3) state r from the SU(3)
 * leg of the Fe-Fe-Tm trilinear couplings, T(S_i, S_j, n_k - r_k), folded into
 * the Hamiltonian so MC, energy and MD agree (this replaces the old implicit
 * subtraction of the Bloch-damping equilibrium in the MD field only; see
 * docs/MIGRATION.md). With `requench` the state is then re-minimised, so the
 * dynamics starts from a stationary state of the modified Hamiltonian.
 * Returns true if the reference was applied.
 */
static bool apply_trilinear_reference(MixedLattice& lattice, const SpinConfig& config, int rank,
                                      bool requench) {
    if (config.get_param("tm_trilinear_reference", 0.0) == 0.0) return false;
    lattice.set_mixed_trilinear_reference_SU3(lattice.spins_SU3);
    if (requench) {
        lattice.greedy_quench();
        if (rank == 0) {
            cout << "Fe-Fe-Tm trilinear couples to n - n_ref (n_ref = annealed SU(3) state); "
                 << "re-minimised, stationarity residual = "
                 << lattice.relative_stationarity_residual(lattice.spins_to_state()) << endl;
        }
    }
    return true;
}

static void configure_dynamics(MixedLattice& lattice, const SpinConfig& config, int rank) {
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
    if (rank == 0 && lattice.alpha_gilbert != 0.0) {
        cout << "SU(2) Gilbert damping alpha = " << lattice.alpha_gilbert << endl;
    }
    lattice.alpha_SU3 = config.get_param("alpha_su3", 0.0);
    if (rank == 0 && lattice.alpha_SU3 != 0.0) {
        cout << "SU(3) Landau-Lifshitz damping alpha_su3 = " << lattice.alpha_SU3
             << " (Casimir-preserving; relaxes to the instantaneous local field)" << endl;
    }

    const double gamma_uniform = config.get_param("gamma_su3", 0.0);
    SpinVector rates = SpinVector::Constant(lattice.spin_dim_SU3, gamma_uniform);
    for (int a = 0; a < static_cast<int>(lattice.spin_dim_SU3); ++a) {
        rates(a) = config.get_param("gamma_su3_lambda" + to_string(a + 1), rates(a));
    }
    lattice.set_damping_SU3(rates);
    lattice.thermal_heat = 0.0;
    if (rates.cwiseAbs().maxCoeff() > 0.0) {
        // Relax toward the current (post-anneal) state, already in the local
        // sublattice frame of the MD state vector. This is only the
        // relaxation target: the Hamiltonian is unchanged.
        lattice.set_equilibrium_SU3(lattice.spins_SU3);
        lattice.thermal_heat = config.get_param("thermal_heat", 0.0);
        lattice.thermal_cap  = config.get_param("thermal_cap", 1.0e30);
        lattice.thermal_cool = config.get_param("thermal_cool", 0.0);
        if (rank == 0) {
            cout << "SU(3) Bloch damping Gamma_a = [";
            for (int a = 0; a < static_cast<int>(lattice.spin_dim_SU3); ++a) {
                cout << (a ? ", " : "") << rates(a);
            }
            cout << "]  (equilibrium = current state)" << endl;
            if (lattice.thermal_heat != 0.0) {
                cout << "Thermal reservoir ON: thermal_heat = " << lattice.thermal_heat
                     << ", cap = " << lattice.thermal_cap << ", cool rate = " << lattice.thermal_cool << endl;
            }
        }
    } else if (rank == 0 && config.get_param("thermal_heat", 0.0) != 0.0) {
        cerr << "Warning: thermal_heat is ignored without SU(3) Bloch damping (gamma_su3*)." << endl;
    }

    const bool linear_torque = config.get_param("linear_drive_torque", 0.0) != 0.0;
    lattice.set_linear_drive_torque_SU2(linear_torque);
    if (rank == 0 && linear_torque) {
        cout << "SU(2) drive torque LINEARIZED about the equilibrium configuration: "
             << "the field-mediated magnon-magnon conversion channel is removed "
             << "(ablation, not a physical model)" << endl;
    }
}

/**
 * Run 2D coherent spectroscopy (2DCS) for mixed lattice
 */
void run_2dcs_spectroscopy_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
    // Determine parallelization strategy
    bool use_tau_parallel = (config.num_trials == 1) && config.parallel_tau && (size > 1);
    
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy (2DCS) on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Delay scan: tau = " << config.tau_start << " to " << config.tau_end 
             << " (step: " << config.tau_step << ")" << endl;
        if (use_tau_parallel) {
            cout << "Parallelization mode: tau-parallel (distributing delay points across ranks)" << endl;
        } else if (config.num_trials > 1) {
            cout << "Parallelization mode: trial-parallel (distributing trials across ranks)" << endl;
        } else {
            cout << "Parallelization mode: single rank" << endl;
        }
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Setup per-sublattice pulse field directions for SU2
    vector<vector<double>> pump_dirs_norm = config.pump_directions;
    for (auto& dir : pump_dirs_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }

    // Optional distinct SU(2) direction for the 2nd pulse (probe @ tau) in 2DCS.
    // Empty => reuse pump_dirs_norm (legacy single-direction behaviour).
    const bool use_distinct_pulse2_dir = !config.pump_directions_2.empty();
    vector<vector<double>> pump_dirs2_norm = config.pump_directions_2;
    for (auto& dir : pump_dirs2_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    if (use_distinct_pulse2_dir &&
        pump_dirs2_norm.size() != 1 && pump_dirs2_norm.size() != lattice.N_atoms_SU2) {
        if (rank == 0) {
            cerr << "Error: pump_direction_2 must have either 1 direction (broadcast to all SU2 sublattices) "
                 << "or N_atoms_SU2 directions. Got " << pump_dirs2_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction count: must be 1 (broadcast to all) or match N_atoms_SU2
    if (pump_dirs_norm.size() != 1 && pump_dirs_norm.size() != lattice.N_atoms_SU2) {
        if (rank == 0) {
            cerr << "Error: pump_direction must have either 1 direction (broadcast to all SU2 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU2 << " directions (one per SU2 sublattice). "
                 << "Got " << pump_dirs_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction dimension matches lattice spin_dim_SU2
    for (const auto& dir : pump_dirs_norm) {
        if (dir.size() != lattice.spin_dim_SU2) {
            if (rank == 0) {
                cerr << "Error: pump_direction dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU2 (" << lattice.spin_dim_SU2 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Prepare per-sublattice pulse directions for SU3 (Gell-Mann basis)
    double local_pump_amplitude_su3 = config.pump_amplitude_su3;
    double local_pump_width_su3 = config.pump_width_su3;
    double local_pump_frequency_su3 = config.pump_frequency_su3;
    
    // Auto-compute SU3 pulse from physical 3D direction using mu_act projection
    vector<vector<double>> pump_dirs_su3_norm = config.pump_directions_su3;
    if (config.auto_su3_pump) {
        const double mu_2x = config.get_param("mu_2x", 0.0);
        const double mu_2y = config.get_param("mu_2y", 0.0);
        const double mu_2z = config.get_param("mu_2z", 5.264);
        const double mu_5x = config.get_param("mu_5x", 2.3915);
        const double mu_5y = config.get_param("mu_5y", -2.7866);
        const double mu_5z = config.get_param("mu_5z", 0.0);
        const double mu_7x = config.get_param("mu_7x", 0.9128);
        const double mu_7y = config.get_param("mu_7y", 0.4655);
        const double mu_7z = config.get_param("mu_7z", 0.0);
        const double g_ratio = config.get_param("g_ratio_tm", 7.0/12.0);
        
        double mu_act[3][3] = {
            {mu_2x, mu_5x, mu_7x},
            {mu_2y, mu_5y, mu_7y},
            {mu_2z, mu_5z, mu_7z}
        };
        const int active_idx[3] = {1, 4, 6};
        
        pump_dirs_su3_norm.clear();
        for (const auto& dir3d : pump_dirs_norm) {
            vector<double> su3_dir(lattice.spin_dim_SU3, 0.0);
            for (int a = 0; a < 3; ++a) {
                for (int al = 0; al < 3; ++al) {
                    su3_dir[active_idx[a]] += mu_act[al][a] * dir3d[al];
                }
            }
            pump_dirs_su3_norm.push_back(su3_dir);
        }
        
        double su3_norm = 0.0;
        for (double v : pump_dirs_su3_norm[0]) su3_norm += v * v;
        su3_norm = sqrt(su3_norm);
        
        local_pump_amplitude_su3 = config.pump_amplitude * g_ratio * su3_norm;
        local_pump_width_su3 = config.pump_width;
        local_pump_frequency_su3 = config.pump_frequency;
        
        if (rank == 0) {
            cout << "Auto-computing SU3 pulse from physical direction:" << endl;
            cout << "  g_ratio_tm = " << g_ratio << endl;
            cout << "  |mu^T * n| = " << su3_norm << endl;
            cout << "  SU3 amplitude = " << local_pump_amplitude_su3 
                 << " (Fe amplitude = " << config.pump_amplitude << ")" << endl;
        }
    }
    
    // Normalize the SU3 pump directions.  A single (broadcast) direction is
    // normalized to unit length as before.  A per-sublattice list is normalized
    // by its COMMON maximum norm so that the relative magnitudes (and hence the
    // sublattice pattern purity) supplied by the user are preserved.
    {
        double norm_max = 0.0;
        for (const auto& dir : pump_dirs_su3_norm) {
            double norm = 0.0;
            for (const auto& comp : dir) norm += comp * comp;
            norm_max = std::max(norm_max, sqrt(norm));
        }
        for (auto& dir : pump_dirs_su3_norm) {
            double norm = norm_max;
            if (pump_dirs_su3_norm.size() == 1) {
                norm = 0.0;
                for (const auto& comp : dir) norm += comp * comp;
                norm = sqrt(norm);
            }
            if (norm > 1e-10) {
                for (auto& comp : dir) comp /= norm;
            }
        }
    }
    
    // Validate SU3 pump direction count: must be 1 (broadcast to all) or match N_atoms_SU3
    if (pump_dirs_su3_norm.size() != 1 && pump_dirs_su3_norm.size() != lattice.N_atoms_SU3) {
        if (rank == 0) {
            cerr << "Error: pump_direction_su3 must have either 1 direction (broadcast to all SU3 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU3 << " directions (one per SU3 sublattice). "
                 << "Got " << pump_dirs_su3_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate SU3 pump direction dimension matches lattice spin_dim_SU3
    for (const auto& dir : pump_dirs_su3_norm) {
        if (dir.size() != lattice.spin_dim_SU3) {
            if (rank == 0) {
                cerr << "Error: pump_direction_su3 dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU3 (" << lattice.spin_dim_SU3 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Create field directions for all sublattices using general spin_dim
    vector<SpinVector> field_dirs_su2(lattice.lattice_size_SU2);
    vector<SpinVector> field_dirs_su3(lattice.lattice_size_SU3);
    
    for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
        size_t atom = i % lattice.N_atoms_SU2;
        // Use per-sublattice direction if provided, otherwise broadcast first direction to all
        size_t dir_idx = (pump_dirs_norm.size() == 1) ? 0 : atom;
        SpinVector pump_dir_su2(lattice.spin_dim_SU2);
        for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
            pump_dir_su2(d) = pump_dirs_norm[dir_idx][d];
        }
        field_dirs_su2[i] = pump_dir_su2;
    }
    // Build the distinct 2nd-pulse (probe @ tau) SU2 field directions. When
    // pump_direction_2 is unset, this is identical to field_dirs_su2 and the
    // solver falls back to legacy same-direction behaviour.
    vector<SpinVector> field_dirs_su2_B;
    if (use_distinct_pulse2_dir) {
        field_dirs_su2_B.resize(lattice.lattice_size_SU2);
        for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
            size_t atom = i % lattice.N_atoms_SU2;
            size_t dir_idx = (pump_dirs2_norm.size() == 1) ? 0 : atom;
            SpinVector pump_dir_su2_B(lattice.spin_dim_SU2);
            for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
                pump_dir_su2_B(d) = pump_dirs2_norm[dir_idx][d];
            }
            field_dirs_su2_B[i] = pump_dir_su2_B;
        }
    }

    // Optional distinct SU3 direction for the 2nd pulse (probe @ tau). Empty =>
    // reuse pump_dirs_su3_norm. All-zeros => SU3 fires only at t=0 (isolates
    // omega_tau to the SU2 mode while omega_t carries the direct E12 drive).
    const bool use_distinct_pulse2_dir_su3 = !config.pump_directions_su3_2.empty();
    vector<vector<double>> pump_dirs2_su3_norm = config.pump_directions_su3_2;
    for (auto& dir : pump_dirs2_su3_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) norm += comp * comp;
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) comp /= norm;
        }
    }
    if (use_distinct_pulse2_dir_su3 &&
        pump_dirs2_su3_norm.size() != 1 && pump_dirs2_su3_norm.size() != lattice.N_atoms_SU3) {
        if (rank == 0) {
            cerr << "Error: pump_direction_su3_2 must have either 1 direction or "
                 << lattice.N_atoms_SU3 << " directions. Got " << pump_dirs2_su3_norm.size() << "." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
        size_t atom = i % lattice.N_atoms_SU3;
        // Use per-sublattice direction if provided, otherwise broadcast first direction to all
        size_t dir_idx = (pump_dirs_su3_norm.size() == 1) ? 0 : atom;
        SpinVector pump_dir_su3(lattice.spin_dim_SU3);
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            pump_dir_su3(d) = pump_dirs_su3_norm[dir_idx][d];
        }
        field_dirs_su3[i] = pump_dir_su3;
    }
    // Build distinct 2nd-pulse (probe @ tau) SU3 field directions.
    vector<SpinVector> field_dirs_su3_B;
    if (use_distinct_pulse2_dir_su3) {
        field_dirs_su3_B.resize(lattice.lattice_size_SU3);
        for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
            size_t atom = i % lattice.N_atoms_SU3;
            size_t dir_idx = (pump_dirs2_su3_norm.size() == 1) ? 0 : atom;
            SpinVector pump_dir_su3_B(lattice.spin_dim_SU3);
            for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
                pump_dir_su3_B(d) = pump_dirs2_su3_norm[dir_idx][d];
            }
            field_dirs_su3_B[i] = pump_dir_su3_B;
        }
    }

    const bool save_spin_traj = (config.get_param("save_spin_trajectories", 0.0) > 0.5);

    // Load tabulated pump pulse if specified.
    // This sets lattice.tabulated_pulse_{times,values,sigma} and later causes
    // drive_envelopes_SU2/SU3 to use linear interpolation instead of Gaussian.
    // The effective width/frequency override the config values so that the
    // pulse-window chunking window correctly covers the tabulated pulse.
    double effective_pump_width = config.pump_width;
    double effective_pump_frequency = config.pump_frequency;
    double effective_pump_width_su3 = local_pump_width_su3;
    double effective_pump_frequency_su3 = local_pump_frequency_su3;
    if (!config.pump_table_file.empty()) {
        lattice.load_tabulated_pulse(config.pump_table_file);
        if (rank == 0) {
            cout << "Tabulated pulse loaded: " << config.pump_table_file << " ("
                 << lattice.tabulated_pulse_times.size() << " points, t in ["
                 << lattice.tabulated_pulse_times.front() << ", " << lattice.tabulated_pulse_times.back()
                 << "], effective width " << lattice.tabulated_pulse_sigma << ")" << endl;
        }
        effective_pump_width     = lattice.tabulated_pulse_sigma;
        effective_pump_frequency = 0.0;  // frequency is encoded in the table
        effective_pump_width_su3     = lattice.tabulated_pulse_sigma;
        effective_pump_frequency_su3 = 0.0;
    }

    // Optional second SU(3) carrier color (two-color CEF drive).  Set on the
    // lattice so drive_envelopes_SU3 superposes cos(freq)+cos(freq2).
    lattice.field_drive_freq_SU3_2 = config.pump_frequency_su3_2;
    // Continue every M01 from the stored M0 state where the probe is still
    // negligible (exact up to the 1.6e-9 Gaussian tail; ~2x fewer M01 steps).
    const bool reuse_m0_for_m01 = config.get_param("reuse_m0_for_m01", 1.0) != 0.0;
    if (use_tau_parallel) {
        // Single trial, parallelize over tau values
        string trial_dir = config.output_dir + "/sample_0";
        filesystem::create_directories(trial_dir);
        
        // Always equilibrate for 2DCS (only rank 0), unless the caller
        // explicitly disabled every relaxation stage (annealing_steps=0 and
        // n_deterministics=0): then the loaded seed is used verbatim, which
        // is required for excited-level (thermal-ensemble) initial states.
        if (rank == 0 && (config.annealing_steps > 0 || config.n_deterministics > 0)) {
            if (!config.initial_spin_config.empty()) {
                cout << "\n[1/2] Equilibrating from loaded seed to true ground state..." << endl;
            } else {
                cout << "\n[1/2] Equilibrating to ground state..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.twist_sweep_count
            );
        }
        
        // Wait for rank 0 to finish annealing
        MPI_Barrier(MPI_COMM_WORLD);
        
        broadcast_spins(lattice, rank);
        // Optional trilinear reference: every rank folds the same broadcast
        // state, rank 0 re-minimises and the result is broadcast again.
        if (apply_trilinear_reference(lattice, config, rank, /*requench=*/rank == 0)) {
            broadcast_spins(lattice, rank);
        }
        // Damping / ablation setup from the synchronized ground state,
        // identically on every rank.
        configure_dynamics(lattice, config, rank);
        if (rank == 0) {
            lattice.save_spin_config_to_dir(trial_dir, "initial_spins");
        }

        if (rank == 0) {
            cout << "\n[2/2] Running MPI-parallel pump-probe spectroscopy..." << endl;
            cout << "  SU2 Pulse: amplitude=" << config.pump_amplitude 
                 << ", width=" << config.pump_width 
                 << ", frequency=" << config.pump_frequency << endl;
            if (pump_dirs_norm.size() == 1) {
                cout << "  SU2 Direction: [" << pump_dirs_norm[0][0] << ", " 
                     << pump_dirs_norm[0][1] << ", " << pump_dirs_norm[0][2] << "]" << endl;
            } else {
                cout << "  SU2 Directions per sublattice:" << endl;
                for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                    cout << "    Sublattice " << i << ": [" << pump_dirs_norm[i][0] << ", " 
                         << pump_dirs_norm[i][1] << ", " << pump_dirs_norm[i][2] << "]" << endl;
                }
            }
            cout << "  SU3 Pulse: amplitude=" << config.pump_amplitude_su3 
                 << ", width=" << config.pump_width_su3 
                 << ", frequency=" << config.pump_frequency_su3 << endl;
            if (pump_dirs_su3_norm.size() == 1) {
                cout << "  SU3 Direction (Gell-Mann): [";
                for (int k = 0; k < 8; ++k) {
                    if (k > 0) cout << ", ";
                    cout << pump_dirs_su3_norm[0][k];
                }
                cout << "]" << endl;
            } else {
                cout << "  SU3 Directions per sublattice (Gell-Mann):" << endl;
                for (size_t i = 0; i < pump_dirs_su3_norm.size(); ++i) {
                    cout << "    Sublattice " << i << ": [";
                    for (int k = 0; k < 8; ++k) {
                        if (k > 0) cout << ", ";
                        cout << pump_dirs_su3_norm[i][k];
                    }
                    cout << "]" << endl;
                }
            }
        }
        
        // Run MPI-parallelized version
        lattice.pump_probe_spectroscopy_mpi(
            field_dirs_su2,
            field_dirs_su3,
            config.pump_amplitude,
            effective_pump_width,
            effective_pump_frequency,
            local_pump_amplitude_su3,
            effective_pump_width_su3,
            effective_pump_frequency_su3,
            config.tau_start,
            config.tau_end,
            config.tau_step,
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            config.T_start,
            config.T_end,
            config.annealing_steps,
            config.T_zero,
            config.n_deterministics,
            trial_dir,
            config.md_integrator,
            config.use_gpu,
            save_spin_traj,
            config.reuse_m0_for_m1,
            config.stationarity_tol,
            config.pulse_window_chunking,
            config.pump_probe_abs_tol,
            config.pump_probe_rel_tol,
            field_dirs_su2_B,  // probe (2nd pulse) SU2 directions; empty => same as the pump
            field_dirs_su3_B,  // probe (2nd pulse) SU3 directions; empty => same as the pump
            reuse_m0_for_m01
        );
        
    } else {
        // Distribute trials across MPI ranks (same physics as the tau-parallel
        // path: probe directions, ablation and damping are honoured here too).
        const TrialStart start(lattice, config);
        for (int trial = rank; trial < config.num_trials; trial += size) {
            string trial_dir = config.output_dir + "/sample_" + to_string(trial);
            filesystem::create_directories(trial_dir);
            
            if (config.num_trials > 1) {
                cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
            }
            
            start.apply(lattice, trial, rank);
            
            // Always equilibrate for 2DCS (need true ground state even if seed loaded)
            if (rank == 0 || config.num_trials == 1) {
                if (!config.initial_spin_config.empty()) {
                    cout << "\n[1/3] Equilibrating from loaded seed to true ground state..." << endl;
                } else {
                    cout << "\n[1/3] Equilibrating to ground state..." << endl;
                }
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                trial_dir,
                config.save_observables,
                config.T_zero,
                config.n_deterministics,
                config.twist_sweep_count
            );
            
            apply_trilinear_reference(lattice, config, rank, /*requench=*/true);
            configure_dynamics(lattice, config, rank);
            // Save the initial spin configuration of the time evolution
            lattice.save_spin_config_to_dir(trial_dir, "initial_spins");

            if (rank == 0 || config.num_trials == 1) {
                cout << "\n[2/3] Pulse configuration:" << endl;                cout << "  SU2 Pulse: amplitude=" << config.pump_amplitude 
                     << ", width=" << config.pump_width 
                     << ", frequency=" << config.pump_frequency << endl;
                if (pump_dirs_norm.size() == 1) {
                    cout << "  SU2 Direction: [" << pump_dirs_norm[0][0] << ", " 
                         << pump_dirs_norm[0][1] << ", " << pump_dirs_norm[0][2] << "]" << endl;
                } else {
                    cout << "  SU2 Directions per sublattice:" << endl;
                    for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                        cout << "    Sublattice " << i << ": [" << pump_dirs_norm[i][0] << ", " 
                             << pump_dirs_norm[i][1] << ", " << pump_dirs_norm[i][2] << "]" << endl;
                    }
                }
                cout << "  SU3 Pulse: amplitude=" << config.pump_amplitude_su3 
                     << ", width=" << config.pump_width_su3 
                     << ", frequency=" << config.pump_frequency_su3 << endl;
                if (pump_dirs_su3_norm.size() == 1) {
                    cout << "  SU3 Direction (Gell-Mann): [";
                    for (int k = 0; k < 8; ++k) {
                        if (k > 0) cout << ", ";
                        cout << pump_dirs_su3_norm[0][k];
                    }
                    cout << "]" << endl;
                } else {
                    cout << "  SU3 Directions per sublattice (Gell-Mann):" << endl;
                    for (size_t i = 0; i < pump_dirs_su3_norm.size(); ++i) {
                        cout << "    Sublattice " << i << ": [";
                        for (int k = 0; k < 8; ++k) {
                            if (k > 0) cout << ", ";
                            cout << pump_dirs_su3_norm[i][k];
                        }
                        cout << "]" << endl;
                    }
                }
                cout << "\n[3/3] Running pump-probe spectroscopy scan..." << endl;
            }
            
            // Run the full 2DCS scan using mixed lattice method
            lattice.pump_probe_spectroscopy(
                field_dirs_su2,
                field_dirs_su3,
                config.pump_amplitude,
                effective_pump_width,
                effective_pump_frequency,
                local_pump_amplitude_su3,
                effective_pump_width_su3,
                effective_pump_frequency_su3,
                config.tau_start,
                config.tau_end,
                config.tau_step,
                config.md_time_start,
                config.md_time_end,
                config.md_timestep,
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.T_zero,
                config.n_deterministics,
                trial_dir,
                config.md_integrator,
                config.use_gpu,
                save_spin_traj,
                config.reuse_m0_for_m1,
                config.stationarity_tol,
                config.pump_probe_omp_threads,
                config.pulse_window_chunking,
                config.pump_probe_abs_tol,
                config.pump_probe_rel_tol,
                field_dirs_su2_B,
                field_dirs_su3_B,
                reuse_m0_for_m01
            );
            
            cout << "[Rank " << rank << "] Trial " << trial << " 2DCS spectroscopy completed!" << endl;
            cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/pump_probe_spectroscopy.h5" << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n2DCS spectroscopy completed (" << config.num_trials << " trials)!" << endl;
        cout << "\nTo analyze:" << endl;
        cout << "  - M0(t): Reference single-pulse response" << endl;
        cout << "  - M1(t,tau): Probe-only response at delay tau" << endl;
        cout << "  - M01(t,tau): Two-pulse response (pump + probe)" << endl;
        cout << "  - Nonlinear signal: M01 - M0 - M1" << endl;
    }
}
