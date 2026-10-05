/**
 * runners_lattice.cpp — spin_solver runners for the regular SU(2) Lattice.
 *
 * This TU was split out of `spin_solver.cpp` to cut its size and allow
 * parallel compilation. See `src/apps/spin_solver_runners.h` for the full
 * list of runners and `docs/refactor_backlog.md` for the broader plan.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/lattice.h"

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

// ============================================================================
// SIMULATION RUNNERS
// ============================================================================

/**
 * Run simulated annealing
 */
void run_simulated_annealing(Lattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing..." << endl;
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

        // Optionally load an initial spin configuration; with annealing_steps == 0
        // this collapses to a plain energy evaluation of the loaded state.
        if (!config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Loading initial spin configuration from "
                     << config.initial_spin_config << endl;
            }
            lattice.load_spin_config(config.initial_spin_config);
        }

        if (config.annealing_steps == 0) {
            if (rank == 0) {
                cout << "annealing_steps == 0: skipping SA, just evaluating energy."
                     << endl;
            }
        } else {
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.use_twist_boundary,
                config.gaussian_move,
                config.cooling_rate,
                trial_dir,
                config.save_observables,
                config.T_zero,
                config.n_deterministics,
                config.twist_sweep_count
            );
        }
        
        // Save final configuration
        lattice.save_positions(trial_dir + "/positions.txt");
        // lattice.save_spin_config(trial_dir + "/spins.txt");
        
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
 * Run parallel tempering
 * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
 */
void run_parallel_tempering(Lattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm) {
    if (rank == 0) {
        cout << "Running parallel tempering with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    vector<size_t> sweeps_per_temp;  // Bittner adaptive sweep schedule (empty = use fixed swap_rate)
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid
        // Phase 1: Katzgraber et al. feedback for uniform acceptance rates
        // Phase 2: Bittner et al. adaptive sweep schedule for minimal round-trip time
        if (rank == 0) {
            bool use_grad = (config.pt_temperature_optimizer == "gradient");
            cout << "Generating optimized temperature grid ("
                 << (use_grad ? "gradient-based, Miyata et al. 2024" : "Katzgraber+Bittner")
                 << ", MPI-distributed)..." << endl;
        }
        bool use_gradient = (config.pt_temperature_optimizer == "gradient");
        OptimizedTempGridResult opt_result = lattice.generate_optimized_temperature_grid_mpi(
            config.T_end,    // Tmin (coldest)
            config.T_start,  // Tmax (hottest)
            config.pt_optimization_warmup,
            config.pt_optimization_sweeps,
            config.pt_optimization_iterations,
            config.gaussian_move,
            config.overrelaxation_rate,
            config.pt_target_acceptance,
            0.05,  // convergence tolerance
            comm,
            use_gradient
        );
        temps = opt_result.temperatures;
        sweeps_per_temp = opt_result.sweeps_per_temp;
        
        // Save optimized temperature grid info to file (rank 0 only)
        if (rank == 0 && !config.output_dir.empty()) {
            filesystem::create_directories(config.output_dir);
            ofstream opt_file(config.output_dir + "/optimized_temperatures.txt");
            opt_file << "# Optimized temperature grid\n";
            opt_file << "# References: Katzgraber et al., PRE 73, 056702 (2006)\n";
            opt_file << "#             Bittner et al., PRL 101, 130603 (2008)\n";
            opt_file << "# Target acceptance rate: " << config.pt_target_acceptance << "\n";
            opt_file << "# Mean acceptance rate: " << opt_result.mean_acceptance_rate << "\n";
            opt_file << "# Converged: " << (opt_result.converged ? "yes" : "no") << "\n";
            opt_file << "# Feedback iterations: " << opt_result.feedback_iterations_used << "\n";
            opt_file << "# Round-trip estimate: " << opt_result.round_trip_estimate << "\n";
            opt_file << "#\n";
            opt_file << "# rank  temperature  acceptance_rate  diffusivity  tau_int  n_sweeps\n";
            for (int i = 0; i < size; ++i) {
                opt_file << i << "  " << scientific << setprecision(12) << temps[i];
                if (i < size - 1) {
                    opt_file << "  " << fixed << setprecision(4) << opt_result.acceptance_rates[i]
                             << "  " << scientific << setprecision(6) << opt_result.local_diffusivities[i];
                } else {
                    opt_file << "  " << fixed << setprecision(4) << 0.0
                             << "  " << scientific << setprecision(6) << 0.0;
                }
                if (!opt_result.autocorrelation_times.empty()) {
                    opt_file << "  " << fixed << setprecision(2) << opt_result.autocorrelation_times[i];
                }
                if (!opt_result.sweeps_per_temp.empty()) {
                    opt_file << "  " << opt_result.sweeps_per_temp[i];
                }
                opt_file << "\n";
            }
            opt_file.close();
        }
    } else {
        // Use geometric (logarithmic) temperature spacing
        if (rank == 0) {
            cout << "Using geometric temperature grid..." << endl;
            temps = Lattice::generate_geometric_temperature_ladder(config.T_end, config.T_start, size);
        }
        // Broadcast temperatures from rank 0 to all ranks
        MPI_Bcast(temps.data(), size, MPI_DOUBLE, 0, comm);
    }
    
    // Re-initialize spins after temperature optimization (or geometric grid setup)
    // This ensures each rank starts with fresh random spins - the optimization
    // phase leaves spins in a "mixed" state from many replica exchanges
    lattice.init_random();
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
            config.annealing_steps,
            config.annealing_steps,
            config.overrelaxation_rate,
            config.pt_exchange_frequency,
            config.probe_rate,
            trial_dir,
            config.ranks_to_write,
            config.gaussian_move,
            comm,
            false,  // verbose
            config.pt_accumulate_correlations,
            config.pt_n_bond_types,
            sweeps_per_temp  // Bittner adaptive sweep schedule
        );
        
        // T=0 deterministic quench for coldest replica (rank 0)
        if (config.T_zero && rank == 0 && config.n_deterministics > 0) {
            cout << "Rank 0: Performing " << config.n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < config.n_deterministics; ++sweep) {
                lattice.deterministic_sweep(1);
                if (sweep % 100 == 0 || sweep == config.n_deterministics - 1) {
                    cout << "Deterministic sweep " << sweep << "/" << config.n_deterministics 
                         << ", E/N = " << lattice.energy_density() << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << lattice.energy_density() << endl;
            // Save the T=0 quenched configuration
            lattice.save_spin_config(trial_dir + "/rank_0/spins_T0_quench.txt");
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

// ============================================================================
// SPIN-DYNAMICS RUNNERS (molecular dynamics, pump-probe, 2DCS)
// ============================================================================

namespace {

/// Damping and bath temperature of the Lattice equation of motion
/// (Hamiltonian-style keys, as for MixedLattice / PhononLattice).
void apply_dynamics_config(Lattice& lattice, const SpinConfig& config) {
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
    lattice.langevin_temperature = config.get_param("langevin_temperature", 0.0);
    if (lattice.alpha_gilbert < 0.0 || lattice.langevin_temperature < 0.0)
        throw invalid_argument("alpha_gilbert and langevin_temperature must be >= 0");
}

void report_dynamics_config(const Lattice& lattice, const SpinConfig& config) {
    cout << "Integrator: " << config.md_integrator << ", dt = " << config.md_timestep
         << ", t = " << config.md_time_start << " -> " << config.md_time_end << endl;
    if (lattice.alpha_gilbert > 0.0) cout << "Damping: alpha = " << lattice.alpha_gilbert << endl;
    if (lattice.langevin_temperature > 0.0)
        cout << "Langevin bath: T = " << lattice.langevin_temperature << endl;
#ifndef CUDA_ENABLED
    if (config.use_gpu) cout << "GPU requested but not compiled in (CUDA_ENABLED); using the CPU" << endl;
#endif
}

#ifdef CUDA_ENABLED
/// Bind this rank to a GPU (round robin over the visible devices).
void bind_gpu(const SpinConfig& config, int rank) {
    if (!config.use_gpu) return;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaSetDevice(rank % device_count);
        cout << "[Rank " << rank << "] GPU " << (rank % device_count) << " of " << device_count << endl;
    } else if (rank == 0) {
        cout << "Warning: no GPU detected, using the CPU" << endl;
    }
}
#endif

/**
 * One polarisation per sublattice from a config list of directions: either
 * one direction (applied to every sublattice) or one per sublattice, each of
 * dimension spin_dim, normalised. Throws std::invalid_argument otherwise.
 */
vector<SpinVector> pulse_directions(const vector<vector<double>>& dirs, const Lattice& lattice,
                                    const string& key) {
    if (dirs.size() != 1 && dirs.size() != lattice.N_atoms)
        throw invalid_argument(key + ": give 1 direction (all sublattices) or " + to_string(lattice.N_atoms) +
                               " (one per sublattice), got " + to_string(dirs.size()));
    vector<SpinVector> out;
    for (size_t a = 0; a < lattice.N_atoms; ++a) {
        const vector<double>& d = dirs[dirs.size() == 1 ? 0 : a];
        if (d.size() != lattice.spin_dim)
            throw invalid_argument(key + ": direction dimension " + to_string(d.size()) +
                                   " does not match spin_dim " + to_string(lattice.spin_dim));
        SpinVector v = Eigen::Map<const Eigen::VectorXd>(d.data(), Eigen::Index(d.size()));
        if (v.norm() > 1e-10) v.normalize();
        out.push_back(v);
    }
    return out;
}

/**
 * Initial state of one dynamics trial. A configuration loaded from
 * initial_spin_config is used as-is in EVERY trial (it used to be replaced by
 * random spins from the second trial on, while the log claimed otherwise);
 * with `polish_seed` and T_zero it is first relaxed to the nearest local
 * minimum by the T = 0 quench (spectroscopy needs a stationary state).
 * Without a seed every trial anneals from random spins, or from the
 * ferromagnetic start when use_ferromagnetic_init is set.
 */
void prepare_trial_state(Lattice& lattice, const SpinConfig& config, const Lattice::SpinConfig& start,
                         int trial, int rank, bool polish_seed) {
    const bool seeded = !config.initial_spin_config.empty();
    if (seeded) {
        lattice.spins = start;
        if (polish_seed && config.T_zero) {
            lattice.greedy_quench();
            cout << "[Rank " << rank << "] trial " << trial << ": loaded configuration relaxed by the T = 0 "
                 << "quench, E/N = " << setprecision(12) << lattice.energy_density() << endl;
        } else {
            cout << "[Rank " << rank << "] trial " << trial << ": using the loaded configuration, E/N = "
                 << setprecision(12) << lattice.energy_density() << endl;
        }
        return;
    }
    if (config.use_ferromagnetic_init) lattice.spins = start;
    else lattice.init_random();
    lattice.simulated_annealing(config.T_start, config.T_end, config.annealing_steps,
                                config.overrelaxation_rate, config.use_twist_boundary,
                                config.gaussian_move, config.cooling_rate, "", false,
                                config.T_zero, config.n_deterministics, config.twist_sweep_count);
    cout << "[Rank " << rank << "] trial " << trial << ": annealed, E/N = " << setprecision(12)
         << lattice.energy_density() << endl;
}

/// Run fn(trial) for this rank's trials; a failing trial is reported with its
/// rank (main only prints rank 0's errors) and turns into an error at the end.
template<class Fn>
void for_each_trial(const SpinConfig& config, int rank, int size, Fn&& fn) {
    string first_error;
    for (int trial = rank; trial < config.num_trials; trial += size) {
        try {
            fn(trial);
        } catch (const exception& e) {
            cerr << "[Rank " << rank << "] trial " << trial << " failed: " << e.what() << endl;
            if (first_error.empty()) first_error = "trial " + to_string(trial) + ": " + e.what();
        }
    }
    if (!first_error.empty()) throw runtime_error(first_error);
}

void write_trajectory_columns(ofstream& out, const Lattice::PumpProbeTrajectory& traj, size_t k) {
    for (int c = 0; c < 3; ++c)
        for (Eigen::Index d = 0; d < traj[k].second[c].size(); ++d) out << ' ' << traj[k].second[c](d);
}

}  // namespace

/**
 * Run molecular dynamics: per trial, prepare the initial state (loaded,
 * annealed or ferromagnetic) and integrate it, writing
 * output_dir/sample_<trial>/trajectory.h5 on the rank that owns the trial.
 */
void run_molecular_dynamics(Lattice& lattice, const SpinConfig& config, int rank, int size) {
    apply_dynamics_config(lattice, config);
    if (rank == 0) {
        cout << "Running molecular dynamics: " << config.num_trials << " trial(s) on " << size << " rank(s)" << endl;
        report_dynamics_config(lattice, config);
    }
#ifdef CUDA_ENABLED
    bind_gpu(config, rank);
#endif
    const Lattice::SpinConfig start = lattice.spins;
    for_each_trial(config, rank, size, [&](int trial) {
        const string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        prepare_trial_state(lattice, config, start, trial, rank, /*polish_seed=*/false);
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        lattice.molecular_dynamics(config.md_time_start, config.md_time_end, config.md_timestep,
                                   trial_dir, config.md_save_interval, config.md_integrator,
                                   config.use_gpu, config.md_abs_tol, config.md_rel_tol);
        cout << "[Rank " << rank << "] trial " << trial << " -> " << trial_dir << "/trajectory.h5" << endl;
    });
    if (rank == 0) cout << "Molecular dynamics completed (" << config.num_trials << " trials)." << endl;
}

/**
 * Run a pump-probe experiment: a pump pulse (pump_time, pump_amplitude,
 * pump_width, pump_frequency, pump_direction) and, when probe_amplitude != 0,
 * a probe pulse with its own parameters (probe_time, probe_amplitude,
 * probe_width, probe_frequency, probe_direction). Each rank writes
 * output_dir/sample_<trial>/pump_probe_trajectory.txt for its trials: time,
 * then [M_antiferro, M_local, M_global] of the pump+probe run and, for
 * deterministic dynamics with a probe, of the pump-only and probe-only runs
 * (M_NL = M_pump_probe - M_pump - M_probe).
 */
void run_pump_probe(Lattice& lattice, const SpinConfig& config, int rank, int size) {
    using classical_spin::dynamics::Pulse;
    using classical_spin::dynamics::TimeGrid;
    apply_dynamics_config(lattice, config);
    const vector<SpinVector> pump_dirs = pulse_directions(config.pump_directions, lattice, "pump_direction");
    const vector<SpinVector> probe_dirs = pulse_directions({config.probe_direction}, lattice, "probe_direction");
    const Pulse pump{config.pump_time, config.pump_amplitude, config.pump_width, config.pump_frequency};
    const Pulse probe{config.probe_time, config.probe_amplitude, config.probe_width, config.probe_frequency};
    const bool with_probe = (config.probe_amplitude != 0.0);
    const bool references = with_probe && lattice.langevin_temperature == 0.0;
    if (rank == 0) {
        cout << "Running pump-probe: " << config.num_trials << " trial(s) on " << size << " rank(s)" << endl;
        report_dynamics_config(lattice, config);
        cout << "Pump:  t = " << pump.t_center << ", A = " << pump.amplitude << ", w = " << pump.width
             << ", omega = " << pump.frequency << endl;
        if (with_probe)
            cout << "Probe: t = " << probe.t_center << ", A = " << probe.amplitude << ", w = " << probe.width
                 << ", omega = " << probe.frequency << endl;
        else
            cout << "Probe: off (probe_amplitude = 0)" << endl;
    }
#ifdef CUDA_ENABLED
    bind_gpu(config, rank);
#endif
    const TimeGrid grid = TimeGrid::covering(config.md_time_start, config.md_time_end, config.md_timestep,
                                             "pump-probe time grid");
    const Lattice::DynamicsSettings settings{config.md_integrator, config.md_timestep,
                                             config.pump_probe_abs_tol, config.pump_probe_rel_tol, 0.0};
    const Lattice::SpinConfig start = lattice.spins;
    for_each_trial(config, rank, size, [&](int trial) {
        const string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        prepare_trial_state(lattice, config, start, trial, rank, /*polish_seed=*/true);
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");

        Lattice::DriveSchedule both = lattice.make_drive();
        lattice.add_pulse(both, pump_dirs, pump);
        if (with_probe) lattice.add_pulse(both, probe_dirs, probe);
        Lattice::PumpProbeTrajectory M01, M0, M1;
        if (config.use_gpu && !with_probe) {
            M01 = lattice.single_pulse_drive(pump_dirs, pump.t_center, pump.amplitude, pump.width,
                                             pump.frequency, config.md_time_start, config.md_time_end,
                                             config.md_timestep, config.md_integrator, true, false,
                                             settings.abs_tol, settings.rel_tol);
        } else {
            M01 = lattice.drive_trajectory(both, grid, settings);
        }
        if (references) {
            Lattice::DriveSchedule d0 = lattice.make_drive(), d1 = lattice.make_drive();
            lattice.add_pulse(d0, pump_dirs, pump);
            lattice.add_pulse(d1, probe_dirs, probe);
            M0 = lattice.drive_trajectory(d0, grid, settings);
            M1 = lattice.drive_trajectory(d1, grid, settings);
        }

        const string path = trial_dir + "/pump_probe_trajectory.txt";
        ofstream out(path);
        if (!out) throw runtime_error("cannot write " + path);
        out << "# t, then [M_antiferro, M_local, M_global] (spin_dim components each) of the pump"
            << (with_probe ? "+probe run" : " run")
            << (references ? ", the pump-only run and the probe-only run" : "") << "\n";
        out << setprecision(17);
        for (size_t k = 0; k < M01.size(); ++k) {
            out << M01[k].first;
            write_trajectory_columns(out, M01, k);
            if (references) {
                write_trajectory_columns(out, M0, k);
                write_trajectory_columns(out, M1, k);
            }
            out << '\n';
        }
        cout << "[Rank " << rank << "] trial " << trial << " -> " << path << endl;
    });
    if (rank == 0) cout << "Pump-probe completed (" << config.num_trials << " trials)." << endl;
}

/**
 * Run 2D coherent spectroscopy (pump-probe delay scan, M0 / M1 / M01).
 * With one trial and parallel_tau, the delays are distributed over the ranks
 * (pump_probe_spectroscopy_mpi; rank 0's ground state is broadcast); with
 * several trials each rank runs whole scans for its trials. The ground state
 * is prepared identically in both modes (prepare_trial_state).
 */
void run_2dcs_spectroscopy(Lattice& lattice, const SpinConfig& config, int rank, int size) {
    apply_dynamics_config(lattice, config);
    const vector<SpinVector> field_dirs = pulse_directions(config.pump_directions, lattice, "pump_direction");
    // GPU batched path: with use_gpu one launch handles every delay, so route
    // there even on a single rank.
    const bool use_tau_parallel = (config.num_trials == 1) && config.parallel_tau &&
                                  ((size > 1) || config.use_gpu);
    // A runner called with size == 1 (e.g. one point of a parameter sweep)
    // must not touch the other ranks.
    MPI_Comm comm = (size > 1) ? MPI_COMM_WORLD : MPI_COMM_SELF;
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy: " << config.num_trials << " trial(s) on " << size
             << " rank(s), delays " << config.tau_start << " .. " << config.tau_end << " step "
             << config.tau_step << (use_tau_parallel ? " (delay-parallel)" : " (trial-parallel)") << endl;
        report_dynamics_config(lattice, config);
        cout << "Pulse: A = " << config.pump_amplitude << ", w = " << config.pump_width
             << ", omega = " << config.pump_frequency << endl;
    }
#ifdef CUDA_ENABLED
    bind_gpu(config, rank);
#endif
    const Lattice::SpinConfig start = lattice.spins;
    auto scan = [&](const string& trial_dir, bool parallel) {
        if (parallel) {
            lattice.pump_probe_spectroscopy_mpi(
                field_dirs, config.pump_amplitude, config.pump_width, config.pump_frequency,
                config.tau_start, config.tau_end, config.tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                config.T_start, config.T_end, config.annealing_steps, config.T_zero, config.n_deterministics,
                trial_dir, config.md_integrator, config.use_gpu,
                config.reuse_m0_for_m1, config.stationarity_tol, config.pulse_window_chunking,
                config.pump_probe_abs_tol, config.pump_probe_rel_tol, comm);
        } else {
            lattice.pump_probe_spectroscopy(
                field_dirs, config.pump_amplitude, config.pump_width, config.pump_frequency,
                config.tau_start, config.tau_end, config.tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                config.T_start, config.T_end, config.annealing_steps, config.T_zero, config.n_deterministics,
                trial_dir, config.md_integrator, config.use_gpu,
                config.reuse_m0_for_m1, config.stationarity_tol, config.pump_probe_omp_threads,
                config.pulse_window_chunking, config.pump_probe_abs_tol, config.pump_probe_rel_tol);
        }
    };

    if (use_tau_parallel) {
        const string trial_dir = config.output_dir + "/sample_0";
        // Rank 0 prepares the ground state (the scan broadcasts it); a failure
        // there must reach every rank before the scan's collectives.
        string error;
        if (rank == 0) {
            try {
                filesystem::create_directories(trial_dir);
                prepare_trial_state(lattice, config, start, 0, rank, /*polish_seed=*/true);
                lattice.save_spin_config(trial_dir + "/initial_spins.txt");
            } catch (const exception& e) {
                error = e.what();
            }
        }
        int failed = error.empty() ? 0 : 1;
        MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
        if (failed) throw runtime_error(rank == 0 ? error : "ground-state preparation failed on rank 0");
        scan(trial_dir, true);
        if (rank == 0) cout << "Results: " << trial_dir << "/pump_probe_spectroscopy.h5" << endl;
    } else {
        for_each_trial(config, rank, size, [&](int trial) {
            const string trial_dir = config.output_dir + "/sample_" + to_string(trial);
            filesystem::create_directories(trial_dir);
            prepare_trial_state(lattice, config, start, trial, rank, /*polish_seed=*/true);
            lattice.save_spin_config(trial_dir + "/initial_spins.txt");
            scan(trial_dir, false);
            cout << "[Rank " << rank << "] trial " << trial << " -> " << trial_dir
                 << "/pump_probe_spectroscopy.h5" << endl;
        });
    }
    if (rank == 0) {
        cout << "\n2DCS spectroscopy completed. Non-linear signal: M_NL(t, tau) = M01 - M0 - M1 "
             << "(all on the same time grid)." << endl;
    }
}
