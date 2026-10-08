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

// ============================================================================
// SIMULATION RUNNERS
// ============================================================================

/**
 * Run simulated annealing: trials round-robin over the ranks of comm. Every
 * trial starts from the configured initial state (initial_spin_config or the
 * ferromagnetic start, else fresh random spins), so trial k does not depend
 * on which rank ran it or what ran before. Every rank writes its trials'
 * sample_<k>/final_energy.txt; rank 0 writes the summary over all trials.
 */
void run_simulated_annealing(Lattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    if (rank == 0) {
        cout << "Running simulated annealing..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }
    const bool fixed_start = !config.initial_spin_config.empty() || config.use_ferromagnetic_init;
    const Lattice::SpinConfig start = lattice.spins;
    if (rank == 0 && !config.initial_spin_config.empty())
        cout << "Every trial starts from " << config.initial_spin_config << endl;

    vector<TrialResult> results;
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        if (fixed_start) lattice.spins = start;
        else lattice.init_random();

        // With annealing_steps == 0 this collapses to a plain energy
        // evaluation of the starting configuration.
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
        
        // Save final configuration and energy (every rank, for its own trials)
        lattice.save_positions(trial_dir + "/positions.txt");
        lattice.save_spin_config(trial_dir + "/spins_final.txt");
        const double e = lattice.energy_density();
        {
            const string path = trial_dir + "/final_energy.txt";
            ofstream energy_file(path);
            energy_file << setprecision(17) << "Energy Density: " << e << "\n";
            energy_file.close();
            if (!energy_file) throw runtime_error("cannot write " + path);
        }
        results.push_back({trial, e, trial_dir + "/spins_final.txt"});
        cout << "[Rank " << rank << "] Trial " << trial << " completed. Final energy: " << setprecision(12) << e
             << endl;
    }
    write_trial_summary(config, "simulated annealing", results, comm);
    
    if (rank == 0) {
        cout << "Simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run parallel tempering on the ranks of comm (one replica per rank). Every
 * trial after the first restarts from the configured initial state (a loaded
 * configuration is no longer replaced by random spins).
 */
void run_parallel_tempering(Lattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    if (rank == 0) {
        cout << "Running parallel tempering with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    const bool fixed_start = !config.initial_spin_config.empty() || config.use_ferromagnetic_init;
    const Lattice::SpinConfig start = lattice.spins;

    // Ladder: geometric, or tuned with the replica chain itself (nrpt by
    // default). The tuned replicas are kept: each rank's configuration is
    // already equilibrated near its production temperature.
    const vector<double> temps = pt_runner::prepare_ladder(config, rank, size, [&](const mc::LadderTuningOptions& o) {
        return lattice.tune_temperature_ladder(o, config.overrelaxation_rate, config.gaussian_move, comm);
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
            if (fixed_start) lattice.spins = start;
            else lattice.init_random();
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
            comm,
            false,  // verbose
            config.pt_accumulate_correlations,
            config.pt_n_bond_types
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
            filesystem::create_directories(trial_dir + "/rank_0");
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
    lattice.damping_form = classical_spin::dynamics::parse_damping_form(config.damping_form);
    if (lattice.alpha_gilbert < 0.0 || lattice.langevin_temperature < 0.0)
        throw invalid_argument("alpha_gilbert and langevin_temperature must be >= 0");
}

void report_dynamics_config(const Lattice& lattice, const SpinConfig& config) {
    cout << "Integrator: " << config.md_integrator << ", dt = " << config.md_timestep
         << ", t = " << config.md_time_start << " -> " << config.md_time_end << endl;
    if (lattice.alpha_gilbert > 0.0)
        cout << "Damping: alpha = " << lattice.alpha_gilbert << " (" << config.damping_form << " form)" << endl;
    if (lattice.langevin_temperature > 0.0)
        cout << "Langevin bath: T = " << lattice.langevin_temperature << endl;
}

/// The GPU flag handed to the drivers: config.use_gpu only when a device is
/// actually usable (this rank is then bound to one, round robin), so a
/// "falling back to the CPU" message is true.
bool select_gpu(const SpinConfig& config, int rank) {
    if (!config.use_gpu) return false;
#ifdef CUDA_ENABLED
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        if (rank == 0) cout << "Warning: no usable GPU; running on the CPU" << endl;
        return false;
    }
    cudaSetDevice(rank % device_count);
    cout << "[Rank " << rank << "] GPU " << (rank % device_count) << " of " << device_count << endl;
    return true;
#else
    if (rank == 0) cout << "GPU requested but not compiled in (CUDA_ENABLED); running on the CPU" << endl;
    return false;
#endif
}

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
/// rank and turns into an error at the end (main then aborts the job).
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
void run_molecular_dynamics(Lattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    apply_dynamics_config(lattice, config);
    if (rank == 0) {
        cout << "Running molecular dynamics: " << config.num_trials << " trial(s) on " << size << " rank(s)" << endl;
        report_dynamics_config(lattice, config);
    }
    const bool gpu = select_gpu(config, rank);
    const Lattice::SpinConfig start = lattice.spins;
    // Dynamical structure factor mode: S(q, w) from thermal samples instead of one trajectory.
    Lattice::DSSFSettings dssf;
    if (config.dssf_samples > 0) {
        const auto b = lattice.reciprocal_vectors();
        for (const auto& hkl : config.dssf_q_points)
            dssf.q_points.push_back(hkl[0] * b[0] + hkl[1] * b[1] + hkl[2] * b[2]);
        dssf.temperature = (config.dssf_temperature >= 0.0) ? config.dssf_temperature : config.T_end;
        dssf.n_samples = config.dssf_samples;
        dssf.t_equilibrate = config.dssf_t_equilibrate;
        dssf.t_decorrelate = config.dssf_t_decorrelate;
        dssf.alpha_sampling = config.dssf_alpha;
        dssf.t_max = config.md_time_end - config.md_time_start;
        dssf.dt = config.md_timestep;
        dssf.save_every = config.md_save_interval;
        dssf.method = config.md_integrator;
        dssf.hann_window = config.dssf_hann_window;
        if (rank == 0)
            cout << "DSSF: " << dssf.q_points.size() << " q points, " << dssf.n_samples << " samples at T = "
                 << dssf.temperature << ", t_max = " << dssf.t_max << endl;
    }
    vector<TrialResult> results;
    for_each_trial(config, rank, size, [&](int trial) {
        const string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        prepare_trial_state(lattice, config, start, trial, rank, /*polish_seed=*/false);
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        const double e0 = lattice.energy_density();
        if (config.dssf_samples > 0) {
            const Lattice::DSSFResult r = lattice.dynamical_structure_factor(dssf);
            Lattice::write_dssf(r, trial_dir + "/dssf.h5");
            cout << "[Rank " << rank << "] trial " << trial << " -> " << trial_dir << "/dssf.h5" << endl;
            results.push_back({trial, e0, trial_dir + "/dssf.h5"});
            return;
        }
        lattice.molecular_dynamics(config.md_time_start, config.md_time_end, config.md_timestep,
                                   trial_dir, config.md_save_interval, config.md_integrator,
                                   gpu, config.md_abs_tol, config.md_rel_tol);
        cout << "[Rank " << rank << "] trial " << trial << " -> " << trial_dir << "/trajectory.h5" << endl;
        results.push_back({trial, e0, trial_dir + "/trajectory.h5"});
    });
    write_trial_summary(config, "molecular dynamics (energy = initial state)", results, comm);
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
void run_pump_probe(Lattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    using classical_spin::dynamics::Pulse;
    using classical_spin::dynamics::TimeGrid;
    apply_dynamics_config(lattice, config);
    const vector<SpinVector> pump_dirs = pulse_directions(config.pump_directions, lattice, "pump_direction");
    const Pulse pump{config.pump_time, config.pump_amplitude, config.pump_width, config.pump_frequency};
    const Pulse probe{config.probe_time, config.probe_amplitude, config.probe_width, config.probe_frequency};
    const bool with_probe = (config.probe_amplitude != 0.0);
    const vector<SpinVector> probe_dirs =
        with_probe ? pulse_directions({config.probe_direction}, lattice, "probe_direction") : pump_dirs;
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
    const bool gpu = select_gpu(config, rank);
    const TimeGrid grid = TimeGrid::covering(config.md_time_start, config.md_time_end, config.md_timestep,
                                             "pump-probe time grid");
    const Lattice::DynamicsSettings settings{config.md_integrator, config.md_timestep,
                                             config.pump_probe_abs_tol, config.pump_probe_rel_tol, 0.0};
    const Lattice::SpinConfig start = lattice.spins;
    vector<TrialResult> results;
    for_each_trial(config, rank, size, [&](int trial) {
        const string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        prepare_trial_state(lattice, config, start, trial, rank, /*polish_seed=*/true);
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        const double e0 = lattice.energy_density();

        Lattice::DriveSchedule both = lattice.make_drive();
        lattice.add_pulse(both, pump_dirs, pump);
        if (with_probe) lattice.add_pulse(both, probe_dirs, probe);
        Lattice::PumpProbeTrajectory M01, M0, M1;
        if (gpu && !with_probe) {
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
        out.close();
        if (!out) throw runtime_error("writing " + path + " failed");
        cout << "[Rank " << rank << "] trial " << trial << " -> " << path << endl;
        results.push_back({trial, e0, path});
    });
    write_trial_summary(config, "pump-probe (energy = initial state)", results, comm);
    if (rank == 0) cout << "Pump-probe completed (" << config.num_trials << " trials)." << endl;
}

/**
 * Run 2D coherent spectroscopy (pump-probe delay scan, M0 / M1 / M01).
 * With one trial and parallel_tau, the delays are distributed over the ranks
 * (pump_probe_spectroscopy_mpi; rank 0's ground state is broadcast); with
 * several trials each rank runs whole scans for its trials. The ground state
 * is prepared identically in both modes (prepare_trial_state).
 */
void run_2dcs_spectroscopy(Lattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    apply_dynamics_config(lattice, config);
    const vector<SpinVector> field_dirs = pulse_directions(config.pump_directions, lattice, "pump_direction");
    // GPU batched path: with use_gpu one launch handles every delay, so route
    // there even on a single rank. The routing depends on the config only, so
    // every rank takes the same branch.
    const bool use_tau_parallel = (config.num_trials == 1) && config.parallel_tau &&
                                  ((size > 1) || config.use_gpu);
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy: " << config.num_trials << " trial(s) on " << size
             << " rank(s), delays " << config.tau_start << " .. " << config.tau_end << " step "
             << config.tau_step << (use_tau_parallel ? " (delay-parallel)" : " (trial-parallel)") << endl;
        report_dynamics_config(lattice, config);
        cout << "Pulse: A = " << config.pump_amplitude << ", w = " << config.pump_width
             << ", omega = " << config.pump_frequency << endl;
    }
    const bool gpu = select_gpu(config, rank);
    const Lattice::SpinConfig start = lattice.spins;
    auto scan = [&](const string& trial_dir, bool parallel) {
        if (parallel) {
            lattice.pump_probe_spectroscopy_mpi(
                field_dirs, config.pump_amplitude, config.pump_width, config.pump_frequency,
                config.tau_start, config.tau_end, config.tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                config.T_start, config.T_end, config.annealing_steps, config.T_zero, config.n_deterministics,
                trial_dir, config.md_integrator, gpu,
                config.reuse_m0_for_m1, config.stationarity_tol, config.pulse_window_chunking,
                config.pump_probe_abs_tol, config.pump_probe_rel_tol, comm);
        } else {
            lattice.pump_probe_spectroscopy(
                field_dirs, config.pump_amplitude, config.pump_width, config.pump_frequency,
                config.tau_start, config.tau_end, config.tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                config.T_start, config.T_end, config.annealing_steps, config.T_zero, config.n_deterministics,
                trial_dir, config.md_integrator, gpu,
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
        const double e0 = lattice.energy_density();
        scan(trial_dir, true);
        if (rank == 0) cout << "Results: " << trial_dir << "/pump_probe_spectroscopy.h5" << endl;
        vector<TrialResult> mine;
        if (rank == 0) mine.push_back({0, e0, trial_dir + "/pump_probe_spectroscopy.h5"});
        write_trial_summary(config, "2DCS, delay-parallel (energy = ground state)", mine, comm);
    } else {
        vector<TrialResult> results;
        for_each_trial(config, rank, size, [&](int trial) {
            const string trial_dir = config.output_dir + "/sample_" + to_string(trial);
            filesystem::create_directories(trial_dir);
            prepare_trial_state(lattice, config, start, trial, rank, /*polish_seed=*/true);
            lattice.save_spin_config(trial_dir + "/initial_spins.txt");
            const double e0 = lattice.energy_density();
            scan(trial_dir, false);
            cout << "[Rank " << rank << "] trial " << trial << " -> " << trial_dir
                 << "/pump_probe_spectroscopy.h5" << endl;
            results.push_back({trial, e0, trial_dir + "/pump_probe_spectroscopy.h5"});
        });
        write_trial_summary(config, "2DCS (energy = ground state)", results, comm);
    }
    if (rank == 0) {
        cout << "\n2DCS spectroscopy completed. Non-linear signal: M_NL(t, tau) = M01 - M0 - M1 "
             << "(all on the same time grid)." << endl;
    }
}
