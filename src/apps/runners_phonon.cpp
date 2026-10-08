/**
 * runners_phonon.cpp — spin_solver runners for PhononLattice (discrete phonon modes).
 *
 * Split out of `spin_solver.cpp` so each lattice family compiles in its
 * own TU. See `src/apps/spin_solver_runners.h`. The model itself is built once by
 * make_ncto_lattice() (classical_spin/lattice/phonon_config.h); the runners only
 * prepare independent trials and drive the simulations.
 *
 * MPI: every runner is collective over `comm` (MPI_COMM_SELF for a sweep point).
 * Trials are distributed round-robin over the ranks of comm; the 2DCS runner with one
 * trial distributes the delay points instead (all ranks share rank 0's ground state).
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/phonon_config.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <string>

using namespace std;

namespace {

/// Small-angle Gaussian proposals: gaussian_move = true or local_update = gaussian.
bool gaussian_proposals(const SpinConfig& config) {
    return config.gaussian_move || config.local_update == "gaussian";
}

/**
 * Start trial `trial` from the same model in an independent state: the seed file is
 * reloaded (initial_spin_config) or the spins are redrawn (ferromagnetic / random), the
 * lattice sector is reset to the configured initial displacement, and — without a seed
 * file — the spins are equilibrated by simulated annealing.
 */
void prepare_trial(PhononLattice& lattice, const SpinConfig& config, bool anneal, bool verbose) {
    if (!config.initial_spin_config.empty()) {
        lattice.load_spin_config(config.initial_spin_config);
        lattice.reset_lattice_sector();
    } else if (config.use_ferromagnetic_init) {
        lattice.init_ferromagnetic(Eigen::Vector3d(config.ferromagnetic_direction[0],
                                                   config.ferromagnetic_direction[1],
                                                   config.ferromagnetic_direction[2]));
    } else {
        lattice.init_random();
    }
    lattice.phonons.Q_x_E1 = config.get_param("initial_eps_x", 0.0);
    lattice.phonons.Q_y_E1 = config.get_param("initial_eps_y", 0.0);
    lattice.phonons.V_x_E1 = config.get_param("initial_v_x", 0.0);
    lattice.phonons.V_y_E1 = config.get_param("initial_v_y", 0.0);

    if (anneal && config.initial_spin_config.empty()) {
        if (verbose) cout << "Equilibrating spin subsystem..." << endl;
        lattice.simulated_annealing(
            config.T_start, config.T_end, config.annealing_steps,
            config.overrelaxation_rate, config.cooling_rate,
            "", false, config.T_zero, config.n_deterministics,
            config.adiabatic_phonons, gaussian_proposals(config),
            config.get_param("preserve_initial_phonons", 0.0) > 0.5);
    } else if (verbose && !config.initial_spin_config.empty()) {
        cout << "Using the spin configuration from " << config.initial_spin_config << endl;
    }
}

/// Joint spin–lattice relaxation before the dynamics (relax_phonons / adiabatic_phonons).
void relax_if_requested(PhononLattice& lattice, const SpinConfig& config, bool verbose) {
    if (config.relax_phonons || config.adiabatic_phonons) {
        if (verbose)
            cout << (config.phonon_only_relax ? "Relaxing phonons only (spins fixed)..."
                                              : "Relaxing spins and phonons to joint equilibrium...") << endl;
        lattice.relax_joint(1e-8, 100, 10, config.phonon_only_relax);   // tolerance per site
    } else if (verbose) {
        cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
    }
}

/// Langevin dynamics needs damping: alpha_gilbert = 0.05 unless set in the config.
void ensure_langevin_damping(PhononLattice& lattice, const SpinConfig& config, bool verbose) {
    if (!config.was_set("alpha_gilbert")) {
        lattice.alpha_gilbert = 0.05;
        if (verbose) cout << "alpha_gilbert not set: using 0.05 for the Langevin thermostat" << endl;
    }
}

/// Stochastic run of one trial with its own noise stream.
void run_langevin_trial(PhononLattice& lattice, const SpinConfig& config, int trial,
                        const string& trial_dir, bool verbose) {
    ensure_langevin_damping(lattice, config, verbose);
    const uint64_t seed = ncto_trial_seed(config, trial);
    if (verbose) {
        cout << "Starting Langevin dynamics: T (k_B T) = " << lattice.langevin_temperature
             << ", alpha_gilbert = " << lattice.alpha_gilbert << ", trial seed = " << seed << endl;
        cout << "  Time range: " << config.md_time_start << " -> " << config.md_time_end
             << ", fixed timestep " << config.md_timestep << endl;
    }
    lattice.integrate_langevin(config.md_time_start, config.md_time_end, config.md_timestep,
                               trial_dir, config.md_save_interval, seed);
}

}  // namespace

/**
 * Run simulated annealing on PhononLattice (spin subsystem only).
 */
void run_simulated_annealing_phonon(PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    if (rank == 0) {
        cout << "Running simulated annealing on PhononLattice (spin subsystem)..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }

    // Distribute trials across MPI ranks
    vector<TrialResult> results;
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }

        prepare_trial(lattice, config, /*anneal=*/false, rank == 0);
        // annealing_steps == 0: a plain energy evaluation of the start (as for Lattice)
        if (config.annealing_steps > 0) {
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                trial_dir,
                config.save_observables,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons,
                gaussian_proposals(config),
                config.get_param("preserve_initial_phonons", 0.0) > 0.5
            );
        } else if (rank == 0) {
            cout << "annealing_steps == 0: skipping SA, just evaluating energy." << endl;
        }

        lattice.save_positions(trial_dir + "/positions.txt");
        lattice.print_state();
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
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    write_trial_summary(config, "PhononLattice simulated annealing", results, comm);

    if (rank == 0) {
        cout << "PhononLattice simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run molecular dynamics for PhononLattice (full spin-phonon dynamics).
 * langevin_temperature > 0 selects the thermostatted (Langevin) integrator.
 */
void run_molecular_dynamics_phonon(PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    if (rank == 0) {
        cout << "Running spin-phonon molecular dynamics on PhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Zone-center E1 effective charge Z* = " << lattice.phonon_params.Z_star
             << ", THz drive: E0_1 = " << lattice.drive_params.E0_1 << ", E0_2 = " << lattice.drive_params.E0_2
             << " (set pump_amplitude / probe_amplitude to drive an MD run)" << endl;
    }

    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }

        prepare_trial(lattice, config, /*anneal=*/true, rank == 0);
        relax_if_requested(lattice, config, rank == 0);

        // Save initial spin configuration and site positions before time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        lattice.save_positions(trial_dir + "/positions.txt");

        if (lattice.langevin_temperature > 0.0) {
            run_langevin_trial(lattice, config, trial, trial_dir, rank == 0);
        } else {
            if (rank == 0) {
                cout << "Starting spin-phonon dynamics..." << endl;
                cout << "Time range: " << config.md_time_start
                     << " -> " << config.md_time_end << endl;
                cout << "Timestep: " << config.md_timestep << endl;
                cout << "Integration method: " << config.md_integrator
                     << ", alpha_gilbert = " << lattice.alpha_gilbert << endl;
            }
            lattice.molecular_dynamics(
                config.md_time_start,
                config.md_time_end,
                config.md_timestep,
                trial_dir,
                config.md_save_interval,
                config.md_integrator,
                config.md_abs_tol,
                config.md_rel_tol
            );
        }

        lattice.print_state();
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }

    MPI_Barrier(comm);

    if (rank == 0) {
        cout << "PhononLattice spin-phonon dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run pump-probe for PhononLattice (THz driving IR phonon)
 */
void run_pump_probe_phonon(PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    if (rank == 0) {
        cout << "Running THz pump-probe on PhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "\nDrive parameters:" << endl;
        cout << "  Pump amplitude: " << lattice.drive_params.E0_1 << endl;
        cout << "  Pump frequency: " << lattice.drive_params.omega_1 << endl;
        cout << "  Pump time: " << lattice.drive_params.t_1 << endl;
        cout << "  Pump width: " << lattice.drive_params.sigma_1 << endl;
        cout << "  Second pulse amplitude: " << lattice.drive_params.E0_2 << endl;
        cout << "  Zone-center E1 effective charge Z* = " << lattice.phonon_params.Z_star << endl;
    }

    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }

        prepare_trial(lattice, config, /*anneal=*/true, rank == 0);
        relax_if_requested(lattice, config, rank == 0);

        // Set ordering pattern AFTER all equilibration is complete
        // This ensures O_custom = 1 at t=0 (spins match the ordering pattern)
        lattice.set_ordering_pattern();

        // Save initial configuration and site positions (for r3 / chirality post-processing)
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        lattice.save_positions(trial_dir + "/positions.txt");

        // The THz drive enters through drive_params inside ode_system() in both branches.
        if (lattice.langevin_temperature > 0.0) {
            run_langevin_trial(lattice, config, trial, trial_dir, rank == 0);
        } else {
            if (rank == 0) {
                cout << "Starting THz-driven spin-phonon dynamics..." << endl;
            }
            lattice.molecular_dynamics(
                config.md_time_start,
                config.md_time_end,
                config.md_timestep,
                trial_dir,
                config.md_save_interval,
                config.md_integrator,
                config.md_abs_tol,
                config.md_rel_tol
            );
        }

        lattice.print_state();
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }

    MPI_Barrier(comm);

    if (rank == 0) {
        cout << "PhononLattice THz pump-probe completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run 2D coherent spectroscopy for PhononLattice
 * Full pump-probe delay scan with nonlinear signal extraction.
 *
 * One trial on several ranks: the delay points are distributed (tau-parallel). Rank 0
 * prepares the ground state and broadcasts it, then every rank of comm enters the
 * collective pump_probe_spectroscopy_mpi. Otherwise trials are distributed round-robin and
 * each rank runs the serial driver (a collective inside a rank-strided loop would deadlock
 * or mix ground states of different trials).
 */
void run_2dcs_phonon(PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    // Pulse shape: the same resolution as the MD/pump-probe drive (explicit key, else the
    // measured NCTO pulse).
    SpinPhononCouplingParams sp; PhononParams ph; DriveParams dr; TimeDependentSpinPhononParams td;
    build_phonon_params(config, sp, ph, dr, td);
    const double pulse_amp = config.pump_amplitude;
    const double pulse_width = dr.sigma_1;
    const double pulse_freq = dr.omega_1;
    const double pulse_theta = config.get_param("pulse_polarization", dr.theta_1);

    // Delay scan: explicit keys, else the historic default 0 → 100 step 5.
    const double tau_start = config.was_set("tau_start") ? config.tau_start : 0.0;
    const double tau_end   = config.was_set("tau_end")   ? config.tau_end   : 100.0;
    const double tau_step  = config.was_set("tau_step")  ? config.tau_step  : 5.0;

    const bool tau_parallel = (config.num_trials == 1) && config.parallel_tau && (size > 1);

    if (rank == 0) {
        cout << "\n===========================================" << endl;
        cout << "2D Coherent Spectroscopy on PhononLattice" << endl;
        cout << "===========================================" << endl;
        cout << "Pulse parameters:" << endl;
        cout << "  Amplitude: " << pulse_amp << endl;
        cout << "  Width: " << pulse_width << endl;
        cout << "  Frequency: " << pulse_freq << endl;
        cout << "  Polarization: " << pulse_theta << " rad" << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end
             << " (step: " << tau_step << ")" << endl;
        cout << "Time evolution: " << config.md_time_start << " → "
             << config.md_time_end << " (step: " << config.md_timestep << ")" << endl;
        cout << "Parallelism: " << (tau_parallel ? "delay points over MPI ranks" : "trials over MPI ranks") << endl;
    }

    if (tau_parallel) {
        const string trial_dir = config.output_dir + "/sample_0";
        // Rank 0 prepares the ground state; a failure there is broadcast before anyone
        // blocks in the state broadcast.
        int ok = 1;
        std::string err;
        if (rank == 0) {
            try {
                filesystem::create_directories(trial_dir);
                prepare_trial(lattice, config, /*anneal=*/true, true);
                relax_if_requested(lattice, config, true);
            } catch (const std::exception& e) {
                ok = 0;
                err = e.what();
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
        if (!ok) throw std::runtime_error("run_2dcs_phonon: ground-state preparation failed on rank 0" +
                                          (err.empty() ? std::string() : ": " + err));
        lattice.broadcast_state(0, comm);
        lattice.pump_probe_spectroscopy_mpi(
            pulse_theta,
            pulse_amp, pulse_width, pulse_freq,
            tau_start, tau_end, tau_step,
            config.md_time_start, config.md_time_end, config.md_timestep,
            trial_dir, config.md_integrator,
            // Ingredient XV (W1/W3 — W2 replaced by MPI here):
            config.reuse_m0_for_m1,
            config.stationarity_tol,
            config.pulse_window_chunking,
            // Ingredient XVIII: pump-probe ODE tolerances (default 1e-8).
            config.pump_probe_abs_tol,
            config.pump_probe_rel_tol,
            comm
        );
    } else {
        for (int trial = rank; trial < config.num_trials; trial += size) {
            string trial_dir = config.output_dir + "/sample_" + to_string(trial);
            filesystem::create_directories(trial_dir);
            if (config.num_trials > 1) {
                cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
            }
            prepare_trial(lattice, config, /*anneal=*/true, rank == 0);
            relax_if_requested(lattice, config, rank == 0);
            lattice.pump_probe_spectroscopy(
                pulse_theta,
                pulse_amp, pulse_width, pulse_freq,
                tau_start, tau_end, tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                trial_dir, config.md_integrator,
                // Ingredient XV (W1/W2/W3):
                config.reuse_m0_for_m1,
                config.stationarity_tol,
                config.pump_probe_omp_threads,
                config.pulse_window_chunking,
                config.pump_probe_abs_tol,
                config.pump_probe_rel_tol
            );
            cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        }
    }

    MPI_Barrier(comm);

    if (rank == 0) {
        cout << "\n===========================================" << endl;
        cout << "PhononLattice 2DCS completed!" << endl;
        cout << "===========================================" << endl;
    }
}
