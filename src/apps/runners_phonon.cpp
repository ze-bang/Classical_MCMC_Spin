/**
 * runners_phonon.cpp — spin_solver runners for PhononLattice (discrete phonon modes).
 *
 * Split out of `spin_solver.cpp` so each lattice family compiles in its
 * own TU. See `src/apps/spin_solver_runners.h`.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/phonon_lattice.h"

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
 * Build spin-phonon / phonon / drive / time-dependent parameters from a SpinConfig.
 */
void build_phonon_params(const SpinConfig& config, 
                         SpinPhononCouplingParams& sp_params,
                         PhononParams& ph_params,
                         DriveParams& dr_params,
                         TimeDependentSpinPhononParams& td_sp_params) {
    // Defaults (when a key is absent) = the Na2Co2TeO6 experimental operating point, i.e. the
    // SpinPhononCouplingParams / PhononParams / DriveParams struct defaults (Krüger fit, J7 at
    // the 3Q/ZZ near-degeneracy, 4.2 THz E1 mode, measured ring-down, measured pulse shape).
    const SpinPhononCouplingParams d_sp;
    const PhononParams d_ph;
    const DriveParams d_dr;
    sp_params.J = config.get_param("J", d_sp.J);
    sp_params.K = config.get_param("K", d_sp.K);
    sp_params.Gamma = config.get_param("Gamma", d_sp.Gamma);
    sp_params.Gammap = config.get_param("Gammap", d_sp.Gammap);

    // 2nd neighbor (J2) - isotropic Heisenberg, sublattice-dependent
    sp_params.J2_A = config.get_param("J2_A", d_sp.J2_A);
    sp_params.J2_B = config.get_param("J2_B", d_sp.J2_B);

    // 3rd neighbor (J3) - isotropic Heisenberg
    sp_params.J3 = config.get_param("J3", d_sp.J3);

    // Six-spin ring exchange on hexagonal plaquettes
    sp_params.J7 = config.get_param("J7", d_sp.J7);
    // Scalar quadratic E1 modulation of ring exchange (phonon effect on J_ring):
    //   J7_eff = J7 + lambda_E1_J7_0 * |epsilon|^2.
    sp_params.lambda_E1_J7_0 = config.get_param("lambda_E1_J7_0",
                                      config.get_param("lambda_J7_0", d_sp.lambda_E1_J7_0));
    
    // E1 magnetoelastic couplings: one isotropic (λ_X,0) and one
    // anisotropic (λ_X,2) coefficient per exchange channel X ∈ {J, K, Γ, Γ'}.
    // The leading symmetry-allowed E1 coupling is quadratic in ε; see
    // δX_γ(ε) = λ_X,0 (ε_x²+ε_y²) + λ_X,2 [(ε_x²-ε_y²) cos2θ_γ + 2ε_x ε_y sin2θ_γ].
    sp_params.lambda_E1_J_0      = config.get_param("lambda_E1_J_0",      config.get_param("lambda_J_0",      d_sp.lambda_E1_J_0));
    sp_params.lambda_E1_J_2      = config.get_param("lambda_E1_J_2",      config.get_param("lambda_J_2",      d_sp.lambda_E1_J_2));
    sp_params.lambda_E1_K_0      = config.get_param("lambda_E1_K_0",      config.get_param("lambda_K_0",      d_sp.lambda_E1_K_0));
    sp_params.lambda_E1_K_2      = config.get_param("lambda_E1_K_2",      config.get_param("lambda_K_2",      d_sp.lambda_E1_K_2));
    sp_params.lambda_E1_Gamma_0  = config.get_param("lambda_E1_Gamma_0",  config.get_param("lambda_Gamma_0",  d_sp.lambda_E1_Gamma_0));
    sp_params.lambda_E1_Gamma_2  = config.get_param("lambda_E1_Gamma_2",  config.get_param("lambda_Gamma_2",  d_sp.lambda_E1_Gamma_2));
    sp_params.lambda_E1_Gammap_0 = config.get_param("lambda_E1_Gammap_0", config.get_param("lambda_Gammap_0", d_sp.lambda_E1_Gammap_0));
    sp_params.lambda_E1_Gammap_2 = config.get_param("lambda_E1_Gammap_2", config.get_param("lambda_Gammap_2", d_sp.lambda_E1_Gammap_2));
    // D3-allowed linear E-channel striction δX_γ = λ_{X,1}[ε_x cos2θ_γ − ε_y sin2θ_γ]
    // (off by default; see SpinPhononCouplingParams).
    sp_params.lambda_E1_J_1      = config.get_param("lambda_E1_J_1",      0.0);
    sp_params.lambda_E1_K_1      = config.get_param("lambda_E1_K_1",      0.0);
    sp_params.lambda_E1_Gamma_1  = config.get_param("lambda_E1_Gamma_1",  0.0);
    sp_params.lambda_E1_Gammap_1 = config.get_param("lambda_E1_Gammap_1", 0.0);

    // Time-dependent E1 magnetoelastic scaling (single multiplicative
    // factor on all 8 quadratic coefficients).
    const double time_mode = config.get_param("lambda_time_mode", 0.0);
    td_sp_params.mode = (time_mode > 0.5) ? "window" : "constant";
    td_sp_params.t_start_E1 = config.get_param("lambda_E1_t_start", 0.0);
    td_sp_params.t_end_E1   = config.get_param("lambda_E1_t_end",   1e30);
    td_sp_params.e1_coupling_scale_target =
        config.get_param("lambda_E1_target", 1.0);

    // Zone-center E1 phonon parameters: ω_E1, γ_E1, optional quartic
    // self-coupling λ (ε²)²/4, and effective charge Z_E1*.
    ph_params.omega_E1          = config.get_param("omega_E1",          config.get_param("omega_E", d_ph.omega_E1));
    ph_params.gamma_E1          = config.get_param("gamma_E1",          config.get_param("gamma_E", d_ph.gamma_E1));
    ph_params.lambda_E1_quartic = config.get_param("lambda_E1_quartic", config.get_param("lambda_E", d_ph.lambda_E1_quartic));
    ph_params.Z_star            = config.get_param("Z_star", d_ph.Z_star);
    // phonon_per_site = 1 (default): magnetoelastic force per site in the E1
    // equation of motion (size-independent dynamics). 0 = legacy extensive force.
    ph_params.per_site_backaction = config.get_param("phonon_per_site", 1.0) > 0.5;

    // Drive parameters (pulse 1 - pump) - couples linearly to ε via -Z*·E·ε.
    // pump_frequency/pump_width/pump_time are SpinConfig members shared with the other
    // systems, so "absent" cannot be detected there; the NCTO operating point (3.0 THz
    // carrier, σ = 0.273, centre 10) is used whenever the config leaves them at the generic
    // SpinConfig defaults (0, 10, 0) or sets them ≤ 0.
    auto pick = [](double cfg, double generic_default, double ncto_default) {
        return (cfg <= 0.0 || cfg == generic_default) ? ncto_default : cfg;
    };
    const SpinConfig generic;
    dr_params.E0_1    = config.pump_amplitude;
    dr_params.omega_1 = pick(config.pump_frequency, generic.pump_frequency, d_dr.omega_1);
    dr_params.t_1     = config.pump_time;   // pulse centre is always taken from the config
    dr_params.sigma_1 = pick(config.pump_width, generic.pump_width, d_dr.sigma_1);
    dr_params.phi_1   = config.get_param("pump_phase", d_dr.phi_1);
    dr_params.theta_1 = config.get_param("pump_polarization", d_dr.theta_1);

    // Drive parameters (pulse 2 - second THz pulse of the coherent-control protocol)
    dr_params.E0_2    = config.probe_amplitude;
    dr_params.omega_2 = pick(config.probe_frequency, generic.probe_frequency, d_dr.omega_2);
    dr_params.t_2     = config.probe_time;
    dr_params.sigma_2 = pick(config.probe_width, generic.probe_width, d_dr.sigma_2);
    dr_params.phi_2   = config.get_param("probe_phase", d_dr.phi_2);
    dr_params.theta_2 = config.get_param("probe_polarization", d_dr.theta_2);
}

/**
 * Complete lattice sector (audit 2026-08): extra zone-centre modes and cubic
 * anharmonic transfers.  Keys (i = 1..n_extra_modes):
 *   n_extra_modes
 *   mode<i>_irrep      0 = E polar/E1 (IR, drives with Zstar), 1 = E2-type (Raman / in-plane
 *                      strain), 2 = A1 (Raman / breathing strain), 3 = A2 (c-polarised)
 *   mode<i>_omega, _gamma, _quartic, _Zstar, _frozen, _Q1, _Q2   (initial or frozen values)
 *   first order : mode<i>_cE<k> (k=1..9), _aA1_<k> (1..5), _dA2_<k> (1..4),
 *                 _lamJ7, _lamJ2A, _lamJ2B, _lamJ3
 *   second order: mode<i>_bEsq<k> (1..9), _aA1sq_<k> (1..5), _lamJ7sq, _lamJ2Asq, _lamJ2Bsq, _lamJ3sq
 * The primary mode (i = 0) accepts the same keys on top of the legacy lambda_E1_* values
 * (e.g. mode0_cE5 .. mode0_cE9 for the five non-channel linear tensors, mode0_aA1sq_5 for
 * the DM-parallel term, mode0_lamJ2A for the J2 nematic).
 * Anharmonic transfers: n_anharmonic, anh<j>_target, anh<j>_lam, anh<j>_lamp, anh<j>_g
 * (mode indices, 0 = primary), energy −N g Q_target (Q_lam ⊗ Q_lamp).
 */
void build_lattice_modes(const SpinConfig& config, PhononLattice& lattice) {
    auto key = [](int i, const std::string& s) { return "mode" + std::to_string(i) + "_" + s; };
    auto fill = [&](int i, LatticeMode& md) {
        for (int k = 0; k < 9; ++k) md.cE[k]     = config.get_param(key(i, "cE" + std::to_string(k + 1)),    md.cE[k]);
        for (int k = 0; k < 5; ++k) md.aA1[k]    = config.get_param(key(i, "aA1_" + std::to_string(k + 1)),  md.aA1[k]);
        for (int k = 0; k < 4; ++k) md.dA2[k]    = config.get_param(key(i, "dA2_" + std::to_string(k + 1)),  md.dA2[k]);
        for (int k = 0; k < 9; ++k) md.bE_sq[k]  = config.get_param(key(i, "bEsq" + std::to_string(k + 1)),  md.bE_sq[k]);
        for (int k = 0; k < 5; ++k) md.aA1_sq[k] = config.get_param(key(i, "aA1sq_" + std::to_string(k + 1)), md.aA1_sq[k]);
        md.lamJ7     = config.get_param(key(i, "lamJ7"),    md.lamJ7);
        md.lamJ2A    = config.get_param(key(i, "lamJ2A"),   md.lamJ2A);
        md.lamJ2B    = config.get_param(key(i, "lamJ2B"),   md.lamJ2B);
        md.lamJ3     = config.get_param(key(i, "lamJ3"),    md.lamJ3);
        md.lamJ7_sq  = config.get_param(key(i, "lamJ7sq"),  md.lamJ7_sq);
        md.lamJ2A_sq = config.get_param(key(i, "lamJ2Asq"), md.lamJ2A_sq);
        md.lamJ2B_sq = config.get_param(key(i, "lamJ2Bsq"), md.lamJ2B_sq);
        md.lamJ3_sq  = config.get_param(key(i, "lamJ3sq"),  md.lamJ3_sq);
    };
    std::vector<LatticeMode> extra;
    const int n = int(config.get_param("n_extra_modes", 0.0));
    for (int i = 1; i <= n; ++i) {
        LatticeMode md;
        const int irr = int(config.get_param(key(i, "irrep"), 0.0));
        md.irrep = (irr == 2) ? LatticeMode::Irrep::A1 : (irr == 3) ? LatticeMode::Irrep::A2 : LatticeMode::Irrep::E;
        md.weight = (irr == 1) ? 2 : 1;
        md.name = (irr == 0) ? "E1 (polar)" : (irr == 1) ? "E2-type" : (irr == 2) ? "A1" : "A2";
        md.omega   = config.get_param(key(i, "omega"), 1.0);
        md.gamma   = config.get_param(key(i, "gamma"), 0.0);
        md.quartic = config.get_param(key(i, "quartic"), 0.0);
        md.Zstar   = config.get_param(key(i, "Zstar"), (irr == 0) ? 1.0 : 0.0);
        md.frozen  = config.get_param(key(i, "frozen"), 0.0) > 0.5;
        md.Q1 = config.get_param(key(i, "Q1"), 0.0);
        md.Q2 = config.get_param(key(i, "Q2"), 0.0);
        fill(i, md);
        extra.push_back(md);
    }
    std::vector<AnharmonicTerm> anh;
    const int na = int(config.get_param("n_anharmonic", 0.0));
    for (int j = 1; j <= na; ++j) {
        AnharmonicTerm t;
        const std::string p = "anh" + std::to_string(j) + "_";
        t.target = int(config.get_param(p + "target", 0.0));
        t.lam    = int(config.get_param(p + "lam", 0.0));
        t.lamp   = int(config.get_param(p + "lamp", 0.0));
        t.g      = config.get_param(p + "g", 0.0);
        anh.push_back(t);
    }
    lattice.set_modes(extra, anh);
    // Primary-mode extras on top of the legacy lambda_E1_* couplings.
    fill(0, lattice.modes[0]);
    lattice.update_modulation_flags();
}

/**
 * Run simulated annealing on PhononLattice (spin subsystem only).
 */
void run_simulated_annealing_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing on PhononLattice (spin subsystem)..." << endl;
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
            config.overrelaxation_rate,
            config.cooling_rate,
            trial_dir,
            config.save_observables,
            config.T_zero,
            config.n_deterministics,
            config.adiabatic_phonons,
            config.gaussian_move,
            config.get_param("preserve_initial_phonons", 0.0) > 0.5
        );
        
        // Save final configuration
        lattice.save_positions(trial_dir + "/positions.txt");
        // lattice.save_spin_config(trial_dir + "/spins.txt");
        lattice.print_state();
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "PhononLattice simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run molecular dynamics for PhononLattice (full spin-phonon dynamics)
 */
void run_molecular_dynamics_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running spin-phonon molecular dynamics on PhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Zone-center E1 effective charge Z* = " << lattice.phonon_params.Z_star << endl;
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
        
        // First equilibrate at low temperature (spin subsystem only)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating spin subsystem..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons,
                config.gaussian_move
            );
        } else {
            lattice.load_spin_config(config.initial_spin_config);
        }
        
        // Relax phonons to equilibrium for the current spin configuration
        // This finds the joint spin-phonon equilibrium before time evolution
        // Skip if adiabatic_phonons was used (phonons already relaxed during SA) and relax_phonons is false
        if (config.relax_phonons || config.adiabatic_phonons) {
            if (rank == 0) {
                if (config.phonon_only_relax) {
                    cout << "Relaxing phonons only (spins fixed)..." << endl;
                } else {
                    cout << "Relaxing spins and phonons to joint equilibrium..." << endl;
                }
            }
            lattice.relax_joint(1e-6, 100, 10, config.phonon_only_relax);
        } else {
            if (rank == 0) {
                cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
            }
        }
        
        // Save initial spin configuration and site positions before time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        lattice.save_positions(trial_dir + "/positions.txt");

        // Decide between deterministic MD and Langevin based on T.
        const double langevin_T_md = config.get_param("langevin_temperature", 0.0);
        const bool use_langevin = (langevin_T_md > 0.0);

        if (use_langevin) {
            lattice.langevin_temperature = langevin_T_md;
            if (lattice.alpha_gilbert <= 0.0)
                lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.01);
            if (rank == 0) {
                cout << "Starting Langevin dynamics (qualitative spin thermostat):" << endl;
                cout << "  T (k_B T) = " << langevin_T_md
                     << ", alpha_gilbert = " << lattice.alpha_gilbert << endl;
                cout << "  Time range: " << config.md_time_start
                     << " -> " << config.md_time_end << endl;
                cout << "  Fixed timestep: " << config.md_timestep << endl;
            }
            const uint64_t seed = static_cast<uint64_t>(
                config.get_param("langevin_seed", 0.0));
            lattice.integrate_langevin(
                config.md_time_start,
                config.md_time_end,
                config.md_timestep,
                trial_dir,
                config.md_save_interval,
                seed
            );
        } else {
            // Deterministic spin-phonon MD
            if (rank == 0) {
                cout << "Starting spin-phonon dynamics..." << endl;
                cout << "Time range: " << config.md_time_start
                     << " -> " << config.md_time_end << endl;
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
                config.md_abs_tol,
                config.md_rel_tol
            );
        }

        lattice.print_state();
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "PhononLattice spin-phonon dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run pump-probe for PhononLattice (THz driving IR phonon)
 */
void run_pump_probe_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running THz pump-probe on PhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "\nDrive parameters:" << endl;
        cout << "  Pump amplitude: " << config.pump_amplitude << endl;
        cout << "  Pump frequency: " << config.pump_frequency << endl;
        cout << "  Pump time: " << config.pump_time << endl;
        cout << "  Pump width: " << config.pump_width << endl;
        cout << "  Zone-center E1 effective charge Z* = " << lattice.phonon_params.Z_star << endl;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        // Equilibrate spin subsystem
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating spin subsystem..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons,
                config.gaussian_move
            );
        } else {
            lattice.load_spin_config(config.initial_spin_config);
        }
        
        // Relax spins and phonons to joint equilibrium
        // Skip if adiabatic_phonons was used (phonons already relaxed during SA) and relax_phonons is false
        if (config.relax_phonons || config.adiabatic_phonons) {
            if (rank == 0) {
                if (config.phonon_only_relax) {
                    cout << "Relaxing phonons only (spins fixed)..." << endl;
                } else {
                    cout << "Relaxing spins and phonons to joint equilibrium..." << endl;
                }
            }
            lattice.relax_joint(1e-6, 100, 10, config.phonon_only_relax);
        } else {
            if (rank == 0) {
                cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
            }
        }
        
        // Set ordering pattern AFTER all equilibration is complete
        // This ensures O_custom = 1 at t=0 (spins match the ordering pattern)
        lattice.set_ordering_pattern();

        // Save initial configuration and site positions (for r3 / chirality post-processing)
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        lattice.save_positions(trial_dir + "/positions.txt");
        
        // Run spin-phonon dynamics with THz drive.  langevin_temperature > 0 selects the
        // stochastic (thermostatted) integrator; the THz drive enters through
        // drive_params inside ode_system() in both branches, so the pulse is applied
        // identically.  (Before this branch existed the key was silently ignored here.)
        const double langevin_T_pp = config.get_param("langevin_temperature", 0.0);
        // Spin–lattice dynamics (explicit in-plane acoustic phonons with exchange striction)
        if (config.get_param("sld_enabled", 0.0) > 0.5) {
            lattice.sld_mass    = config.get_param("sld_mass", lattice.sld_mass);
            lattice.sld_k       = config.get_param("sld_k", lattice.sld_k);
            lattice.sld_k2      = config.get_param("sld_k2", lattice.sld_k2);
            lattice.sld_g       = config.get_param("sld_g", 0.0);
            lattice.sld_v3      = config.get_param("sld_v3", 0.0);
            lattice.sld_gamma   = config.get_param("sld_gamma", 0.0);
            lattice.sld_T       = config.get_param("sld_T", -1.0);
            lattice.sld_init_T  = config.get_param("sld_init_T", 0.0);
            lattice.sld_quantum = config.get_param("sld_quantum", 0.0) > 0.5;
            lattice.sld_relax   = static_cast<int>(config.get_param("sld_relax", 200.0));
            lattice.langevin_temperature = langevin_T_pp;
            lattice.enable_sld(true);
        }
        if (langevin_T_pp > 0.0) {
            lattice.langevin_temperature = langevin_T_pp;
            // two-reservoir bath profile (scenario 1): optional keys
            lattice.langevin_dT      = config.get_param("langevin_dT", 0.0);
            lattice.langevin_t_step  = config.get_param("langevin_t_step", config.pump_time);
            lattice.langevin_tau_on  = config.get_param("langevin_tau_on", 5.0);
            lattice.langevin_tau_off = config.get_param("langevin_tau_off", 0.0);
            lattice.langevin_quantum = config.get_param("langevin_quantum", 0.0) > 0.5;
            lattice.langevin_block   = static_cast<int>(config.get_param("langevin_block", 4096.0));
            lattice.langevin_bath_C  = config.get_param("langevin_bath_C", 0.0);
            if (lattice.alpha_gilbert <= 0.0)
                lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.01);
            if (rank == 0) {
                cout << "Starting THz-driven spin-phonon LANGEVIN dynamics:" << endl;
                cout << "  T (k_B T) = " << langevin_T_pp
                     << ", alpha_gilbert = " << lattice.alpha_gilbert << endl;
                cout << "  Time range: " << config.md_time_start
                     << " -> " << config.md_time_end
                     << ", fixed timestep " << config.md_timestep << endl;
            }
            lattice.integrate_langevin(
                config.md_time_start,
                config.md_time_end,
                config.md_timestep,
                trial_dir,
                config.md_save_interval,
                static_cast<uint64_t>(config.get_param("langevin_seed", 0.0))
            );
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "PhononLattice THz pump-probe completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run 2D coherent spectroscopy for PhononLattice
 * Full pump-probe delay scan with nonlinear signal extraction
 */
void run_2dcs_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    // Get 2DCS-specific parameters
    double pulse_amp = config.pump_amplitude;
    double pulse_width = config.pump_width;
    double pulse_freq = config.pump_frequency > 0 ? config.pump_frequency 
                        : config.get_param("omega_E", 1.0);
    double pulse_theta = config.get_param("pulse_polarization", 0.0);
    
    double tau_start = config.get_param("tau_start", 0.0);
    double tau_end = config.get_param("tau_end", 100.0);
    double tau_step = config.get_param("tau_step", 5.0);
    
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
    }
    
    // Distribute trials across MPI ranks (typically 1 trial for 2DCS)
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        // Equilibrate spin subsystem if no initial config provided
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "\nEquilibrating spin subsystem via simulated annealing..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons,
                config.gaussian_move
            );
        } else {
            lattice.load_spin_config(config.initial_spin_config);
        }
        
        // Relax spins and phonons to joint equilibrium before 2DCS
        // Skip if adiabatic_phonons was used (phonons already relaxed during SA) and relax_phonons is false
        if (config.relax_phonons || config.adiabatic_phonons) {
            if (rank == 0) {
                if (config.phonon_only_relax) {
                    cout << "Relaxing phonons only (spins fixed)..." << endl;
                } else {
                    cout << "Relaxing spins and phonons to joint equilibrium..." << endl;
                }
            }
            lattice.relax_joint(1e-6, 100, 10, config.phonon_only_relax);
        } else {
            if (rank == 0) {
                cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
            }
        }
        
        // Run 2DCS workflow
        if (size > 1) {
            // Use MPI-parallel version
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
                config.pump_probe_rel_tol
            );
        } else {
            // Single-rank version
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
        }
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n===========================================" << endl;
        cout << "PhononLattice 2DCS completed!" << endl;
        cout << "===========================================" << endl;
    }
}
