/**
 * phonon_config.cpp — SpinConfig → PhononLattice (NCTO model) construction.
 * See phonon_config.h.
 */

#include "classical_spin/lattice/phonon_config.h"
#include "classical_spin/core/unitcell_builders.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using std::cout;
using std::endl;

void build_phonon_params(const SpinConfig& config,
                         SpinPhononCouplingParams& sp_params,
                         PhononParams& ph_params,
                         DriveParams& dr_params,
                         TimeDependentSpinPhononParams& td_sp_params) {
    // Defaults (when a key is absent) = the Na2Co2TeO6 experimental operating point, i.e. the
    // SpinPhononCouplingParams / PhononParams / DriveParams struct defaults (Krüger fit, J7 at
    // the 3Q/ZZ near-degeneracy, 4.2 THz E1 mode, measured ring-down, measured pulse shape).
    // The struct defaults are the single source of these values.
    const SpinPhononCouplingParams d_sp;
    const PhononParams d_ph;
    const DriveParams d_dr;
    sp_params = SpinPhononCouplingParams{};
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
    // anisotropic (λ_X,2) coefficient per exchange channel X ∈ {J, K, Γ, Γ'}:
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
    // (off by default; see SpinPhononCouplingParams for the operating-point estimate).
    sp_params.lambda_E1_J_1      = config.get_param("lambda_E1_J_1",      d_sp.lambda_E1_J_1);
    sp_params.lambda_E1_K_1      = config.get_param("lambda_E1_K_1",      d_sp.lambda_E1_K_1);
    sp_params.lambda_E1_Gamma_1  = config.get_param("lambda_E1_Gamma_1",  d_sp.lambda_E1_Gamma_1);
    sp_params.lambda_E1_Gammap_1 = config.get_param("lambda_E1_Gammap_1", d_sp.lambda_E1_Gammap_1);

    // Spin storage frame (the same key selects the builder's frame)
    sp_params.frame = classical_spin::kitaev::ncto_frame_from_flag(
        config.get_param(classical_spin::kitaev::kLegacyFrameKey, 0.0));

    // Time-dependent magnetoelastic scale s(t) on H_ME.
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
    if (!(ph_params.omega_E1 > 0.0) || ph_params.gamma_E1 < 0.0)
        throw std::invalid_argument("build_phonon_params: omega_E1 must be > 0 and gamma_E1 >= 0");

    // Pulse shapes: a key given in the file wins; absent keys take the measured NCTO pulse
    // (3.0 THz carrier, σ = 0.273). The pulse centres are the config's pump_time/probe_time.
    auto pick = [&](const char* key, double value, double ncto_default) {
        if (!config.was_set(key)) return ncto_default;
        if (!(value > 0.0)) throw std::invalid_argument(std::string("build_phonon_params: ") + key + " must be > 0");
        return value;
    };
    dr_params = DriveParams{};
    dr_params.E0_1    = config.pump_amplitude;
    dr_params.omega_1 = pick("pump_frequency", config.pump_frequency, d_dr.omega_1);
    dr_params.t_1     = config.pump_time;   // pulse centre always from the config (default 0)
    dr_params.sigma_1 = pick("pump_width", config.pump_width, d_dr.sigma_1);
    dr_params.phi_1   = config.get_param("pump_phase", d_dr.phi_1);
    dr_params.theta_1 = config.get_param("pump_polarization", d_dr.theta_1);

    // Pulse 2 (second THz pulse of the coherent-control protocol)
    dr_params.E0_2    = config.probe_amplitude;
    dr_params.omega_2 = pick("probe_frequency", config.probe_frequency, d_dr.omega_2);
    dr_params.t_2     = config.probe_time;
    dr_params.sigma_2 = pick("probe_width", config.probe_width, d_dr.sigma_2);
    dr_params.phi_2   = config.get_param("probe_phase", d_dr.phi_2);
    dr_params.theta_2 = config.get_param("probe_polarization", d_dr.theta_2);
}

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
    if (n < 0) throw std::invalid_argument("build_lattice_modes: n_extra_modes must be >= 0");
    for (int i = 1; i <= n; ++i) {
        LatticeMode md;
        const int irr = int(config.get_param(key(i, "irrep"), 0.0));
        if (irr < 0 || irr > 3)
            throw std::invalid_argument("build_lattice_modes: " + key(i, "irrep") + " must be 0 (E1), 1 (E2), 2 (A1) or 3 (A2)");
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

DriveParams resolve_ncto_drive(const SpinConfig& config, SimulationType simulation) {
    SpinPhononCouplingParams sp; PhononParams ph; DriveParams dr; TimeDependentSpinPhononParams td;
    build_phonon_params(config, sp, ph, dr, td);
    switch (simulation) {
        case SimulationType::PUMP_PROBE:
            // the pump is the protocol; the second pulse only on request
            if (!config.was_set("probe_amplitude")) dr.E0_2 = 0.0;
            break;
        case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
            dr.E0_1 = dr.E0_2 = 0.0;                      // set per run by the 2DCS driver
            break;
        default:
            if (!config.was_set("pump_amplitude")) dr.E0_1 = 0.0;
            if (!config.was_set("probe_amplitude")) dr.E0_2 = 0.0;
            break;
    }
    auto check = [&](const char* name, double E0, double tc, double sigma) {
        if (E0 != 0.0 && std::abs(tc - config.md_time_start) < 4.0 * sigma)
            cout << "WARNING: " << name << " pulse centred at t = " << tc << " lies within 4σ = " << 4.0 * sigma
                 << " of md_time_start = " << config.md_time_start
                 << ": the drive is switched on abruptly (move the pulse centre)" << endl;
    };
    check("pump", dr.E0_1, dr.t_1, dr.sigma_1);
    check("probe", dr.E0_2, dr.t_2, dr.sigma_2);
    return dr;
}

PhononLattice make_ncto_lattice(const SpinConfig& config) {
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, config.lattice_size[0], config.lattice_size[1],
                          config.lattice_size[2], config.spin_length);

    SpinPhononCouplingParams sp_params;
    PhononParams ph_params;
    DriveParams dr_params;
    TimeDependentSpinPhononParams td_sp_params;
    build_phonon_params(config, sp_params, ph_params, dr_params, td_sp_params);
    const SimulationType sim = (config.simulation == SimulationType::PARAMETER_SWEEP)
                             ? config.sweep_base_simulation : config.simulation;
    dr_params = resolve_ncto_drive(config, sim);

    lattice.set_parameters(sp_params, ph_params, dr_params);
    build_lattice_modes(config, lattice);
    lattice.apply_nn_exchange_disorder_from_file(config.nn_exchange_disorder_config);
    lattice.apply_nn_exchange_channel_disorder_from_file(config.nn_exchange_channel_disorder_config);
    lattice.apply_plaquette_j7_disorder_from_file(config.plaquette_j7_disorder_config);
    lattice.set_time_dependent_spin_phonon(td_sp_params);

    // One damping default for every NCTO path: deterministic dynamics is conservative unless
    // alpha_gilbert is given (the Langevin runners substitute 0.05 when it is absent).
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
    if (lattice.alpha_gilbert < 0.0) throw std::invalid_argument("alpha_gilbert must be >= 0");

    // Zeeman field in the storage frame (crystal axes by default) + site pinning fields
    if (config.field_direction.size() < 3) throw std::invalid_argument("field_direction needs 3 components");
    lattice.set_field(Eigen::Vector3d(config.field_strength * config.field_direction[0],
                                      config.field_strength * config.field_direction[1],
                                      config.field_strength * config.field_direction[2]));
    lattice.add_pinning_fields_from_file(config.pinning_field_config);

    // Spin–lattice dynamics (explicit in-plane acoustic phonons with exchange striction)
    lattice.langevin_temperature = config.get_param("langevin_temperature", 0.0);
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
        lattice.enable_sld(true);
    }
    // Langevin bath settings (used only by integrate_langevin)
    lattice.langevin_dT      = config.get_param("langevin_dT", 0.0);
    lattice.langevin_t_step  = config.get_param("langevin_t_step", dr_params.t_1);
    lattice.langevin_tau_on  = config.get_param("langevin_tau_on", 5.0);
    lattice.langevin_tau_off = config.get_param("langevin_tau_off", 0.0);
    lattice.langevin_quantum = config.get_param("langevin_quantum", 0.0) > 0.5;
    lattice.langevin_block   = static_cast<int>(config.get_param("langevin_block", 4096.0));
    lattice.langevin_bath_C  = config.get_param("langevin_bath_C", 0.0);
    lattice.phonon_langevin_T = config.get_param("phonon_langevin_T", -1.0);

    lattice.mc_sample_lattice = config.get_param("mc_sample_lattice", 0.0) > 0.5;
    if (config.local_update == "heat_bath" || config.local_update == "heatbath")
        lattice.local_update = PhononLattice::LocalUpdate::HeatBath;
    else if (config.local_update == "metropolis" || config.local_update == "gaussian")
        lattice.local_update = PhononLattice::LocalUpdate::Metropolis;
    else
        throw std::invalid_argument("local_update must be 'metropolis', 'gaussian' or 'heat_bath', got '" +
                                    config.local_update + "'");

    // Initial spins (same priority as every trial start: seed file, ferromagnetic, random)
    if (!config.initial_spin_config.empty()) {
        lattice.load_spin_config(config.initial_spin_config);
    } else if (config.use_ferromagnetic_init) {
        if (config.ferromagnetic_direction.size() < 3)
            throw std::invalid_argument("ferromagnetic_direction needs 3 components");
        lattice.init_ferromagnetic(Eigen::Vector3d(config.ferromagnetic_direction[0],
                                                   config.ferromagnetic_direction[1],
                                                   config.ferromagnetic_direction[2]));
    } else {
        lattice.init_random();
    }

    // Krüger et al. (PRL 131, 146702) local-field stand-in for the ring exchange,
    // H_nbl = −h Σ n_i·S_i, n_i the triple-q directions (S = 1/2: h = 0.88 meV): added ONCE
    // to the static field, from local_field_config if given, else the loaded seed.
    const double h_loc = config.get_param("local_field_h", 0.0);
    if (h_loc != 0.0) {
        std::vector<Eigen::Vector3d> dirs;
        std::string src;
        if (!config.local_field_config.empty()) {
            dirs = PhononLattice::read_vec3_file(config.local_field_config, lattice.lattice_size);
            src = config.local_field_config;
        } else if (!config.initial_spin_config.empty()) {
            for (size_t s = 0; s < lattice.lattice_size; ++s) dirs.push_back(Eigen::Vector3d(lattice.spins[s]));
            src = "the loaded spin directions (" + config.initial_spin_config + ")";
        } else {
            throw std::invalid_argument("local_field_h needs local_field_config or initial_spin_config for its directions");
        }
        for (size_t s = 0; s < lattice.lattice_size; ++s) {
            if (!(dirs[s].norm() > 1e-12))
                throw std::invalid_argument("local_field_h: zero direction at site " + std::to_string(s) + " in " + src);
            lattice.field[s] += h_loc * dirs[s].normalized();
        }
        cout << "Local field h = " << h_loc << " meV along " << src << " (triple-q stabiliser)" << endl;
    }

    // Optional: prescribe a non-zero initial E1 phonon displacement (curvature of the
    // BO surface at ε = 0, cooperative pseudo-Jahn-Teller test).
    lattice.phonons.Q_x_E1 = config.get_param("initial_eps_x", 0.0);
    lattice.phonons.Q_y_E1 = config.get_param("initial_eps_y", 0.0);
    lattice.phonons.V_x_E1 = config.get_param("initial_v_x", 0.0);
    lattice.phonons.V_y_E1 = config.get_param("initial_v_y", 0.0);
    return lattice;
}

uint64_t ncto_trial_seed(const SpinConfig& config, int trial) {
    const uint64_t base = config.was_set("langevin_seed")
                        ? static_cast<uint64_t>(config.get_param("langevin_seed", 0.0))
                        : static_cast<uint64_t>(config.seed);
    return splitmix64(base ^ splitmix64(static_cast<uint64_t>(trial) + 0x4C414E4745564EULL));
}
