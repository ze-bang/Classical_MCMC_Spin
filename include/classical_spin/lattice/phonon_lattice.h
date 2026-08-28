/**
 * @file phonon_lattice.h
 * @brief Spin-phonon coupled honeycomb lattice (NCTO E1 magnetoelastic model)
 *
 * Implements the in-plane E1 magnetoelastic model for Na2Co2TeO6 derived from
 * symmetry under the full hexagonal C6 of the honeycomb layer. See
 * docs/tmfeo3_notes.tex (or the standalone "Leading-order E1 magnetoelastic
 * coupling in NCTO" memo) for the full derivation. The model contains:
 *
 *   - Standard J–K–Γ–Γ' nearest-neighbour spin Hamiltonian (with optional 2nd
 *     and 3rd NN Heisenberg, sublattice-dependent J2, and 6-spin ring exchange).
 *   - A SINGLE zone-center two-component E1 optical-strain coordinate
 *         ε = (ε_x, ε_y)
 *     with conjugate velocities (V_x, V_y). Total phonon DOF = 4.
 *   - Symmetry-allowed leading magnetoelastic coupling, which is QUADRATIC in
 *     ε (a linear coupling is forbidden by the full C6 symmetry once the bond
 *     bilinears are summed):
 *         δX_γ(ε) = λ_{X,0}(ε_x²+ε_y²)
 *                 + λ_{X,2}[(ε_x²-ε_y²)cos(2θ_γ)
 *                           + 2 ε_x ε_y sin(2θ_γ)]
 *     for X ∈ {J, K, Γ, Γ'} and bond-axis angles θ_x=0, θ_y=2π/3, θ_z=4π/3.
 *   - Single global polar THz drive coupling linearly to the E1 coordinate:
 *         H_drive = -Z* [E_x(t) ε_x + E_y(t) ε_y].
 *
 * Hamiltonian:
 *
 *   H = H_spin + H_E1 + H_drive + H_sp-ph
 *
 *   H_spin   = Σ_<ij>γ Si · J1_global · Sj + Σ_<<ij>>_A J2_A Si·Sj
 *              + Σ_<<ij>>_B J2_B Si·Sj + Σ_<<<ij>>> J3 Si·Sj
 *              + H_7  - Σ_i B·S_i
 *
 *   H_E1     = (1/2)(V_x² + V_y²) + (1/2) ω_E1² (ε_x² + ε_y²)
 *              + (λ_E1_quartic / 4) (ε_x² + ε_y²)²
 *
 *   H_drive  = -Z* [E_x(t) ε_x + E_y(t) ε_y]
 *
 *   H_sp-ph  = Σ_<ij>γ Σ_X δX_γ(ε) O_{ij,γ}^{(X)}
 *              + λ_{J7,0}(ε_x²+ε_y²) R_7
 *              where R_7 is the ring-exchange operator without the J7 prefactor.
 *
 * COORDINATE FRAMES
 * -----------------
 * The exchange matrix J^{(γ)} is defined in the LOCAL Kitaev frame and the
 * δX_γ(ε) modulation rides on those local-frame coefficients. Spins are stored
 * and evolved in the GLOBAL Cartesian frame; the spin–phonon coupling is
 * computed by first rotating the spins back to the local Kitaev frame, applying
 * the modulated J^{(γ)}, and rotating the resulting effective field back to the
 * global frame.
 *
 * Equations of motion (Euler–Lagrange):
 *   - Spins:  dS/dt = S × H_eff  (LLG, optional Gilbert damping)
 *   - E1:     d²ε_a/dt² = -ω_E1² ε_a - λ_E1_quartic (ε_x²+ε_y²) ε_a
 *                          - γ_E1 dε_a/dt - ∂H_sp-ph/∂ε_a + Z* E_a(t)
 *             (a = x, y; uniform zone-center field).
 */

#ifndef PHONON_LATTICE_H
#define PHONON_LATTICE_H

#include "unitcell.h"
#include "unitcell_builders.h"
#include "simple_linear_alg.h"
#include "kitaev_bonds.h"
#include "classical_spin/lattice/ncto_me_tensors.h"   // D3-covariant magnetoelastic tensors
#include "classical_spin/lattice/pulse_chunking.h"  // Ingredient XVIII tols + W3 helper
#include <string>
#include <stdexcept>
#include <vector>
#include <array>
#include <functional>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <boost/numeric/odeint.hpp>

// Include Boost uBLAS for implicit solvers (rosenbrock4, implicit_euler)
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp>
#include <boost/numeric/odeint/stepper/implicit_euler.hpp>

#ifdef HDF5_ENABLED
#include "classical_spin/io/hdf5_io.h"
#endif
#include "classical_spin/core/spin_config.h"  // For should_rank_write
#include "classical_spin/mc/mc_common.h"       // Common MC types and algorithms

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::function;
using std::array;

/**
 * A zone-centre lattice coordinate of the honeycomb layer with its complete
 * D3-symmetry-allowed magnetoelastic couplings (audit 2026-08, see
 * ncto_phonon/audit/NCTO_full_model_hamiltonian.tex and ncto_me_tensors.h).
 *
 *   irrep E  : in-plane doublet (Q1,Q2). weight = 1: polar / IR-active E1 mode
 *              (couples to the THz field through Zstar); weight = 2: E2-type
 *              (Raman mode or in-plane shear strain). The weight fixes the sign
 *              with which Q2 enters the tensor tables (q2sign()).
 *   irrep A1 : scalar (Raman A1 mode, eps_xx+eps_yy, eps_zz)
 *   irrep A2 : scalar, c-polarised optical displacement u_z
 *
 * First order:  dM_gamma = Σ_k cE[k] (Q1 T1_k + t2 Q2 T2_k)   (E, 9 tensors)
 *                        = Q Σ_k aA1[k] TA1_k                  (A1, 5 tensors incl. DM ∥ bond)
 *                        = Q Σ_k dA2[k] TA2_k                  (A2, 4 tensors)
 *               J7 += lamJ7 Q (A1);  J2A/J2B/J3 += lamJ2A.. Q (A1) or bond-projected nematic (E)
 * Second order (E only): |Q|² Σ aA1_sq TA1 + (Q⊗Q)_E · Σ bE_sq (T1,T2);  J7 += lamJ7_sq |Q|²;
 *               J2/J3 += lam.._sq |Q|².  For the primary E1 mode these are the legacy
 *               lambda_E1_X_{0,1,2} and lambda_E1_J7_0.
 * frozen = true turns the coordinate into a static parameter (uniform strain).
 */
struct LatticeMode {
    enum class Irrep { E, A1, A2 };
    Irrep irrep = Irrep::E;
    int weight = 1;
    std::string name;
    double omega = 1.0, gamma = 0.0, quartic = 0.0, Zstar = 0.0;
    bool frozen = false;
    std::array<double, 9> cE{};
    std::array<double, 5> aA1{};
    std::array<double, 4> dA2{};
    double lamJ7 = 0.0, lamJ2A = 0.0, lamJ2B = 0.0, lamJ3 = 0.0;
    std::array<double, 5> aA1_sq{};
    std::array<double, 9> bE_sq{};
    double lamJ7_sq = 0.0, lamJ2A_sq = 0.0, lamJ2B_sq = 0.0, lamJ3_sq = 0.0;
    // coordinates and velocities (mode 0 = primary E1 keeps its state in PhononState)
    double Q1 = 0.0, Q2 = 0.0, V1 = 0.0, V2 = 0.0;
    int ncoord() const { return irrep == Irrep::E ? 2 : 1; }
    int ndof() const { return frozen ? 0 : 2 * ncoord(); }
    double q2sign() const { return weight == 2 ? -1.0 : 1.0; }
};

/// Cubic anharmonic transfer −N g Q_target (Q_lam ⊗ Q_lamp)_{irrep(target)}:
/// target A1: Q (Q_lam · Q_lamp); target E (weight 2): Q1 P1 + Q2 P2 with
/// P = (Q_lam1 Q_lamp1 − Q_lam2 Q_lamp2, Q_lam1 Q_lamp2 + Q_lam2 Q_lamp1).
struct AnharmonicTerm {
    int target = 0, lam = 0, lamp = 0;
    double g = 0.0;
};

/**
 * Zone-center E1 phonon state.
 *
 * NORMALISATION OF Q (one dictionary, used everywhere — energy, drive, coupling, EOM):
 *   Q is the mass-normalised amplitude of the zone-centre mode, one coordinate per unit
 *   cell, intensive.  With ħ = 1, energies in meV and time in ħ/meV,
 *     phonon energy   E_ph    = N_sites · [ ½|Q̇|² + ½ ω² |Q|² ]          (ω in meV)
 *     drive energy    E_drive = − N_sites · Z* E(t)·Q
 *     coupling        δX_γ    = λ_{X,1} Q·d̂_γ + λ_{X,2} (Q⊗Q)_E ...        (meV per unit Q, Q²)
 *     equation of motion  Q̈ = −ω²Q − γQ̇ + Z*E(t) − (1/N_sites) ∂H_ME/∂Q
 *   so Q² carries units of 1/meV (= ħ²/(meV) in SI).  Physical dictionary for a mode with
 *   effective mass M_cell per unit cell and atomic displacement u:
 *     Q = sqrt(M_cell/2) · u / ħ     (the ½ because N_sites = 2 N_cell)
 *     e.g. M_cell = 30 amu, u = 2 pm  →  Q ≈ 0.012;  7 µeV per Co of absorbed energy at
 *     4.2 THz  →  |Q| = sqrt(2·0.007/17.37²) ≈ 0.007.
 *     Z* E0 = (N_cell/N_sites) Z_e e E0_phys ħ / sqrt(M_cell/2): the impulsive response of the
 *     matched pulse is |Q|max = 0.0097 per unit Z*E0, hence Z*E0 ≈ 1 ↔ 300 kV/cm.
 *
 * The E1 optical strain field is a single in-plane two-component coordinate
 * ε = (Q_x, Q_y) with conjugate velocities (V_x, V_y). Total DOF = 4.
 *
 * Layout in the flat ODE buffer: [Q_x, Q_y, V_x, V_y].
 */
struct PhononState {
    // Zone-center E1 coordinates and velocities
    double Q_x_E1 = 0.0;
    double Q_y_E1 = 0.0;
    double V_x_E1 = 0.0;
    double V_y_E1 = 0.0;

    static constexpr size_t N_DOF = 4;

    void to_array(double* arr) const {
        arr[0] = Q_x_E1;
        arr[1] = Q_y_E1;
        arr[2] = V_x_E1;
        arr[3] = V_y_E1;
    }

    void from_array(const double* arr) {
        Q_x_E1 = arr[0];
        Q_y_E1 = arr[1];
        V_x_E1 = arr[2];
        V_y_E1 = arr[3];
    }

    /// Kinetic energy of the E1 mode (unit mass).
    double kinetic_energy() const {
        return 0.5 * (V_x_E1 * V_x_E1 + V_y_E1 * V_y_E1);
    }

    /// |ε| = sqrt(Q_x² + Q_y²).
    double E1_amplitude() const {
        return std::sqrt(Q_x_E1 * Q_x_E1 + Q_y_E1 * Q_y_E1);
    }
};

/**
 * E1 phonon parameters.
 */
struct PhononParams {
    // Defaults = the Na2Co2TeO6 experimental operating point (NCTO_phonon_v0, 6 K):
    // dominant IR-active E1 line at 4.2 THz (h·ν = 17.37 meV), amplitude ring-down
    // 2/γ = 6.6 ps (measured 5–10 ps).  Time unit ħ/meV = 0.658 ps.
    double omega_E1 = 17.37;        ///< E1 mode frequency ω_E1 (meV) = 4.2 THz
    double gamma_E1 = 0.20;         ///< E1 mode damping γ_E1 (1/code time): 2/γ = 6.6 ps
    double lambda_E1_quartic = 0.0; ///< Optional quartic self-coupling λ (ε²)²/4
    double Z_star = 1.0;            ///< Effective charge for E(t) coupling
    /// Normalise the magnetoelastic force in the phonon equation of motion per
    /// site (true, physical: zone-centre mode has extensive inertia N, so
    /// ε̈ = ... − (1/N)∂H_sp-ph/∂ε and E_phonon = N(½ε̇²+½ω²ε²)).  false
    /// reproduces the legacy extensive back-action, whose effective phonon
    /// frequency depends on lattice size and goes soft at N ~ ω²/λ.
    bool per_site_backaction = true;
};

/**
 * Spin Hamiltonian + leading E1 magnetoelastic coupling parameters.
 *
 * Spin Hamiltonian: bond-dependent J–K–Γ–Γ' on the honeycomb nearest neighbours,
 * plus optional sublattice-dependent J2, J3, ring-exchange J7, and Zeeman field.
 * The exchange matrices are defined in the LOCAL Kitaev frame and rotated to the
 * GLOBAL Cartesian frame for spin storage:  J_global = R · J_local · Rᵀ.
 *
 * E1 magnetoelastic coupling (in the LOCAL Kitaev frame):
 *   H_sp-ph = Σ_<ij>_γ Σ_X δX_γ(ε) O_{ij,γ}^{(X)},  X ∈ {J, K, Γ, Γ'}
 *   δX_γ(ε) = λ_{X,0} (ε_x² + ε_y²)
 *           + λ_{X,2} [(ε_x² - ε_y²) cos(2θ_γ) + 2 ε_x ε_y sin(2θ_γ)]
 * with bond-axis angles θ_x = 0, θ_y = 2π/3, θ_z = 4π/3 (i.e.
 * cos(2θ_γ) = (1, -1/2, -1/2) and sin(2θ_γ) = (0, -√3/2, +√3/2)).
 *
 * Symmetry (audit 2026-08, ncto_phonon/audit): a single SOC honeycomb layer
 * of Na2Co2TeO6 has point symmetry D3 (C3 ⊥ plane, three C2 axes along the
 * bonds); C2 about the plane normal is the 6_3/2_1 screw that exchanges the
 * two layers and is NOT a symmetry of one layer's J–K–Γ–Γ' Hamiltonian.
 * Under D3 the polar coordinate ε (E) couples to the bond channels
 * (A1 ⊕ E per channel) as
 *   quadratic:  λ_{X,0}|ε|²  and  λ_{X,2}[(ε_x²−ε_y²)cos2θ + 2ε_xε_y sin2θ]
 *   LINEAR   :  λ_{X,1}[ε_x cos2θ_γ − ε_y sin2θ_γ]   (see below; off by default)
 * and to the ring operator (A1) only through |ε|².  The "no linear term"
 * statement holds for the layer-summed q=0 response under the bilayer D6,
 * not for the single-layer model integrated here.
 */
struct SpinPhononCouplingParams {
    // Defaults = Na2Co2TeO6 operating point: Krüger et al. neutron fit (meV) with the ring
    // exchange J7 at the corrected 3Q/zigzag near-degeneracy (fixed J2/J3 bond lists).
    double J = 0.68;       ///< Heisenberg coupling
    double K = -7.89;      ///< Kitaev coupling
    double Gamma = 3.07;   ///< Γ (off-diagonal symmetric)
    double Gammap = -2.94; ///< Γ' (off-diagonal asymmetric)

    // 2nd NN exchange (isotropic Heisenberg, sublattice-dependent: Co1 (2b) ≠ Co2 (2d))
    double J2_A = -0.06;
    double J2_B = -0.70;

    // 3rd NN exchange (isotropic Heisenberg)
    double J3 = 0.52;

    // Six-spin ring exchange on hexagonal plaquettes (3Q ground state for J7 < -0.4096)
    double J7 = -0.4096;
    // Scalar quadratic E1 modulation of ring exchange:
    //   J7_eff(ε) = J7 + lambda_E1_J7_0 (ε_x² + ε_y²)   (the phonon effect on J_ring;
    // λ > 0 drives |J7| down, i.e. towards zigzag, for any polarization).  The ring operator
    // is A1, so this |Q|² term is its LEADING coupling; off by default (scenario 2 turns it on).
    double lambda_E1_J7_0 = 0.0;

    // Quadratic E1 exchange-modulation coefficients δX_γ(ε):
    //   λ_{X,0} multiplies the rotational invariant ε_x² + ε_y²
    //   λ_{X,2} multiplies the bond-dependent rank-2 piece
    // Second-order (|Q|² and (Q⊗Q)_E) bilinear couplings: OFF by default.  They are not the
    // leading term for a polar E1 mode of one D3 layer — the linear channel below is — and the
    // rectified E2-shear physics they were meant to model arises automatically, at O(λ1²/ω²),
    // from the time-dependent linear modulation once the dynamics is integrated.
    double lambda_E1_J_0      = 0.0;
    double lambda_E1_J_2      = 0.0;
    double lambda_E1_K_0      = 0.0;
    double lambda_E1_K_2      = 0.0;
    double lambda_E1_Gamma_0  = 0.0;
    double lambda_E1_Gamma_2  = 0.0;
    double lambda_E1_Gammap_0 = 0.0;
    double lambda_E1_Gammap_2 = 0.0;
    // LINEAR E-channel striction, allowed by the D3 symmetry of one SOC
    // honeycomb layer (P6_322 Co site symmetry 32, twofold axes along bonds):
    //   δX_γ^{(1)}(ε) = λ_{X,1} [ε_x cos2θ_γ − ε_y sin2θ_γ] = λ_{X,1}|ε| cos(θ_pol + 2θ_γ).
    // It is the unique D3 invariant Re[ε₊N₊] (rotational weight 1+2 = 3);
    // it is odd in ε (no static rectification at first order) but it is FIRST order in the
    // phonon amplitude: the leading magnetoelastic term of a polar E1 mode of one D3 layer.
    // Forbidden only by the layer-exchanging C6 of the bilayer (layer-summed response).
    // Defaults = the experimental operating point with a stipulated Grüneisen-type scale:
    // δX/X common to all channels, λ_{X,1} = (X/K) λ_{K,1}, λ_{K,1} = 40 meV per unit Q, i.e.
    // a 5 % modulation of every exchange at the physical amplitude |Q| ≈ 0.01 (γ_G ≈ 10 with a
    // Co1–Co2 relative displacement of ~1 pm at 300 kV/cm).  Replace by DFT/Raman-anomaly values.
    double lambda_E1_J_1      = -3.447;
    double lambda_E1_K_1      = 40.0;
    double lambda_E1_Gamma_1  = -15.56;
    double lambda_E1_Gammap_1 = 14.91;

    /// Kitaev local-to-global rotation matrix R.
    static SpinMatrix get_kitaev_rotation() {
        return classical_spin::kitaev::kitaev_rotation();
    }

    /// Transform local-frame exchange matrix to the global frame.
    static SpinMatrix to_global_frame(const SpinMatrix& J_local) {
        return classical_spin::kitaev::to_global_frame(J_local);
    }

    // Bond-dependent exchange matrices in the LOCAL Kitaev frame.
    SpinMatrix get_Jx_local() const {
        return classical_spin::kitaev::make_Jx_local(J, K, Gamma, Gammap);
    }
    SpinMatrix get_Jy_local() const {
        return classical_spin::kitaev::make_Jy_local(J, K, Gamma, Gammap);
    }
    SpinMatrix get_Jz_local() const {
        return classical_spin::kitaev::make_Jz_local(J, K, Gamma, Gammap);
    }

    // Bond-dependent exchange matrices in the GLOBAL Cartesian frame.
    SpinMatrix get_Jx() const { return to_global_frame(get_Jx_local()); }
    SpinMatrix get_Jy() const { return to_global_frame(get_Jy_local()); }
    SpinMatrix get_Jz() const { return to_global_frame(get_Jz_local()); }

    // J2 / J3 are isotropic Heisenberg → invariant under rotation.
    SpinMatrix get_J3_matrix()   const { return classical_spin::kitaev::heisenberg_matrix(J3);   }
    SpinMatrix get_J2_A_matrix() const { return classical_spin::kitaev::heisenberg_matrix(J2_A); }
    SpinMatrix get_J2_B_matrix() const { return classical_spin::kitaev::heisenberg_matrix(J2_B); }
};

/**
 * Time-dependent multiplicative scale on the E1 magnetoelastic coupling.
 *
 * Multiplies all eight quadratic E1 coupling coefficients (λ_{X,0}, λ_{X,2})
 * by a time-dependent factor s(t):
 *   - mode == "constant" (default):  s(t) = 1
 *   - mode == "window"            :  s(t) = e1_coupling_scale_target for
 *                                    t_start_E1 ≤ t ≤ t_end_E1, else 1.
 *
 * The model in the LaTeX notes uses a constant coupling, so the default leaves
 * the physics unchanged. The window option is provided for protocol-style
 * pump–probe experiments where the magnetoelastic coupling is briefly
 * suppressed or enhanced by an external knob.
 */
struct TimeDependentSpinPhononParams {
    std::string mode = "constant";

    double t_start_E1 = 0.0;
    double t_end_E1   = 1e30;
    double e1_coupling_scale_target = 1.0;

    /// Multiplicative scale applied to all 8 quadratic E1 coefficients at time t.
    double get_e1_coupling_scale(double t) const {
        if (mode == "window" && t >= t_start_E1 && t <= t_end_E1) {
            return e1_coupling_scale_target;
        }
        return 1.0;
    }
};

/**
 * Two-pulse polar THz drive on the E1 phonon coordinate.
 *
 *   E_a(t) = Σ_{n=1,2} E0_n · exp[-(t-t_n)²/(2 σ_n²)] · cos[ω_n(t-t_n)+φ_n] · ê_n,
 *
 * where ê_n = (cos θ_n, sin θ_n) is the in-plane polarization of pulse n. This
 * polar field couples linearly to the zone-center E1 coordinate via
 *   H_drive = -Z* [E_x(t) ε_x + E_y(t) ε_y].
 */
struct DriveParams {
    // Defaults = the measured NCTO THz transient (SI Fig. S1b,c): ~1.5-cycle pulse, envelope
    // FWHM ≈ 0.6 ps, spectrum 1–5 THz peaking at 3–4 THz  →  Gaussian σ = 0.18 ps = 0.273 code
    // units, carrier 3.0 THz = 12.41 meV, CEP 0, centred 6.6 ps (10 code units) after t = 0.
    // Pulse 1 (pump)
    double E0_1    = 0.0;
    double omega_1 = 12.41;
    double t_1     = 10.0;
    double sigma_1 = 0.273;
    double phi_1   = 0.0;
    double theta_1 = 0.0;  ///< Polarization angle from the x-bond line (0 = x, π/2 = y)

    // Pulse 2 (second THz pulse of the coherent-control protocol; E2 ≈ 0.6 E1 in the experiment)
    double E0_2    = 0.0;
    double omega_2 = 12.41;
    double t_2     = 10.0;
    double sigma_2 = 0.273;
    double phi_2   = 0.0;
    double theta_2 = 0.0;

    /// Compute the in-plane THz field components at time t.
    void E_field(double t, double& Ex, double& Ey) const {
        const double dt1 = t - t_1;
        const double dt2 = t - t_2;
        const double env1 = std::exp(-0.5 * dt1 * dt1 / (sigma_1 * sigma_1));
        const double env2 = std::exp(-0.5 * dt2 * dt2 / (sigma_2 * sigma_2));
        const double E1 = E0_1 * env1 * std::cos(omega_1 * dt1 + phi_1);
        const double E2 = E0_2 * env2 * std::cos(omega_2 * dt2 + phi_2);
        Ex = E1 * std::cos(theta_1) + E2 * std::cos(theta_2);
        Ey = E1 * std::sin(theta_1) + E2 * std::sin(theta_2);
    }
};

// ============================================================
// MC types from common library (replaces PL_-prefixed structs)
// ============================================================
using mc::BinningResult;
using mc::Observable;
using mc::VectorObservable;
using mc::ThermodynamicObservables;
using mc::OptimizedTempGridResult;
using mc::AutocorrelationResult;

// Legacy aliases for backward compatibility
using PL_BinningResult             = mc::BinningResult;
using PL_Observable                = mc::Observable;
using PL_VectorObservable          = mc::VectorObservable;
using PL_ThermodynamicObservables  = mc::ThermodynamicObservables;
using PL_OptimizedTempGridResult   = mc::OptimizedTempGridResult;

/**
 * PhononLattice: honeycomb lattice with E1 zone-center magnetoelastic coupling.
 *
 * Degrees of freedom:
 *   - N_spin = N_atoms · dim1 · dim2 · dim3 classical spins (spin_dim = 3)
 *   - 4 zone-center E1 phonon DOF: (Q_x, Q_y, V_x, V_y)
 *
 * Total ODE state size: spin_dim · N_spin + 4.
 */
class PhononLattice {
public:
    using SpinConfig = vector<SpinVector>;
    using ODEState = vector<double>;
    
    // Lattice properties
    static constexpr size_t spin_dim = 3;    // 3D classical spins
    size_t N_atoms;                          // Atoms per unit cell (from UnitCell)
    size_t dim1, dim2, dim3;                 // Lattice dimensions
    size_t lattice_size;                     // Total spin sites
    float spin_length = 1.0;                 // Spin magnitude
    
    // Unit cell (stored for reference)
    UnitCell unit_cell;
    
    // Spin configuration
    SpinConfig spins;
    vector<Eigen::Vector3d> site_positions;
    
    // Phonon state (primary E1 mode)
    PhononState phonons;

    // Complete lattice sector: modes[0] mirrors the primary E1 mode (its
    // coordinates live in `phonons`), modes[1..] are extra E/A1/A2 modes or
    // frozen strains; anharmonic = cubic transfer terms between them.
    vector<LatticeMode> modes;
    vector<AnharmonicTerm> anharmonic;
    bool has_further_modulation = false;
    // In-plane doublet frame of the layer: e_x = A→B direction of the x bond,
    // e_y = n × e_x. Further-neighbour bonds are grouped into ≤3 line-angle
    // classes per sublattice with form factors (cos2θ, sin2θ).
    Eigen::Vector3d ex_code = Eigen::Vector3d::UnitX();
    Eigen::Vector3d ey_code = Eigen::Vector3d::UnitY();
    vector<vector<int>> j2_cls, j3_cls;
    std::array<std::pair<double, double>, 3> j2_cs{}, j3_cs{};
    int n_j2_cls = 0, n_j3_cls = 0;
    struct Coords { std::vector<double> q1, q2; };

    // NN interactions (stored per site to avoid double counting)
    vector<vector<SpinMatrix>> nn_interaction;      // J1 matrices
    vector<vector<size_t>> nn_partners;             // NN partner indices
    vector<vector<int>> nn_bond_types;              // Bond type (0,1,2 for x,y,z bonds)
    
    // 2nd NN interactions (sublattice-dependent)
    vector<vector<SpinMatrix>> j2_interaction;      // J2 matrices (J2_A or J2_B depending on sublattice)
    vector<vector<size_t>> j2_partners;             // 2nd NN partner indices
    
    // 3rd NN interactions
    vector<vector<SpinMatrix>> j3_interaction;      // J3 matrices  
    vector<vector<size_t>> j3_partners;             // 3rd NN partner indices
    
    // Hexagonal plaquettes for ring exchange
    // Each hexagon stores 6 site indices in order (i,j,k,l,m,n) going around the ring
    vector<std::array<size_t, 6>> hexagons;
    // Optional static per-plaquette J7 offsets, used to model Na/stacking-induced
    // ring-exchange landscapes without adding direct spin pinning.
    vector<double> plaquette_j7_offsets;
    // For each site, list of hexagons it belongs to and its position (0-5) within each hexagon
    vector<vector<std::pair<size_t, size_t>>> site_hexagons;
    
    // External field
    vector<SpinVector> field;
    
    // Parameters
    PhononParams phonon_params;
    SpinPhononCouplingParams spin_phonon_params;
    TimeDependentSpinPhononParams time_dep_spin_phonon_params;
    DriveParams drive_params;
    
    // LLG damping
    double alpha_gilbert = 0.0;

    // Langevin thermostat (spins). When > 0 and integrate_langevin() is
    // invoked, a per-step Gaussian noise field with σ = sqrt(2 α k_B T / (|S| dt))
    // is added to H_eff at each spin during integration. Phonons are evolved
    // deterministically (E1 phonon thermal noise neglected at this stage).
    double langevin_temperature = 0.0;

    // Two-reservoir ("scenario 1") bath profile for integrate_langevin(): the spin bath
    // temperature is  T_b(t) = T0 + dT * Θ(t - t_step) (1 - e^{-(t-t_step)/tau_on}) e^{-(t-t_step)/tau_off}
    // i.e. a hot phonon reservoir filled within tau_on (the E1 ring-down) and, optionally,
    // cooling with tau_off (0 = no decay).  dT = 0 reproduces the constant-T thermostat.
    // Semi-quantum thermostat (Barker & Bauer, PRB 100, 140401 (2019)): the stochastic field is
    // coloured Gaussian noise whose power spectrum is the Bose energy of a mode at frequency ω,
    //     P(ω) = P_classical(T) · F(ω,T),   F = x/(e^x − 1),  x = ħω/k_B T,
    // (no zero-point term), while the Gilbert damping stays Markovian.  For every harmonic mode the
    // steady state is then E_k = ħω_k n_B(ω_k): quantum thermal occupation without a quantum solver.
    // The noise is generated in FFT blocks of langevin_block steps, overlap-added with sine windows
    // (Σ w² = 1) so that the process is stationary; the block temperature is the bath temperature at
    // the block centre.  langevin_quantum = false reproduces the classical white-noise thermostat.
    bool langevin_quantum = false;
    int langevin_block = 4096;
    // Finite-capacity, energy-conserving bath (two-temperature model built from the dynamics):
    // when langevin_bath_C > 0 (heat capacity of the bath in k_B per spin) the bath temperature is a
    // dynamical variable, T_b(t+dt) = T_b(t) - [E_sys(t+dt) - E_sys(t) - W_drive]/(N C_l), i.e. every
    // unit of energy the system loses (phonon damping γ, Gilbert damping) heats the bath and every
    // unit the noise injects cools it; the E1 ring-down is then the deposit itself and no
    // langevin_dT step is needed.  0 = infinite bath with the prescribed profile.
    double langevin_bath_C = 0.0;
    double langevin_dT = 0.0;
    double langevin_t_step = 0.0;
    double langevin_tau_on = 5.0;     // code units (5 = 3.3 ps)
    double langevin_tau_off = 0.0;
    double langevin_bath_T(double t) const {
        if (langevin_dT == 0.0 || t < langevin_t_step) return langevin_temperature;
        const double s = t - langevin_t_step;
        double f = (langevin_tau_on > 0.0) ? 1.0 - std::exp(-s / langevin_tau_on) : 1.0;
        if (langevin_tau_off > 0.0) f *= std::exp(-s / langevin_tau_off);
        return langevin_temperature + langevin_dT * f;
    }

    // External "noise field" used by integrate_langevin(): one 3-vector per
    // site, regenerated at every Langevin time step and added to H_eff in
    // ode_system() when use_langevin_noise == true.
    vector<Eigen::Vector3d> langevin_noise;
    bool use_langevin_noise = false;

    // ---- Spin–lattice dynamics (SLD): in-plane site displacements u_i and momenta p_i ----
    // Harmonic NN (k) and 2nd-NN (k2) central springs; exchange striction δM_ij = g δr_ij M_ij with
    // δr_ij = (u_j − u_i)·d̂_ij (g = ∂ln X/∂r, one common Grüneisen fraction for J, K, Γ, Γ′, same
    // convention as the E1 coupling λ_X1); optional cubic E1–acoustic vertex
    // H3 = v3 Σ_bonds (Q·d̂_γ) δr_ij² (decay of the E1 mode into acoustic pairs); lattice friction γ_l
    // with the matching Langevin noise on p in integrate_langevin (classical or Bose-coloured).
    // Units: length = lattice constant a, energy meV, time ħ/meV, mass in meV·(ħ/meV)²/a²
    // (= 0.150 amu for a = 5.27 Å).  ODE layout: after the mode block,
    // [u_0(3), …, u_{N−1}(3), p_0(3), …, p_{N−1}(3)]; z components are kept at zero.
    bool sld_enabled = false;
    double sld_mass = 1000.0, sld_k = 1.5e5, sld_k2 = 4.0e4, sld_g = 0.0, sld_v3 = 0.0;
    double sld_gamma = 0.0, sld_T = -1.0, sld_init_T = 0.0;
    bool sld_quantum = false;
    vector<Eigen::Vector3d> u_site, p_site;
    vector<vector<Eigen::Vector3d>> nn_bond_vec, j2_bond_vec;   // unit vectors i→j per site per neighbour
    vector<vector<Eigen::Vector2d>> nn_bond_cq;                  // A→B direction of the bond in the doublet frame (cx, cy)
    void enable_sld(bool on);
    /// Static relaxation of the site displacements to the magnetostrictive equilibrium of the current
    /// spins (Jacobi iteration on the lattice forces; p = 0).  Returns the final max |F|.  Without this
    /// u = 0 carries an elastic energy F²/2k per site that is released as lattice heat at t = 0.
    double relax_sld_static(int max_iter = 500, double tol = 1e-6);
    int sld_relax = 200;
    size_t mode_dof() const;
    size_t sld_offset() const { return spin_dim * lattice_size + mode_dof(); }
    double sld_energy() const;
    double sld_kinetic_energy() const;
    double sld_spring_energy() const;
    double sld_striction_energy() const;
    /// Lattice temperature from equipartition of the 2N in-plane momenta: T_l = E_kin/N.
    double sld_lattice_temperature() const {
        return (sld_enabled && lattice_size) ? sld_kinetic_energy() / double(lattice_size) : 0.0;
    }

    // ODE state size
    size_t state_size;
    
    // Sublattice local frames for global-to-local spin transformations
    // For Kitaev honeycomb, transforms from local Kitaev basis to global cubic frame
    // sublattice_frames[atom] is a 3x3 rotation matrix: S_global = R * S_local
    vector<SpinMatrix> sublattice_frames;
    vector<double> afm_sublattice_signs;   // AFM sublattice signs for Bertaut modes
    
    // Custom ordering vector (set from initial spin configuration)
    // Used to compute order parameter along the ground state ordering direction
    SpinConfig ordering_pattern;
    bool has_ordering_pattern = false;
    
    /**
     * Constructor: Build from a UnitCell (consistent with Lattice interface)
     * 
     * @param uc        Unit cell defining lattice structure (positions, interactions, bond types)
     * @param d1        Lattice size in first dimension
     * @param d2        Lattice size in second dimension
     * @param d3        Lattice size in third dimension
     * @param spin_l    Magnitude of spin vectors
     */
    PhononLattice(const UnitCell& uc, size_t d1, size_t d2, size_t d3 = 1, float spin_l = 1.0);
    
    // ============================================================
    // LATTICE CONSTRUCTION
    // ============================================================
    
    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom) const {
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }
    
    /**
     * Periodic boundary condition
     */
    int periodic_boundary(int coord, size_t dim_size) const {
        if (coord < 0) return coord + dim_size;
        if (coord >= (int)dim_size) return coord - dim_size;
        return coord;
    }
    
    /**
     * Flatten with periodic boundaries
     */
    size_t flatten_index_periodic(int i, int j, int k, size_t atom) const {
        return flatten_index(
            periodic_boundary(i, dim1),
            periodic_boundary(j, dim2),
            periodic_boundary(k, dim3),
            atom
        );
    }
    
    // ============================================================
    // PARAMETER SETTING
    // ============================================================
    
    /**
     * Set all parameters and rebuild interaction matrices
     */
    void set_parameters(const SpinPhononCouplingParams& sp_params,
                       const PhononParams& ph_params,
                       const DriveParams& dr_params);
    
    /**
     * Set time-dependent E1 magnetoelastic scale parameters.
     */
    void set_time_dependent_spin_phonon(const TimeDependentSpinPhononParams& td_params) {
        time_dep_spin_phonon_params = td_params;
        if (td_params.mode != "constant") {
            std::cout << "Time-dependent E1 magnetoelastic scaling enabled (mode: "
                      << td_params.mode << ")" << std::endl;
            if (td_params.mode == "window") {
                std::cout << "  E1 scale = " << td_params.e1_coupling_scale_target
                          << " for t∈[" << td_params.t_start_E1
                          << ", " << td_params.t_end_E1 << "]" << std::endl;
            }
        }
    }

    /// Multiplicative scale on the 8 quadratic E1 coefficients at time t.
    double get_e1_coupling_scale(double t) const {
        return time_dep_spin_phonon_params.get_e1_coupling_scale(t);
    }
    
    /**
     * Set external magnetic field (uniform)
     */
    void set_field(const Eigen::Vector3d& B) {
        for (size_t i = 0; i < lattice_size; ++i) {
            field[i] = B;
        }
    }

    /**
     * Apply additive plaquette-resolved J7 disorder.
     *
     * Format: plaquette_index dJ7, with optional '#' comments. The offsets are
     * static in time and added to the uniform/pump-dependent J7 on each hexagon.
     */
    void apply_plaquette_j7_disorder_from_file(const string& filename) {
        if (filename.empty()) return;
        ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open plaquette J7 disorder file: " + filename);
        }
        if (plaquette_j7_offsets.size() != hexagons.size()) {
            plaquette_j7_offsets.assign(hexagons.size(), 0.0);
        }

        string line;
        size_t line_no = 0;
        size_t n_loaded = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const size_t hash = line.find('#');
            if (hash != string::npos) line = line.substr(0, hash);
            std::istringstream iss(line);
            size_t plaquette;
            double dJ7;
            if (!(iss >> plaquette >> dJ7)) {
                continue;
            }
            if (plaquette >= plaquette_j7_offsets.size()) {
                throw std::runtime_error(
                    "Plaquette J7 disorder index out of range at line " +
                    std::to_string(line_no) + " in " + filename);
            }
            plaquette_j7_offsets[plaquette] += dJ7;
            ++n_loaded;
        }
        std::cout << "Applied " << n_loaded << " plaquette J7 disorder offsets from "
                  << filename << std::endl;
    }

    /**
     * Add site-resolved pinning fields from a text file.
     *
     * Format: site Bx By Bz, with optional '#' comments.  The fields are added
     * on top of the current uniform field, so call set_field() first.
     */
    void add_pinning_fields_from_file(const string& filename) {
        if (filename.empty()) return;
        ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open pinning field file: " + filename);
        }

        string line;
        size_t line_no = 0;
        size_t n_loaded = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const size_t hash = line.find('#');
            if (hash != string::npos) line = line.substr(0, hash);
            std::istringstream iss(line);
            size_t site;
            double bx, by, bz;
            if (!(iss >> site >> bx >> by >> bz)) {
                continue;
            }
            if (site >= lattice_size) {
                throw std::runtime_error(
                    "Pinning field site index out of range at line " +
                    std::to_string(line_no) + " in " + filename);
            }
            field[site](0) += bx;
            field[site](1) += by;
            field[site](2) += bz;
            ++n_loaded;
        }
        cout << "Loaded " << n_loaded << " site-resolved pinning fields from "
             << filename << endl;
    }

    /**
     * Apply quenched nearest-neighbour exchange disorder from a text file.
     *
     * Format: site partner scale, with optional '#' comments.  Each row should
     * refer to one unique NN bond.  The full 3x3 exchange matrix on that bond is
     * multiplied by scale and the reverse directed entry is updated by the same
     * transpose convention.  This perturbs the Hamiltonian without templating a
     * spin direction.
     */
    void apply_nn_exchange_disorder_from_file(const string& filename) {
        if (filename.empty()) return;
        ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open NN exchange disorder file: " + filename);
        }

        auto find_neighbor_index = [&](size_t site, size_t partner) -> size_t {
            for (size_t n = 0; n < nn_partners[site].size(); ++n) {
                if (nn_partners[site][n] == partner) {
                    return n;
                }
            }
            throw std::runtime_error(
                "NN exchange disorder references non-NN bond " +
                std::to_string(site) + " " + std::to_string(partner) +
                " in " + filename);
        };

        string line;
        size_t line_no = 0;
        size_t n_loaded = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const size_t hash = line.find('#');
            if (hash != string::npos) line = line.substr(0, hash);
            std::istringstream iss(line);
            size_t site, partner;
            double scale;
            if (!(iss >> site >> partner >> scale)) {
                continue;
            }
            if (site >= lattice_size || partner >= lattice_size) {
                throw std::runtime_error(
                    "NN exchange disorder site index out of range at line " +
                    std::to_string(line_no) + " in " + filename);
            }
            if (!(scale > 0.0)) {
                throw std::runtime_error(
                    "NN exchange disorder scale must be positive at line " +
                    std::to_string(line_no) + " in " + filename);
            }

            const size_t n_forward = find_neighbor_index(site, partner);
            const size_t n_reverse = find_neighbor_index(partner, site);
            nn_interaction[site][n_forward] *= scale;
            nn_interaction[partner][n_reverse] *= scale;
            ++n_loaded;
        }
        cout << "Applied " << n_loaded << " NN exchange disorder bond scalings from "
             << filename << endl;
    }

    /**
     * Apply additive, channel-resolved quenched NN exchange disorder.
     *
     * Format: site partner dJ dK dGamma dGammap, with optional '#' comments.
     * The increments are interpreted in the local Kitaev channel basis for the
     * corresponding NN bond type, transformed to the global frame used by
     * PhononLattice, and added to the stored exchange matrix.  This allows
     * physically sharper tests such as K-only or Γ-only bond disorder without
     * applying a spin-direction pinning field.
     */
    void apply_nn_exchange_channel_disorder_from_file(const string& filename) {
        if (filename.empty()) return;
        ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open NN exchange channel disorder file: " + filename);
        }

        auto find_neighbor_index = [&](size_t site, size_t partner) -> size_t {
            for (size_t n = 0; n < nn_partners[site].size(); ++n) {
                if (nn_partners[site][n] == partner) {
                    return n;
                }
            }
            throw std::runtime_error(
                "NN exchange channel disorder references non-NN bond " +
                std::to_string(site) + " " + std::to_string(partner) +
                " in " + filename);
        };

        auto channel_matrix = [](int bond_type, double dJ, double dK,
                                 double dGamma, double dGammap) -> SpinMatrix {
            SpinPhononCouplingParams delta_params;
            delta_params.J = dJ;
            delta_params.K = dK;
            delta_params.Gamma = dGamma;
            delta_params.Gammap = dGammap;

            SpinMatrix local = SpinMatrix::Zero(3, 3);
            if (bond_type == 0) {
                local = delta_params.get_Jx_local();
            } else if (bond_type == 1) {
                local = delta_params.get_Jy_local();
            } else if (bond_type == 2) {
                local = delta_params.get_Jz_local();
            } else {
                throw std::runtime_error("Invalid NN bond type in channel disorder");
            }
            return SpinPhononCouplingParams::to_global_frame(local);
        };

        string line;
        size_t line_no = 0;
        size_t n_loaded = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const size_t hash = line.find('#');
            if (hash != string::npos) line = line.substr(0, hash);
            std::istringstream iss(line);
            size_t site, partner;
            double dJ, dK, dGamma, dGammap;
            if (!(iss >> site >> partner >> dJ >> dK >> dGamma >> dGammap)) {
                continue;
            }
            if (site >= lattice_size || partner >= lattice_size) {
                throw std::runtime_error(
                    "NN exchange channel disorder site index out of range at line " +
                    std::to_string(line_no) + " in " + filename);
            }

            const size_t n_forward = find_neighbor_index(site, partner);
            const size_t n_reverse = find_neighbor_index(partner, site);
            const int bond_type = nn_bond_types[site][n_forward];
            const SpinMatrix dM = channel_matrix(bond_type, dJ, dK, dGamma, dGammap);
            nn_interaction[site][n_forward] += dM;
            nn_interaction[partner][n_reverse] += dM.transpose();
            ++n_loaded;
        }
        cout << "Applied " << n_loaded << " NN exchange channel disorder increments from "
             << filename << endl;
    }
    
    /**
     * Set external field for a specific site (consistent with Lattice)
     */
    void set_uniform_field(const Eigen::Vector3d& B) {
        set_field(B);
    }
    
    // ============================================================
    // INITIALIZATION
    // ============================================================
    
    /**
     * Generate random spin on 2-sphere
     */
    SpinVector gen_random_spin() {
        return gen_random_spin(spin_length);
    }
    
    /**
     * Generate random spin on 2-sphere with specified magnitude
     */
    SpinVector gen_random_spin(float spin_l) {
        SpinVector spin(3);
        double z = uniform_dist(rng) * 2.0 - 1.0;
        double phi = uniform_dist(rng) * 2.0 * M_PI;
        double r = std::sqrt(1.0 - z*z);
        spin(0) = r * std::cos(phi);
        spin(1) = r * std::sin(phi);
        spin(2) = z;
        return spin * spin_l;
    }
    
    /**
     * Gaussian move around current spin (small-angle perturbation)
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma) {
        SpinVector perturbation = gen_random_spin(1.0) * sigma;
        SpinVector new_spin = current_spin + perturbation;
        double norm = new_spin.norm();
        if (norm < 1e-10) return current_spin;
        return new_spin * (spin_length / norm);
    }
    
    /**
     * Initialize random spins
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = gen_random_spin();
        }
        phonons = PhononState();
    }
    
    /**
     * Initialize ferromagnetic state
     */
    void init_ferromagnetic(const Eigen::Vector3d& direction) {
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = dir;
        }
        phonons = PhononState();
    }
    
    /**
     * Initialize Néel state (antiferromagnetic on sublattices)
     */
    void init_neel(const Eigen::Vector3d& direction) {
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % N_atoms == 0) ? 1.0 : -1.0;
            spins[i] = sign * dir;
        }
        phonons = PhononState();
    }
    
    /**
     * Set a specific spin (consistent with Lattice)
     */
    void set_spin(size_t site_index, const SpinVector& spin_in) {
        spins[site_index] = spin_in;
    }
    
    /**
     * Get a specific spin (consistent with Lattice)
     */
    const SpinVector& get_spin(size_t site_index) const {
        return spins[site_index];
    }
    
    /**
     * Print lattice info (consistent with Lattice)
     */
    void print_info() const {
        cout << "PhononLattice: " << dim1 << "x" << dim2 << "x" << dim3
             << ", N_atoms=" << N_atoms << ", lattice_size=" << lattice_size
             << ", spin_dim=" << spin_dim << ", spin_length=" << spin_length << endl;
    }
    
    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================
    
    /**
     * Compute local energy contribution for a hypothetical spin at @a site.
     *
     * Includes Zeeman, full nearest-neighbour bond energies (with the E1
     * quadratic exchange modulation δX_γ(ε) folded into the local-frame
     * exchange matrix), and 2nd / 3rd NN couplings. Ring exchange is NOT
     * included here; for the correctly counted total energy use
     * total_energy().
     *
     * Implemented out-of-line so we can reuse the namespaced E1 helpers.
     */
    double site_energy(const Eigen::Vector3d& spin_here, size_t site) const;
    
    /**
     * Compute energy difference for a proposed spin flip (optimized for Metropolis)
     * dE = E(new_spin) - E(old_spin)
     * 
     * Includes:
     * - Zeeman energy change
     * - NN, 2nd NN, 3rd NN spin-spin interaction changes
     * - Spin-phonon coupling energy change (if phonons are non-zero)
     * - Ring exchange energy change
     */
    double site_energy_diff(const Eigen::Vector3d& new_spin, 
                           const Eigen::Vector3d& old_spin, 
                           size_t site) const;
    
    /**
     * Pure spin energy (NN + 2nd NN + 3rd NN + Zeeman + ring exchange)
     */
    double spin_energy() const;
    
    /**
     * Six-spin ring exchange energy on hexagonal plaquettes
     * 
     * H_7 = (J_7/6) Σ_{hex} [2(S_i·S_j)(S_k·S_l)(S_m·S_n)
     *                       -6(S_i·S_k)(S_j·S_l)(S_m·S_n)
     *                       +3(S_i·S_l)(S_j·S_k)(S_m·S_n)
     *                       +3(S_i·S_k)(S_j·S_m)(S_l·S_n)
     *                       -(S_i·S_l)(S_j·S_m)(S_k·S_n)
     *                       + cyclic permutations of (i,j,k,l,m,n)]
     */
    double ring_exchange_energy() const;
    /// Ring-exchange operator R_7, defined by H_7 = J7_eff * R_7.
    double ring_exchange_normalized() const;
    
    /**
     * E1 phonon energy (kinetic + harmonic potential + optional quartic):
     *   E_ph = (1/2)(V_x² + V_y²)
     *        + (1/2) ω_E1² (ε_x² + ε_y²)
     *        + (λ_E1_quartic / 4) (ε_x² + ε_y²)².
     */
    double phonon_energy() const;

    /**
     * E1 magnetoelastic coupling energy (quadratic in ε):
     *   H_sp-ph = Σ_<ij>γ Σ_X δX_γ(ε) O_{ij,γ}^{(X)}, X ∈ {J, K, Γ, Γ'}.
     */
    double spin_phonon_energy() const;
    
    /**
     * Total energy
     */
    double total_energy() const {
        return spin_energy() + phonon_energy() + spin_phonon_energy() + anharmonic_energy() + sld_energy();
    }

    /// Extensivity factor of the zone-centre phonon sector: N_sites when the
    /// back-action is normalised per site (physical), 1 in the legacy mode.
    double phonon_norm() const {
        return phonon_params.per_site_backaction ? double(lattice_size) : 1.0;
    }
    
    /**
     * Energy per site
     */
    double energy_density() const {
        return total_energy() / lattice_size;
    }
    
    // ============================================================
    // DERIVATIVES FOR EQUATIONS OF MOTION
    // ============================================================
    
    /// ∂H_sp-ph/∂ε_x for the zone-center E1 coordinate.
    double dH_dQx_E1() const;
    /// ∂H_sp-ph/∂ε_y for the zone-center E1 coordinate.
    double dH_dQy_E1() const;
    // ---- complete lattice sector (all modes) ----
    size_t phonon_dof() const;
    void recompute_state_size();
    void pack_lattice(double* arr) const;
    void unpack_lattice(const double* arr);
    Coords coords_current() const;
    Coords coords_from_state(const double* arr) const;
    void bond_increments_local(const Coords& c, double scale, Eigen::Matrix3d dM[3]) const;
    void bond_increment_derivs_local(const Coords& c, double scale, size_t m, int comp, Eigen::Matrix3d dD[3]) const;
    void bond_increments_global(const Coords& c, double scale, Eigen::Matrix3d dM[3]) const;
    void bond_increment_derivs_global(const Coords& c, double scale, size_t m, int comp, Eigen::Matrix3d dD[3]) const;
    void bond_correlations(Eigen::Matrix3d C[3]) const;
    void further_correlations(double Cj2[2][3], double Cj3[3]) const;
    double further_bond_modulation(const Coords& c, double c2, double s2, int which, int sub) const;
    double further_bond_modulation_deriv(const Coords& c, double c2, double s2, int which, int sub, size_t m, int comp) const;
    double further_neighbour_modulation_energy(const Coords& c) const;
    /// Effective ring exchange J7 + Σ_E λ_J7|Q|² + Σ_A1 λ_J7 Q (+ frozen strains).
    double effective_J7(const Coords& c) const;
    double effective_J7() const;
    double effective_J7_for_hexagon(size_t hex_idx, double J7eff) const {
        const double offset = hex_idx < plaquette_j7_offsets.size()
                                ? plaquette_j7_offsets[hex_idx] : 0.0;
        return J7eff + offset;
    }
    double dJ7_dq(const Coords& c, size_t m, int comp) const;
    double anharmonic_energy(const Coords& c) const;
    double anharmonic_energy() const;
    double anharmonic_deriv(const Coords& c, size_t m, int comp) const;
    /// Raw (extensive) ∂H_ME/∂q for mode m, component comp.
    double lattice_force_raw(const Coords& c, double scale, const Eigen::Matrix3d C[3],
                             const double Cj2[2][3], const double Cj3[3], double R7,
                             size_t m, int comp) const;
    /// Raw ∂H_ME/∂q for every coordinate, in mode order (q1[,q2] per mode).
    std::vector<double> lattice_forces_raw() const;
    void rebuild_primary_mode();
    void update_modulation_flags();
    void set_modes(const std::vector<LatticeMode>& extra, const std::vector<AnharmonicTerm>& anh);
    
    /**
     * Compute ring exchange contribution to effective field on spin at given site
     * 
     * H_eff_ring = -∂H_7/∂S_site
     * 
     * For each hexagon containing the site, computes the derivative of the
     * ring exchange term with respect to that spin.
     */
    SpinVector get_ring_exchange_field(size_t site) const;
    SpinVector get_ring_exchange_field(size_t site, double J7eff) const;
    
    /**
     * Compute effective field on spin i (for spin EOM)
     * 
     * H_eff = -∂H/∂Si = B + Σ_j [NN contributions] + [spin-phonon contributions] + [ring exchange]
     */
    SpinVector get_local_field(size_t site) const;
    
    // ============================================================
    // EQUATIONS OF MOTION
    // ============================================================
    
    /**
     * E1 phonon EOM derivatives (zone-center, IR-driven):
     *   dε_x/dt = V_x
     *   dV_x/dt = -ω_E1² ε_x - λ_E1_quartic (ε_x²+ε_y²) ε_x
     *             - γ_E1 V_x - ∂H_sp-ph/∂ε_x + Z* E_x(t)
     * (and the same for the y-component).
     *
     * @param ph         Input phonon state
     * @param t          Current time
     * @param dHsp_dQx   ∂H_sp-ph/∂ε_x evaluated for the current spin config
     * @param dHsp_dQy   ∂H_sp-ph/∂ε_y evaluated for the current spin config
     * @param dph_dt     Output phonon derivatives
     */
    void phonon_derivatives(const PhononState& ph, double t,
                           double dHsp_dQx, double dHsp_dQy,
                           PhononState& dph_dt) const;
    
    /**
     * Full ODE system for coupled spin-phonon dynamics
     * State: [S0_x, S0_y, S0_z, ..., SN_z, Qx, Qy, Q_R, Vx, Vy, V_R]
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);
    
    /**
     * Spin derivative (LLG equation, Landau-Lifshitz form).
     *   dS/dt = S × H_eff − α/|S| · S × (S × H_eff)
     * The damping term has a MINUS sign so that energy decreases under
     * α > 0 (the term −α S × (S × H) drives S towards H).
     * NOTE: prior to this fix the sign was +, which made damping pump energy
     * INTO the spin sector instead of dissipating it — a latent bug that
     * was masked by all production runs using α = 0. Sign now matches
     * StrainPhononLattice::spin_derivative.
     */
    Eigen::Vector3d spin_derivative(const Eigen::Vector3d& S, 
                                    const Eigen::Vector3d& H_eff) const {
        Eigen::Vector3d dSdt = S.cross(H_eff);
        if (alpha_gilbert > 0) {
            dSdt -= alpha_gilbert * S.cross(S.cross(H_eff)) / spin_length;
        }
        return dSdt;
    }
    
    // ============================================================
    // STATE CONVERSION (consistent with Lattice: spins_to_state / state_to_spins)
    // ============================================================
    
    /**
     * Pack current state to flat ODE state vector
     */
    ODEState spins_to_state() const {
        ODEState state(state_size);
        size_t idx = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                state[idx++] = spins[i](d);
            }
        }
        pack_lattice(&state[idx]);
        return state;
    }
    
    /**
     * Unpack flat ODE state to internal variables
     */
    void state_to_spins(const ODEState& state) {
        size_t idx = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                spins[i](d) = state[idx++];
            }
            // Renormalize spins
            spins[i] = spins[i].normalized() * spin_length;
        }
        unpack_lattice(&state[idx]);
    }
    
    // Legacy aliases for backward compatibility
    ODEState to_state() const { return spins_to_state(); }
    void from_state(const ODEState& state) { state_to_spins(state); }
    
    // ============================================================
    // OBSERVABLES
    // ============================================================
    
    /**
     * Total magnetization per spin (consistent with Lattice: magnetization_local)
     */
    Eigen::Vector3d magnetization_local() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (const auto& s : spins) {
            M += s;
        }
        return M / lattice_size;
    }
    
    /**
     * Staggered magnetization (consistent with Lattice: magnetization_local_antiferro)
     */
    Eigen::Vector3d magnetization_local_antiferro() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % N_atoms == 0) ? 1.0 : -1.0;
            M += sign * spins[i];
        }
        return M / lattice_size;
    }
    
    // Legacy aliases
    Eigen::Vector3d magnetization() const { return magnetization_local(); }
    Eigen::Vector3d staggered_magnetization() const { return magnetization_local_antiferro(); }
    
    /**
     * Global magnetization (transformed from local Kitaev frame to global cubic frame)
     * M_global = Σ R * S_local / N
     * where R is the sublattice frame transformation matrix
     */
    Eigen::Vector3d magnetization_global() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            // Transform spin from local to global frame
            M += sublattice_frames[atom] * spins[i];
        }
        return M / lattice_size;
    }
    
    /**
     * Set the ordering pattern from current spin configuration
     * This should be called after simulated annealing/equilibration to capture
     * the ground state ordering for computing custom order parameters
     */
    void set_ordering_pattern() {
        ordering_pattern = spins;
        has_ordering_pattern = true;
    }
    
    /**
     * Set ordering pattern from provided spin configuration
     */
    void set_ordering_pattern(const SpinConfig& pattern) {
        if (pattern.size() != lattice_size) {
            throw std::invalid_argument("Ordering pattern size mismatch");
        }
        ordering_pattern = pattern;
        has_ordering_pattern = true;
    }
    
    /**
     * Compute custom order parameter based on the ordering pattern
     * Projects current spin configuration onto the initial ordering pattern
     * O = Σ S_i · S_i^(0) / N
     * where S_i^(0) is the initial ordering pattern
     */
    double custom_order_parameter() const {
        if (!has_ordering_pattern) {
            return 0.0;
        }
        double O = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            O += spins[i].dot(ordering_pattern[i]);
        }
        return O / lattice_size;
    }
    
    /**
     * Compute custom magnetization projected onto ordering pattern (per sublattice)
     * Returns vector of order parameters: [O_total, O_A, O_B]
     * where O_A = Σ_{i∈A} S_i · S_i^(0) / N_A and similarly for O_B
     */
    Eigen::Vector3d custom_order_parameter_sublattice() const {
        if (!has_ordering_pattern) {
            return Eigen::Vector3d::Zero();
        }
        double O_total = 0.0;
        double O_A = 0.0;
        double O_B = 0.0;
        size_t N_A = 0, N_B = 0;
        
        for (size_t i = 0; i < lattice_size; ++i) {
            double proj = spins[i].dot(ordering_pattern[i]);
            O_total += proj;
            if (i % N_atoms == 0) {
                O_A += proj;
                N_A++;
            } else {
                O_B += proj;
                N_B++;
            }
        }
        
        Eigen::Vector3d result;
        result << O_total / lattice_size,
                  (N_A > 0) ? O_A / N_A : 0.0,
                  (N_B > 0) ? O_B / N_B : 0.0;
        return result;
    }
    
    /// |ε| for the E1 zone-center coordinate.
    double E1_amplitude() const {
        return phonons.E1_amplitude();
    }

    // ============================================================
    // SIMULATION
    // ============================================================
    
private:
    /**
     * Generic ODE integrator with support for multiple methods
     * 
     * Available methods:
     * 
     * EXPLICIT METHODS (recommended for non-stiff problems):
     * - "euler": Explicit Euler (1st order)
     * - "rk2" or "midpoint": Runge-Kutta 2nd order
     * - "rk4": Classic Runge-Kutta 4th order (fixed step)
     * - "rk5" or "rkck54": Cash-Karp 5(4) adaptive
     * - "rk54" or "rkf54": Runge-Kutta-Fehlberg 5(4) adaptive
     * - "dopri5": Dormand-Prince 5(4) adaptive (default, recommended)
     * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy)
     * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy)
     * - "adams_bashforth" or "ab": Adams-Bashforth 5-step multistep
     * - "adams_moulton" or "am": Adams-Bashforth-Moulton predictor-corrector
     * 
     * IMPLICIT METHODS (recommended for stiff problems):
     * - "rosenbrock4" or "rb4": Rosenbrock 4th order (stiff systems, uses numerical Jacobian)
     * - "implicit_euler" or "ie": Implicit Euler (1st order, very stable for stiff systems)
     * 
     * Note: Implicit methods use numerical Jacobian approximation via finite differences.
     * They are more stable for stiff problems but computationally more expensive.
     */
    template<typename System, typename Observer>
    void integrate_ode_system(System system_func, ODEState& state,
                             double T_start, double T_end, double dt_step,
                             Observer observer, const string& method,
                             bool use_adaptive = false,
                             double abs_tol = 1e-6, double rel_tol = 1e-6);

public:
    /**
     * Run molecular dynamics simulation
     * 
     * @param T_start        Start time
     * @param T_end          End time
     * @param dt_initial     Initial/fixed time step
     * @param out_dir        Output directory for trajectories
     * @param save_interval  Steps between saves
     * @param method         Integration method: euler, rk2, rk4, rk5, dopri5 (default),
     *                       rk78, bulirsch_stoer, adams_bashforth, adams_moulton
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5",
                           // Ingredient XVIII: MD tolerance overrides. Negative
                           // values fall back to the legacy method-aware defaults.
                           double abs_tol = -1.0, double rel_tol = -1.0);

    /**
     * Stochastic spin-Langevin dynamics (qualitative thermal-decay mode).
     *
     * Iterates fixed-step RK4 (with the Boost.Odeint adaptive error control
     * disabled) while injecting a per-step Gaussian noise field on every
     * spin so the effective field becomes
     *     H_eff_total = H_eff + ξ_i,
     *     ξ_i ~ N(0, σ_spin² I_3),  σ_spin = sqrt(2 α k_B T / (|S| dt)).
     *
     * The noise is held constant across the four RK4 sub-stages of a single
     * macro step (Euler-Maruyama-on-a-step approximation). This is exact
     * only in the dt → 0 limit but produces qualitatively correct thermal
     * decay statistics for moderate dt and is sufficient for observing
     * spin-state hopping (3Q ↔ ZZ) at finite T. Phonons evolve
     * deterministically; their thermal noise is neglected (justified by the
     * symmetry-protected ε_BO = 0).
     *
     * @param t_start          start time
     * @param t_end            end time
     * @param dt               fixed time step
     * @param output_dir       directory for trajectory output (HDF5 + text)
     * @param save_every       save observables every N steps
     * @param seed             RNG seed; if 0, uses random_device
     */
    void integrate_langevin(double t_start, double t_end, double dt,
                            const string& output_dir = "",
                            size_t save_every = 100,
                            uint64_t seed = 0);
    
    // ============================================================
    // MONTE CARLO METHODS (consistent with Lattice / StrainPhononLattice)
    // ============================================================
    
    /**
     * Single Metropolis sweep over all spins
     * @param T             Temperature
     * @param gaussian_move If true, use Gaussian perturbation; if false, propose random spin
     * @param sigma         Width of Gaussian perturbation (only used if gaussian_move=true)
     * @return Acceptance rate (0.0 to 1.0)
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0);
    
    // Legacy alias
    size_t metropolis_sweep(double T) {
        double rate = metropolis(T);
        return static_cast<size_t>(rate * lattice_size);
    }
    
    /**
     * Single overrelaxation sweep over all spins (consistent with Lattice: overrelaxation)
     * Reflects each spin about its local field (energy-conserving)
     */
    void overrelaxation();
    
    /**
     * Greedy quench: T=0 deterministic alignment with convergence check
     * Delegates to mc::greedy_quench template.
     */
    void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000) {
        mc::greedy_quench(*this, rel_tol, max_sweeps);
    }
    
    /**
     * Simulated annealing for spin subsystem
     * 
     * @param T_start              Starting temperature
     * @param T_end                Final temperature
     * @param n_steps              Number of MC sweeps per temperature
     * @param overrelax_rate       Overrelaxation frequency (0 = disabled)
     * @param cooling_rate         Temperature cooling factor (T *= cooling_rate each step)
     * @param out_dir              Output directory for saving configs
     * @param save_observables     Whether to save observables to HDF5
     * @param T_zero               Whether to perform deterministic sweeps at T=0
     * @param n_deterministics     Number of deterministic sweeps at T=0
     * @param adiabatic_phonons    If true, relax phonons to equilibrium at each temperature step
     *                             (Born-Oppenheimer approximation for phonons)
     * @param gaussian_move        If true, use Gaussian moves instead of uniform random
     */
    void simulated_annealing(double T_start, double T_end, size_t n_steps,
                            size_t overrelax_rate = 0,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = true,
                            bool T_zero = false,
                            size_t n_deterministics = 1000,
                            bool adiabatic_phonons = false,
                            bool gaussian_move = false,
                            bool preserve_initial_phonons = false);
    
    /**
     * Deterministic T=0 sweep: align each spin with its local field
     * 
     * @param num_sweeps  Number of sweeps to perform
     */
    void deterministic_sweep(size_t num_sweeps);
    
    /**
     * Relax the zone-center E1 coordinate to its static equilibrium for the
     * current spin configuration. The equilibrium satisfies
     *
     *   ω_E1² ε_a + λ_E1_quartic (ε_x²+ε_y²) ε_a + ∂H_sp-ph/∂ε_a = 0,
     *   a = x, y.
     *
     * @param tol      convergence tolerance for the residual
     * @param max_iter maximum Newton iterations
     * @param damping  damped Newton step size (1.0 = full Newton)
     * @return true on convergence, false if @c max_iter is exhausted.
     */
    bool relax_phonons(double tol = 1e-8, size_t max_iter = 10000, double damping = 1.0);
    
    /**
     * Joint spin-phonon relaxation to find true steady state.
     * 
     * Iterates between:
     * 1. Relaxing phonons to equilibrium for current spin configuration
     * 2. Relaxing spins (deterministic sweeps) for current phonon configuration
     * 
     * This finds the self-consistent equilibrium where both spins and phonons
     * are stationary. Required for proper energy conservation in dynamics.
     * 
     * @param tol                   Convergence tolerance for energy and Q changes
     * @param max_iter              Maximum joint relaxation iterations
     * @param spin_sweeps_per_iter  Number of deterministic spin sweeps per iteration
     * @param phonon_only           If true, only relax phonons (keep spins fixed)
     * @return true if converged, false if max_iter reached
     */
    bool relax_joint(double tol = 1e-6, size_t max_iter = 100, size_t spin_sweeps_per_iter = 10, bool phonon_only = false);
    
    // ============================================================
    // SINGLE/DOUBLE PULSE DRIVE (for 2DCS)
    // ============================================================
    
    /**
     * Magnetization trajectory data type
     * Returns: (time, [M_antiferro, M_local, M_global, (O_custom, 0, 0)])
     * The 4th element stores the custom order parameter in the x-component
     */
    using MagTrajectory = vector<std::pair<double, std::array<Eigen::Vector3d, 4>>>;
    
    /**
     * Single pulse THz drive on phonon E1 mode
     * Matches lattice.h::single_pulse_drive signature (adapted for phonon drive)
     * 
     * @param polarization  THz field polarization angle (0=x, π/2=y)
     * @param t_B           Center time of pulse
     * @param pulse_amp     Pulse amplitude (E-field strength)
     * @param pulse_width   Gaussian width (sigma)
     * @param pulse_freq    Carrier frequency
     * @param T_start       Integration start time
     * @param T_end         Integration end time
     * @param step_size     Integration timestep
     * @param method        ODE integration method
     * @return Trajectory of (time, [M_antiferro, M_local, M_global])
     */
    MagTrajectory single_pulse_drive(double polarization, double t_B,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double T_start, double T_end, double step_size,
                                     const string& method = "dopri5",
                                     // W3: pulse-window-aware chunked integration.
                                     bool pulse_window_chunking = true,
                                     // Ingredient XVIII: pump-probe ODE tolerances.
                                     double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                     double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);
    
    /**
     * Double pulse THz drive (pump + probe)
     * Matches lattice.h::double_pulse_drive signature (adapted for phonon drive)
     * Both pulses share the same amplitude, width, and frequency
     * 
     * @param polarization_1  Pump pulse polarization angle
     * @param t_B_1           Pump pulse center time
     * @param polarization_2  Probe pulse polarization angle
     * @param t_B_2           Probe pulse center time
     * @param pulse_amp       Pulse amplitude (shared)
     * @param pulse_width     Gaussian width (shared)
     * @param pulse_freq      Carrier frequency (shared)
     * @param T_start         Integration start time
     * @param T_end           Integration end time
     * @param step_size       Integration timestep
     * @param method          ODE integration method
     * @return Trajectory of (time, [M_antiferro, M_local, M_global])
     */
    MagTrajectory double_pulse_drive(double polarization_1, double t_B_1,
                                     double polarization_2, double t_B_2,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double T_start, double T_end, double step_size,
                                     const string& method = "dopri5",
                                     bool pulse_window_chunking = true,
                                     double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                     double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);
    
    /**
     * Complete 2D coherent spectroscopy (2DCS) workflow
     * Matches lattice.h::pump_probe_spectroscopy signature (adapted for phonon drive)
     * 
     * Performs pump-probe spectroscopy with THz pulses driving the E1 phonon mode:
     * 1. Uses current spin configuration as ground state
     * 2. Runs reference single-pulse dynamics M0 (pump at t=0)
     * 3. Scans delay times (tau) to measure:
     *    - M1(t, tau): Response to probe pulse at time tau only
     *    - M01(t, tau): Response to pump (t=0) + probe (t=tau)
     * 
     * Nonlinear signal extraction: M_NL = M01 - M0 - M1
     * 
     * @param polarization  THz field polarization angle
     * @param pulse_amp     THz pulse amplitude
     * @param pulse_width   Gaussian pulse width
     * @param pulse_freq    Pulse carrier frequency
     * @param tau_start     Initial delay time
     * @param tau_end       Final delay time  
     * @param tau_step      Delay time step
     * @param T_start       Integration start time
     * @param T_end         Integration end time
     * @param T_step        Integration timestep
     * @param dir_name      Output directory
     * @param method        ODE integration method
     */
    void pump_probe_spectroscopy(double polarization,
                                double pulse_amp, double pulse_width, double pulse_freq,
                                double tau_start, double tau_end, double tau_step,
                                double T_start, double T_end, double T_step,
                                const string& dir_name = "spectroscopy",
                                const string& method = "dopri5",
                                // W1/W2/W3 controls. Same semantics as
                                // Lattice::pump_probe_spectroscopy.
                                bool reuse_m0_for_m1 = true,
                                double stationarity_tol = 1e-6,
                                int outer_omp_threads = 0,
                                bool pulse_window_chunking = true,
                                double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * MPI-parallelized 2DCS spectroscopy
     * Distributes tau values across MPI ranks
     */
    void pump_probe_spectroscopy_mpi(double polarization,
                                    double pulse_amp, double pulse_width, double pulse_freq,
                                    double tau_start, double tau_end, double tau_step,
                                    double T_start, double T_end, double T_step,
                                    const string& dir_name = "spectroscopy",
                                    const string& method = "dopri5",
                                    bool reuse_m0_for_m1 = true,
                                    double stationarity_tol = 1e-6,
                                    bool pulse_window_chunking = true,
                                    double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                    double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    // ------------------------------------------------------------------
    // W1 (time-translation) helpers — see lattice.h doc.
    // ------------------------------------------------------------------
    using PumpProbeTrajectory = MagTrajectory;

    /**
     * Maximum |dS/dt|_∞ across the spin sector with the THz drive
     * disabled. Returns a runtime stationarity bound for W1.
     *
     * Note: we deliberately ignore the phonon-sector RHS here, because
     * a non-zero phonon velocity at t = T_start does NOT block the W1
     * synthesis as long as the *spin* sector returns to the same
     * trajectory shape after the pulse. Phonons couple to spins only
     * through `spin_phonon_params`, so a quiet spin sector with a noisy
     * phonon initial condition would still violate the synthesis. To
     * keep it simple and conservative, we report only the spin part —
     * users with non-stationary phonons should pass
     * `reuse_m0_for_m1 = false` explicitly.
     */
    double max_dSdt_norm_no_drive() const;

    /**
     * Time-shift M_pulse (t_B = 0 trajectory) by τ to obtain M_1(τ).
     * Pre-pulse samples are filled with M_ground.
     */
    PumpProbeTrajectory synthesize_M1_from_M0(
        const PumpProbeTrajectory& M_pulse_trajectory,
        const std::array<Eigen::Vector3d, 4>& M_ground,
        double tau, double T_step) const;
    
    // ============================================================
    // I/O
    // ============================================================
    
#ifdef HDF5_ENABLED
    void save_spin_config_hdf5(const string& filename) const;
    void load_spin_config_hdf5(const string& filename);
    void save_state_hdf5(const string& filename) const;
    void load_state_hdf5(const string& filename);
#endif
    
    void save_spin_config(const string& filename) const;
    void load_spin_config(const string& filename);
    void read_spins_from_file(const string& filename) { load_spin_config(filename); }
    void save_positions(const string& filename) const;
    
    void print_state() const {
        cout << "=== PhononLattice State ===" << endl;
        cout << "E1: Qx=" << phonons.Q_x_E1 << ", Qy=" << phonons.Q_y_E1
             << ", Vx=" << phonons.V_x_E1 << ", Vy=" << phonons.V_y_E1
             << ", |ε|=" << E1_amplitude() << endl;
        cout << "Magnetization: " << magnetization_local().transpose() << endl;
        cout << "Staggered M: " << magnetization_local_antiferro().transpose() << endl;
        cout << "Energy: " << energy_density() << " per site" << endl;
        cout << "===========================" << endl;
    }
    
    // ============================================================
    // OBSERVABLES — SUBLATTICE & STRUCTURE FACTOR
    // ============================================================
    
    /**
     * Compute magnetization for each sublattice separately (consistent with Lattice)
     * @return Vector of SpinVectors, one per sublattice (N_atoms sublattices)
     */
    vector<SpinVector> magnetization_sublattice() const {
        vector<SpinVector> M(N_atoms, SpinVector::Zero(spin_dim));
        vector<size_t> counts(N_atoms, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            M[atom] += sublattice_frames[atom] * spins[i];
            counts[atom]++;
        }
        for (size_t a = 0; a < N_atoms; ++a) {
            if (counts[a] > 0) M[a] /= counts[a];
        }
        return M;
    }
    
    /**
     * Compute static spin structure factor S(q) = |Σ_i S_i exp(-i q·r_i)|² / N
     * (consistent with Lattice)
     */
    double structure_factor(const Eigen::Vector3d& q) const {
        std::complex<double> Sq_x(0, 0), Sq_y(0, 0), Sq_z(0, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            double phase = q.dot(site_positions[i]);
            std::complex<double> exp_factor(std::cos(phase), -std::sin(phase));
            Sq_x += spins[i](0) * exp_factor;
            Sq_y += spins[i](1) * exp_factor;
            Sq_z += spins[i](2) * exp_factor;
        }
        return (std::norm(Sq_x) + std::norm(Sq_y) + std::norm(Sq_z)) / lattice_size;
    }
    
    /**
     * Measure observables: returns (total_energy, sublattice_magnetizations)
     * (consistent with Lattice)
     */
    std::pair<double, vector<SpinVector>> measure_observables() const {
        return {total_energy(), magnetization_sublattice()};
    }
    
    // ============================================================
    // PARALLEL TEMPERING (consistent with Lattice / StrainPhononLattice)
    // ============================================================
    
    // ============================================================
    // PARALLEL TEMPERING & DIAGNOSTICS — delegated to mc::
    // ============================================================
    
    /** Parallel tempering with MPI (delegates to mc::parallel_tempering). */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false, const vector<size_t>& sweeps_per_temp = {}) {
        // Seed lattice RNG per rank
        int rank; MPI_Comm_rank(comm, &rank);
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rng.seed(static_cast<unsigned int>(seed + rank * 1000));
        mc::parallel_tempering(*this, temp, n_anneal, n_measure,
            overrelaxation_rate, swap_rate, probe_rate, dir_name, rank_to_write,
            gaussian_move, comm, verbose, sweeps_per_temp);
    }
    
    /** Generate optimized temperature grid (delegates to mc::). */
    mc::OptimizedTempGridResult generate_optimized_temperature_grid_mpi(
        double Tmin, double Tmax,
        size_t warmup_sweeps = 500, size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20, bool gaussian_move = false,
        size_t overrelaxation_rate = 0, double target_acceptance = 0.45,
        double convergence_tol = 0.05, MPI_Comm comm = MPI_COMM_WORLD,
        bool use_gradient = true) {
        int rank; MPI_Comm_rank(comm, &rank);
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rng.seed(static_cast<unsigned int>(seed + rank * 12345));
        return mc::generate_optimized_temperature_grid_mpi(*this, Tmin, Tmax,
            warmup_sweeps, sweeps_per_iter, feedback_iters, gaussian_move,
            overrelaxation_rate, target_acceptance, convergence_tol, comm, use_gradient);
    }
    
    /** Geometric temperature ladder (delegates to mc::). */
    static vector<double> generate_geometric_temperature_ladder(
        double Tmin, double Tmax, size_t R) {
        return mc::generate_geometric_temperature_ladder(Tmin, Tmax, R);
    }
    
    /** Binning analysis (delegates to mc::). */
    static mc::BinningResult binning_analysis(const vector<double>& data) {
        return mc::binning_analysis(data);
    }
    
    /** Autocorrelation estimation (delegates to mc::). */
    static void estimate_autocorrelation_time(const vector<double>& energies,
            size_t base_interval, double& tau_int_out, size_t& sampling_interval_out) {
        mc::estimate_autocorrelation_time(energies, base_interval,
                                          tau_int_out, sampling_interval_out);
    }
    
    /** Thermodynamic observables (delegates to mc::). */
    mc::ThermodynamicObservables compute_thermodynamic_observables(
        const vector<double>& energies,
        const vector<vector<SpinVector>>& sublattice_mags,
        double temperature) const {
        return mc::compute_thermodynamic_observables<SpinVector>(
            energies, sublattice_mags, temperature, lattice_size);
    }
    
private:
    // RNG members for reproducible per-rank seeding (needed for parallel tempering)
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
    std::normal_distribution<double> normal_dist{0.0, 1.0};
};

#endif // PHONON_LATTICE_H
