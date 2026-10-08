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
 * The exchange matrices J^{(γ)} and the magnetoelastic tensor tables are defined
 * in the cubic Kitaev frame and rotated once into the spin STORAGE frame, which by
 * default is the crystal frame (a, b, c*) — the frame of the site positions, with
 * c* along z (see kitaev_bonds.h; `legacy_kitaev_frame = 1` restores the pre-2026-10
 * R·cubic frame exactly). Fields (field_direction), stored spins and M_local are in
 * the storage frame; the "global" outputs (M_global, M_antiferro) are always crystal
 * components.
 *
 * Equations of motion (Euler–Lagrange):
 *   - Spins:  dS/dt = S × H_eff − (α/|S|) S × (S × H_eff)  (Landau–Lifshitz–Gilbert)
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
#include <cstdlib>
#include <mpi.h>

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
 * The exchange matrices are defined in the LOCAL (cubic) Kitaev frame and rotated
 * into the spin storage frame `frame` (default: crystal a, b, c*):
 * J_storage = U J_local Uᵀ with U = kitaev::storage_from_cubic(frame) — Rᵀ for the
 * crystal frame, R for the legacy frame (see kitaev_bonds.h).
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
    // OFF by default (as are all phonon couplings: the struct, the config defaults of
    // build_phonon_params and the regression tests agree).  Operating-point estimate with a
    // stipulated Grüneisen-type scale: δX/X common to all channels, λ_{X,1} = (X/K) λ_{K,1},
    // λ_{K,1} = 40 meV per unit Q (a 5 % modulation of every exchange at |Q| ≈ 0.01, γ_G ≈ 10
    // for a Co1–Co2 relative displacement of ~1 pm at 300 kV/cm), i.e.
    //   lambda_E1_J_1 = -3.447, lambda_E1_K_1 = 40, lambda_E1_Gamma_1 = -15.56, lambda_E1_Gammap_1 = 14.91.
    // Replace by DFT/Raman-anomaly values.
    double lambda_E1_J_1      = 0.0;
    double lambda_E1_K_1      = 0.0;
    double lambda_E1_Gamma_1  = 0.0;
    double lambda_E1_Gammap_1 = 0.0;

    /// Spin storage frame (crystal by default; Legacy reproduces the pre-2026-10 R·cubic frame).
    classical_spin::kitaev::Frame frame = classical_spin::kitaev::Frame::Crystal;

    /// R: crystal axes (a, b, c*) in cubic Kitaev coordinates (S_cubic = R S_crystal).
    static SpinMatrix get_kitaev_rotation() {
        return classical_spin::kitaev::kitaev_rotation();
    }

    /// U with S_storage = U S_local for this parameter set's frame.
    Eigen::Matrix3d storage_from_local() const {
        return classical_spin::kitaev::storage_from_cubic(frame);
    }

    /// Local (cubic Kitaev) exchange matrix → crystal frame, Rᵀ J R.
    static SpinMatrix to_global_frame(const SpinMatrix& J_local) {
        return classical_spin::kitaev::to_global_frame(J_local);
    }

    /// Local (cubic Kitaev) exchange matrix → this parameter set's storage frame.
    Eigen::Matrix3d to_storage_frame(const Eigen::Matrix3d& J_local) const {
        return classical_spin::kitaev::to_storage_frame(J_local, frame);
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

    // Bond-dependent exchange matrices in the spin storage frame.
    SpinMatrix get_Jx() const { return to_storage_frame(get_Jx_local()); }
    SpinMatrix get_Jy() const { return to_storage_frame(get_Jy_local()); }
    SpinMatrix get_Jz() const { return to_storage_frame(get_Jz_local()); }
    /// Storage-frame exchange of NN bond type γ ∈ {0, 1, 2}.
    Eigen::Matrix3d nn_exchange(int bond_type) const {
        return to_storage_frame(classical_spin::kitaev::make_J_local(bond_type, J, K, Gamma, Gammap));
    }

    // J2 / J3 are isotropic Heisenberg → invariant under rotation.
    SpinMatrix get_J3_matrix()   const { return classical_spin::kitaev::heisenberg_matrix(J3);   }
    SpinMatrix get_J2_A_matrix() const { return classical_spin::kitaev::heisenberg_matrix(J2_A); }
    SpinMatrix get_J2_B_matrix() const { return classical_spin::kitaev::heisenberg_matrix(J2_B); }
};

/**
 * Time-dependent multiplicative scale s(t) on the magnetoelastic coupling.
 *
 * s(t) multiplies the WHOLE magnetoelastic Hamiltonian H_ME(S, Q) of every lattice
 * mode — all bond increments δM_γ(Q) (linear and quadratic, E/A1/A2), the ring
 * modulation J7_eff(Q) − J7 and the J2/J3 modulations — consistently in the energy,
 * the spin field and the lattice force (so the coupled dynamics conserves
 * E(t) + ∫ ∂_t H dt):
 *   - mode == "constant" (default):  s(t) = 1
 *   - mode == "window"            :  s(t) = e1_coupling_scale_target for
 *                                    t_start_E1 ≤ t ≤ t_end_E1, else 1.
 * Before 2026-10 the scale acted on the bilinear increments only and was ignored by
 * every energy function.
 */
struct TimeDependentSpinPhononParams {
    std::string mode = "constant";

    double t_start_E1 = 0.0;
    double t_end_E1   = 1e30;
    double e1_coupling_scale_target = 1.0;

    /// Multiplicative magnetoelastic scale at time t.
    double get_e1_coupling_scale(double t) const {
        if (mode == "window" && t >= t_start_E1 && t <= t_end_E1) {
            return e1_coupling_scale_target;
        }
        return 1.0;
    }

    /// True when s(t) ≡ 1 (time-translation invariant Hamiltonian).
    bool is_constant() const {
        return mode == "constant" || e1_coupling_scale_target == 1.0;
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
 *   - N_spin = 2 · dim1 · dim2 · dim3 classical spins (spin_dim = 3)
 *   - the zone-centre lattice sector: the primary E1 doublet (Q_x, Q_y, V_x, V_y),
 *     optional extra modes (set_modes) and, with spin–lattice dynamics, the
 *     in-plane site displacements and momenta.
 *
 * ODE state: [S_0 .. S_{N-1} (3 each), lattice sector (pack_lattice layout)].
 *
 * Kernel structure. Every term of H is linear in each individual spin (bond
 * bilinears, the six-spin ring term — each product contains every hexagon spin
 * once —, Zeeman, the magnetoelastic increments and the exchange striction; there
 * is no single-ion term and self-bonds are rejected at construction), so
 *     E(S_i') − E(S_i) = −(S_i' − S_i) · H_i,   H_i = −∂E/∂S_i
 * exactly. One field kernel (exchange + magnetoelastic + further-neighbour + striction)
 * and one hexagon kernel (ring energy, all six gradients and R_hex in one pass) serve
 * the energy, the Monte Carlo increments, the heat bath, overrelaxation and the
 * equations of motion, so these can no longer disagree. The coordinate-dependent
 * couplings δM_γ(Q), the J2/J3 modulations and J7_eff(Q) are computed once per lattice
 * configuration (Couplings) instead of once per site.
 */
class PhononLattice {
public:
    using SpinConfig = vector<SpinVector>;
    using ODEState = vector<double>;
    using Frame = classical_spin::kitaev::Frame;

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
    // After changing a mode's couplings directly, call update_modulation_flags()
    // (it also refreshes the cached couplings).
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

    // NN interactions (stored per site to avoid double counting). nn_interaction is
    // the effective storage-frame exchange = (NN disorder scale) × clean + (channel
    // disorder increment); it is rebuilt from the coupling parameters by
    // set_parameters() and keeps the quenched disorder across rebuilds.
    vector<vector<Eigen::Matrix3d>> nn_interaction;  // J1 matrices
    vector<vector<size_t>> nn_partners;              // NN partner indices
    vector<vector<int>> nn_bond_types;               // Bond type (0,1,2 for x,y,z bonds)

    // 2nd / 3rd NN (isotropic Heisenberg) couplings. The geometric bond lists are
    // always present — independent of the build-time J2/J3 values — so the SLD
    // second-neighbour springs and the J2/J3 phonon modulations never silently vanish.
    vector<vector<double>> j2_coupling;              // J2_A (sublattice 0) or J2_B (sublattice 1)
    vector<vector<size_t>> j2_partners;              // 2nd NN partner indices
    vector<vector<double>> j3_coupling;              // J3
    vector<vector<size_t>> j3_partners;              // 3rd NN partner indices

    // Hexagonal plaquettes for ring exchange (built once, in the constructor).
    // Each hexagon stores 6 site indices in order going around the ring.
    vector<std::array<size_t, 6>> hexagons;
    // Optional static per-plaquette J7 offsets, used to model Na/stacking-induced
    // ring-exchange landscapes without adding direct spin pinning.
    vector<double> plaquette_j7_offsets;
    // For each site, list of hexagons it belongs to and its position (0-5) within each hexagon
    vector<vector<std::pair<size_t, size_t>>> site_hexagons;

    // External field (storage frame = crystal frame by default)
    vector<SpinVector> field;

    // Parameters
    PhononParams phonon_params;
    SpinPhononCouplingParams spin_phonon_params;
    TimeDependentSpinPhononParams time_dep_spin_phonon_params;
    DriveParams drive_params;

    // LLG damping (Landau–Lifshitz form, see spin_derivative)
    double alpha_gilbert = 0.0;

    // Langevin thermostat. integrate_langevin() samples the Gibbs state at
    // T = langevin_temperature: the spin noise ξ enters the precession AND the
    // damping term (Stratonovich) with <ξ_a(t) ξ_b(t')> = 2D δ_ab δ(t−t'),
    //     D = α T / (|S| (1 + α²))
    // (García-Palacios & Lázaro, PRB 58, 14937 (1998); the variance 2αT/|S| used
    // before 2026-10 heated the spins to T(1+α²)), and every damped lattice
    // coordinate (E1 and extra modes, SLD momenta) receives the matching
    // fluctuation–dissipation noise, so the coupled system is thermostatted.
    double langevin_temperature = 0.0;
    /// Temperature of the noise on the zone-centre modes (< 0: follow the bath).
    double phonon_langevin_T = -1.0;

    // Two-reservoir ("scenario 1") bath profile for integrate_langevin(): the spin bath
    // temperature is  T_b(t) = T0 + dT * Θ(t - t_step) (1 - e^{-(t-t_step)/tau_on}) e^{-(t-t_step)/tau_off}
    // i.e. a hot phonon reservoir filled within tau_on (the E1 ring-down) and, optionally,
    // cooling with tau_off (0 = no decay).  dT = 0 reproduces the constant-T thermostat.
    // Semi-quantum thermostat (Barker & Bauer, PRB 100, 140401 (2019)): the stochastic field is
    // coloured Gaussian noise whose power spectrum is the Bose energy of a mode at frequency ω,
    //     P(ω) = P_classical(T) · F(ω,T),   F = x/(e^x − 1),  x = ħω/k_B T,
    // (no zero-point term), while the Gilbert damping stays Markovian.  For every harmonic mode the
    // steady state is then E_k = ħω_k n_B(ω_k): quantum thermal occupation without a quantum solver.
    // P_classical is the white level of the classical thermostat above (same D), so F → 1 recovers
    // it exactly. The noise is generated in FFT blocks of langevin_block steps, overlap-added with
    // sine windows (Σ w² = 1) so that the process is stationary; the block temperature is the bath
    // temperature at the block centre.  langevin_quantum = false reproduces the classical white-noise
    // thermostat.
    bool langevin_quantum = false;
    int langevin_block = 4096;
    // Finite-capacity, energy-conserving bath (two-temperature model built from the dynamics):
    // when langevin_bath_C > 0 (heat capacity of the bath in k_B per spin) the bath temperature is a
    // dynamical variable, T_b(t+dt) = T_b(t) - [E_sys(t+dt) - E_sys(t) - W_drive]/(N C_l), i.e. every
    // unit of energy the system loses (phonon damping γ, Gilbert damping) heats the bath and every
    // unit the noise injects cools it; the E1 ring-down is then the deposit itself and no
    // langevin_dT step is needed.  W_drive is the work of the THz field on every polar mode.
    // 0 = infinite bath with the prescribed profile.
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
    /// Switch spin–lattice dynamics on/off. Resets u and p, and (when on) schedules one static
    /// relaxation of the displacements (sld_relax iterations) for the next integrate_langevin().
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

    // Sublattice frames: sublattice_frames[atom] maps a stored spin to crystal (a, b, c*)
    // components, S_crystal = F · S_storage ("global" outputs). Identity in the default
    // crystal frame; set by set_parameters() from the coupling-parameter frame.
    vector<SpinMatrix> sublattice_frames;
    vector<double> afm_sublattice_signs;   // Néel signs (+1, −1) for the staggered magnetisation

    // Custom ordering vector (set from initial spin configuration)
    // Used to compute order parameter along the ground state ordering direction
    SpinConfig ordering_pattern;
    bool has_ordering_pattern = false;

    /// Local Monte Carlo kernel used by local_sweep() (and so by simulated_annealing).
    enum class LocalUpdate { Metropolis, HeatBath };
    LocalUpdate local_update = LocalUpdate::Metropolis;

    /// Lattices with at least this many sites evaluate the RHS with OpenMP.
    size_t parallel_min_sites = 1024;

    /**
     * Constructor: Build from a UnitCell (consistent with Lattice interface)
     *
     * Validates the topology (two-site honeycomb basis, NN bond types 0..2,
     * isotropic J2/J3, no bond that wraps onto its own site) and builds the
     * hexagonal plaquettes. Throws std::invalid_argument otherwise.
     *
     * @param uc        Unit cell defining lattice structure (positions, interactions, bond types)
     * @param d1        Lattice size in first dimension (≥ 2)
     * @param d2        Lattice size in second dimension (≥ 2)
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
     * Periodic boundary condition (Euclidean modulo: any offset, not only |offset| ≤ L)
     */
    int periodic_boundary(int coord, size_t dim_size) const {
        const int L = static_cast<int>(dim_size);
        return ((coord % L) + L) % L;
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
     * Set the coupling, phonon and drive parameters.
     *
     * Rebuilds the NN exchange (J, K, Γ, Γ' rotated into sp_params.frame) and the
     * isotropic J2_A/J2_B/J3 couplings from sp_params — the UnitCell supplies only the
     * topology and bond types — and mirrors the legacy E1 parameters into modes[0].
     * Quenched disorder (NN scales / channel increments, plaquette J7 offsets) and the
     * extra lattice modes set by set_modes() are preserved, so this may be called
     * repeatedly (e.g. in a J7 scan).
     */
    void set_parameters(const SpinPhononCouplingParams& sp_params,
                       const PhononParams& ph_params,
                       const DriveParams& dr_params);

    /// Spin storage frame of the exchange and magnetoelastic tensors.
    Frame frame() const { return spin_phonon_params.frame; }

    /**
     * Set time-dependent magnetoelastic scale parameters.
     */
    void set_time_dependent_spin_phonon(const TimeDependentSpinPhononParams& td_params) {
        if (td_params.mode != "constant" && td_params.mode != "window")
            throw std::invalid_argument("TimeDependentSpinPhononParams.mode must be 'constant' or 'window', got '" +
                                        td_params.mode + "'");
        time_dep_spin_phonon_params = td_params;
        invalidate_couplings();
        if (td_params.mode != "constant") {
            std::cout << "Time-dependent magnetoelastic scaling enabled (mode: "
                      << td_params.mode << ")" << std::endl;
            if (td_params.mode == "window") {
                std::cout << "  H_ME scale = " << td_params.e1_coupling_scale_target
                          << " for t∈[" << td_params.t_start_E1
                          << ", " << td_params.t_end_E1 << "]" << std::endl;
            }
        }
    }

    /// Multiplicative magnetoelastic scale s(t) (see TimeDependentSpinPhononParams).
    double get_e1_coupling_scale(double t) const {
        return time_dep_spin_phonon_params.get_e1_coupling_scale(t);
    }

    /**
     * Set external magnetic field (uniform, storage frame = crystal frame by default)
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
    void apply_plaquette_j7_disorder_from_file(const string& filename);

    /**
     * Add site-resolved pinning fields from a text file.
     *
     * Format: site Bx By Bz, with optional '#' comments.  The fields are added
     * on top of the current uniform field, so call set_field() first.
     */
    void add_pinning_fields_from_file(const string& filename);

    /**
     * Apply quenched nearest-neighbour exchange disorder from a text file.
     *
     * Format: site partner scale, with optional '#' comments.  Each row should
     * refer to one unique NN bond.  The clean 3x3 exchange matrix on that bond is
     * multiplied by scale (both directed entries, transpose convention).  This
     * perturbs the Hamiltonian without templating a spin direction. Scales are
     * stored and survive set_parameters().
     */
    void apply_nn_exchange_disorder_from_file(const string& filename);

    /**
     * Apply additive, channel-resolved quenched NN exchange disorder.
     *
     * Format: site partner dJ dK dGamma dGammap, with optional '#' comments.
     * The increments are interpreted in the local Kitaev channel basis for the
     * corresponding NN bond type, transformed to the spin storage frame, and added
     * to the stored exchange matrix (after the NN scale, independent of the order in
     * which the two disorder files are applied).  This allows physically sharper tests
     * such as K-only or Γ-only bond disorder without applying a spin-direction pinning
     * field. Increments are stored and survive set_parameters().
     */
    void apply_nn_exchange_channel_disorder_from_file(const string& filename);

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
     * Generate random spin uniformly on the 2-sphere
     */
    SpinVector gen_random_spin() {
        return gen_random_spin(spin_length);
    }

    /**
     * Generate random spin uniformly on the 2-sphere with specified magnitude
     */
    SpinVector gen_random_spin(float spin_l) {
        return SpinVector(random_unit_vector() * double(spin_l));
    }

    /**
     * Symmetric small-angle move around the current spin: S + σ û (û uniform on S²),
     * renormalised. The proposal density depends only on the angle to S, so it is
     * symmetric and needs no Hastings factor.
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma) {
        return SpinVector(gaussian_move3(Eigen::Vector3d(current_spin), sigma));
    }

    /**
     * Initialize random spins (and reset the lattice sector, see reset_lattice_sector)
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = random_unit_vector() * double(spin_length);
        }
        reset_lattice_sector();
    }

    /**
     * Initialize ferromagnetic state
     */
    void init_ferromagnetic(const Eigen::Vector3d& direction) {
        if (!(direction.norm() > 0.0))
            throw std::invalid_argument("init_ferromagnetic: direction must be non-zero");
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = dir;
        }
        reset_lattice_sector();
    }

    /**
     * Initialize Néel state (antiferromagnetic on sublattices)
     */
    void init_neel(const Eigen::Vector3d& direction) {
        if (!(direction.norm() > 0.0))
            throw std::invalid_argument("init_neel: direction must be non-zero");
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = afm_sublattice_signs[i % N_atoms] * dir;
        }
        reset_lattice_sector();
    }

    /**
     * Zero every non-frozen lattice coordinate and velocity (primary E1 doublet, extra
     * modes) and the SLD displacements/momenta, so that successive trials start from
     * the same undisplaced lattice. Frozen strains keep their prescribed values.
     */
    void reset_lattice_sector();

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

    /**
     * Reseed the private Monte Carlo / noise engine from (master, stream) through
     * splitmix64, e.g. set_seed(config.seed, rank) for independent per-rank chains.
     */
    void set_seed(uint64_t master, uint64_t stream) {
        rng.seed(static_cast<std::mt19937::result_type>(
            splitmix64(master ^ splitmix64(stream + 0x50484F4E4C415454ULL))));
    }

    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================

    /**
     * Energy of every term that involves the spin at @a site, evaluated with
     * @a spin_here in place of the stored spin: Zeeman, NN exchange with the
     * magnetoelastic increments, J2/J3 with their modulations, the exchange
     * striction and the six-spin ring term of the site's three hexagons
     * (−spin_here · H_site, exact because H is linear in each spin).
     */
    double site_energy(const Eigen::Vector3d& spin_here, size_t site) const;

    /**
     * Exact energy difference for replacing the spin at @a site:
     *   dE = E(new_spin) − E(old_spin) = −(new_spin − old_spin) · H_site,
     * with H_site = −∂E/∂S_site the full local field (every term of H is linear in
     * each individual spin). Includes every Hamiltonian term — Zeeman, exchange,
     * magnetoelastic increments, further-neighbour modulations, ring exchange and the
     * SLD exchange striction.
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
    /// Ring-exchange operator R_7 = Σ_hex R_hex, defined by H_7 = J7_eff * R_7 (no plaquette offsets).
    double ring_exchange_normalized() const;

    /**
     * Lattice-sector energy of every mode (kinetic + harmonic + quartic), extensive:
     *   E_ph = N Σ_m [ (1/2)|V_m|² + (1/2) ω_m² |Q_m|² + (λ4_m / 4) |Q_m|⁴ ].
     */
    double phonon_energy() const;

    /**
     * Magnetoelastic coupling energy Σ_γ ⟨δM_γ(Q), C_γ⟩ + further-neighbour modulations
     * (the ring modulation J7_eff(Q) − J7 is part of ring_exchange_energy()).
     */
    double spin_phonon_energy() const;

    /**
     * Total energy at coupling scale s = 1 (the Hamiltonian for a constant schedule)
     */
    double total_energy() const {
        return spin_energy() + phonon_energy() + spin_phonon_energy() + anharmonic_energy() + sld_energy();
    }

    /**
     * Total energy of the Hamiltonian at time t, i.e. with the magnetoelastic scale
     * s(t) of time_dep_spin_phonon_params (identical to total_energy() for a constant
     * schedule). The drive term −N Z* E(t)·Q is not included (it is external work).
     */
    double total_energy(double t) const;

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
    /// Effective ring exchange J7 + Σ_E λ_J7|Q|² + Σ_A1 λ_J7 Q (+ frozen strains), at scale 1.
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
     * Every magnetoelastic quantity that depends only on the lattice coordinates and the
     * coupling scale s: the storage-frame bond increments δM_γ (and transposes, for the
     * B end of a bond), the J2/J3 modulations per (sublattice, bond class) and J7_eff.
     */
    struct Couplings {
        Eigen::Matrix3d dM[3], dMT[3];
        double dJ2[2][3] = {{0, 0, 0}, {0, 0, 0}};
        double dJ3[3] = {0, 0, 0};
        double J7eff = 0.0;
        bool ring_active = false;     ///< any hexagon has a non-zero J7
    };
    void compute_couplings(const Coords& c, double scale, Couplings& out) const;
    /// Couplings of the committed lattice coordinates at s = 1 (cached; refreshed when the
    /// coordinates change or after set_parameters / set_modes / update_modulation_flags).
    const Couplings& couplings() const;
    void invalidate_couplings() const { ++coupling_epoch_; }

    /**
     * Compute ring exchange contribution to effective field on spin at given site
     *
     * H_eff_ring = -∂H_7/∂S_site
     */
    SpinVector get_ring_exchange_field(size_t site) const;
    SpinVector get_ring_exchange_field(size_t site, double J7eff) const;

    /**
     * Compute effective field on spin i (for spin EOM)
     *
     * H_eff = -∂H/∂Si = B + Σ_j [NN contributions] + [spin-phonon contributions] + [ring exchange]
     */
    SpinVector get_local_field(size_t site) const;
    /// Same as get_local_field(), fixed-size result (hot paths).
    Eigen::Vector3d local_field(size_t site) const;

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
     * Full ODE system for the coupled spin–lattice dynamics, a pure function of the
     * flat state x (the committed spins/phonons are not touched, so rejected trial
     * stages of adaptive steppers leave no trace). OpenMP-parallel over sites and
     * hexagons for lattices with ≥ parallel_min_sites sites.
     * State: [S_0 .. S_{N-1}, lattice sector (pack_lattice layout)].
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t) const;
    /// ode_system with optional noise: spin_noise (3N) is added to the effective fields,
    /// lattice_noise (phonon_dof() entries, pack_lattice layout) to the lattice derivatives.
    void rhs(const ODEState& x, ODEState& dxdt, double t,
             const double* spin_noise, const double* lattice_noise) const;

    /**
     * Spin derivative (LLG equation, Landau-Lifshitz form).
     *   dS/dt = S × H_eff − α/|S| · S × (S × H_eff)
     * The damping term has a MINUS sign so that energy decreases under α > 0.
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
        if (state.size() != state_size)
            throw std::invalid_argument("state_to_spins: state has " + std::to_string(state.size()) +
                                        " entries, expected state_size = " + std::to_string(state_size));
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

    // Replica exchange: the whole lattice sector travels with the spins, so the
    // exchanged total energies are those of the exchanged states (mc::has_extra_dof).
    size_t extra_dof_size() const { return phonon_dof(); }
    void pack_extra_dof(double* buf) const { pack_lattice(buf); }
    void unpack_extra_dof(const double* buf) { unpack_lattice(buf); }

    // ============================================================
    // OBSERVABLES
    // ============================================================

    /**
     * Total magnetization per spin in the storage frame (consistent with Lattice: magnetization_local)
     */
    Eigen::Vector3d magnetization_local() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (const auto& s : spins) {
            M += s;
        }
        return M / lattice_size;
    }

    /**
     * Staggered (Néel) magnetization in the storage frame, Σ σ_i S_i / N with
     * σ = afm_sublattice_signs (consistent with Lattice: magnetization_local_antiferro)
     */
    Eigen::Vector3d magnetization_local_antiferro() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            M += afm_sublattice_signs[i % N_atoms] * spins[i];
        }
        return M / lattice_size;
    }

    // Legacy aliases
    Eigen::Vector3d magnetization() const { return magnetization_local(); }
    Eigen::Vector3d staggered_magnetization() const { return magnetization_local_antiferro(); }

    /**
     * Magnetization in crystal (a, b, c*) components, M = Σ F_atom S_i / N with
     * F = sublattice_frames (identity in the default crystal storage frame).
     */
    Eigen::Vector3d magnetization_global() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            M += sublattice_frames[i % N_atoms] * spins[i];
        }
        return M / lattice_size;
    }

    /**
     * The four magnetisation observables of the dynamics drivers from a flat spin
     * array x (3N doubles): {M_antiferro (crystal frame, Néel signs), M_local (storage
     * frame), M_global (crystal frame), (O_custom, 0, 0)}. Shared by MD, the pulse
     * drives and pump-probe so every output uses one definition.
     */
    std::array<Eigen::Vector3d, 4> magnetization_observables(const double* x) const;

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
     * IMPLICIT METHODS (stiff problems; dense finite-difference Jacobian, so limited to
     * small systems — the call throws above 1200 state entries):
     * - "rosenbrock4" or "rb4": Rosenbrock 4th order
     * - "implicit_euler" or "ie": Implicit Euler (1st order)
     *
     * Unknown method names throw std::invalid_argument.
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
     * Observables are written on the uniform grid T_start + k · save_interval · dt_initial
     * (adaptive methods subdivide internally between grid points).
     *
     * @param T_start        Start time
     * @param T_end          End time
     * @param dt_initial     Output grid step (and initial step of adaptive methods)
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
     * Stochastic (Langevin) spin–lattice dynamics at the bath temperature.
     *
     * Each step dt draws the noise and integrates the full system with it held constant
     * over the step (Wong–Zakai / frozen-noise RK4, consistent with the Stratonovich SDE):
     *   - spins: h ~ N(0, 2D/dt) per component is added to every effective field, so it
     *     enters precession and damping, D = αT/(|S|(1+α²)) (García-Palacios & Lázaro 1998);
     *   - every damped zone-centre mode velocity gets a force η ~ N(0, 2γ_m T_ph/(N dt)) and
     *     every SLD momentum N(0, 2γ_l m T_l/dt) — the fluctuation–dissipation partners of the
     *     friction terms −γV, −γ_l p, so the Gibbs state of the COUPLED system is stationary.
     * Spin norms are restored after each step. Integrating noise and drift together removes
     * the O(α|H| dt) bias of the former "deterministic step, then noise kick" splitting
     * (verified against Monte Carlo in tests/test_phonon_dynamics.cpp).
     * With langevin_quantum the white noise is replaced by Bose-coloured noise of the same
     * classical level (see langevin_quantum).
     *
     * Invalid arguments throw std::invalid_argument; neither alpha_gilbert nor any other
     * member is modified (beyond the evolved state). The SLD static relaxation scheduled by
     * enable_sld() runs once, on the first call.
     *
     * @param t_start          start time
     * @param t_end            end time
     * @param dt               fixed time step
     * @param output_dir       directory for trajectory output (text + HDF5); empty = none
     * @param save_every       save observables every N steps
     * @param seed             RNG seed; 0 = derived from the process master seed (config `seed`)
     * @param on_save          optional observer invoked every save_every steps, after the
     *                         state has been synced back into spins[]/phonons, so that a
     *                         caller can emit arbitrary observables WITHOUT chopping the run
     *                         into separate integrate_langevin() calls. Chopping restarts the
     *                         noise stream, which truncates its correlations at the chunk
     *                         length — fatal for the Bose-coloured thermostat, whose
     *                         correlation time is hbar/k_B T (1.9 code units at 6 K).
     */
    void integrate_langevin(double t_start, double t_end, double dt,
                            const string& output_dir = "",
                            size_t save_every = 100,
                            uint64_t seed = 0,
                            const std::function<void(double)>& on_save = {});

    // ============================================================
    // MONTE CARLO METHODS (consistent with Lattice)
    // ============================================================

    /**
     * Single Metropolis sweep over all spins (random site order). The lattice
     * coordinates are held fixed (they are sampled by relax_phonons / dynamics only).
     * @param T             Temperature
     * @param gaussian_move If true, use a symmetric small-angle move of width sigma;
     *                      if false, propose a uniform random spin
     * @param sigma         Width of the small-angle move (only used if gaussian_move=true)
     * @return Acceptance rate (0.0 to 1.0)
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0);

    /**
     * Single heat-bath sweep (rejection-free): each spin is drawn from its exact
     * conditional distribution ∝ exp(β S·H) about its local field H (Miyatake et al.,
     * J. Phys. C 19, 2539 (1986)); exact because the local energy is −S·H.
     * @return 1 (every proposal is accepted)
     */
    double heat_bath(double T);

    /**
     * Monte Carlo sweep of the zone-centre lattice coordinates at fixed spins (Metropolis
     * moves of thermal width, exact Maxwell velocities). With mc_sample_lattice = true it is
     * appended to every metropolis()/heat_bath() sweep, so SA and parallel tempering sample
     * the joint spin–lattice Gibbs state instead of P(S | Q fixed). SLD displacements are not
     * sampled. @return acceptance of the coordinate moves
     */
    double lattice_mc_sweep(double T);
    /// Sample the lattice coordinates in Monte Carlo (config key mc_sample_lattice); default off.
    bool mc_sample_lattice = false;

    /// One local sweep with the kernel selected by `local_update`.
    double local_sweep(double T, bool gaussian_move = false, double sigma = 60.0) {
        return local_update == LocalUpdate::HeatBath ? heat_bath(T)
                                                     : metropolis(T, gaussian_move, sigma);
    }

    // Legacy alias
    size_t metropolis_sweep(double T) {
        double rate = metropolis(T);
        return static_cast<size_t>(rate * lattice_size);
    }

    /**
     * Single overrelaxation sweep over all spins (consistent with Lattice: overrelaxation)
     * Reflects each spin about its local field — exactly energy-conserving, since the
     * local energy is −S·H.
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
     * Temperatures follow mc::annealing_schedule(T_start, T_end, cooling_rate) (validated:
     * 0 < T_end ≤ T_start, 0 < cooling_rate < 1). Each of the n_steps sweeps per temperature
     * is one local sweep (kernel: `local_update`); with overrelax_rate = k > 0 every sweep
     * is preceded by one overrelaxation sweep and the local sweep runs every k-th sweep —
     * the same meaning as in mc::parallel_tempering. Gaussian moves adapt their width to
     * 45 % acceptance during the first half of each temperature (mc::StepSizeController).
     *
     * @param T_start              Starting temperature
     * @param T_end                Final temperature
     * @param n_steps              Number of MC sweeps per temperature
     * @param overrelax_rate       Overrelaxation frequency (0 = disabled)
     * @param cooling_rate         Temperature cooling factor (T *= cooling_rate each step)
     * @param out_dir              Output directory for saving configs
     * @param save_observables     Whether to save observables to HDF5
     * @param T_zero               Whether to perform deterministic sweeps at T=0
     * @param n_deterministics     Maximum number of deterministic sweeps at T=0
     * @param adiabatic_phonons    If true, relax phonons to equilibrium at each temperature step
     *                             (Born-Oppenheimer approximation for phonons)
     * @param gaussian_move        If true, use Gaussian moves instead of uniform random
     * @param preserve_initial_phonons  Keep the prescribed lattice coordinates instead of
     *                             resetting the lattice sector at the start
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
     * Deterministic T=0 sweeps: align each spin with its local field (sequential
     * Gauss–Seidel order; every step lowers the energy). Stops early when the largest
     * spin change of a sweep falls below 1e-12 |S|.
     *
     * @param num_sweeps  Maximum number of sweeps to perform
     * @return largest |ΔS| of the last sweep
     */
    double deterministic_sweep(size_t num_sweeps);

    /**
     * Relax the zone-centre lattice coordinates (all non-frozen modes) to their
     * static equilibrium for the current spin configuration:
     *
     *   ω² q + λ4 |q|² q + (1/N) ∂H/∂q = 0
     *
     * by Newton iteration with the exact Hessian of the lattice energy (harmonic +
     * quartic + magnetoelastic + anharmonic, by central differences of the analytic
     * forces) and a backtracking line search on the energy, so it converges near soft
     * modes where the diagonal fixed-point iteration diverged.
     *
     * @param tol      convergence tolerance for the residual (per-site force norm)
     * @param max_iter maximum Newton iterations
     * @param damping  initial Newton step fraction (1.0 = full Newton)
     * @return true on convergence, false if @c max_iter is exhausted.
     */
    bool relax_phonons(double tol = 1e-8, size_t max_iter = 10000, double damping = 1.0);

    /**
     * Joint spin-phonon relaxation to find true steady state.
     *
     * Iterates between relaxing the lattice coordinates for the current spins and
     * deterministic spin sweeps for the current coordinates, until the energy change
     * per site and the change of |Q| fall below tol.
     *
     * @param tol                   Convergence tolerance (energy per site and |Q|)
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
     * (see magnetization_observables for the frames)
     */
    using MagTrajectory = vector<std::pair<double, std::array<Eigen::Vector3d, 4>>>;

    /**
     * Single pulse THz drive on phonon E1 mode
     * Matches lattice.h::single_pulse_drive signature (adapted for phonon drive).
     * The final state is committed to spins/phonons/modes.
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
     * @return Trajectory of (time, [M_antiferro, M_local, M_global, O_custom])
     */
    MagTrajectory single_pulse_drive(double polarization, double t_B,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double T_start, double T_end, double step_size,
                                     const string& method = "dopri5",
                                     // ignored: one integration on the exact grid
                                     // T_start + k step_size (kept for API compatibility)
                                     bool pulse_window_chunking = true,
                                     // Ingredient XVIII: pump-probe ODE tolerances.
                                     double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                     double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * Double pulse THz drive (pump + probe)
     * Matches lattice.h::double_pulse_drive signature (adapted for phonon drive)
     * Both pulses share the same amplitude, width, and frequency.
     * The final state is committed to spins/phonons/modes.
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
     * @return Trajectory of (time, [M_antiferro, M_local, M_global, O_custom])
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
     * Output: dir_name/pump_probe_spectroscopy.h5 with /reference and /delay_scan/tau_i/{M1,M01},
     * each holding time, M_antiferro, M_local, M_global, O_custom (identical layout for the
     * MPI driver).
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
     * MPI-parallel 2DCS spectroscopy: the delay points are distributed over the ranks
     * of @p comm. COLLECTIVE over comm — every rank must call it with the same
     * arguments and the same ground state (spins, lattice sector); rank 0 of comm
     * writes the output file. The full trajectory records are gathered, so the file is
     * identical to the one of pump_probe_spectroscopy().
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
                                    double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol,
                                    MPI_Comm comm = MPI_COMM_WORLD);

    /// Broadcast the spin configuration and the lattice sector from root to every rank of comm.
    void broadcast_state(int root, MPI_Comm comm);

    // ------------------------------------------------------------------
    // W1 (time-translation) helpers — see lattice.h doc.
    // ------------------------------------------------------------------
    using PumpProbeTrajectory = MagTrajectory;

    /**
     * Stationarity measure for W1: the max-norm of the FULL right-hand side
     * (spins, every lattice coordinate and velocity, SLD) with the THz drive
     * disabled. M1(τ) may be synthesised from M0 by a time shift only if the
     * unperturbed state does not evolve at all; a non-relaxed lattice (e.g. a linear
     * λ1 coupling at Q = 0) evolves even when dS/dt vanishes. Returns +∞ when the
     * magnetoelastic schedule is time dependent.
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

    /// Write N lines "Sx Sy Sz" (storage frame); throws if the file cannot be written.
    void save_spin_config(const string& filename) const;
    /**
     * Read exactly lattice_size finite triplets (storage frame) and normalise them to
     * spin_length. Throws std::runtime_error on a missing file, a short or non-numeric
     * file, extra data, or a (near-)zero vector; the stored spins are untouched on error.
     */
    void load_spin_config(const string& filename);
    void read_spins_from_file(const string& filename) { load_spin_config(filename); }
    void save_positions(const string& filename) const;

    /**
     * Read exactly n finite 3-vectors from a whitespace-separated text file (lines
     * starting with '#' are comments). Throws std::runtime_error naming the file on any
     * malformed, short or overlong input.
     */
    static vector<Eigen::Vector3d> read_vec3_file(const string& filename, size_t n);

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
     * Magnetization of each sublattice in crystal components (consistent with Lattice)
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
    // PARALLEL TEMPERING & DIAGNOSTICS — delegated to mc::
    // ============================================================

    /**
     * Parallel tempering with MPI (delegates to mc::parallel_tempering). Each replica
     * exchanges its spins TOGETHER with its lattice sector (extra_dof hooks), so the
     * swap criterion uses the total energies of the states actually exchanged.
     */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false, const vector<size_t>& sweeps_per_temp = {}) {
        // Independent per-rank Monte Carlo streams derived from the process seed.
        int rank; MPI_Comm_rank(comm, &rank);
        set_seed(lehman_master_seed_value(), 0x5054ULL + static_cast<uint64_t>(rank));
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
        set_seed(lehman_master_seed_value(), 0x4F5054ULL + static_cast<uint64_t>(rank));
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
    // Private Monte Carlo / noise engine (seeded from the process seed; set_seed()).
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
    std::normal_distribution<double> normal_dist{0.0, 1.0};

    // Quenched NN disorder, kept apart from the clean exchange so that set_parameters()
    // can rebuild nn_interaction = nn_scale_ * nn_clean_ + nn_delta_.
    vector<vector<Eigen::Matrix3d>> nn_clean_, nn_delta_;
    vector<vector<double>> nn_scale_;
    bool has_j2_ = false, has_j3_ = false;   // any non-zero J2 / J3 (hot-loop skip)
    bool sld_relax_pending_ = false;         // enable_sld(true) → relax once in integrate_langevin
    void rebuild_nn_exchange();
    void update_exchange_flags();
    void report_stability() const;
    double lattice_potential(const Coords& c, const Eigen::Matrix3d C[3], const double Cj2[2][3],
                             const double Cj3[3], double R7) const;
    void build_hexagons();
    void validate_topology() const;
    size_t nn_index(size_t site, size_t partner, const string& context) const;

    // Couplings cache for the committed coordinates (couplings()).
    mutable Couplings coupling_cache_;
    mutable std::vector<double> coupling_cache_q_;
    mutable uint64_t coupling_epoch_ = 1;
    mutable uint64_t coupling_cache_epoch_ = 0;

    // Scratch for ode_system (sized on first use; ode_system is not reentrant on one object).
    mutable std::vector<double> ring_grad_scratch_, ring_val_scratch_;
    mutable std::vector<double> corr_scratch_;

    Eigen::Vector3d random_unit_vector();
    Eigen::Vector3d gaussian_move3(const Eigen::Vector3d& current, double sigma);

    // Kernels (defined in phonon_lattice.cpp). SpinOf(j) and DispOf(j) return the spin /
    // displacement of site j (Eigen expressions or Map<const Vector3d>).
    template <class SpinOf, class DispOf>
    Eigen::Vector3d field_without_ring(size_t i, const SpinOf& S, const DispOf& U,
                                       const Couplings& cc) const;
    template <class SpinOf>
    Eigen::Vector3d ring_field_site(size_t i, const SpinOf& S, double J7eff) const;
    template <class SpinOf>
    double ring_operator(const SpinOf& S, double J7eff, double* weighted_energy) const;
    template <class SpinOf>
    void correlations_of(const SpinOf& S, Eigen::Matrix3d C[3], double Cj2[2][3], double Cj3[3]) const;

    // Shared trajectory serialisation / HDF5 writer of the pump-probe drivers.
    static void pack_trajectory(const MagTrajectory& traj, vector<double>& out);
    static MagTrajectory unpack_trajectory(const double* buf, size_t n_times);
    void write_pump_probe_hdf5(const string& filename, double polarization,
                               double pulse_amp, double pulse_width, double pulse_freq,
                               double E_ground, const vector<double>& tau_values,
                               const MagTrajectory& M0,
                               const vector<MagTrajectory>& M1,
                               const vector<MagTrajectory>& M01) const;
};

#endif // PHONON_LATTICE_H
