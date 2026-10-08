#ifndef MIXED_LATTICE_REFACTORED_H
#define MIXED_LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "classical_spin/core/spin_config.h"  // For should_rank_write
#include "classical_spin/core/su3_coherent_state.h"  // SU(3) coherent-state utilities
#include "classical_spin/core/su3_mc.h"              // SU(3) Monte Carlo moves on CP^2
                                                     // (Zhang & Batista, PRB 104, 104409 (2021))
#include "classical_spin/mc/mc_common.h"      // Common MC structs & templates
#include "classical_spin/mc/parallel_tempering.h"  // replica-exchange engine + ladder tuning
#include "classical_spin/lattice/pulse_chunking.h"  // default pump-probe tolerances
#include "classical_spin/dynamics/grid_integrate.h"   // exact-grid ODE integration
#include "classical_spin/io/spin_table.h"              // spin configuration text files
#include <vector>
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
#include <cstring>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <mpi.h>
#include <boost/numeric/odeint.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HDF5_ENABLED
#include "hdf5_io.h"
#endif

// GPU support: API header for all C++ TUs, full .cuh only for CUDA TUs
#ifdef CUDA_ENABLED
#include "mixed_lattice_gpu_api.h"
#include "classical_spin/gpu/gpu_handle.h"
#endif

// Optional profiling instrumentation
#ifdef ENABLE_PROFILING
    #define PROFILE_START(name) auto __profile_start_##name = std::chrono::high_resolution_clock::now()
    #define PROFILE_END(name) do { \
        auto __profile_end_##name = std::chrono::high_resolution_clock::now(); \
        auto __profile_duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(__profile_end_##name - __profile_start_##name).count(); \
        std::cout << "[PROFILE] " << #name << ": " << __profile_duration_##name << " us" << std::endl; \
    } while(0)
#else
    #define PROFILE_START(name)
    #define PROFILE_END(name)
#endif

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::function;
using std::array;

// Common MC structs (from mc_common.h)
using mc::BinningResult;
using mc::Observable;
using mc::VectorObservable;
using mc::AutocorrelationResult;
using mc::OptimizedTempGridResult;

// Legacy type aliases for backward compatibility
using MixedBinningResult = mc::BinningResult;
using MixedObservable = mc::Observable;
using MixedVectorObservable = mc::VectorObservable;

// Complete set of thermodynamic observables for mixed lattice with uncertainties
// (Extended version with SU2/SU3 split — MixedLattice-specific)
struct MixedThermodynamicObservables {
    double temperature;
    
    // Energy observables
    Observable energy_total;         // <E>/N_total
    Observable energy_SU2;           // <E_SU2>/N_SU2
    Observable energy_SU3;           // <E_SU3>/N_SU3
    Observable specific_heat;        // C_V = (<E²> - <E>²) / (T² N)
    
    // SU(2) sublattice observables
    vector<VectorObservable> sublattice_magnetization_SU2;  // <S_α> for each SU(2) sublattice
    vector<VectorObservable> energy_sublattice_cross_SU2;   // <E * S_α> - <E><S_α>
    
    // SU(3) sublattice observables
    vector<VectorObservable> sublattice_magnetization_SU3;  // <S_α> for each SU(3) sublattice
    vector<VectorObservable> energy_sublattice_cross_SU3;   // <E * S_α> - <E><S_α>
};

/**
 * MixedLattice class: Template-free implementation for coupled SU(2) and SU(3) systems
 * 
 * This class manages a periodic lattice with two types of spins:
 * - SU(2) spins (typically spin-1/2 with 3 components)
 * - SU(3) spins (typically with 8 components - Gell-Mann generators)
 * 
 * Supports:
 * - Bilinear and trilinear interactions within each sublattice
 * - Mixed bilinear and trilinear interactions between sublattices
 * - Monte Carlo sampling (Metropolis, heat bath, overrelaxation; SU(3)
 *   sites on CP^2 with the Fubini-Study measure)
 * - Parallel tempering
 * - Molecular dynamics (Landau-Lifshitz equations)
 * - Time-dependent external fields
 */
class MixedLattice {
public:
    // Type aliases for clarity
    using SpinConfigSU2 = vector<SpinVector>;  // SU(2) spin configuration
    using SpinConfigSU3 = vector<SpinVector>;  // SU(3) spin configuration
    using ODEState = vector<double>;            // Flat state vector for Boost.Odeint
    // Magnetisation observables of one sample:
    // ([M_antiferro, M_local, M_global] SU(2), [M_antiferro, M_local, M_global] SU(3)).
    using Observables = pair<array<SpinVector, 3>, array<SpinVector, 3>>;
    // (time, observables) per sample of a pulse-drive trajectory.
    using PumpProbeTrajectory = vector<pair<double, Observables>>;

    // Core lattice properties
    size_t spin_dim_SU2;         // Dimension of SU(2) spins (typically 3)
    size_t spin_dim_SU3;         // Dimension of SU(3) spins (typically 8)
    size_t N_atoms_SU2;          // Number of SU(2) atoms per unit cell
    size_t N_atoms_SU3;          // Number of SU(3) atoms per unit cell
    size_t dim1, dim2, dim3;     // Lattice dimensions
    size_t lattice_size_SU2;     // Total SU(2) sites = N_atoms_SU2 * dim1 * dim2 * dim3
    size_t lattice_size_SU3;     // Total SU(3) sites = N_atoms_SU3 * dim1 * dim2 * dim3
    float spin_length_SU2;       // Magnitude of SU(2) spin vectors
    float spin_length_SU3;       // Radius of the legacy S^7 SU(3) manifold (unused on CP^2, |n| = 2/sqrt3)

    // Spin configurations and positions
    SpinConfigSU2 spins_SU2;                    // Current SU(2) spins
    SpinConfigSU3 spins_SU3;                    // Current SU(3) spins
    vector<Eigen::Vector3d> site_positions_SU2; // Real-space positions for SU(2) sites
    vector<Eigen::Vector3d> site_positions_SU3; // Real-space positions for SU(3) sites

    // SU(2) interaction lookup tables
    vector<SpinVector> field_SU2;                               // External field at each SU(2) site
    vector<SpinMatrix> onsite_interaction_SU2;                  // On-site anisotropy for SU(2)
    vector<vector<SpinMatrix>> bilinear_interaction_SU2;        // SU(2)-SU(2) bilinear coupling
    vector<vector<SpinTensor3>> trilinear_interaction_SU2;      // SU(2)-SU(2)-SU(2) trilinear coupling
    vector<vector<size_t>> bilinear_partners_SU2;               // SU(2) bilinear partner indices
    vector<vector<array<size_t, 2>>> trilinear_partners_SU2;   // SU(2) trilinear partner pairs

    // SU(3) interaction lookup tables
    vector<SpinVector> field_SU3;                               // External field at each SU(3) site
    vector<SpinMatrix> onsite_interaction_SU3;                  // On-site anisotropy for SU(3)
    vector<vector<SpinMatrix>> bilinear_interaction_SU3;        // SU(3)-SU(3) bilinear coupling
    vector<vector<SpinTensor3>> trilinear_interaction_SU3;      // SU(3)-SU(3)-SU(3) trilinear coupling
    vector<vector<size_t>> bilinear_partners_SU3;               // SU(3) bilinear partner indices
    vector<vector<array<size_t, 2>>> trilinear_partners_SU3;   // SU(3) trilinear partner pairs

    // Mixed SU(2)-SU(3) interaction lookup tables
    vector<vector<Eigen::MatrixXd>> mixed_bilinear_interaction_SU2;  // SU(2)-SU(3) bilinear (from SU(2) side)
    vector<vector<Eigen::MatrixXd>> mixed_bilinear_interaction_SU3;  // SU(3)-SU(2) bilinear (from SU(3) side)
    vector<vector<size_t>> mixed_bilinear_partners_SU2;              // SU(3) partner indices for SU(2)
    vector<vector<size_t>> mixed_bilinear_partners_SU3;              // SU(2) partner indices for SU(3)

    // Pulse-envelope-modulated mixed SU(2)-SU(3) bilinears (field-assisted
    // Fe-Tm exchange H_{E chi} and H_{B chi}).  Same storage shape as the
    // static mixed bilinears, plus a per-entry `envelope` selector:
    //   0 -> SU(3) pulse envelope (electric / THz field E(t), scalar)
    //   1 -> SU(2) pulse envelope (magnetic field B(t), scalar magnitude)
    //   2 -> global B_x(t) component   (polarization-resolved magnetic gate)
    //   3 -> global B_y(t) component
    //   4 -> global B_z(t) component
    // At each RHS evaluation the contribution is scaled by the matching pulse
    // envelope scalar, so it vanishes outside the pulse window.  The
    // component-resolved tags (2..4) gate a vertex by the *actual* driving-field
    // component along a lab (global) axis, so e.g. a B_z-gated kappaB vertex is
    // identically zero for an H||a (B along x) pump and active for H||c.  This
    // makes a single Hamiltonian self-select by measurement geometry.
    vector<vector<Eigen::MatrixXd>> mixed_bilinear_drive_interaction_SU2;
    vector<vector<Eigen::MatrixXd>> mixed_bilinear_drive_interaction_SU3;
    vector<vector<size_t>> mixed_bilinear_drive_partners_SU2;
    vector<vector<size_t>> mixed_bilinear_drive_partners_SU3;
    vector<vector<int>>    mixed_bilinear_drive_envelope_SU2;   // per-entry envelope tag
    vector<vector<int>>    mixed_bilinear_drive_envelope_SU3;   // per-entry envelope tag
    bool has_mixed_bilinear_drive = false;   // true if any field-assisted bond is set
    vector<vector<SpinTensor3>> mixed_trilinear_interaction_SU2;  // SU(2)-SU(2)-SU(3) trilinear (vector of matrices)
    vector<vector<SpinTensor3>> mixed_trilinear_interaction_SU3;  // SU(3)-SU(2)-SU(2) trilinear (vector of matrices)
    vector<vector<array<size_t, 2>>> mixed_trilinear_partners_SU2;             // (SU(2), SU(3)) partner pairs
    vector<vector<array<size_t, 2>>> mixed_trilinear_partners_SU3;             // (SU(2), SU(2)) partner pairs

    // ============================================================
    // PACKED INTERACTION BUFFERS (SoA, row-major, double*)
    // ============================================================
    // Mirror data already in {bilinear,trilinear,mixed_*}_interaction_*
    // but laid out as one contiguous double[] per site, row-major in
    // (a,b) for bilinear and (a,b,c) for trilinear. This:
    //   - removes the vector<MatrixXd> pointer-chase per bond,
    //   - flips the inner-loop stride from column-major MatrixXd
    //     (stride = rows) to unit-stride row-major (stride = 1),
    //   - lets the compiler vectorize and unroll fixed-size kernels
    //     (3x3, 3x8, 8x8, 3x3x3, 3x3x8, 8x3x3, 8x8x8) used by the
    //     SU(2)/SU(3) MD RHS and Monte Carlo local fields.
    // Built in build_packed_interaction_buffers() at the end of the
    // constructor (and after any change of the coupling tables); the
    // original SpinMatrix / SpinTensor3 storage is the source of rebuilds.
    //
    // Layout per site (n indexes bonds at this site):
    //   bilinear_packed_*[site][n*da*db + a*db + b]
    //   trilinear_packed_*[site][n*da*db*dc + (a*db + b)*dc + c]
    vector<vector<double>> bilinear_packed_SU2;             // da=db=spin_dim_SU2
    vector<vector<double>> bilinear_packed_SU3;             // da=db=spin_dim_SU3
    vector<vector<double>> mixed_bilinear_packed_SU2;       // da=spin_dim_SU2, db=spin_dim_SU3
    vector<vector<double>> mixed_bilinear_packed_SU3;       // da=spin_dim_SU3, db=spin_dim_SU2
    vector<vector<double>> mixed_bilinear_drive_packed_SU2; // field-assisted: da=spin_dim_SU2, db=spin_dim_SU3
    vector<vector<double>> mixed_bilinear_drive_packed_SU3; // field-assisted: da=spin_dim_SU3, db=spin_dim_SU2
    vector<vector<double>> trilinear_packed_SU2;            // da=db=dc=spin_dim_SU2
    vector<vector<double>> trilinear_packed_SU3;            // da=db=dc=spin_dim_SU3
    vector<vector<double>> mixed_trilinear_packed_SU2;      // da=db=spin_dim_SU2, dc=spin_dim_SU3
    vector<vector<double>> mixed_trilinear_packed_SU3;      // da=spin_dim_SU3, db=dc=spin_dim_SU2

    // Sublattice frame transformations
    vector<SpinMatrix> sublattice_frames_SU2;  // Frame transformations for SU(2) sublattices
    vector<SpinMatrix> sublattice_frames_SU3;  // Frame transformations for SU(3) sublattices
    vector<double> afm_sublattice_signs_SU2;   // AFM sublattice signs for SU(2) (Bertaut modes)
    vector<double> afm_sublattice_signs_SU3;   // AFM sublattice signs for SU(3)

    // Interaction counts per site
    size_t num_bi_SU2;       // Number of SU(2)-SU(2) bilinear neighbors per site
    size_t num_tri_SU2;      // Number of SU(2)-SU(2)-SU(2) trilinear interactions per site
    size_t num_bi_SU3;       // Number of SU(3)-SU(3) bilinear neighbors per site
    size_t num_tri_SU3;      // Number of SU(3)-SU(3)-SU(3) trilinear interactions per site
    size_t num_bi_SU2_SU3;   // Number of mixed bilinear interactions per site
    size_t num_tri_SU2_SU3;  // Number of mixed trilinear interactions per site

    // ------------------------------------------------------------------
    // Sublattice colouring of the bond graph for the parallel coloured
    // Metropolis / over-relaxation sweeps. Built once at init() time.
    //
    // SU(2) and SU(3) sublattices are coloured **independently** because the
    // parallel sweep updates each species in its own pass: while updating
    // SU(2) sites, the SU(3) configuration is frozen, so only intra-SU(2)
    // interactions create write-vs-read races. The relevant edges for the
    // SU(2) graph are: SU(2)-SU(2) bilinear, SU(2)-SU(2)-SU(2) trilinear,
    // and the SU(2)-SU(2) pair inside any SU(2)-SU(2)-SU(3) mixed trilinear.
    // Symmetric story for SU(3).
    //
    // Mixed bilinear (SU(2)-SU(3)) does NOT add intra-sublattice edges.
    //
    // Layout: same CSR pattern as Lattice (cf. lattice.h::color_of_site).
    // ------------------------------------------------------------------
    vector<uint16_t>           color_of_site_SU2;
    vector<size_t>             sites_by_color_csr_off_SU2;   // size n_colors_SU2 + 1
    vector<size_t>             sites_by_color_csr_SU2;       // size lattice_size_SU2
    size_t                     n_colors_SU2 = 0;

    vector<uint16_t>           color_of_site_SU3;
    vector<size_t>             sites_by_color_csr_off_SU3;   // size n_colors_SU3 + 1
    vector<size_t>             sites_by_color_csr_SU3;       // size lattice_size_SU3
    size_t                     n_colors_SU3 = 0;

    // Time-dependent fields for molecular dynamics.
    //
    // The drive is a train of at most two physical pulses k = 0, 1 centred at
    // t_pulse_*[k]; slot k acts (drives the spins AND gates the field-assisted
    // Fe-Tm exchange) only for k < n_active_pulses. A single-pulse experiment
    // therefore has no phantom second pulse, whatever its direction vectors.
    size_t n_active_pulses = 0;
    array<SpinVector, 2> field_drive_SU2;     // Two pulse components for SU(2)
    // Global (lab-frame) representative of the two SU(2) pulse field vectors,
    // BEFORE the per-sublattice local-frame transform.  Used only to compute
    // the polarization-resolved magnetic gates B_x/B_y/B_z(t) for the
    // component-tagged (2..4) field-assisted bilinears.  The THz pump is a
    // uniform plane wave, so a single lab-frame vector per pulse suffices.
    array<SpinVector, 2> field_drive_global_SU2{SpinVector::Zero(3), SpinVector::Zero(3)};
    array<SpinVector, 2> field_drive_SU3;     // Two pulse components for SU(3)
    array<double, 2> t_pulse_SU2;             // Pulse center times for SU(2)
    array<double, 2> t_pulse_SU3;             // Pulse center times for SU(3)
    double field_drive_amp_SU2;               // Pulse amplitude for SU(2)
    double field_drive_freq_SU2;              // Pulse frequency for SU(2)
    double field_drive_width_SU2;             // Pulse width (Gaussian) for SU(2)
    double field_drive_amp_SU3;               // Pulse amplitude for SU(3)
    double field_drive_freq_SU3;              // Pulse frequency for SU(3)
    double field_drive_width_SU3;             // Pulse width (Gaussian) for SU(3)
    // Optional SECOND SU(3) carrier color (two-color CEF drive).  When
    // nonzero the SU(3) envelope carries cos(freq*dt) + cos(freq2*dt) under
    // the same Gaussian, so one pulse drives two CEF lines (e.g. E13=1.20 and
    // E23=0.70) resonantly.  This enables the f_257 Raman product to land on
    // E12 = E13 - E23 = 0.50 THz.  0.0 disables (single-color, legacy behavior).
    double field_drive_freq_SU3_2 = 0.0;

    // Tabulated (experimental) pulse — loaded by load_tabulated_pulse().
    // When non-empty, drive_envelopes_SU2/SU3 use linear interpolation instead
    // of the Gaussian model.  Times are centered at the pulse peak (ps); values
    // are normalized so that max|E| = 1.  tabulated_pulse_sigma is the
    // Gaussian-equivalent σ = max_extent / kPulseWindowSigmas used for chunking.
    std::vector<double> tabulated_pulse_times;   // Centered time axis (ps)
    std::vector<double> tabulated_pulse_values;  // Normalized E-field values
    double tabulated_pulse_sigma = 0.0;          // Effective σ for pulse-window chunking

    // SU(2) Gilbert damping: dS/dt += (alpha/|S|) * S × (S × H)
    double alpha_gilbert = 0.0;

    // Lie-Poisson coefficient c of the SU(3) equation of motion
    //   dn^a/dt = c f_{abc} (∂E/∂n^b) n^c,
    // c = 2 for n = <λ> (core/su3_coherent_state.h); taken from the SU(3) unit
    // cell (UnitCell::poisson_bracket), 1 only under su3_legacy_convention.
    double su3_bracket = classical_spin::su3::kGellMannBracket;

    // SU(3) Landau-Lifshitz damping (Casimir-preserving double bracket; the
    // Bloch-vector form of the SU(N) LL damping of Dahlbom et al.,
    // PRB 106, 235154 (2022)):
    //     dn/dt += -(alpha_SU3 / |n|) c f(n, P),   P = c f(H, n) the precession,
    // the direct analogue of the SU(2) Gilbert term -(alpha/|S|) S x (H x S).
    // dE/dt = -(alpha_SU3/|n|) |P|^2 <= 0 and, because the update is an
    // adjoint action, |n|^2 and d_abc n^a n^b n^c are conserved: a pure state
    // stays pure and relaxes towards the ground state of its local field
    // (unlike the Bloch relaxation below, which targets a fixed n_eq and can
    // leave the physical state space for non-uniform rates).
    double alpha_SU3 = 0.0;

    // SU(3) Bloch damping/relaxation (tmfeo3_notes.tex Eq. blochdampedfull)
    // dn^a/dt = c f_{abc} h^b n^c  −  Γ_a (n^a − n^a_eq)
    // damping_rates_SU3[a]: phenomenological relaxation rates Γ_a for each Gell-Mann channel
    // equilibrium_SU3[site]: thermal equilibrium Bloch vector n^a_eq for each SU(3) site
    SpinVector damping_rates_SU3;               // 8-component, one Γ per Gell-Mann channel
    SpinConfigSU3 equilibrium_SU3;              // Per-site equilibrium Bloch vector

    // Thermal repopulation (tau-labeled heating). A reservoir energy E_dep
    // is integrated as ONE EXTRA ODE VARIABLE appended to the spin state
    // (spins_to_state() adds it, initialised to 0, whenever thermal_heat != 0):
    //     dE_dep/dt = P(n) − thermal_cool · E_dep,
    //     P(n) = Σ_sites Σ_{a ∉ {3,8}} Γ_a (n^a − n^a_eq)^2,
    // a positive-definite proxy for the power dissipated by the coherence
    // relaxation (the population channels λ3, λ8 are excluded: their
    // relaxation returns energy to the bath, and re-heating from them would
    // close a runaway loop). The λ3 relaxation target is shifted by
    //     −min(thermal_cap, thermal_heat · E_dep),
    // so the population rise time is 1/Γ_3 and the pump-FID × probe
    // interference in P carries the τ label into M_NL. Being part of the ODE
    // state, E_dep is integrated by the same (adaptive) scheme as the spins
    // and the right-hand side stays a pure function of (state, t).
    double thermal_heat = 0.0;                  // coupling (0 = feature off)
    double thermal_cap  = 1.0e30;               // max lambda3 eq shift (saturation)
    double thermal_cool = 0.0;                  // heat-reservoir cooling rate (1/time; 0 = no cooling)

    // ============================================================
    // MONTE CARLO: STATE SPACE, UPDATE POLICY, LOCAL FORMS
    // ============================================================

    /**
     * State space sampled for the SU(3) sites.
     *  - CP2 (default): qutrit pure states psi in C^3, stored as their
     *    Gell-Mann expectations n^a = <psi|lambda^a|psi> (|n|^2 = 4/3, cubic
     *    Casimir 8/9) and sampled with the Fubini-Study (Haar) measure: the
     *    manifold the SU(3) dynamics conserves, on which E(n) = <psi|H|psi>
     *    is the energy of a product state (Zhang & Batista, PRB 104, 104409
     *    (2021); Dahlbom et al., PRB 106, 235154 (2022)).
     *  - Sphere (legacy): the 7-sphere |n| = spin_length_SU3 with its uniform
     *    measure (most of its points are not density matrices). Default only
     *    under su3_legacy_convention (SU(3) UnitCell::poisson_bracket == 1).
     */
    enum class SU3Manifold { CP2, Sphere };
    SU3Manifold su3_mc_manifold = SU3Manifold::CP2;

    /// "cp2" / "sphere" (case-sensitive); throws std::invalid_argument otherwise.
    static SU3Manifold parse_su3_manifold(const string& name) {
        if (name == "cp2" || name == "CP2") return SU3Manifold::CP2;
        if (name == "sphere" || name == "S7") return SU3Manifold::Sphere;
        throw std::invalid_argument("su3_mc_manifold: unknown manifold '" + name +
                                    "' (valid: cp2, sphere)");
    }

    /**
     * Select the SU(3) Monte Carlo manifold by name; "" or "auto" restores the
     * default of the SU(3) convention (CP2, or Sphere under
     * su3_legacy_convention). The stored spins are not changed: call
     * init_random() or project_SU3_to_manifold() afterwards (the SA and PT
     * drivers project automatically).
     */
    void set_su3_mc_manifold(const string& name) {
        if (name.empty() || name == "auto") {
            su3_mc_manifold = (su3_bracket == classical_spin::su3::kLegacyBracket) ? SU3Manifold::Sphere
                                                                                 : SU3Manifold::CP2;
        } else {
            su3_mc_manifold = parse_su3_manifold(name);
        }
    }
    bool su3_on_cp2() const { return su3_mc_manifold == SU3Manifold::CP2; }

    /**
     * Local update used by local_sweep() (SA, PT, final measurements):
     *  - Metropolis: independent uniform proposals (SU(2): sphere; SU(3):
     *    Haar on CP^2, or uniform on S^7);
     *  - Gaussian: symmetric small moves of width sigma (SU(2):
     *    S' ∝ S + sigma L u; SU(3) on CP^2: psi' ∝ psi + sigma xi, xi a complex
     *    Gaussian 3-vector), sigma adapted by the drivers;
     *  - HeatBath: rejection-free draw from the linear part of the local
     *    energy (SU(2): Miyatake et al., J. Phys. C 19, 2539 (1986); SU(3)
     *    on CP^2: populations of the local eigenbasis uniform on the simplex
     *    under Fubini-Study), Metropolis-corrected for the quadratic self
     *    terms (single-ion anisotropy, self-coupled trilinears). SU(3) sites
     *    on the legacy sphere fall back to Metropolis.
     * Config key `local_update` (as for Lattice).
     */
    enum class LocalUpdate { Metropolis, Gaussian, HeatBath };
    LocalUpdate local_update = LocalUpdate::Metropolis;
    /// Coloured OpenMP sweeps are used from this many sites (and > 1 thread) on.
    size_t parallel_sweep_min_sites = 4096;

    static LocalUpdate parse_local_update(const string& name) {
        if (name == "metropolis" || name == "uniform") return LocalUpdate::Metropolis;
        if (name == "gaussian" || name == "adaptive") return LocalUpdate::Gaussian;
        if (name == "heat_bath" || name == "heatbath") return LocalUpdate::HeatBath;
        throw std::invalid_argument("unknown local update '" + name +
                                    "' (valid: metropolis, gaussian, heat_bath)");
    }

    // ------------------------------------------------------------------
    // Local forms. With every other spin frozen, the energy of SU(2) site i
    // as a function of its own spin is
    //     E_i(S) = h_i . S + S^T A_i S  (+ cubic self terms, if any):
    // h_i collects every coupling that contains S_i exactly once (field,
    // bilinear, mixed bilinear, trilinear entries with two other partners),
    // A_i the symmetrised on-site matrix plus every coupling that contains
    // S_i twice, with its third partner contracted (the TmFeO3 vertex
    // W(S_i, S_i, n_k) and any trilinear that wraps onto itself on a small
    // lattice). Couplings with S_i in all three slots are kept as cubic
    // entries. The same holds for SU(3) sites with n_j. Each trilinear entry
    // stored at site i carries weight 1/(1 + number of its partner slots equal
    // to i) in the local energy, which reproduces total_energy() exactly.
    //
    // The tables below hold the self-coupled part, merged per partner and
    // symmetrised in the two self slots, so a site visit is one pass over
    // its packed couplings with no allocation (rebuilt by
    // build_packed_interaction_buffers()).
    struct SelfQuadTerm {
        size_t partner;      // partner site index
        size_t offset;       // into selfq_coef_SU{2,3}: (a <= b) upper-triangle rows x dim, row-major
        uint8_t partner_su3; // partner species: 0 = SU(2), 1 = SU(3)
        uint8_t dim;         // partner components (3 or 8)
    };
    vector<double>           onsite_sym_SU2, onsite_sym_SU3;    // sym(onsite), d x d per site
    vector<size_t>           selfq_off_SU2, selfq_off_SU3;      // CSR offsets, size N + 1
    vector<SelfQuadTerm>     selfq_SU2, selfq_SU3;
    vector<double>           selfq_coef_SU2, selfq_coef_SU3;
    vector<vector<uint32_t>> cubic_self_SU2, cubic_self_SU3;    // trilinear entries with S_i in every slot
    vector<uint8_t>          self_mode_SU2, self_mode_SU3;      // 0: no self energy, 1: quadratic, 2: + cubic
    vector<uint8_t>          self_isotropic_SU2, self_isotropic_SU3;  // self energy constant on the manifold

    /**
     * Constructor: Build a mixed lattice from two unit cells
     * 
     * @param uc_SU2         Unit cell defining the SU(2) sublattice structure
     * @param uc_SU3         Unit cell defining the SU(3) sublattice structure
     * @param dim1           Lattice size in first dimension
     * @param dim2           Lattice size in second dimension
     * @param dim3           Lattice size in third dimension
     * @param spin_l_SU2     Magnitude of SU(2) spin vectors
     * @param spin_l_SU3     Magnitude of SU(3) spin vectors
     */
    MixedLattice(const UnitCell& uc_SU2, const UnitCell& uc_SU3,
                 size_t dim1, size_t dim2, size_t dim3,
                 float spin_l_SU2 = 1.0, float spin_l_SU3 = 1.0)
        : MixedLattice(MixedUnitCell(uc_SU2, uc_SU3), dim1, dim2, dim3, spin_l_SU2, spin_l_SU3)
    {
        // Delegating constructor - mixed interactions will be empty
        cout << "Note: Using separate unit cells - mixed SU(2)-SU(3) interactions not set." << endl;
    }

    /**
     * Constructor: Build a mixed lattice from a MixedUnitCell
     * 
     * @param mixed_uc       Mixed unit cell defining both sublattices and mixed interactions
     * @param dim1           Lattice size in first dimension
     * @param dim2           Lattice size in second dimension
     * @param dim3           Lattice size in third dimension
     * @param spin_l_SU2     Magnitude of SU(2) spin vectors
     * @param spin_l_SU3     Magnitude of SU(3) spin vectors
     */
    MixedLattice(const MixedUnitCell& mixed_uc, size_t dim1, size_t dim2, size_t dim3,
                 float spin_l_SU2 = 1.0, float spin_l_SU3 = 1.0)
        : spin_dim_SU2(mixed_uc.SU2_cell.N),
          spin_dim_SU3(mixed_uc.SU3_cell.N),
          N_atoms_SU2(mixed_uc.SU2_cell.N_atoms),
          N_atoms_SU3(mixed_uc.SU3_cell.N_atoms),
          dim1(dim1), dim2(dim2), dim3(dim3),
          spin_length_SU2(spin_l_SU2),
          spin_length_SU3(spin_l_SU3)
    {
        // The dynamics, the packed field kernels and the observables are
        // written for SU(2) vectors (3 components) coupled to SU(3) Gell-Mann
        // vectors (8 components); any other shape used to run silently with
        // frozen spins or out-of-bounds structure constants.
        if (spin_dim_SU2 != 3 || spin_dim_SU3 != 8) {
            throw std::invalid_argument(
                "MixedLattice: requires 3-component SU(2) and 8-component SU(3) spins (got spin_dim_SU2="
                + std::to_string(spin_dim_SU2) + ", spin_dim_SU3=" + std::to_string(spin_dim_SU3) + ")");
        }
        su3_bracket = mixed_uc.SU3_cell.poisson_bracket;
        set_su3_mc_manifold("");   // CP^2, or the legacy S^7 under su3_legacy_convention
        lattice_size_SU2 = N_atoms_SU2 * dim1 * dim2 * dim3;
        lattice_size_SU3 = N_atoms_SU3 * dim1 * dim2 * dim3;
        
        cout << "Initializing mixed lattice with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "SU(2): " << lattice_size_SU2 << " sites (" << N_atoms_SU2 << " atoms/cell, spin_dim=" << spin_dim_SU2 << ", spin_length=" << spin_length_SU2 << ")" << endl;
        cout << "SU(3): " << lattice_size_SU3 << " sites (" << N_atoms_SU3 << " atoms/cell, spin_dim=" << spin_dim_SU3 << ", spin_length=" << spin_length_SU3 << ")" << endl;

        // Initialize arrays
        spins_SU2.resize(lattice_size_SU2);
        spins_SU3.resize(lattice_size_SU3);
        site_positions_SU2.resize(lattice_size_SU2);
        site_positions_SU3.resize(lattice_size_SU3);
        
        field_SU2.resize(lattice_size_SU2);
        field_SU3.resize(lattice_size_SU3);
        onsite_interaction_SU2.resize(lattice_size_SU2);
        onsite_interaction_SU3.resize(lattice_size_SU3);
        
        bilinear_interaction_SU2.resize(lattice_size_SU2);
        bilinear_interaction_SU3.resize(lattice_size_SU3);
        trilinear_interaction_SU2.resize(lattice_size_SU2);
        trilinear_interaction_SU3.resize(lattice_size_SU3);
        
        bilinear_partners_SU2.resize(lattice_size_SU2);
        bilinear_partners_SU3.resize(lattice_size_SU3);
        trilinear_partners_SU2.resize(lattice_size_SU2);
        trilinear_partners_SU3.resize(lattice_size_SU3);
        
        mixed_bilinear_interaction_SU2.resize(lattice_size_SU2);
        mixed_bilinear_interaction_SU3.resize(lattice_size_SU3);
        mixed_bilinear_partners_SU2.resize(lattice_size_SU2);
        mixed_bilinear_partners_SU3.resize(lattice_size_SU3);

        mixed_bilinear_drive_interaction_SU2.resize(lattice_size_SU2);
        mixed_bilinear_drive_interaction_SU3.resize(lattice_size_SU3);
        mixed_bilinear_drive_partners_SU2.resize(lattice_size_SU2);
        mixed_bilinear_drive_partners_SU3.resize(lattice_size_SU3);
        mixed_bilinear_drive_envelope_SU2.resize(lattice_size_SU2);
        mixed_bilinear_drive_envelope_SU3.resize(lattice_size_SU3);
        
        mixed_trilinear_interaction_SU2.resize(lattice_size_SU2);
        mixed_trilinear_interaction_SU3.resize(lattice_size_SU3);
        mixed_trilinear_partners_SU2.resize(lattice_size_SU2);
        mixed_trilinear_partners_SU3.resize(lattice_size_SU3);
        
        sublattice_frames_SU2.resize(N_atoms_SU2);
        sublattice_frames_SU3.resize(N_atoms_SU3);
        
        // Copy sublattice frames
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            sublattice_frames_SU2[atom] = mixed_uc.SU2_cell.sublattice_frames[atom];
        }
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            sublattice_frames_SU3[atom] = mixed_uc.SU3_cell.sublattice_frames[atom];
        }
        afm_sublattice_signs_SU2 = mixed_uc.SU2_cell.afm_sublattice_signs;
        afm_sublattice_signs_SU3 = mixed_uc.SU3_cell.afm_sublattice_signs;

        // Initialize time-dependent fields
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        t_pulse_SU2[0] = 0.0;
        t_pulse_SU2[1] = 0.0;
        t_pulse_SU3[0] = 0.0;
        t_pulse_SU3[1] = 0.0;
        field_drive_amp_SU2 = 0.0;
        field_drive_freq_SU2 = 0.0;
        field_drive_width_SU2 = 1.0;
        field_drive_amp_SU3 = 0.0;
        field_drive_freq_SU3 = 0.0;
        field_drive_width_SU3 = 1.0;

        // SU(3) Bloch damping: default zero rates (no damping)
        damping_rates_SU3 = SpinVector::Zero(spin_dim_SU3);
        // equilibrium_SU3 will be sized after lattice is built (see below)

        // Initialize random seed
        // The RNG is seeded once per process (config key `seed`); constructing
        // a lattice must not reseed it.

        // A lattice no wider than 2|offset| along a bonded direction maps
        // distinct bonds onto the same pair of sites (their couplings add) or
        // onto the site itself (folded into the on-site term): a valid
        // periodic Hamiltonian, but rarely the intended one.
        {
            array<long, 3> reach = {0, 0, 0};
            auto widen = [&](const auto& off) {
                for (int d = 0; d < 3; ++d) reach[d] = std::max(reach[d], long(std::abs(int(off[d]))));
            };
            for (const UnitCell* uc : {&mixed_uc.SU2_cell, &mixed_uc.SU3_cell}) {
                for (const auto& [atom, b] : uc->bilinear_interaction) widen(b.offset);
                for (const auto& [atom, t] : uc->trilinear_interaction) { widen(t.offset1); widen(t.offset2); }
            }
            for (const auto& [atom, b] : mixed_uc.bilinear_SU2_SU3) widen(b.offset);
            for (const auto& [atom, b] : mixed_uc.bilinear_drive_SU2_SU3) widen(b.offset);
            for (const auto& [atom, t] : mixed_uc.trilinear_SU2_SU3) { widen(t.offset1); widen(t.offset2); }
            const array<size_t, 3> dims = {dim1, dim2, dim3};
            for (int d = 0; d < 3; ++d)
                if (reach[d] > 0 && long(dims[d]) < 2 * reach[d] + 1)
                    cout << "Warning: lattice extent " << dims[d] << " along a" << d + 1 << " is below 2*"
                         << reach[d] << "+1; periodic images of distinct bonds coincide (their couplings add)"
                         << endl;
        }

        // Build SU(2) sublattice
        build_sublattice(mixed_uc.SU2_cell, spins_SU2, site_positions_SU2, field_SU2,
                        onsite_interaction_SU2, bilinear_interaction_SU2,
                        trilinear_interaction_SU2, bilinear_partners_SU2,
                        trilinear_partners_SU2, num_bi_SU2, num_tri_SU2,
                        spin_length_SU2, spin_dim_SU2, N_atoms_SU2);

        // Build SU(3) sublattice
        build_sublattice(mixed_uc.SU3_cell, spins_SU3, site_positions_SU3, field_SU3,
                        onsite_interaction_SU3, bilinear_interaction_SU3,
                        trilinear_interaction_SU3, bilinear_partners_SU3,
                        trilinear_partners_SU3, num_bi_SU3, num_tri_SU3,
                        spin_length_SU3, spin_dim_SU3, N_atoms_SU3);

        // Build mixed SU(2)-SU(3) interactions
        build_mixed_interactions(mixed_uc, num_bi_SU2_SU3, num_tri_SU2_SU3);

        // Initialize SU(3) Bloch damping equilibrium (default: zero = infinite temperature)
        equilibrium_SU3.resize(lattice_size_SU3, SpinVector::Zero(spin_dim_SU3));

        // Build per-sublattice colour partition for the parallel coloured
        // Metropolis / over-relaxation sweeps. See header doc on
        // color_of_site_SU{2,3} for the edge-set we use.
        build_color_partition();

        // Pack {bi,tri}linear interaction tensors into row-major double[]
        // buffers (MD and MC hot paths) and build the MC local-form tables.
        build_packed_interaction_buffers();

        // Random initial state on the Monte Carlo manifolds.
        init_random();

        cout << "Mixed lattice initialization complete!" << endl;
        cout << "SU(2) - Max bilinear: " << num_bi_SU2 << ", Max trilinear: " << num_tri_SU2 << endl;
        cout << "SU(3) - Max bilinear: " << num_bi_SU3 << ", Max trilinear: " << num_tri_SU3 << endl;
        cout << "Mixed - Bilinear: " << num_bi_SU2_SU3 << ", Trilinear: " << num_tri_SU2_SU3 << endl;
        cout << "SU(2) sublattice colouring: " << n_colors_SU2 << " colour(s)" << endl;
        cout << "SU(3) sublattice colouring: " << n_colors_SU3 << " colour(s)" << endl;
    }

    // ============================================================
    // UTILITY METHODS
    // ============================================================

    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom, size_t N_atoms) const {
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }

    /**
     * Periodic boundary condition: Euclidean (floor) reduction of a cell
     * coordinate onto [0, dim_size), correct for bonds longer than the
     * lattice (the previous single-step `coord ± L` returned an out-of-range
     * index for |coord| > L, a heap overflow in the constructor).
     */
    size_t periodic_boundary(int coord, size_t dim_size) const {
        const long L = long(dim_size);
        long r = long(coord) % L;
        if (r < 0) r += L;
        return size_t(r);
    }

    /**
     * Flatten with periodic boundaries
     */
    size_t flatten_index_periodic(int i, int j, int k, size_t atom, size_t N_atoms) const {
        return flatten_index(periodic_boundary(i, dim1),
                           periodic_boundary(j, dim2),
                           periodic_boundary(k, dim3),
                           atom, N_atoms);
    }

    /**
     * Uniformly random spin on the sphere of radius spin_l in R^spin_dim
     * (see random_point_on_sphere).
     */
    SpinVector gen_random_spin(float spin_l, size_t spin_dim) {
        SpinVector spin(spin_dim);
        random_point_on_sphere(spin.data(), spin_dim, double(spin_l));
        return spin;
    }

    /**
     * Build a sublattice from a unit cell
     */
    void build_sublattice(const UnitCell& uc,
                         SpinConfigSU2& spins,
                         vector<Eigen::Vector3d>& positions,
                         vector<SpinVector>& field,
                         vector<SpinMatrix>& onsite,
                         vector<vector<SpinMatrix>>& bilinear,
                         vector<vector<SpinTensor3>>& trilinear,
                         vector<vector<size_t>>& bi_partners,
                         vector<vector<array<size_t, 2>>>& tri_partners,
                         size_t& num_bi, size_t& num_tri,
                         [[maybe_unused]] float spin_length, size_t spin_dim, size_t N_atoms)
    {
        const size_t lattice_size = N_atoms * dim1 * dim2 * dim3;

        // Phase 1: Count interactions per site
        vector<size_t> bi_count(lattice_size, 0);
        vector<size_t> tri_count(lattice_size, 0);
        
        size_t site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t l = 0; l < N_atoms; ++l) {
                        // Calculate position
                        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
                        for (int d = 0; d < 3; d++) {
                            pos(d) = uc.lattice_vectors[0](d) * int(i) + 
                                    uc.lattice_vectors[1](d) * int(j) + 
                                    uc.lattice_vectors[2](d) * int(k) + 
                                    uc.lattice_pos[l](d);
                        }
                        positions[site_idx] = pos;
                        
                        // Spins are drawn by init_random() once the lattice is built.
                        spins[site_idx] = SpinVector::Zero(spin_dim);
                        
                        // Copy field and onsite interaction
                        field[site_idx] = uc.field[l];
                        onsite[site_idx] = uc.onsite_interaction[l];
                        
                        // Count bilinear interactions
                        auto bi_range = uc.bilinear_interaction.equal_range(l);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& J = it->second;
                            bi_count[site_idx]++;
                            size_t partner = flatten_index_periodic(
                                int(i) + J.offset[0], int(j) + J.offset[1], int(k) + J.offset[2], J.partner, N_atoms);
                            bi_count[partner]++;
                        }
                        
                        // Count trilinear interactions
                        auto tri_range = uc.trilinear_interaction.equal_range(l);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& J = it->second;
                            size_t partner1 = flatten_index_periodic(
                                int(i) + J.offset1[0], int(j) + J.offset1[1], int(k) + J.offset1[2], J.partner1, N_atoms);
                            size_t partner2 = flatten_index_periodic(
                                int(i) + J.offset2[0], int(j) + J.offset2[1], int(k) + J.offset2[2], J.partner2, N_atoms);
                            tri_count[site_idx]++;
                            tri_count[partner1]++;
                            tri_count[partner2]++;
                        }
                        
                        site_idx++;
                    }
                }
            }
        }

        // Phase 2: Allocate storage
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            bilinear[idx].reserve(bi_count[idx]);
            bi_partners[idx].reserve(bi_count[idx]);
            trilinear[idx].reserve(tri_count[idx]);
            tri_partners[idx].reserve(tri_count[idx]);
        }

        // Phase 3: Build interactions
        site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t l = 0; l < N_atoms; ++l) {
                        // Bilinear interactions
                        auto bi_range = uc.bilinear_interaction.equal_range(l);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& J = it->second;
                            size_t partner = flatten_index_periodic(
                                int(i) + J.offset[0], int(j) + J.offset[1], int(k) + J.offset[2], J.partner, N_atoms);

                            // A bond onto the site's own periodic image (lattice
                            // extent 1 along a bonded direction) is the single-ion
                            // term S^T J S: fold it into the on-site matrix so the
                            // energy, the MC energy differences, the local field
                            // and the dynamics all count it once (pushing J and J^T
                            // with partner == site made the Metropolis dE miss the
                            // quadratic part, max |dE - ΔE| = 2.4 on 1x1x1 TmFeO3).
                            if (partner == site_idx) {
                                onsite[site_idx] += 0.5 * (J.interaction + J.interaction.transpose());
                                continue;
                            }

                            bilinear[site_idx].push_back(J.interaction);
                            bi_partners[site_idx].push_back(partner);
                            
                            // Symmetric interaction
                            bilinear[partner].push_back(J.interaction.transpose());
                            bi_partners[partner].push_back(site_idx);
                        }
                        
                        // Trilinear interactions
                        auto tri_range = uc.trilinear_interaction.equal_range(l);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& J = it->second;
                            size_t partner1 = flatten_index_periodic(
                                int(i) + J.offset1[0], int(j) + J.offset1[1], int(k) + J.offset1[2], J.partner1, N_atoms);
                            size_t partner2 = flatten_index_periodic(
                                int(i) + J.offset2[0], int(j) + J.offset2[1], int(k) + J.offset2[2], J.partner2, N_atoms);
                            
                            // Add interaction T_abc S_a^(0) S_b^(1) S_c^(2)
                            trilinear[site_idx].push_back(J.interaction);
                            tri_partners[site_idx].push_back({partner1, partner2});
                            
                            // Add symmetric contributions for energy conservation
                            // For partner1: T_bac S_b^(1) S_a^(0) S_c^(2) (swap first two indices)
                            // T_permuted[b](a,c) = T_original[a](b,c)
                            SpinTensor3 tensor_p1(spin_dim);
                            for (size_t b = 0; b < spin_dim; ++b) {
                                tensor_p1[b] = SpinMatrix(spin_dim, spin_dim);
                                for (size_t a = 0; a < spin_dim; ++a) {
                                    for (size_t c = 0; c < spin_dim; ++c) {
                                        tensor_p1[b](a, c) = J.interaction[a](b, c);
                                    }
                                }
                            }
                            trilinear[partner1].push_back(tensor_p1);
                            tri_partners[partner1].push_back({site_idx, partner2});
                            
                            // For partner2: T_cab S_c^(2) S_a^(0) S_b^(1) (cyclic permutation)
                            // T_permuted[c](a,b) = T_original[a](b,c)
                            SpinTensor3 tensor_p2(spin_dim);
                            for (size_t c = 0; c < spin_dim; ++c) {
                                tensor_p2[c] = SpinMatrix(spin_dim, spin_dim);
                                for (size_t a = 0; a < spin_dim; ++a) {
                                    for (size_t b = 0; b < spin_dim; ++b) {
                                        tensor_p2[c](a, b) = J.interaction[a](b, c);
                                    }
                                }
                            }
                            trilinear[partner2].push_back(tensor_p2);
                            tri_partners[partner2].push_back({site_idx, partner1});
                        }
                        
                        site_idx++;
                    }
                }
            }
        }

        // Only the symmetric part of an on-site matrix enters S^T A S.
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            onsite[idx] = 0.5 * (onsite[idx] + onsite[idx].transpose()).eval();
        }

        num_bi = 0;
        for (const auto& b : bi_partners) num_bi = std::max(num_bi, b.size());
        num_tri = *std::max_element(tri_count.begin(), tri_count.end());
    }

    /**
     * Build mixed SU(2)-SU(3) interactions from MixedUnitCell
     */
    void build_mixed_interactions(const MixedUnitCell& mixed_uc, 
                                  size_t& num_bi_mixed, size_t& num_tri_mixed)
    {
        // Compute max mixed interaction counts
        num_bi_mixed = 0;
        num_tri_mixed = 0;
        
        // Count bilinear interactions
        for (auto it = mixed_uc.bilinear_SU2_SU3.begin(); 
             it != mixed_uc.bilinear_SU2_SU3.end(); ) {
            int source = it->first;
            auto range = mixed_uc.bilinear_SU2_SU3.equal_range(source);
            num_bi_mixed = std::max(num_bi_mixed, static_cast<size_t>(std::distance(range.first, range.second)));
            it = range.second;
        }
        
        // Count trilinear interactions
        for (auto it = mixed_uc.trilinear_SU2_SU3.begin();
             it != mixed_uc.trilinear_SU2_SU3.end(); ) {
            int source = it->first;
            auto range = mixed_uc.trilinear_SU2_SU3.equal_range(source);
            num_tri_mixed = std::max(num_tri_mixed, static_cast<size_t>(std::distance(range.first, range.second)));
            it = range.second;
        }

        cout << "Building mixed interactions: max " << num_bi_mixed 
             << " bilinear, " << num_tri_mixed << " trilinear per site" << endl;

        // Build mixed bilinear interactions
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    // Process SU(2) sites as sources
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);
                        
                        auto bi_range = mixed_uc.bilinear_SU2_SU3.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            // Partner is in SU(3) sublattice
                            int pi = static_cast<int>(i) + bi.offset(0);
                            int pj = static_cast<int>(j) + bi.offset(1);
                            int pk = static_cast<int>(k) + bi.offset(2);
                            size_t partner_idx = flatten_index_periodic(pi, pj, pk, bi.partner, N_atoms_SU3);
                            
                            // bi.interaction is N_SU2 x N_SU3 (3x8)
                            // For SU2 energy: S2.dot(J * S3), need J to be N_SU2 x N_SU3 (3x8)
                            // For SU3 energy: S3.dot(J^T * S2), need J^T to be N_SU3 x N_SU2 (8x3)
                            mixed_bilinear_interaction_SU2[site_idx].push_back(bi.interaction);
                            mixed_bilinear_partners_SU2[site_idx].push_back(partner_idx);
                            
                            // Add transposed contribution to SU(3) side
                            mixed_bilinear_interaction_SU3[partner_idx].push_back(bi.interaction.transpose());
                            mixed_bilinear_partners_SU3[partner_idx].push_back(site_idx);
                        }
                    }
                }
            }
        }

        // Build pulse-modulated mixed bilinear interactions (field-assisted
        // Fe-Tm exchange H_{E chi}, H_{B chi}).  Identical geometry to the
        // static mixed bilinears, but stored in the parallel `*_drive_*`
        // tables together with the per-bond pulse-envelope selector.
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);

                        auto bi_range = mixed_uc.bilinear_drive_SU2_SU3.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            int pi = static_cast<int>(i) + bi.offset(0);
                            int pj = static_cast<int>(j) + bi.offset(1);
                            int pk = static_cast<int>(k) + bi.offset(2);
                            size_t partner_idx = flatten_index_periodic(pi, pj, pk, bi.partner, N_atoms_SU3);

                            mixed_bilinear_drive_interaction_SU2[site_idx].push_back(bi.interaction);
                            mixed_bilinear_drive_partners_SU2[site_idx].push_back(partner_idx);
                            mixed_bilinear_drive_envelope_SU2[site_idx].push_back(bi.envelope);

                            mixed_bilinear_drive_interaction_SU3[partner_idx].push_back(bi.interaction.transpose());
                            mixed_bilinear_drive_partners_SU3[partner_idx].push_back(site_idx);
                            mixed_bilinear_drive_envelope_SU3[partner_idx].push_back(bi.envelope);

                            has_mixed_bilinear_drive = true;
                        }
                    }
                }
            }
        }

        // Build mixed trilinear interactions
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    // Process SU(2) sites as sources
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);
                        
                        auto tri_range = mixed_uc.trilinear_SU2_SU3.equal_range(atom);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& tri = it->second;
                            
                            // First partner (SU2)
                            int p1i = static_cast<int>(i) + tri.offset1(0);
                            int p1j = static_cast<int>(j) + tri.offset1(1);
                            int p1k = static_cast<int>(k) + tri.offset1(2);
                            size_t partner1_idx = flatten_index_periodic(p1i, p1j, p1k, tri.partner1, N_atoms_SU2);
                            
                            // Second partner (SU3)
                            int p2i = static_cast<int>(i) + tri.offset2(0);
                            int p2j = static_cast<int>(j) + tri.offset2(1);
                            int p2k = static_cast<int>(k) + tri.offset2(2);
                            size_t partner2_idx = flatten_index_periodic(p2i, p2j, p2k, tri.partner2, N_atoms_SU3);
                            
                            // Original: K[a](b,c) for site[a] with SU2[b] and SU3[c]
                            // H = Σ_abc K[a](b,c) S_source^a S_partner1^b λ_partner2^c
                            mixed_trilinear_interaction_SU2[site_idx].push_back(tri.interaction);
                            mixed_trilinear_partners_SU2[site_idx].push_back({partner1_idx, partner2_idx});
                            
                            // Symmetric contribution to SU2 partner1: K_bac[b](a,c) = K[a](b,c)
                            // ∂H/∂S_partner1^b = Σ_ac K[a](b,c) S_source^a λ^c
                            // For partner1: T_p1[b] is (spin_dim_SU2 × spin_dim_SU3)
                            //   with T_p1[b](a,c) = K[a](b,c)
                            SpinTensor3 K_bac(spin_dim_SU2);
                            for (size_t b = 0; b < spin_dim_SU2; ++b) {
                                K_bac[b] = Eigen::MatrixXd(spin_dim_SU2, spin_dim_SU3);
                                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                                        K_bac[b](a, c) = tri.interaction[a](b, c);
                                    }
                                }
                            }
                            mixed_trilinear_interaction_SU2[partner1_idx].push_back(K_bac);
                            mixed_trilinear_partners_SU2[partner1_idx].push_back({site_idx, partner2_idx});
                            
                            // Symmetric contribution to SU3 site: K_cab[c](a,b) = K[a](b,c)
                            // ∂H/∂λ^c = Σ_ab K[a](b,c) S_source^a S_partner1^b
                            SpinTensor3 K_cab(spin_dim_SU3);
                            for (size_t c = 0; c < spin_dim_SU3; ++c) {
                                K_cab[c] = Eigen::MatrixXd(spin_dim_SU2, spin_dim_SU2);
                                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                                        K_cab[c](a, b) = tri.interaction[a](b, c);
                                    }
                                }
                            }
                            mixed_trilinear_interaction_SU3[partner2_idx].push_back(K_cab);
                            mixed_trilinear_partners_SU3[partner2_idx].push_back({site_idx, partner1_idx});
                        }
                    }
                }
            }
        }

        cout << "Mixed interactions built successfully!" << endl;
    }

    // ============================================================
    // PACKED INTERACTION BUFFER BUILDER
    // ============================================================
    /**
     * Pack the bilinear / trilinear / mixed interaction tensors into
     * contiguous row-major double[] buffers used by the MD and MC hot paths
     * (`get_local_field_*_flat_into`, `linear_field_*`), then rebuild the MC
     * local-form tables (build_mc_tables).
     *
     * Layout (per site, n indexes the bond):
     *   bilinear_packed_*[site][n*da*db + a*db + b]              = J^n(a,b)
     *   trilinear_packed_*[site][n*da*db*dc + (a*db + b)*dc + c] = T^n[a](b,c)
     *
     * The original `bilinear_interaction_*`, `trilinear_interaction_*`,
     * `mixed_*_interaction_*` storage is preserved (source of every rebuild). Only one rebuild call is required after the
     * full interaction graph is set; in our setup that is at the end of
     * the `MixedLattice` constructor (after `build_color_partition()`).
     *
     * Idempotent and safe to call again after the user mutates the
     * interaction tables (e.g. via `set_*` helpers); cost is O(total
     * coupling tensor entries), trivially ~ms even for large lattices.
     */
    void build_packed_interaction_buffers() {
        const size_t d2 = spin_dim_SU2;
        const size_t d3 = spin_dim_SU3;
        const size_t d22  = d2 * d2;
        const size_t d23  = d2 * d3;
        const size_t d33  = d3 * d3;
        const size_t d222 = d22 * d2;
        const size_t d223 = d22 * d3;   // SU2-SU2-SU3 mixed trilinear (from SU2 side)
        const size_t d322 = d3 * d22;   // SU3-SU2-SU2 mixed trilinear (from SU3 side)
        const size_t d333 = d33 * d3;

        bilinear_packed_SU2.assign(lattice_size_SU2, {});
        mixed_bilinear_packed_SU2.assign(lattice_size_SU2, {});
        trilinear_packed_SU2.assign(lattice_size_SU2, {});
        mixed_trilinear_packed_SU2.assign(lattice_size_SU2, {});

        bilinear_packed_SU3.assign(lattice_size_SU3, {});
        mixed_bilinear_packed_SU3.assign(lattice_size_SU3, {});
        trilinear_packed_SU3.assign(lattice_size_SU3, {});
        mixed_trilinear_packed_SU3.assign(lattice_size_SU3, {});

        mixed_bilinear_drive_packed_SU2.assign(lattice_size_SU2, {});
        mixed_bilinear_drive_packed_SU3.assign(lattice_size_SU3, {});

        // ---- SU(2) sublattice ----
        for (size_t site = 0; site < lattice_size_SU2; ++site) {
            // Bilinear SU(2)-SU(2): J(a,b)
            const size_t n_bi = bilinear_interaction_SU2[site].size();
            bilinear_packed_SU2[site].assign(n_bi * d22, 0.0);
            for (size_t n = 0; n < n_bi; ++n) {
                const auto& J = bilinear_interaction_SU2[site][n];
                double* p = bilinear_packed_SU2[site].data() + n * d22;
                for (size_t a = 0; a < d2; ++a)
                    for (size_t b = 0; b < d2; ++b)
                        p[a * d2 + b] = J(a, b);
            }
            // Mixed bilinear SU(2)-SU(3): J(a,c)
            const size_t n_mb = mixed_bilinear_interaction_SU2[site].size();
            mixed_bilinear_packed_SU2[site].assign(n_mb * d23, 0.0);
            for (size_t n = 0; n < n_mb; ++n) {
                const auto& J = mixed_bilinear_interaction_SU2[site][n];
                double* p = mixed_bilinear_packed_SU2[site].data() + n * d23;
                for (size_t a = 0; a < d2; ++a)
                    for (size_t c = 0; c < d3; ++c)
                        p[a * d3 + c] = J(a, c);
            }
            // Field-assisted (pulse-modulated) mixed bilinear SU(2)-SU(3): J(a,c)
            const size_t n_mbd = mixed_bilinear_drive_interaction_SU2[site].size();
            mixed_bilinear_drive_packed_SU2[site].assign(n_mbd * d23, 0.0);
            for (size_t n = 0; n < n_mbd; ++n) {
                const auto& J = mixed_bilinear_drive_interaction_SU2[site][n];
                double* p = mixed_bilinear_drive_packed_SU2[site].data() + n * d23;
                for (size_t a = 0; a < d2; ++a)
                    for (size_t c = 0; c < d3; ++c)
                        p[a * d3 + c] = J(a, c);
            }
            // Trilinear SU(2)-SU(2)-SU(2): T[a](b,c)
            const size_t n_tri = trilinear_interaction_SU2[site].size();
            trilinear_packed_SU2[site].assign(n_tri * d222, 0.0);
            for (size_t n = 0; n < n_tri; ++n) {
                const auto& T = trilinear_interaction_SU2[site][n];
                double* p = trilinear_packed_SU2[site].data() + n * d222;
                for (size_t a = 0; a < d2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d2; ++b)
                        for (size_t c = 0; c < d2; ++c)
                            p[(a * d2 + b) * d2 + c] = Ta(b, c);
                }
            }
            // Mixed trilinear SU(2)-SU(2)-SU(3): T[a](b,c)
            const size_t n_mtri = mixed_trilinear_interaction_SU2[site].size();
            mixed_trilinear_packed_SU2[site].assign(n_mtri * d223, 0.0);
            for (size_t n = 0; n < n_mtri; ++n) {
                const auto& T = mixed_trilinear_interaction_SU2[site][n];
                double* p = mixed_trilinear_packed_SU2[site].data() + n * d223;
                for (size_t a = 0; a < d2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d2; ++b)
                        for (size_t c = 0; c < d3; ++c)
                            p[(a * d2 + b) * d3 + c] = Ta(b, c);
                }
            }
        }

        // ---- SU(3) sublattice ----
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            // Bilinear SU(3)-SU(3): J(a,b)
            const size_t n_bi = bilinear_interaction_SU3[site].size();
            bilinear_packed_SU3[site].assign(n_bi * d33, 0.0);
            for (size_t n = 0; n < n_bi; ++n) {
                const auto& J = bilinear_interaction_SU3[site][n];
                double* p = bilinear_packed_SU3[site].data() + n * d33;
                for (size_t a = 0; a < d3; ++a)
                    for (size_t b = 0; b < d3; ++b)
                        p[a * d3 + b] = J(a, b);
            }
            // Mixed bilinear SU(3)-SU(2): J(a,b) with a in SU3, b in SU2
            const size_t n_mb = mixed_bilinear_interaction_SU3[site].size();
            mixed_bilinear_packed_SU3[site].assign(n_mb * d3 * d2, 0.0);
            for (size_t n = 0; n < n_mb; ++n) {
                const auto& J = mixed_bilinear_interaction_SU3[site][n];
                double* p = mixed_bilinear_packed_SU3[site].data() + n * d3 * d2;
                for (size_t a = 0; a < d3; ++a)
                    for (size_t b = 0; b < d2; ++b)
                        p[a * d2 + b] = J(a, b);
            }
            // Field-assisted (pulse-modulated) mixed bilinear SU(3)-SU(2): J(a,b)
            const size_t n_mbd = mixed_bilinear_drive_interaction_SU3[site].size();
            mixed_bilinear_drive_packed_SU3[site].assign(n_mbd * d3 * d2, 0.0);
            for (size_t n = 0; n < n_mbd; ++n) {
                const auto& J = mixed_bilinear_drive_interaction_SU3[site][n];
                double* p = mixed_bilinear_drive_packed_SU3[site].data() + n * d3 * d2;
                for (size_t a = 0; a < d3; ++a)
                    for (size_t b = 0; b < d2; ++b)
                        p[a * d2 + b] = J(a, b);
            }
            // Trilinear SU(3)-SU(3)-SU(3): T[a](b,c)
            const size_t n_tri = trilinear_interaction_SU3[site].size();
            trilinear_packed_SU3[site].assign(n_tri * d333, 0.0);
            for (size_t n = 0; n < n_tri; ++n) {
                const auto& T = trilinear_interaction_SU3[site][n];
                double* p = trilinear_packed_SU3[site].data() + n * d333;
                for (size_t a = 0; a < d3; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d3; ++b)
                        for (size_t c = 0; c < d3; ++c)
                            p[(a * d3 + b) * d3 + c] = Ta(b, c);
                }
            }
            // Mixed trilinear SU(3)-SU(2)-SU(2): T[a](b,c), a in SU3, b,c in SU2
            const size_t n_mtri = mixed_trilinear_interaction_SU3[site].size();
            mixed_trilinear_packed_SU3[site].assign(n_mtri * d322, 0.0);
            for (size_t n = 0; n < n_mtri; ++n) {
                const auto& T = mixed_trilinear_interaction_SU3[site][n];
                double* p = mixed_trilinear_packed_SU3[site].data() + n * d322;
                for (size_t a = 0; a < d3; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d2; ++b)
                        for (size_t c = 0; c < d2; ++c)
                            p[(a * d2 + b) * d2 + c] = Ta(b, c);
                }
            }
        }

        build_mc_tables();
    }

    /**
     * Build the per-sublattice colour partition used by metropolis_parallel /
     * overrelaxation_parallel.
     *
     * SU(2) edges considered:
     *   - SU(2)-SU(2) bilinear: bilinear_partners_SU2
     *   - SU(2)-SU(2)-SU(2) trilinear: both partners in trilinear_partners_SU2
     *   - intra-SU(2) pair inside SU(2)-SU(2)-SU(3) mixed trilinear:
     *     mixed_trilinear_partners_SU2[i][.][0] is the other SU(2) site
     * (mixed *bilinear* SU(2)-SU(3) does NOT add intra-SU(2) edges; the SU(3)
     * is read but not written during the SU(2) parallel pass.)
     *
     * SU(3) edges considered: mirror of the above.
     *
     * Algorithm: greedy first-fit on natural site order. Builds the CSR
     * sites_by_color_csr_off_SU{2,3} / sites_by_color_csr_SU{2,3} tables.
     */
    void build_color_partition() {
        auto greedy = [&](size_t N,
                          const std::vector<std::vector<size_t>>& edges,
                          std::vector<uint16_t>& color_of_site,
                          std::vector<size_t>& csr_off,
                          std::vector<size_t>& csr,
                          size_t& n_colors_out) {
            if (N == 0) {
                n_colors_out = 0;
                color_of_site.clear();
                csr_off.clear();
                csr.clear();
                return;
            }
            color_of_site.assign(N, std::numeric_limits<uint16_t>::max());
            std::vector<uint8_t> forbidden(64, 0);
            size_t max_color = 0;
            for (size_t i = 0; i < N; ++i) {
                std::fill(forbidden.begin(), forbidden.end(), uint8_t(0));
                for (size_t j : edges[i]) {
                    if (j >= N) continue;
                    const uint16_t c = color_of_site[j];
                    if (c == std::numeric_limits<uint16_t>::max()) continue;
                    if (size_t(c) >= forbidden.size()) forbidden.resize(size_t(c) + 1, 0);
                    forbidden[c] = 1;
                }
                uint16_t chosen = 0;
                while (size_t(chosen) < forbidden.size() && forbidden[chosen]) ++chosen;
                color_of_site[i] = chosen;
                if (size_t(chosen) > max_color) max_color = chosen;
            }
            n_colors_out = max_color + 1;
            csr_off.assign(n_colors_out + 1, 0);
            for (size_t i = 0; i < N; ++i) csr_off[color_of_site[i] + 1] += 1;
            for (size_t c = 0; c < n_colors_out; ++c) csr_off[c + 1] += csr_off[c];
            csr.assign(N, 0);
            std::vector<size_t> cursor(n_colors_out, 0);
            for (size_t i = 0; i < N; ++i) {
                const uint16_t c = color_of_site[i];
                csr[csr_off[c] + cursor[c]] = i;
                ++cursor[c];
            }
        };

        // Build undirected adjacency for SU(2) sublattice.
        std::vector<std::vector<size_t>> adj_SU2(lattice_size_SU2);
        auto add_edge_SU2 = [&](size_t a, size_t b) {
            if (a == b || a >= lattice_size_SU2 || b >= lattice_size_SU2) return;
            adj_SU2[a].push_back(b);
            adj_SU2[b].push_back(a);
        };
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j : bilinear_partners_SU2[i]) add_edge_SU2(i, j);
            for (const auto& pr : trilinear_partners_SU2[i]) {
                add_edge_SU2(i, pr[0]);
                add_edge_SU2(i, pr[1]);
                add_edge_SU2(pr[0], pr[1]);
            }
            for (const auto& pr : mixed_trilinear_partners_SU2[i]) {
                add_edge_SU2(i, pr[0]);  // pr[0] is another SU(2); pr[1] is SU(3)
            }
        }
        greedy(lattice_size_SU2, adj_SU2,
               color_of_site_SU2, sites_by_color_csr_off_SU2,
               sites_by_color_csr_SU2, n_colors_SU2);

        // Build undirected adjacency for SU(3) sublattice.
        std::vector<std::vector<size_t>> adj_SU3(lattice_size_SU3);
        auto add_edge_SU3 = [&](size_t a, size_t b) {
            if (a == b || a >= lattice_size_SU3 || b >= lattice_size_SU3) return;
            adj_SU3[a].push_back(b);
            adj_SU3[b].push_back(a);
        };
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j : bilinear_partners_SU3[i]) add_edge_SU3(i, j);
            for (const auto& pr : trilinear_partners_SU3[i]) {
                add_edge_SU3(i, pr[0]);
                add_edge_SU3(i, pr[1]);
                add_edge_SU3(pr[0], pr[1]);
            }
            // mixed_trilinear_partners_SU3 stores (SU(2), SU(2)) pairs from the
            // SU(3) site's perspective, so they create no intra-SU(3) edges.
        }
        greedy(lattice_size_SU3, adj_SU3,
               color_of_site_SU3, sites_by_color_csr_off_SU3,
               sites_by_color_csr_SU3, n_colors_SU3);
    }

    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================

    /**
     * Build the Monte Carlo local-form tables (see SelfQuadTerm): symmetrised
     * on-site matrices, the self-coupled trilinear entries merged per partner,
     * and the per-site self-energy class. Called by
     * build_packed_interaction_buffers() after every change of the coupling
     * tables.
     */
    void build_mc_tables() {
        build_mc_tables_species(false);
        build_mc_tables_species(true);
    }

private:
    void build_mc_tables_species(bool su3) {
        const size_t N = su3 ? lattice_size_SU3 : lattice_size_SU2;
        const size_t d = su3 ? spin_dim_SU3 : spin_dim_SU2;
        const auto& onsite = su3 ? onsite_interaction_SU3 : onsite_interaction_SU2;
        const auto& tri = su3 ? trilinear_interaction_SU3 : trilinear_interaction_SU2;
        const auto& tri_p = su3 ? trilinear_partners_SU3 : trilinear_partners_SU2;
        auto& osym = su3 ? onsite_sym_SU3 : onsite_sym_SU2;
        auto& off = su3 ? selfq_off_SU3 : selfq_off_SU2;
        auto& terms = su3 ? selfq_SU3 : selfq_SU2;
        auto& coef = su3 ? selfq_coef_SU3 : selfq_coef_SU2;
        auto& cubic = su3 ? cubic_self_SU3 : cubic_self_SU2;
        auto& mode = su3 ? self_mode_SU3 : self_mode_SU2;
        auto& iso = su3 ? self_isotropic_SU3 : self_isotropic_SU2;

        osym.assign(N * d * d, 0.0);
        off.assign(N + 1, 0);
        terms.clear();
        coef.clear();
        cubic.assign(N, {});
        mode.assign(N, 0);
        iso.assign(N, 1);

        for (size_t i = 0; i < N; ++i) {
            // On-site matrix: symmetric part, and whether it is a multiple of
            // the identity (then S^T A S is constant on the manifold).
            double* O = &osym[i * d * d];
            double scale = 0.0, trace = 0.0;
            for (size_t a = 0; a < d; ++a)
                for (size_t b = 0; b < d; ++b) {
                    O[a * d + b] = 0.5 * (onsite[i](a, b) + onsite[i](b, a));
                    scale = std::max(scale, std::abs(O[a * d + b]));
                }
            for (size_t a = 0; a < d; ++a) trace += O[a * d + a];
            trace /= double(d);
            bool onsite_scalar = true;
            for (size_t a = 0; a < d; ++a)
                for (size_t b = 0; b < d; ++b)
                    if (std::abs(O[a * d + b] - (a == b ? trace : 0.0)) > 1e-14 * scale) onsite_scalar = false;

            // Self-coupled entries, merged per (species, partner). Each entry
            // with one partner slot equal to i is quadratic in S_i with weight
            // 1/2; Q[a][b][c] multiplies S_a S_b x_c, symmetrised in (a, b) and
            // stored for a <= b only (row upper_index(a, b, d)).
            const size_t first = terms.size();
            const size_t n_rows = d * (d + 1) / 2;
            auto term_offset = [&](bool partner_su3, size_t partner) -> size_t {
                for (size_t t = first; t < terms.size(); ++t)
                    if (terms[t].partner == partner && bool(terms[t].partner_su3) == partner_su3)
                        return terms[t].offset;
                const size_t dim = partner_su3 ? spin_dim_SU3 : spin_dim_SU2;
                terms.push_back({partner, coef.size(), uint8_t(partner_su3), uint8_t(dim)});
                coef.resize(coef.size() + n_rows * dim, 0.0);
                return terms.back().offset;
            };
            for (size_t n = 0; n < tri[i].size(); ++n) {
                const size_t p1 = tri_p[i][n][0], p2 = tri_p[i][n][1];
                const bool s1 = (p1 == i), s2 = (p2 == i);
                if (s1 && s2) { cubic[i].push_back(uint32_t(n)); continue; }
                if (!s1 && !s2) continue;
                const auto& T = tri[i][n];   // T[a](b, c): slots (i, p1, p2)
                const size_t q = term_offset(su3, s1 ? p2 : p1);
                for (size_t a = 0; a < d; ++a)
                    for (size_t b = a; b < d; ++b)
                        for (size_t c = 0; c < d; ++c) {
                            const double v_ab = s1 ? T[a](b, c) : T[a](c, b);
                            const double v_ba = s1 ? T[b](a, c) : T[b](c, a);
                            coef[q + upper_index(a, b, d) * d + c] += 0.25 * (v_ab + v_ba);
                        }
            }
            if (!su3) {
                // SU(2)-SU(2)-SU(3): only the SU(2) partner can coincide with i.
                for (size_t n = 0; n < mixed_trilinear_partners_SU2[i].size(); ++n) {
                    if (mixed_trilinear_partners_SU2[i][n][0] != i) continue;
                    const auto& K = mixed_trilinear_interaction_SU2[i][n];   // K[a](b, c), c in SU(3)
                    const size_t q = term_offset(true, mixed_trilinear_partners_SU2[i][n][1]);
                    for (size_t a = 0; a < d; ++a)
                        for (size_t b = a; b < d; ++b)
                            for (size_t c = 0; c < spin_dim_SU3; ++c)
                                coef[q + upper_index(a, b, d) * spin_dim_SU3 + c] += 0.25 * (K[a](b, c) + K[b](a, c));
                }
            }
            off[i + 1] = terms.size();
            const bool has_self = terms.size() > first || !cubic[i].empty();
            mode[i] = !cubic[i].empty() ? 2 : ((has_self || scale > 0.0) ? 1 : 0);
            iso[i] = (!has_self && onsite_scalar) ? 1 : 0;
        }
    }

    // Row of the pair a <= b in a row-major upper triangle of a d x d matrix.
    static constexpr size_t upper_index(size_t a, size_t b, size_t d) { return a * (2 * d - a - 1) / 2 + b; }

    // Symmetric D x D matrix from its upper triangle (row-major, a <= b).
    template <int D>
    static inline void fill_symmetric(const double* __restrict up, double* __restrict A) {
        int r = 0;
        for (int a = 0; a < D; ++a)
            for (int b = a; b < D; ++b, ++r) A[a * D + b] = A[b * D + a] = up[r];
    }

    // h[a] += sum_b J[a DB + b] x[b]  (row-major packed bilinear entry).
    // Two partial sums per row halve the dependent add chain (the kernels are
    // latency bound at these sizes; ~20 % faster than one accumulator).
    template <int DA, int DB>
    static inline void contract_bilinear(const double* __restrict J, const double* __restrict x,
                                         double* __restrict h) {
        for (int a = 0; a < DA; ++a) {
            const double* row = J + a * DB;
            double s0 = 0.0, s1 = 0.0;
            int b = 0;
            for (; b + 1 < DB; b += 2) {
                s0 += row[b] * x[b];
                s1 += row[b + 1] * x[b + 1];
            }
            if (b < DB) s0 += row[b] * x[b];
            h[a] += s0 + s1;
        }
    }

    // h[a] += sum_bc T[(a DB + b) DC + c] x[b] y[c]  (row-major packed trilinear
    // entry): the outer product x y^T, then one DA x (DB DC) matrix-vector product.
    template <int DA, int DB, int DC>
    static inline void contract_trilinear(const double* __restrict T, const double* __restrict x,
                                          const double* __restrict y, double* __restrict h) {
        double xy[DB * DC];
        for (int b = 0; b < DB; ++b)
            for (int c = 0; c < DC; ++c) xy[b * DC + c] = x[b] * y[c];
        contract_bilinear<DA, DB * DC>(T, xy, h);
    }

    template <int D>
    static inline double quadratic_form(const double* __restrict A, const double* __restrict S) {
        double e = 0.0;
        for (int a = 0; a < D; ++a) {
            double row = 0.0;
            for (int b = 0; b < D; ++b) row += A[a * D + b] * S[b];
            e += S[a] * row;
        }
        return e;
    }

public:
    /**
     * Linear coefficient h_i of the local energy of SU(2) site i (every term
     * that contains S_i exactly once): the local energy is
     * h_i . S + S^T A_i S (+ cubic self terms). Writes h[0..2]. Excludes the
     * on-site and self-coupled terms, so unlike get_local_field_SU2 (the full
     * gradient dE/dS) it does not depend on S_i itself.
     */
    inline void linear_field_SU2(size_t i, double* __restrict h) const {
        const double* B = field_SU2[i].data();
        h[0] = -B[0]; h[1] = -B[1]; h[2] = -B[2];
        const auto& bp = bilinear_partners_SU2[i];
        if (!bp.empty()) {
            const double* J = bilinear_packed_SU2[i].data();
            for (size_t n = 0; n < bp.size(); ++n) contract_bilinear<3, 3>(J + 9 * n, spins_SU2[bp[n]].data(), h);
        }
        const auto& mbp = mixed_bilinear_partners_SU2[i];
        if (!mbp.empty()) {
            const double* J = mixed_bilinear_packed_SU2[i].data();
            for (size_t n = 0; n < mbp.size(); ++n) contract_bilinear<3, 8>(J + 24 * n, spins_SU3[mbp[n]].data(), h);
        }
        const auto& tp = trilinear_partners_SU2[i];
        if (!tp.empty()) {
            const double* T = trilinear_packed_SU2[i].data();
            for (size_t n = 0; n < tp.size(); ++n) {
                if (tp[n][0] == i || tp[n][1] == i) continue;   // self-coupled: in A_i
                contract_trilinear<3, 3, 3>(T + 27 * n, spins_SU2[tp[n][0]].data(), spins_SU2[tp[n][1]].data(), h);
            }
        }
        const auto& mtp = mixed_trilinear_partners_SU2[i];
        if (!mtp.empty()) {
            const double* T = mixed_trilinear_packed_SU2[i].data();
            for (size_t n = 0; n < mtp.size(); ++n) {
                if (mtp[n][0] == i) continue;                   // W(S_i, S_i, n_k): in A_i
                contract_trilinear<3, 3, 8>(T + 72 * n, spins_SU2[mtp[n][0]].data(), spins_SU3[mtp[n][1]].data(), h);
            }
        }
    }

    /// SU(3) analogue of linear_field_SU2; writes h[0..7] (h . n is the linear part of the local energy).
    inline void linear_field_SU3(size_t j, double* __restrict h) const {
        const double* B = field_SU3[j].data();
        for (int a = 0; a < 8; ++a) h[a] = -B[a];
        const auto& bp = bilinear_partners_SU3[j];
        if (!bp.empty()) {
            const double* J = bilinear_packed_SU3[j].data();
            for (size_t n = 0; n < bp.size(); ++n) contract_bilinear<8, 8>(J + 64 * n, spins_SU3[bp[n]].data(), h);
        }
        const auto& mbp = mixed_bilinear_partners_SU3[j];
        if (!mbp.empty()) {
            const double* J = mixed_bilinear_packed_SU3[j].data();
            for (size_t n = 0; n < mbp.size(); ++n) contract_bilinear<8, 3>(J + 24 * n, spins_SU2[mbp[n]].data(), h);
        }
        const auto& tp = trilinear_partners_SU3[j];
        if (!tp.empty()) {
            const double* T = trilinear_packed_SU3[j].data();
            for (size_t n = 0; n < tp.size(); ++n) {
                if (tp[n][0] == j || tp[n][1] == j) continue;
                contract_trilinear<8, 8, 8>(T + 512 * n, spins_SU3[tp[n][0]].data(), spins_SU3[tp[n][1]].data(), h);
            }
        }
        const auto& mtp = mixed_trilinear_partners_SU3[j];
        if (!mtp.empty()) {   // partners are SU(2) sites: never self-coupled
            const double* T = mixed_trilinear_packed_SU3[j].data();
            for (size_t n = 0; n < mtp.size(); ++n)
                contract_trilinear<8, 3, 3>(T + 72 * n, spins_SU2[mtp[n][0]].data(), spins_SU2[mtp[n][1]].data(), h);
        }
    }

    /// Quadratic self form A_i of SU(2) site i (row-major 3x3, symmetric): sym(onsite) plus the
    /// self-coupled trilinear entries with their partner contracted.
    inline void self_form_SU2(size_t i, double* __restrict A) const {
        const double* O = &onsite_sym_SU2[i * 9];
        double up[6] = {O[0], O[1], O[2], O[4], O[5], O[8]};
        for (size_t t = selfq_off_SU2[i]; t < selfq_off_SU2[i + 1]; ++t) {
            const SelfQuadTerm& q = selfq_SU2[t];
            const double* x = q.partner_su3 ? spins_SU3[q.partner].data() : spins_SU2[q.partner].data();
            if (q.dim == 8) contract_bilinear<6, 8>(&selfq_coef_SU2[q.offset], x, up);
            else contract_bilinear<6, 3>(&selfq_coef_SU2[q.offset], x, up);
        }
        fill_symmetric<3>(up, A);
    }

    /// Quadratic self form A_j of SU(3) site j (row-major 8x8, symmetric).
    inline void self_form_SU3(size_t j, double* __restrict A) const {
        const double* O = &onsite_sym_SU3[j * 64];
        double up[36];
        for (int a = 0, r = 0; a < 8; ++a)
            for (int b = a; b < 8; ++b, ++r) up[r] = O[a * 8 + b];
        for (size_t t = selfq_off_SU3[j]; t < selfq_off_SU3[j + 1]; ++t) {
            const SelfQuadTerm& q = selfq_SU3[t];
            contract_bilinear<36, 8>(&selfq_coef_SU3[q.offset], spins_SU3[q.partner].data(), up);
        }
        fill_symmetric<8>(up, A);
    }

    /// Self energy S^T A S of SU(2) site i at S, plus its cubic self entries (weight 1/3 each).
    inline double self_energy_SU2(size_t i, const double* A, const double* S) const {
        double e = quadratic_form<3>(A, S);
        if (self_mode_SU2[i] == 2) {
            for (uint32_t n : cubic_self_SU2[i]) {
                double g[3] = {0.0, 0.0, 0.0};
                contract_trilinear<3, 3, 3>(trilinear_packed_SU2[i].data() + 27 * n, S, S, g);
                e += (g[0] * S[0] + g[1] * S[1] + g[2] * S[2]) / 3.0;
            }
        }
        return e;
    }

    /// Self energy n^T A n of SU(3) site j at n, plus its cubic self entries.
    inline double self_energy_SU3(size_t j, const double* A, const double* n) const {
        double e = quadratic_form<8>(A, n);
        if (self_mode_SU3[j] == 2) {
            for (uint32_t m : cubic_self_SU3[j]) {
                double g[8] = {};
                contract_trilinear<8, 8, 8>(trilinear_packed_SU3[j].data() + 512 * m, n, n, g);
                double s = 0.0;
                for (int a = 0; a < 8; ++a) s += g[a] * n[a];
                e += s / 3.0;
            }
        }
        return e;
    }

    /// Exact local energy change of SU(2) site i from So to Sn, given its linear field h.
    inline double local_energy_change_SU2(size_t i, const double* h, const double* Sn, const double* So) const {
        double dE = h[0] * (Sn[0] - So[0]) + h[1] * (Sn[1] - So[1]) + h[2] * (Sn[2] - So[2]);
        if (self_mode_SU2[i]) {
            double A[9];
            self_form_SU2(i, A);
            dE += self_energy_SU2(i, A, Sn) - self_energy_SU2(i, A, So);
        }
        return dE;
    }

    /// Exact local energy change of SU(3) site j from no to nn, given its linear field h.
    inline double local_energy_change_SU3(size_t j, const double* h, const double* nn, const double* no) const {
        double dE = 0.0;
        for (int a = 0; a < 8; ++a) dE += h[a] * (nn[a] - no[a]);
        if (self_mode_SU3[j]) {
            double A[64];
            self_form_SU3(j, A);
            dE += self_energy_SU3(j, A, nn) - self_energy_SU3(j, A, no);
        }
        return dE;
    }

    /**
     * Change of total_energy() when SU(2) site `site_index` goes from
     * old_spin to new_spin with every other spin fixed. Exact for every
     * coupling, including on-site anisotropy, self-bonds (folded into the
     * on-site matrix) and trilinear terms with S_i in more than one slot;
     * new_spin and old_spin need not have the same length.
     */
    double site_energy_SU2_diff(const SpinVector& new_spin, const SpinVector& old_spin, size_t site_index) const {
        double h[3];
        linear_field_SU2(site_index, h);
        return local_energy_change_SU2(site_index, h, new_spin.data(), old_spin.data());
    }

    /**
     * Change of total_energy() when SU(3) site `site_index` goes from
     * old_spin to new_spin (see site_energy_SU2_diff).
     */
    double site_energy_SU3_diff(const SpinVector& new_spin, const SpinVector& old_spin, size_t site_index) const {
        double h[8];
        linear_field_SU3(site_index, h);
        return local_energy_change_SU3(site_index, h, new_spin.data(), old_spin.data());
    }

    /**
     * Compute total energy of the system as sum of SU2 and SU3 contributions
     */
    double total_energy() const {
        return total_energy_SU2() + total_energy_SU3();
    }

    /**
     * Compute energy density (energy per site)
     */
    double energy_density() const {
        return total_energy() / (lattice_size_SU2 + lattice_size_SU3);
    }

    /**
     * Compute energy density for SU(2) sector (energy per SU2 site)
     */
    double energy_density_SU2() const {
        return total_energy_SU2() / lattice_size_SU2;
    }

    /**
     * Compute energy density for SU(3) sector (energy per SU3 site)
     */
    double energy_density_SU3() const {
        return total_energy_SU3() / lattice_size_SU3;
    }

    /**
     * Compute total energy of the SU(2) sublattice only
     * Includes SU2-SU2 interactions, SU2 field/onsite, half of each mixed
     * bilinear and one third of each mixed-trilinear entry stored at SU(2)
     * sites (only the sum total_energy() is convention-free).
     */
    double total_energy_SU2() const {
        double energy = 0.0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const double* S = spins_SU2[i].data();
            const double* B = field_SU2[i].data();
            double half[3] = {0.0, 0.0, 0.0}, third[3] = {0.0, 0.0, 0.0};
            const auto& bp = bilinear_partners_SU2[i];
            for (size_t n = 0; n < bp.size(); ++n)
                contract_bilinear<3, 3>(bilinear_packed_SU2[i].data() + 9 * n, spins_SU2[bp[n]].data(), half);
            const auto& mbp = mixed_bilinear_partners_SU2[i];
            for (size_t n = 0; n < mbp.size(); ++n)
                contract_bilinear<3, 8>(mixed_bilinear_packed_SU2[i].data() + 24 * n, spins_SU3[mbp[n]].data(), half);
            const auto& tp = trilinear_partners_SU2[i];
            for (size_t n = 0; n < tp.size(); ++n)
                contract_trilinear<3, 3, 3>(trilinear_packed_SU2[i].data() + 27 * n, spins_SU2[tp[n][0]].data(),
                                            spins_SU2[tp[n][1]].data(), third);
            const auto& mtp = mixed_trilinear_partners_SU2[i];
            for (size_t n = 0; n < mtp.size(); ++n)
                contract_trilinear<3, 3, 8>(mixed_trilinear_packed_SU2[i].data() + 72 * n, spins_SU2[mtp[n][0]].data(),
                                            spins_SU3[mtp[n][1]].data(), third);
            double e = quadratic_form<3>(&onsite_sym_SU2[i * 9], S);
            for (int a = 0; a < 3; ++a) e += S[a] * (-B[a] + 0.5 * half[a] + third[a] / 3.0);
            energy += e;
        }
        return energy;
    }

    /**
     * Compute total energy of the SU(3) sublattice only
     * Includes SU3-SU3 interactions, SU3 field/onsite, and the remaining half
     * (mixed bilinear) / third (mixed trilinear) of the mixed terms.
     */
    double total_energy_SU3() const {
        double energy = 0.0;
        for (size_t j = 0; j < lattice_size_SU3; ++j) {
            const double* n3 = spins_SU3[j].data();
            const double* B = field_SU3[j].data();
            double half[8] = {}, third[8] = {};
            const auto& bp = bilinear_partners_SU3[j];
            for (size_t n = 0; n < bp.size(); ++n)
                contract_bilinear<8, 8>(bilinear_packed_SU3[j].data() + 64 * n, spins_SU3[bp[n]].data(), half);
            const auto& mbp = mixed_bilinear_partners_SU3[j];
            for (size_t n = 0; n < mbp.size(); ++n)
                contract_bilinear<8, 3>(mixed_bilinear_packed_SU3[j].data() + 24 * n, spins_SU2[mbp[n]].data(), half);
            const auto& tp = trilinear_partners_SU3[j];
            for (size_t n = 0; n < tp.size(); ++n)
                contract_trilinear<8, 8, 8>(trilinear_packed_SU3[j].data() + 512 * n, spins_SU3[tp[n][0]].data(),
                                            spins_SU3[tp[n][1]].data(), third);
            const auto& mtp = mixed_trilinear_partners_SU3[j];
            for (size_t n = 0; n < mtp.size(); ++n)
                contract_trilinear<8, 3, 3>(mixed_trilinear_packed_SU3[j].data() + 72 * n, spins_SU2[mtp[n][0]].data(),
                                            spins_SU2[mtp[n][1]].data(), third);
            double e = quadratic_form<8>(&onsite_sym_SU3[j * 64], n3);
            for (int a = 0; a < 8; ++a) e += n3[a] * (-B[a] + 0.5 * half[a] + third[a] / 3.0);
            energy += e;
        }
        return energy;
    }

    /**
     * Compute total energy directly from flat state array (zero-allocation version)
     * State layout: [SU2_site0_components... SU2_siteN... SU3_site0_components... SU3_siteM...]
     * Includes all interaction terms with proper double-counting avoidance
     */
    double total_energy_flat(const double* state_flat) const {
        double energy = 0.0;
        const size_t offset_SU3 = lattice_size_SU2 * spin_dim_SU2;
        
        // SU(2) contributions
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const double* spin = &state_flat[i * spin_dim_SU2];
            
            // Field
            for (size_t d = 0; d < spin_dim_SU2; ++d) {
                energy -= spin[d] * field_SU2[i](d);
            }
            
            // Onsite
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    energy += spin[a] * onsite_interaction_SU2[i](a, b) * spin[b];
                }
            }
            
            // Bilinear (half-counted)
            for (size_t j = 0; j < bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU2[i][j];
                const double* partner_spin = &state_flat[partner * spin_dim_SU2];
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        energy += 0.5 * spin[a] * bilinear_interaction_SU2[i][j](a, b) * partner_spin[b];
                    }
                }
            }
            
            // Mixed bilinear (half-counted)
            for (size_t j = 0; j < mixed_bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU2[i][j];
                const double* partner_spin = &state_flat[offset_SU3 + partner * spin_dim_SU3];
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        energy += 0.5 * spin[a] * mixed_bilinear_interaction_SU2[i][j](a, b) * partner_spin[b];
                    }
                }
            }

            // Mixed trilinear
            for (size_t j = 0; j < mixed_trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU2[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU2[i][j][1];
                const double* spin1 = &state_flat[p1 * spin_dim_SU2];
                const double* spin2 = &state_flat[offset_SU3 + p2 * spin_dim_SU3];
                const auto& T = mixed_trilinear_interaction_SU2[i][j];

                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0 / 3.0) * spin[a] * temp;
                }
            }
            
            // Trilinear
            for (size_t j = 0; j < trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU2[i][j][0];
                const size_t p2 = trilinear_partners_SU2[i][j][1];
                const double* spin1 = &state_flat[p1 * spin_dim_SU2];
                const double* spin2 = &state_flat[p2 * spin_dim_SU2];
                const auto& T = trilinear_interaction_SU2[i][j];
                
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0/3.0) * spin[a] * temp;
                }
            }
        }
        
        // SU(3) contributions
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            const double* spin = &state_flat[offset_SU3 + i * spin_dim_SU3];
            
            // Field
            for (size_t d = 0; d < spin_dim_SU3; ++d) {
                energy -= spin[d] * field_SU3[i](d);
            }
            
            // Onsite
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    energy += spin[a] * onsite_interaction_SU3[i](a, b) * spin[b];
                }
            }
            
            // Bilinear (half-counted)
            for (size_t j = 0; j < bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU3[i][j];
                const double* partner_spin = &state_flat[offset_SU3 + partner * spin_dim_SU3];
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        energy += 0.5 * spin[a] * bilinear_interaction_SU3[i][j](a, b) * partner_spin[b];
                    }
                }
            }
            
            // Mixed bilinear (half-counted)
            for (size_t j = 0; j < mixed_bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU3[i][j];
                const double* partner_spin = &state_flat[partner * spin_dim_SU2];
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        energy += 0.5 * spin[a] * mixed_bilinear_interaction_SU3[i][j](a, b) * partner_spin[b];
                    }
                }
            }

            // Mixed trilinear
            for (size_t j = 0; j < mixed_trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU3[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU3[i][j][1];
                const double* spin1 = &state_flat[p1 * spin_dim_SU2];
                const double* spin2 = &state_flat[p2 * spin_dim_SU2];
                const auto& T = mixed_trilinear_interaction_SU3[i][j];

                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0 / 3.0) * spin[a] * temp;
                }
            }
            
            // Trilinear
            for (size_t j = 0; j < trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU3[i][j][0];
                const size_t p2 = trilinear_partners_SU3[i][j][1];
                const double* spin1 = &state_flat[offset_SU3 + p1 * spin_dim_SU3];
                const double* spin2 = &state_flat[offset_SU3 + p2 * spin_dim_SU3];
                const auto& T = trilinear_interaction_SU3[i][j];
                
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0/3.0) * spin[a] * temp;
                }
            }
        }
        
        return energy;
    }

    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================
    //
    // Every sweep visits each site exactly once in a fixed order: natural
    // order (SU(2) sites, then SU(3) sites), unit cell by unit cell for the
    // *_interleaved variants (so the two species see each other's updates
    // within the sweep), or colour by colour in the OpenMP kernels. Each
    // single-site update leaves the Boltzmann distribution invariant, hence
    // so does any fixed scan (balance); the previous random-site selection
    // with replacement left ~37 % of the sites untouched per sweep. The site
    // kernels are allocation-free: proposals live in stack arrays, couplings
    // are read from the packed buffers (local forms above).
    //
    // SU(3) sites follow su3_mc_manifold. On CP^2 every update writes
    // n = <psi|lambda|psi> of a normalised psi, so |n|^2 = 4/3 and the cubic
    // Casimir 8/9 hold to round-off after any number of moves; psi is
    // recovered from the stored n without an eigensolver
    // (su3::psi_from_pure_expectations).

    /// Symmetric SU(2) proposal: uniform on the sphere, or S + sigma L u (u uniform) renormalised.
    inline void propose_SU2(const double* S, bool gaussian, double sigma, double* out) const {
        const double L = double(spin_length_SU2);
        if (!gaussian) { random_point_on_sphere(out, 3, L); return; }
        double u[3];
        random_point_on_sphere(u, 3, L);
        double s2 = 0.0;
        for (int d = 0; d < 3; ++d) { out[d] = S[d] + sigma * u[d]; s2 += out[d] * out[d]; }
        if (s2 < 1e-300) { for (int d = 0; d < 3; ++d) out[d] = S[d]; return; }   // measure zero
        const double f = L / std::sqrt(s2);
        for (int d = 0; d < 3; ++d) out[d] *= f;
    }

    /**
     * Symmetric SU(3) proposal on the configured manifold.
     * CP^2: psi' = normalise(z) (Haar / Fubini-Study) or
     * psi' = normalise(psi + sigma z) with z complex Gaussian: the kernel is
     * invariant under the U(2) x U(1) stabiliser of [psi] (and z -> e^{ia} z),
     * so its density depends only on |<psi|psi'>| and is symmetric.
     * Sphere: uniform on |n| = spin_length_SU3, or n + sigma L u renormalised.
     */
    inline void propose_SU3(const double* n, bool gaussian, double sigma, double* out) const {
        if (su3_on_cp2()) {
            if (gaussian) classical_spin::su3::propose_cp2(n, sigma, out);
            else classical_spin::su3::random_cp2(out);
            return;
        }
        const double L = double(spin_length_SU3);
        if (!gaussian) { random_point_on_sphere(out, 8, L); return; }
        double u[8];
        random_point_on_sphere(u, 8, L);
        double s2 = 0.0;
        for (int a = 0; a < 8; ++a) { out[a] = n[a] + sigma * u[a]; s2 += out[a] * out[a]; }
        if (s2 < 1e-300) { for (int a = 0; a < 8; ++a) out[a] = n[a]; return; }
        const double f = L / std::sqrt(s2);
        for (int a = 0; a < 8; ++a) out[a] *= f;
    }

    /// Uniformly random state of one SU(3) site on the configured manifold, written to n[0..7].
    void random_SU3_state(double* n) const { propose_SU3(n, false, 0.0, n); }

    /**
     * Exact draw from P(S) ∝ exp(-beta h.S) on the sphere |S| = L:
     * u = cos(S, -h) has density ∝ exp(b u) on [-1, 1], b = beta |h| L,
     * u = 1 + log1p(xi expm1(-2b)) / b, azimuth uniform.
     */
    static void sample_linear_sphere(const double* h, double beta, double L, double* out) {
        const double hn = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
        const double b = beta * hn * L;
        if (b < 1e-12) { random_point_on_sphere(out, 3, L); return; }
        const double xi = random_double_lehman(0.0, 1.0);
        const double u = std::clamp(1.0 + std::log1p(xi * std::expm1(-2.0 * b)) / b, -1.0, 1.0);
        const double phi = random_double_lehman(0.0, 2.0 * M_PI);
        const double e[3] = {-h[0] / hn, -h[1] / hn, -h[2] / hn};
        double e1[3];
        if (std::abs(e[0]) < 0.9) { e1[0] = 0.0; e1[1] = -e[2]; e1[2] = e[1]; }
        else                      { e1[0] = e[2]; e1[1] = 0.0; e1[2] = -e[0]; }
        const double n1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
        for (double& c : e1) c /= n1;
        const double e2[3] = {e[1] * e1[2] - e[2] * e1[1], e[2] * e1[0] - e[0] * e1[2],
                              e[0] * e1[1] - e[1] * e1[0]};
        const double r = std::sqrt(std::max(0.0, 1.0 - u * u));
        const double c = std::cos(phi), s = std::sin(phi);
        for (int d = 0; d < 3; ++d) out[d] = L * (u * e[d] + r * (c * e1[d] + s * e2[d]));
    }

    /**
     * Exact draw from P(psi) ∝ exp(-beta <psi|H|psi>) on CP^2 with the
     * Fubini-Study measure, H = h.lambda. In the eigenbasis H v_k = e_k v_k
     * (e_0 <= e_1 <= e_2), psi = sum_k sqrt(p_k) e^{i phi_k} v_k; the
     * Fubini-Study measure is uniform in the populations p on the simplex
     * and in the phases (moment map of the torus action, Duistermaat-
     * Heckman), and the energy is sum_k p_k e_k. With a = beta(e_1 - e_0),
     * b = beta(e_2 - e_0): p_2 has the marginal ∝ e^{-b p_2}(1 - e^{-a(1-p_2)}),
     * drawn from the truncated exponential e^{-b p_2} and accepted with
     * (1 - e^{-a(1-p_2)}) / (1 - e^{-a}) (>= 1/2 on average); then p_1 | p_2
     * is a truncated exponential on [0, 1 - p_2]. Writes n[0..7].
     */
    static void sample_linear_cp2(const double* h, double beta, double* n) {
        classical_spin::su3::sample_linear_cp2(h, beta, n);
    }

    // ---- single-site kernels (return true if the site changed) ----

    /// Metropolis update of SU(2) site i (uniform or Gaussian proposal).
    inline bool metropolis_site_SU2(size_t i, double beta, bool gaussian, double sigma) {
        double* S = spins_SU2[i].data();
        double Sn[3], h[3];
        propose_SU2(S, gaussian, sigma, Sn);
        linear_field_SU2(i, h);
        const double dE = local_energy_change_SU2(i, h, Sn, S);
        if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) return false;
        S[0] = Sn[0]; S[1] = Sn[1]; S[2] = Sn[2];
        return true;
    }

    /// Metropolis update of SU(3) site j on the configured manifold.
    inline bool metropolis_site_SU3(size_t j, double beta, bool gaussian, double sigma) {
        double* n = spins_SU3[j].data();
        double nn[8], h[8];
        propose_SU3(n, gaussian, sigma, nn);
        linear_field_SU3(j, h);
        const double dE = local_energy_change_SU3(j, h, nn, n);
        if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) return false;
        std::memcpy(n, nn, sizeof(nn));
        return true;
    }

    /**
     * Heat-bath update of SU(2) site i: exact draw from exp(-beta h.S); the
     * self energy S^T A S (+ cubic) is then accepted with
     * min(1, exp(-beta ΔE_self)) (independence Metropolis-Hastings with the
     * linear heat bath as proposal), skipped when it is constant on the sphere.
     */
    inline bool heat_bath_site_SU2(size_t i, double beta) {
        double* S = spins_SU2[i].data();
        double Sn[3], h[3];
        linear_field_SU2(i, h);
        sample_linear_sphere(h, beta, double(spin_length_SU2), Sn);
        if (!self_isotropic_SU2[i]) {
            double A[9];
            self_form_SU2(i, A);
            const double dE = self_energy_SU2(i, A, Sn) - self_energy_SU2(i, A, S);
            if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) return false;
        }
        S[0] = Sn[0]; S[1] = Sn[1]; S[2] = Sn[2];
        return true;
    }

    /// Heat-bath update of SU(3) site j on CP^2 (sample_linear_cp2, Metropolis-corrected self energy).
    inline bool heat_bath_site_SU3(size_t j, double beta) {
        double* n = spins_SU3[j].data();
        double nn[8], h[8];
        linear_field_SU3(j, h);
        sample_linear_cp2(h, beta, nn);
        if (!self_isotropic_SU3[j]) {
            double A[64];
            self_form_SU3(j, A);
            const double dE = self_energy_SU3(j, A, nn) - self_energy_SU3(j, A, n);
            if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) return false;
        }
        std::memcpy(n, nn, sizeof(nn));
        return true;
    }

    /**
     * Overrelaxation of SU(2) site i: reflect S about its linear field h
     * (which does not depend on S), S' = 2 (S.h) h / |h|^2 - S. The reflection
     * is an isometric involution of the sphere and conserves h.S, so it is an
     * exact microcanonical move when the self energy is constant on the
     * sphere. Otherwise (single-ion anisotropy, self-coupled W) it is a
     * symmetric proposal accepted with min(1, exp(-ΔE_self / T)) for T > 0 and
     * skipped for T <= 0 (as Lattice::overrelax_site). The old kernel
     * reflected about the full gradient h + 2 A S, which conserves neither
     * the energy nor the measure (E 33 -> 14 in 20 sweeps on 2x2x2 TmFeO3).
     */
    inline bool overrelax_site_SU2(size_t i, double T) {
        double* S = spins_SU2[i].data();
        double h[3];
        linear_field_SU2(i, h);
        const double hh = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
        if (hh <= 0.0) return false;
        const double k = 2.0 * (S[0] * h[0] + S[1] * h[1] + S[2] * h[2]) / hh;
        const double Sn[3] = {k * h[0] - S[0], k * h[1] - S[1], k * h[2] - S[2]};
        if (!self_isotropic_SU2[i]) {
            if (T <= 0.0) return false;
            double A[9];
            self_form_SU2(i, A);
            const double dE = self_energy_SU2(i, A, Sn) - self_energy_SU2(i, A, S);
            if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-dE / T)) return false;
        }
        S[0] = Sn[0]; S[1] = Sn[1]; S[2] = Sn[2];
        return true;
    }

    /**
     * Overrelaxation of SU(3) site j.
     * CP^2: the S^7 reflection leaves CP^2, so instead the state gets random
     * relative phases in the eigenbasis of H = h.lambda:
     * psi' = V diag(1, e^{i phi_1}, e^{i phi_2}) V^dagger psi, phi uniform.
     * The populations |<v_k|psi>|^2, hence h.n = sum_k p_k e_k, are
     * conserved; the move is a random unitary drawn from a distribution
     * invariant under inversion (phi -> -phi), and unitaries preserve the
     * Fubini-Study measure, so the kernel is symmetric (detailed balance)
     * and maps CP^2 onto itself. Sphere: reflection about h in R^8.
     * Non-constant self energies are Metropolis-corrected as for SU(2).
     */
    inline bool overrelax_site_SU3(size_t j, double T) {
        double* n = spins_SU3[j].data();
        double h[8], nn[8];
        linear_field_SU3(j, h);
        if (su3_on_cp2()) {
            classical_spin::su3::randomize_phases_cp2(h, n, nn);
        } else {
            double hh = 0.0, nh = 0.0;
            for (int a = 0; a < 8; ++a) { hh += h[a] * h[a]; nh += n[a] * h[a]; }
            if (hh <= 0.0) return false;
            const double k = 2.0 * nh / hh;
            for (int a = 0; a < 8; ++a) nn[a] = k * h[a] - n[a];
        }
        if (!self_isotropic_SU3[j]) {
            if (T <= 0.0) return false;
            double A[64];
            self_form_SU3(j, A);
            const double dE = self_energy_SU3(j, A, nn) - self_energy_SU3(j, A, n);
            if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-dE / T)) return false;
        }
        std::memcpy(n, nn, sizeof(nn));
        return true;
    }

    /**
     * Exact single-site minimisation of SU(2) site i (T = 0 descent step):
     * the global minimiser of h.S + S^T A S on |S| = L (trust-region
     * subproblem, Lattice::minimize_quadratic_on_sphere), -L h/|h| when the
     * self energy is constant; with cubic self terms a gradient-aligned
     * candidate accepted only if it lowers the local energy. Never raises
     * the energy. Returns |ΔS|.
     */
    double deterministic_site_SU2(size_t i);

    /**
     * Exact single-site minimisation of SU(3) site j: on CP^2 the ground
     * state of H = h.lambda (local exact diagonalisation; with quadratic self
     * terms the ground state of the linearised H, accepted only if it lowers
     * the local energy), on the sphere -L h/|h| or the trust-region
     * minimiser. `force_cp2` uses the CP^2 rule whatever the manifold.
     * Returns |Δn|.
     */
    double deterministic_site_SU3(size_t j, bool force_cp2 = false);

    // ---- sweep drivers ----

    /// Apply f2 to every SU(2) site and f3 to every SU(3) site once; returns the summed results.
    template <class F2, class F3>
    size_t sweep_sites(bool interleaved, F2&& f2, F3&& f3) {
        size_t n = 0;
        if (interleaved) {
            const size_t n_cells = dim1 * dim2 * dim3;
            for (size_t c = 0; c < n_cells; ++c) {
                for (size_t a = 0; a < N_atoms_SU2; ++a) n += f2(c * N_atoms_SU2 + a);
                for (size_t a = 0; a < N_atoms_SU3; ++a) n += f3(c * N_atoms_SU3 + a);
            }
        } else {
            for (size_t i = 0; i < lattice_size_SU2; ++i) n += f2(i);
            for (size_t j = 0; j < lattice_size_SU3; ++j) n += f3(j);
        }
        return n;
    }

    /**
     * Coloured OpenMP sweep: SU(2) colours, then SU(3) colours; within a
     * colour no two sites share a coupling of their own species, and the
     * other species is frozen during the pass, so the site kernels run
     * concurrently without races (see build_color_partition). RNG: per-thread
     * Lehmer streams (reproducible for a fixed thread count).
     */
    template <class F2, class F3>
    size_t sweep_sites_coloured(F2&& f2, F3&& f3) {
        size_t n = 0;
#ifdef _OPENMP
        #pragma omp parallel reduction(+:n)
#endif
        {
            for (size_t c = 0; c < n_colors_SU2; ++c) {
                const size_t lo = sites_by_color_csr_off_SU2[c], hi = sites_by_color_csr_off_SU2[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (size_t off = lo; off < hi; ++off) n += f2(sites_by_color_csr_SU2[off]);
            }
            for (size_t c = 0; c < n_colors_SU3; ++c) {
                const size_t lo = sites_by_color_csr_off_SU3[c], hi = sites_by_color_csr_off_SU3[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (size_t off = lo; off < hi; ++off) n += f3(sites_by_color_csr_SU3[off]);
            }
        }
        return n;
    }

    bool can_run_coloured() const {
#ifdef _OPENMP
        return (n_colors_SU2 > 0 || n_colors_SU3 > 0) && omp_get_max_threads() > 1;
#else
        return false;
#endif
    }

    /// True if local_sweep() uses the coloured OpenMP kernels (large lattice, > 1 thread).
    bool use_parallel_sweeps() const {
        return can_run_coloured() && lattice_size_SU2 + lattice_size_SU3 >= parallel_sweep_min_sites;
    }

    double acceptance_ratio(size_t accepted) const {
        const size_t N = lattice_size_SU2 + lattice_size_SU3;
        return N ? double(accepted) / double(N) : 0.0;
    }

    /**
     * Metropolis sweep: every SU(2) site, then every SU(3) site, in natural
     * order. `gaussian_move` selects the small symmetric moves of width
     * `sigma` (see propose_SU2 / propose_SU3), otherwise independent uniform
     * proposals. Returns the acceptance ratio.
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        const double beta = 1.0 / T;
        return acceptance_ratio(sweep_sites(false,
            [&](size_t i) { return metropolis_site_SU2(i, beta, gaussian_move, sigma); },
            [&](size_t j) { return metropolis_site_SU3(j, beta, gaussian_move, sigma); }));
    }

    /**
     * Metropolis sweep visiting the lattice unit cell by unit cell (the
     * cell's SU(2) sites, then its SU(3) sites), so that with mixed
     * couplings each species responds to the other's updates within the
     * same sweep.
     */
    double metropolis_interleaved(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        const double beta = 1.0 / T;
        return acceptance_ratio(sweep_sites(true,
            [&](size_t i) { return metropolis_site_SU2(i, beta, gaussian_move, sigma); },
            [&](size_t j) { return metropolis_site_SU3(j, beta, gaussian_move, sigma); }));
    }

    /// Coloured OpenMP Metropolis sweep (serial metropolis() with one thread or no colouring).
    double metropolis_parallel(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (!can_run_coloured()) return metropolis(T, gaussian_move, sigma);
        if (T <= 0) return 0.0;
        const double beta = 1.0 / T;
        return acceptance_ratio(sweep_sites_coloured(
            [&](size_t i) { return metropolis_site_SU2(i, beta, gaussian_move, sigma); },
            [&](size_t j) { return metropolis_site_SU3(j, beta, gaussian_move, sigma); }));
    }

    /**
     * Heat-bath sweep: heat bath for SU(2) sites and, on CP^2, for SU(3)
     * sites (exact for the linear part of the local energy, Metropolis-
     * corrected self energy). SU(3) sites on the legacy sphere get a
     * Metropolis update instead (uniform proposals, or Gaussian of width
     * sigma when gaussian_move). Returns the acceptance ratio (1 unless self
     * energies reject draws).
     */
    double heat_bath(double T, bool interleaved = false, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        const double beta = 1.0 / T;
        return acceptance_ratio(sweep_sites(interleaved,
            [&](size_t i) { return heat_bath_site_SU2(i, beta); },
            [&](size_t j) {
                return su3_on_cp2() ? heat_bath_site_SU3(j, beta)
                                    : metropolis_site_SU3(j, beta, gaussian_move, sigma);
            }));
    }

    /// Coloured OpenMP heat-bath sweep (see heat_bath).
    double heat_bath_parallel(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (!can_run_coloured()) return heat_bath(T, false, gaussian_move, sigma);
        if (T <= 0) return 0.0;
        const double beta = 1.0 / T;
        return acceptance_ratio(sweep_sites_coloured(
            [&](size_t i) { return heat_bath_site_SU2(i, beta); },
            [&](size_t j) {
                return su3_on_cp2() ? heat_bath_site_SU3(j, beta)
                                    : metropolis_site_SU3(j, beta, gaussian_move, sigma);
            }));
    }

    /**
     * Overrelaxation sweep in natural order (see overrelax_site_SU2 /
     * overrelax_site_SU3): microcanonical where the local energy is linear
     * in the site's own spin; sites with a non-constant self energy get a
     * Metropolis-corrected move at temperature T > 0 and are left untouched
     * at the default T = 0.
     */
    void overrelaxation(double T = 0.0) {
        sweep_sites(false, [&](size_t i) { return overrelax_site_SU2(i, T); },
                           [&](size_t j) { return overrelax_site_SU3(j, T); });
    }

    /// Overrelaxation sweep unit cell by unit cell (see metropolis_interleaved).
    void overrelaxation_interleaved(double T = 0.0) {
        sweep_sites(true, [&](size_t i) { return overrelax_site_SU2(i, T); },
                          [&](size_t j) { return overrelax_site_SU3(j, T); });
    }

    /// Coloured OpenMP overrelaxation sweep (serial with one thread or no colouring).
    void overrelaxation_parallel(double T = 0.0) {
        if (!can_run_coloured()) { overrelaxation(T); return; }
        sweep_sites_coloured([&](size_t i) { return overrelax_site_SU2(i, T); },
                             [&](size_t j) { return overrelax_site_SU3(j, T); });
    }

    /**
     * One local-update sweep at temperature T with the configured policy
     * (local_update; the legacy `gaussian_move` flag selects Gaussian
     * proposals of width sigma under Metropolis). Coloured OpenMP kernels for
     * lattices of at least parallel_sweep_min_sites sites when more than one
     * thread is available; otherwise natural or cell-interleaved order.
     * Returns the acceptance ratio.
     */
    double local_sweep(double T, bool gaussian_move, double sigma, bool interleaved = false) {
        const bool par = use_parallel_sweeps();
        const bool gauss = gaussian_move || local_update == LocalUpdate::Gaussian;
        if (local_update == LocalUpdate::HeatBath)
            return par ? heat_bath_parallel(T) : heat_bath(T, interleaved);
        if (par) return metropolis_parallel(T, gauss, sigma);
        return interleaved ? metropolis_interleaved(T, gauss, sigma) : metropolis(T, gauss, sigma);
    }

    /// Overrelaxation sweep through the same serial / interleaved / coloured selection.
    void overrelaxation_sweep(double T, bool interleaved = false) {
        if (use_parallel_sweeps()) overrelaxation_parallel(T);
        else if (interleaved) overrelaxation_interleaved(T);
        else overrelaxation(T);
    }

    /// Whether local_sweep() uses Gaussian proposals whose width `sigma` the drivers should adapt.
    bool uses_adaptive_step(bool gaussian_move) const {
        if (local_update == LocalUpdate::HeatBath) return false;
        return gaussian_move || local_update == LocalUpdate::Gaussian;
    }

    /**
     * T = 0 block-coordinate descent: every site in natural order is set to
     * its exact single-site minimiser (deterministic_site_SU2 / _SU3), so the
     * energy never increases. Returns the largest site change |ΔS| of the
     * last of `num_sweeps` sweeps. (Formerly each spin was aligned with
     * its full local field at randomly drawn sites: wrong with anisotropy,
     * and off the qutrit state space for SU(3).)
     */
    double deterministic_sweep(size_t num_sweeps = 1);

    /// T = 0 descent visiting the lattice unit cell by unit cell; returns the largest change.
    double deterministic_sweep_interleaved();

    // -------------------------------------------------------------------
    // SU(3) coherent-state utilities (qutrit-only, spin_dim_SU3 == 8).
    //
    // Storage convention. The stored `spins_SU3[i](a)` are the Gell-Mann
    // expectation values n^a = <psi|lambda^a|psi>. The qutrit density
    // matrix is
    //   rho = (1/3) I + (1/2) sum_a n^a lambda^a,
    // whose trace is identically 1 regardless of |n|. Every pure state has
    // Tr rho^2 = 1/3 + |n|^2/2 = 1, i.e. |n|^2 = 2(N-1)/N = 4/3 and
    // |n| = 2/sqrt(3) (e.g. |E1>: n_3 = 1, n_8 = 1/sqrt(3)), and the cubic
    // Casimir d_abc n^a n^b n^c = 8/9 (su3::casimir2 / su3::casimir3).
    // With su3_mc_manifold = CP2 (default) every Monte Carlo update and the
    // T = 0 descent stay on this manifold; `spin_length_SU3` is then not
    // used (for SU(N > 2) it is not a free normalisation: no continuous
    // family of states has the same Casimir spectrum). It sets the radius
    // of the legacy S^7 manifold only.
    // -------------------------------------------------------------------

    /**
     * Project every stored SU(3) Bloch vector onto its closest physical
     * pure-state representation. For each site:
     *   1. Build  rho = (1/3) I + (1/2) sum_a n^a lambda^a  from the
     *      stored n^a.
     *   2. Diagonalise rho; let psi be the eigenvector with the largest
     *      eigenvalue (the closest pure-state projector).
     *   3. Overwrite n^a by <psi|lambda^a|psi>.
     * Returns the minimum top-eigenvalue (purity) encountered across
     * sites. Values below 1 indicate that the original stored state was
     * not a physical qutrit pure state (e.g. produced by the legacy S^7
     * sampling).
     */
    double physicalize_SU3_state() {
        if (spin_dim_SU3 != 8) return 1.0;
        double min_purity = 1.0;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            classical_spin::su3::Vector8r n;
            for (int a = 0; a < 8; ++a) n(a) = spins_SU3[i](a);
            double purity = 0.0;
            auto psi = classical_spin::su3::psi_from_expectations(n, &purity);
            auto n_phys = classical_spin::su3::expectations_from_psi(psi);
            for (int a = 0; a < 8; ++a) spins_SU3[i](a) = n_phys(a);
            if (purity < min_purity) min_purity = purity;
        }
        return min_purity;
    }

    /**
     * Put the SU(3) states on the Monte Carlo manifold where they are not
     * (CP^2: closest pure state, as physicalize_SU3_state, for every site
     * whose Casimirs deviate from 4/3, 8/9 by more than `tol`; sphere:
     * rescale to |n| = spin_length_SU3). Called by the SA, quench and PT
     * drivers, so a loaded or legacy configuration is sampled on the right
     * state space. Returns the number of sites changed.
     */
    size_t project_SU3_to_manifold(double tol = 1e-9) {
        size_t changed = 0;
        for (size_t j = 0; j < lattice_size_SU3; ++j) {
            double* n = spins_SU3[j].data();
            if (su3_on_cp2()) {
                const double c2 = classical_spin::su3::casimir2(n), c3 = classical_spin::su3::casimir3(n);
                if (std::abs(c2 - 4.0 / 3.0) <= tol && std::abs(c3 - 8.0 / 9.0) <= tol) continue;
                classical_spin::su3::Vector8r v;
                for (int a = 0; a < 8; ++a) v(a) = n[a];
                const auto psi = classical_spin::su3::psi_from_expectations(v);
                std::complex<double> p[3] = {psi(0), psi(1), psi(2)};
                classical_spin::su3::pure_expectations(p, n);
            } else {
                const double L = double(spin_length_SU3);
                const double norm = std::sqrt(classical_spin::su3::casimir2(n));
                if (std::abs(norm - L) <= tol * std::max(1.0, L)) continue;
                if (norm > 0.0) for (int a = 0; a < 8; ++a) n[a] *= L / norm;
                else random_SU3_state(n);
            }
            ++changed;
        }
        return changed;
    }

    /**
     * Report the worst-case (smallest) qutrit density-matrix eigenvalue
     * across all SU(3) sites at the current configuration. Negative
     * values prove the stored 8-vector parameterization is unphysical at
     * that site (no qutrit density matrix can produce those expectation
     * values). Useful to audit seeds / annealed states.
     */
    double min_SU3_density_eigenvalue() const {
        if (spin_dim_SU3 != 8) return 1.0;
        double worst = 1.0;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            classical_spin::su3::Vector8r n;
            for (int a = 0; a < 8; ++a) n(a) = spins_SU3[i](a);
            const Eigen::Vector3d ev = classical_spin::su3::density_eigenvalues(n);
            if (ev(0) < worst) worst = ev(0);
        }
        return worst;
    }

    /**
     * T = 0 descent sweep with the SU(3) sites set to the ground state of
     * their local Hamiltonian H = h.lambda (local exact diagonalisation,
     * Zhang & Batista) whatever su3_mc_manifold is; SU(2) sites as in
     * deterministic_sweep. On CP^2 this is deterministic_sweep().
     */
    double deterministic_sweep_SU3_exact_diag();

    /**
     * Zero-temperature quench to a local minimum: deterministic_sweep()
     * until the energy change per sweep is below rel_tol |E| and no site
     * moves by more than sqrt(rel_tol) (in units of its length). Projects
     * the SU(3) states onto the Monte Carlo manifold first.
     */
    void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000);

    /**
     * Simulated annealing on the schedule mc::annealing_schedule(T_start,
     * T_end, cooling_rate) (validated, ends exactly at T_end) with n_anneal
     * sweeps per temperature of local_sweep() (policy `local_update`,
     * cell-interleaved order when mixed couplings exist). With Gaussian
     * proposals the width is adapted (mc::StepSizeController, Robbins-Monro
     * toward 45 % acceptance) during the first half of the sweeps at each
     * temperature and frozen for the second half. SU(3) sites are projected
     * onto su3_mc_manifold first. The optional T = 0 stage runs
     * deterministic_sweep() (exact block-coordinate descent) until converged
     * or n_deterministics sweeps. twist_sweep_count is accepted for interface
     * compatibility (MixedLattice has no twisted boundaries).
     */
    void simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            bool gaussian_move = false,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = false,
                            bool T_zero = false,
                            size_t n_deterministics = 1000,
                            size_t twist_sweep_count = 100);

    /**
     * Perform detailed measurements at final temperature
     * Computes: energy, specific heat, sublattice magnetizations (SU2 and SU3), 
     * and cross-correlations. All with binning analysis for error estimation.
     */
    void perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   const string& out_dir);

    /** Compute autocorrelation — delegates to mc::compute_autocorrelation */
    AutocorrelationResult compute_autocorrelation(const vector<double>& energies, 
                                                   size_t base_interval = 10);

    // ============================================================
    // BINNING ANALYSIS (delegated to mc::* functions)
    // ============================================================

    /** Binning analysis — delegates to mc::binning_analysis */
    static BinningResult binning_analysis(const vector<double>& data) {
        return mc::binning_analysis(data);
    }

    /** Component-wise binning analysis for vector observable — delegates to mc::binning_analysis_vector */
    static vector<BinningResult> binning_analysis_vector(const vector<SpinVector>& data) {
        return mc::binning_analysis_vector<SpinVector>(data);
    }

    // ============================================================
    // SUBLATTICE MAGNETIZATION
    // ============================================================

    /**
     * Compute magnetization for each SU(2) sublattice separately
     * 
     * @return Vector of SpinVectors, one per SU(2) sublattice (N_atoms_SU2 sublattices)
     */
    vector<SpinVector> magnetization_sublattice_SU2() const {
        vector<SpinVector> M_sub(N_atoms_SU2);
        size_t n_cells = dim1 * dim2 * dim3;
        
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim_SU2);
        }
        
        // Sum over all unit cells for each sublattice
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);
                        
                        // Transform to global frame using sublattice frame
                        SpinVector spin_global = SpinVector::Zero(spin_dim_SU2);
                        for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                            for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                                spin_global(mu) += sublattice_frames_SU2[atom](mu, nu) * spins_SU2[site_idx](nu);
                            }
                        }
                        M_sub[atom] += spin_global;
                    }
                }
            }
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            M_sub[atom] /= double(n_cells);
        }
        
        return M_sub;
    }

    /**
     * Compute magnetization for each SU(3) sublattice separately
     * 
     * @return Vector of SpinVectors, one per SU(3) sublattice (N_atoms_SU3 sublattices)
     */
    vector<SpinVector> magnetization_sublattice_SU3() const {
        vector<SpinVector> M_sub(N_atoms_SU3);
        size_t n_cells = dim1 * dim2 * dim3;
        
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim_SU3);
        }
        
        // Sum over all unit cells for each sublattice
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU3);
                        
                        // Transform to global frame using sublattice frame
                        SpinVector spin_global = SpinVector::Zero(spin_dim_SU3);
                        for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                            for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                                spin_global(mu) += sublattice_frames_SU3[atom](mu, nu) * spins_SU3[site_idx](nu);
                            }
                        }
                        M_sub[atom] += spin_global;
                    }
                }
            }
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            M_sub[atom] /= double(n_cells);
        }
        
        return M_sub;
    }

    // ============================================================
    // COMPREHENSIVE OBSERVABLE COLLECTION
    // ============================================================

    /**
     * Collect a single measurement of all thermodynamic observables
     * Returns: (energy, energy_SU2, energy_SU3, sublattice_mags_SU2, sublattice_mags_SU3)
     */
    struct MixedMeasurement {
        double energy;
        double energy_SU2;
        double energy_SU3;
        vector<SpinVector> sublattice_mags_SU2;
        vector<SpinVector> sublattice_mags_SU3;
    };

    MixedMeasurement measure_all_observables() const {
        MixedMeasurement m;
        m.energy_SU2 = total_energy_SU2();
        m.energy_SU3 = total_energy_SU3();
        m.energy = m.energy_SU2 + m.energy_SU3;   // = total_energy(), one pass
        m.sublattice_mags_SU2 = magnetization_sublattice_SU2();
        m.sublattice_mags_SU3 = magnetization_sublattice_SU3();
        return m;
    }

    /**
     * Compute comprehensive thermodynamic observables with binning error analysis
     * 
     * @param measurements Vector of MixedMeasurement from MC sampling
     * @param T Temperature
     * @return MixedThermodynamicObservables struct with all observables and uncertainties
     */
    MixedThermodynamicObservables compute_thermodynamic_observables(
        const vector<MixedMeasurement>& measurements,
        double T) const;

    /**
     * Save comprehensive thermodynamic observables to files
     */
    void save_thermodynamic_observables(const string& out_dir,
                                         const MixedThermodynamicObservables& obs) const;

    /**
     * Print thermodynamic observables summary to stdout
     */
    void print_thermodynamic_observables(const MixedThermodynamicObservables& obs) const;

    /**
     * Save thermodynamic observables to HDF5 format for mixed lattice
     * Single file per rank with all data organized in groups
     */
    void save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const MixedThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<pair<SpinVector, SpinVector>>& magnetizations,
                                              const vector<MixedMeasurement>& measurements,
                                              size_t n_anneal,
                                              size_t n_measure,
                                              size_t probe_rate,
                                              size_t swap_rate,
                                              size_t overrelaxation_rate,
                                              double acceptance_rate,
                                              double swap_acceptance_rate) const;

    /**
     * Save aggregated heat capacity data from all temperatures to HDF5 format
     * Called by rank 0 to save temperature-dependent thermodynamic data
     */
    void save_heat_capacity_hdf5(const string& out_dir,
                                  const vector<double>& temperatures,
                                  const vector<double>& heat_capacity,
                                  const vector<double>& dHeat) const;

    /**
     * Save sublattice magnetization time series to files
     */
    void save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<MixedMeasurement>& measurements) const;

    /**
     * Compute and save thermodynamic observables for mixed lattice
     */
    void compute_and_save_observables(const vector<double>& energies,
                                     const vector<pair<SpinVector, SpinVector>>& magnetizations,
                                     double T, const string& out_dir);

    /**
     * Save observables for mixed lattice
     */
    void save_observables(const string& dir_path,
                         const vector<double>& energies,
                         const vector<pair<SpinVector, SpinVector>>& magnetizations);

    /**
     * Save autocorrelation results
     */
    void save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf);

    // ============================================================
    // PARALLEL TEMPERING (shared engine: mc/parallel_tempering.h)
    // ============================================================

    /**
     * Tune the temperature ladder with the replica chain (one replica per
     * rank, collective) — see mc::tune_temperature_ladder. The MC step is the
     * one used by parallel_tempering() (interleaved SU(2)/SU(3) sweeps when
     * mixed couplings exist).
     */
    mc::LadderTuningResult tune_temperature_ladder(const mc::LadderTuningOptions& options,
                                                   size_t overrelaxation_rate, bool gaussian_move,
                                                   bool use_interleaved = true,
                                                   MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * Parallel tempering with MPI for the mixed lattice on the shared engine
     * mc::run_parallel_tempering (DEO exchange with labelled replicas, adaptive
     * proposal width while equilibrating, Gamma-method / jackknife statistics).
     * Records the SU(2) and SU(3) magnetisations as order parameters and writes
     * the MixedLattice per-rank HDF5 file (SU2/SU3 split observables).
     * Uses interleaved sweeps when mixed interactions are present.
     */
    mc::PTResult parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, bool use_interleaved = true,
                           MPI_Comm comm = MPI_COMM_WORLD, bool verbose = false);

public:
    // ============================================================
    // MOLECULAR DYNAMICS
    // ============================================================

    /**
     * Get local field for SU(2) site
     */
    SpinVector get_local_field_SU2(size_t site_index) const {
        SpinVector H = -field_SU2[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU2[site_index] * spins_SU2[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU2[site_index].size(); ++i) {
            H += bilinear_interaction_SU2[site_index][i] * spins_SU2[bilinear_partners_SU2[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU2[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU2[site_index][i] * spins_SU3[mixed_bilinear_partners_SU2[site_index][i]];
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            
            // Contract tensor with partner spins: H[a] = sum_bc T[a](b,c) * S1[b] * S2[c]
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear SU(2)-SU(2)-SU(3) contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            
            // Contract: H[a] = sum_bc T[a](b,c) * SU2[b] * SU3[c]
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

    /**
     * Get local field for SU(3) site
     */
    SpinVector get_local_field_SU3(size_t site_index) const {
        SpinVector H = -field_SU3[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU3[site_index] * spins_SU3[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU3[site_index].size(); ++i) {
            H += bilinear_interaction_SU3[site_index][i] * spins_SU3[bilinear_partners_SU3[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU3[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU3[site_index][i] * spins_SU2[mixed_bilinear_partners_SU3[site_index][i]];
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) contributions
        for (size_t i = 0; i < trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            
            // Contract tensor with partner spins: H[a] = sum_bc T[a](b,c) * S1[b] * S2[c]
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * spins_SU3[p1_idx](b) * spins_SU3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear SU(3)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            
            // Contract: H[a] = sum_bc T[a](b,c) * SU2_1[b] * SU2_2[c]
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

    /**
     * Two-pulse drive for SU(2): pulse k is centred at t_Bk with per-atom
     * global-frame directions field_ink (transformed to the local frame,
     * B_local = F^T B_global), amplitude `amp`, Gaussian width `width` and
     * carrier `freq`. Marks both pulse slots active; single-pulse experiments
     * go through single_pulse_drive(), which activates only slot 0. Silent:
     * drivers print the pulse configuration once.
     */
    void set_pulse_SU2(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq);

    /**
     * Two-pulse drive for SU(3) (same conventions as set_pulse_SU2).
     */
    void set_pulse_SU3(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq);

    /**
     * Remove the pulse train (no active pulses, zero directions and
     * amplitudes). The pulse SHAPE configuration that the runners set once per
     * run -- second SU(3) colour, tabulated waveform -- is kept.
     */
    void reset_pulse() {
        n_active_pulses = 0;
        for (int k = 0; k < 2; ++k) {
            field_drive_SU2[k].setZero();
            field_drive_SU3[k].setZero();
            field_drive_global_SU2[k].setZero();
        }
        t_pulse_SU2 = {0.0, 0.0};
        t_pulse_SU3 = {0.0, 0.0};
        field_drive_amp_SU2 = 0.0;
        field_drive_freq_SU2 = 0.0;
        field_drive_width_SU2 = 1.0;
        field_drive_amp_SU3 = 0.0;
        field_drive_freq_SU3 = 0.0;
        field_drive_width_SU3 = 1.0;
    }

    /**
     * Set SU(3) Bloch damping rates Γ_a for each Gell-Mann channel.
     * See tmfeo3_notes.tex Eq. blochdampedfull:
     *   dn^a/dt += −Γ_a (n^a − n^a_eq)
     * Rates are physical relaxation rates (1/time) and are independent of the
     * bracket convention.
     * @param rates  8-component vector of non-negative damping rates
     */
    void set_damping_SU3(const SpinVector& rates) {
        if (rates.size() != static_cast<Eigen::Index>(spin_dim_SU3)) {
            throw std::invalid_argument("set_damping_SU3: rates must have " +
                                        std::to_string(spin_dim_SU3) + " components (got " +
                                        std::to_string(rates.size()) + ")");
        }
        for (Eigen::Index a = 0; a < rates.size(); ++a) {
            if (!std::isfinite(rates(a)) || rates(a) < 0.0) {
                throw std::invalid_argument("set_damping_SU3: rates must be finite and >= 0");
            }
        }
        damping_rates_SU3 = rates;
    }

    /**
     * Set per-site SU(3) equilibrium Bloch vectors for Bloch damping.
     * Only the relaxation target: the Hamiltonian is not affected (see
     * set_mixed_trilinear_reference_SU3 for an explicit reference subtraction).
     * @param eq  Per-site equilibrium vectors (size = lattice_size_SU3)
     */
    void set_equilibrium_SU3(const SpinConfigSU3& eq) {
        if (eq.size() != lattice_size_SU3) {
            throw std::invalid_argument("set_equilibrium_SU3: expected " + std::to_string(lattice_size_SU3) +
                                        " vectors (got " + std::to_string(eq.size()) + ")");
        }
        for (const auto& v : eq) {
            if (v.size() != static_cast<Eigen::Index>(spin_dim_SU3)) {
                throw std::invalid_argument("set_equilibrium_SU3: every vector must have " +
                                            std::to_string(spin_dim_SU3) + " components");
            }
        }
        equilibrium_SU3 = eq;
    }

    /**
     * Subtract an explicit SU(3) reference configuration r from the SU(3) leg
     * of every SU(2)-SU(2)-SU(3) trilinear coupling:
     *     T(S_i, S_j, n_k)  ->  T(S_i, S_j, n_k - r_k).
     * The subtraction is the Fe-Fe bilinear -T(S_i, S_j, r_k) (on-site when
     * j = i); it is folded into the SU(2) bilinear / on-site tables, so the
     * total energy, the Monte Carlo
     * energy differences / local fields and the dynamics all see the same
     * modified Hamiltonian. (It used to be applied to the MD field only, with
     * r silently taken from the Bloch-damping equilibrium, so an annealed
     * ground state was not stationary once damping was switched on.)
     * Calling it again replaces the reference; an empty argument removes it.
     * Re-minimise afterwards if a stationary initial state is needed.
     */
    void set_mixed_trilinear_reference_SU3(const SpinConfigSU3& reference);
    const SpinConfigSU3& mixed_trilinear_reference_SU3() const { return trilinear_reference_SU3_; }

    /**
     * Set uniform equilibrium Bloch vector for all SU(3) sites.
     * Transforms to local sublattice frame via F^T.
     * @param eq_global  Single equilibrium vector in global (Bertaut) frame
     */
    void set_equilibrium_SU3_uniform(const SpinVector& eq_global) {
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            size_t atom = site % N_atoms_SU3;
            equilibrium_SU3[site] = sublattice_frames_SU3[atom].transpose() * eq_global;
        }
    }

    /**
     * Diagnostic ablation: evaluate the SU(2) drive torque on the equilibrium
     * spin configuration rather than the instantaneous one.
     *
     * The Zeeman torque is S x h(t).  Writing S = S0 + dS, the piece S0 x h
     * creates the magnon while the piece dS x h converts one magnon into
     * another while the pulse is present -- the field-mediated (field-assisted)
     * magnon-magnon channel.  Enabling this flag keeps only S0 x h, which
     * removes that channel exactly while leaving the intrinsic anharmonicity of
     * the spin Hamiltonian (anisotropy, Dzyaloshinskii-Moriya, exchange in the
     * canted structure) untouched.  This is an ablation, not a physical model:
     * the modified torque is no longer perpendicular to S, so |S| drifts at
     * O(h |dS|); the drift is reported by the caller and is ~1e-9 at the
     * perturbative drive amplitudes used for the 2DCS reference runs.
     *
     * Must be called after the ground state is loaded/annealed and
     * synchronized, since it snapshots spins_SU2 as the reference S0.
     */
    bool linear_drive_torque_SU2 = false;
    std::vector<double> drive_ref_SU2;   // flat lattice_size_SU2 * spin_dim_SU2

    void set_linear_drive_torque_SU2(bool on) {
        linear_drive_torque_SU2 = on;
        if (!on) { drive_ref_SU2.clear(); return; }
        drive_ref_SU2.assign(lattice_size_SU2 * spin_dim_SU2, 0.0);
        for (size_t i = 0; i < lattice_size_SU2; ++i)
            for (size_t d = 0; d < spin_dim_SU2; ++d)
                drive_ref_SU2[i * spin_dim_SU2 + d] = spins_SU2[i](d);
    }

    /**
     * Compute time-dependent drive field for SU(2) site (pre-transformed to local frame)
     */
    SpinVector drive_field_SU2_at_time(double t, size_t site_index) const;

    /**
     * Compute time-dependent drive field for SU(3) site (pre-transformed to local frame)
     */
    SpinVector drive_field_SU3_at_time(double t, size_t site_index) const;

    /**
     * Drive-field envelope helpers — return the two pulse factors at time
     * `t` without touching any per-site data. Used by `landau_lifshitz` to
     * hoist the two `exp + cos` calls per pulse out of the per-site loop;
     * total cost goes from O(2 * lattice_size) transcendentals per RHS to
     * O(2) per RHS.
     */
    void drive_envelopes_SU2(double t, double& factor1, double& factor2) const;
    void drive_envelopes_SU3(double t, double& factor1, double& factor2) const;
    double interp_tabulated_pulse(double dt) const;

    /// Load a two-column text/CSV file (time_ps, E_field) as the pump pulse.
    /// The time axis is shifted so the peak is at t=0; the amplitude is
    /// normalized to max|E|=1.  Sets tabulated_pulse_sigma for chunking.
    void load_tabulated_pulse(const std::string& filename);

    /// Number of spin components in the flat state [SU(2) | SU(3)].
    size_t spin_state_size() const {
        return lattice_size_SU2 * spin_dim_SU2 + lattice_size_SU3 * spin_dim_SU3;
    }

    /// True when the thermal reservoir E_dep is integrated (state has one extra entry).
    bool has_thermal_reservoir() const { return thermal_heat != 0.0; }

    /// Size of the ODE state: spins, plus E_dep when has_thermal_reservoir().
    size_t ode_state_size() const { return spin_state_size() + (has_thermal_reservoir() ? 1 : 0); }

    /**
     * Convert spin configurations to the flat ODE state
     * [SU(2) spins | SU(3) spins | E_dep (only with the thermal reservoir, = 0)].
     */
    ODEState spins_to_state() const {
        ODEState state(ode_state_size(), 0.0);
        
        size_t idx = 0;
        // Pack SU(2) spins
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < spin_dim_SU2; ++j) {
                state[idx++] = spins_SU2[i](j);
            }
        }
        // Pack SU(3) spins
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < spin_dim_SU3; ++j) {
                state[idx++] = spins_SU3[i](j);
            }
        }
        
        return state;
    }

    /**
     * Convert flat state vector to spin configurations
     */
    void state_to_spins(const ODEState& state, SpinConfigSU2& spins2, SpinConfigSU3& spins3) const {
        // Resize output vectors
        spins2.resize(lattice_size_SU2);
        spins3.resize(lattice_size_SU3);
        
        size_t idx = 0;
        // Unpack SU(2) spins
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins2[i] = SpinVector(spin_dim_SU2);
            for (size_t j = 0; j < spin_dim_SU2; ++j) {
                spins2[i](j) = state[idx++];
            }
        }
        // Unpack SU(3) spins
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins3[i] = SpinVector(spin_dim_SU3);
            for (size_t j = 0; j < spin_dim_SU3; ++j) {
                spins3[i](j) = state[idx++];
            }
        }
    }

    /**
     * Integrate `system_func` over [T_start, T_end] and call
     * observer(state, t_k) on the exact grid t_k = T_start + k dt_step
     * (k = 0..n, the last point not beyond T_end). Kept for source
     * compatibility; the drivers use dynamics::integrate_on_time_grid
     * directly. Unknown method names throw std::invalid_argument.
     * `use_adaptive` is accepted for compatibility: adaptive methods always
     * use error control and dense output, fixed-step methods never do.
     */
    template<typename System, typename Observer>
    void integrate_ode_system(System system_func, ODEState& state,
                             double T_start, double T_end, double dt_step,
                             Observer observer, const string& method,
                             bool use_adaptive = false,
                             double abs_tol = 1e-6, double rel_tol = 1e-6) const {
        (void) use_adaptive;
        namespace dyn = classical_spin::dynamics;
        const auto grid = dyn::TimeGrid::covering(T_start, T_end, dt_step, "integrate_ode_system");
        dyn::integrate_on_time_grid(system_func, state, grid, 1, dyn::parse_ode_method(method),
                                    abs_tol, rel_tol,
                                    [&](const ODEState& x, size_t k) { observer(x, grid[k]); });
    }

    /**
     * ODE system function for Boost.Odeint (alias of landau_lifshitz).
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t) const;

    /**
     * Equations of motion, a pure function of (state, t) and the current drive:
     *   SU(2):  dS/dt = H × S + (α/|S|) S × (S × H)            (LL-Gilbert)
     *   SU(3):  dn^a/dt = P^a − (α_SU3/|n|) c f_{abc} n^b P^c − Γ_a (n^a − n^a_eq),
     *           P^a = c f_{abc} H^b n^c,  c = su3_bracket
     *   E_dep:  dE_dep/dt = P(n) − thermal_cool E_dep           (thermal reservoir)
     * with H = ∂E/∂S (∂E/∂n) the local field including the pulse drive.
     * State layout: [SU2 site0 .. SU2 siteN | SU3 site0 .. SU3 siteM | E_dep?];
     * `state` may omit E_dep (then E_dep = 0 and dsdt has the same size).
     */
    void landau_lifshitz(const ODEState& state, ODEState& dsdt, double t) const;

public:
    /**
     * Heap-free, drive-hoisted variants of `get_local_field_SU{2,3}_flat`.
     * The full local field is written directly into the caller-supplied
     * `H_out[0..spin_dim_SU{2,3}-1]` and the time-dependent envelope
     * factors are passed in by the caller (typically computed once per RHS
     * evaluation in `landau_lifshitz`). These are the form used by the LLG
     * hot loop; together they eliminate one `Eigen::VectorXd` heap
     * allocation per site per RHS call and the redundant `exp + cos` calls
     * per site that the legacy `..._flat(t, site)` interface incurs.
     *
     * `env_E` / `env_B` are the SU(3) (electric) and SU(2) (magnetic) pulse
     * envelope scalars that modulate the field-assisted Fe-Tm exchange
     * (H_{E chi} / H_{B chi}); they default to 0 (assisted terms inactive).
     * Public so dynamics diagnostics can probe the field-assisted channels.
     */
    void get_local_field_SU2_flat_into(size_t site, const ODEState& state,
                                       size_t offset_SU3,
                                       double drive_factor1, double drive_factor2,
                                       double* H_out,
                                       double env_E = 0.0, double env_B = 0.0,
                                       double env_Bx = 0.0, double env_By = 0.0,
                                       double env_Bz = 0.0) const;
    void get_local_field_SU3_flat_into(size_t site, const ODEState& state,
                                       size_t offset_SU3,
                                       double drive_factor1, double drive_factor2,
                                       double* H_out,
                                       double env_E = 0.0, double env_B = 0.0,
                                       double env_Bx = 0.0, double env_By = 0.0,
                                       double env_Bz = 0.0) const;

    /**
     * Dimensionless stationarity residual of `state` without any drive:
     *     max_i |dS_i/dt| / (c_i |H_i| |S_i|),
     * the largest sine of the angle between a spin and its local field
     * (c_i = 1 for SU(2), su3_bracket for SU(3)); the SU(3) rate includes the
     * Bloch relaxation towards n_eq (Gilbert damping vanishes with the
     * torque). Independent of the energy scale, so one tolerance fits every
     * model.
     */
    double relative_stationarity_residual(const ODEState& state) const;

    /**
     * Empty string if the GPU MD path implements every term of the current
     * model and drive, otherwise a description of what it would silently drop
     * (trilinear couplings, field-assisted exchange, Gilbert/Bloch damping,
     * the thermal reservoir, two-colour or tabulated pulses, the linearised
     * drive torque, a non-default SU(3) bracket, spin-state output).
     */
    string gpu_unsupported_reason(bool want_spin_states = false) const;

    /// Throws std::invalid_argument when gpu_unsupported_reason() is non-empty.
    void check_gpu_supported(bool want_spin_states = false) const {
        const string why = gpu_unsupported_reason(want_spin_states);
        if (!why.empty()) {
            throw std::invalid_argument("MixedLattice GPU dynamics would silently drop: " + why +
                                        "; run with use_gpu = false");
        }
    }

private:
    /**
     * Helper: Safely create directories if path is non-empty
     */
    static void ensure_directory_exists(const string& dir_path) {
        if (!dir_path.empty()) {
            std::filesystem::create_directories(dir_path);  // (was infinite self-recursion)
        }
    }

    /**
     * Helper: Compute local and antiferromagnetic magnetization from flat state for a sublattice
     * @param x Flat state array
     * @param offset Starting index in flat array
     * @param lattice_size Number of sites
     * @param spin_dim Spin dimension
     * @param M_local_arr Output array for local magnetization
     * @param M_antiferro_arr Output array for antiferromagnetic magnetization
     */
    // SU(3) reference subtraction folded into the SU(2) bilinear / on-site
    // tables: reference_bond_slots_[i] holds, for SU(2) site i, the bilinear
    // indices appended by set_mixed_trilinear_reference_SU3 and the
    // mixed-trilinear entry each one was derived from.
    SpinConfigSU3 trilinear_reference_SU3_;
    vector<vector<std::pair<size_t, size_t>>> reference_bond_slots_;
    vector<SpinMatrix> reference_saved_onsite_SU2_;   // on-site matrices before the fold

    // Pulse envelope factors of one right-hand-side evaluation (zero = no drive).
    struct DriveFactors {
        double su2[2] = {0.0, 0.0};   // amplitude x envelope x carrier, pulse k
        double su3[2] = {0.0, 0.0};
        double env_E = 0.0, env_B = 0.0;                 // field-assisted exchange gates
        double env_Bx = 0.0, env_By = 0.0, env_Bz = 0.0; // lab-frame B(t) components
    };
    DriveFactors drive_factors(double t) const;
    // Equations of motion for given drive factors (see landau_lifshitz).
    void evaluate_rhs(const ODEState& state, ODEState& dsdt, const DriveFactors& d) const;

    // Install a train of n_pulses (1 or 2) pulses; slot 1 is inert for n = 1.
    void configure_pulse_train(size_t n_pulses,
                               const vector<SpinVector>& field1_SU2, const vector<SpinVector>& field1_SU3,
                               double t1,
                               const vector<SpinVector>& field2_SU2, const vector<SpinVector>& field2_SU3,
                               double t2,
                               double amp_SU2, double width_SU2, double freq_SU2,
                               double amp_SU3, double width_SU3, double freq_SU3);
    // Integrate state x from grid index k_start to the end of `grid` under the
    // installed pulse train and return observe() at every sample k_start..n-1
    // (the spin state too if requested; the states at the global indices
    // `capture_at` go to `captured`). The pulse train is removed afterwards,
    // also on error.
    PumpProbeTrajectory integrate_pulse_train(const classical_spin::dynamics::TimeGrid& grid,
                                              size_t k_start, ODEState x, const string& method,
                                              vector<vector<double>>* spin_state_out,
                                              double abs_tol, double rel_tol,
                                              const vector<size_t>& capture_at = {},
                                              vector<ODEState>* captured = nullptr);

    static void compute_sublattice_magnetizations_from_flat(const double* x, size_t offset,
                                                            size_t lattice_size, size_t spin_dim,
                                                            double* M_local_arr, double* M_antiferro_arr) {
        std::fill(M_local_arr, M_local_arr + spin_dim, 0.0);
        std::fill(M_antiferro_arr, M_antiferro_arr + spin_dim, 0.0);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            size_t idx = offset + i * spin_dim;
            for (size_t d = 0; d < spin_dim; ++d) {
                M_local_arr[d] += x[idx + d];
                M_antiferro_arr[d] += x[idx + d] * sign;
            }
        }
    }

    /**
     * Helper: n_sweeps MC steps at temperature T. One step is one
     * overrelaxation sweep (when overrelaxation_rate = k > 0) plus a
     * local_sweep() every k-th step (every step when k = 0), the convention
     * of the PT engine. Returns the mean acceptance of the local sweeps.
     *
     * @param n_sweeps Number of steps to perform
     * @param T Temperature
     * @param gaussian_move Use Gaussian moves
     * @param sigma Gaussian move width
     * @param overrelaxation_rate Local sweep every k-th step, overrelaxation every step (0 = no OR)
     * @param interleaved Cell-interleaved site order when mixed couplings exist
     */
    double perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate = 0,
                            bool interleaved = true);

public:

    // ============================================================
    // OBSERVABLES
    // ============================================================

    /**
     * Compute SU(2) magnetization
     */
    SpinVector magnetization_SU2() const {
        SpinVector mag = SpinVector::Zero(spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            mag += spins_SU2[i];
        }
        return mag / double(lattice_size_SU2);
    }

    /**
     * Compute SU(3) magnetization
     */
    SpinVector magnetization_SU3() const {
        SpinVector mag = SpinVector::Zero(spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            mag += spins_SU3[i];
        }
        return mag / double(lattice_size_SU3);
    }

    /**
     * Helper function to compute SU(2) global magnetization from flat state
     */
    void compute_magnetization_global_SU2_from_flat(const double* x, double* M_global_arr) const {
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_global_arr[d] = 0.0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            size_t atom = i % N_atoms_SU2;
            size_t idx = i * spin_dim_SU2;
            for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                    M_global_arr[mu] += sublattice_frames_SU2[atom](mu, nu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_global_arr[d] /= double(lattice_size_SU2);
    }

    /**
     * Helper function to compute SU(3) global magnetization from flat state
     */
    void compute_magnetization_global_SU3_from_flat(const double* x, double* M_global_arr) const {
        for (size_t d = 0; d < spin_dim_SU3; ++d) M_global_arr[d] = 0.0;
        size_t offset = lattice_size_SU2 * spin_dim_SU2;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            size_t atom = i % N_atoms_SU3;
            size_t idx = offset + i * spin_dim_SU3;
            for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                    M_global_arr[mu] += sublattice_frames_SU3[atom](mu, nu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU3; ++d) M_global_arr[d] /= double(lattice_size_SU3);
    }

    /**
     * Helper function to compute SU(2) staggered magnetization from flat state
     * Uses sublattice frames and AFM signs (Bertaut modes)
     */
    void compute_magnetization_staggered_SU2_from_flat(const double* x, double* M_stag_arr) const {
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_stag_arr[d] = 0.0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            size_t atom = i % N_atoms_SU2;
            double sign = afm_sublattice_signs_SU2[atom];
            size_t idx = i * spin_dim_SU2;
            for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                    M_stag_arr[mu] += sign * sublattice_frames_SU2[atom](mu, nu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_stag_arr[d] /= double(lattice_size_SU2);
    }

    /**
     * SU(3) staggered magnetisation from a flat state: sublattice frames and
     * the unit cell's AFM signs, exactly like the SU(2) version. (It used to
     * alternate signs by flat site index, which depends on the site ordering;
     * for the TmFeO3 cell, whose Tm signs are (+,-,+,-) with identity frames,
     * both definitions coincide.)
     */
    void compute_magnetization_staggered_SU3_from_flat(const double* x, double* M_stag_arr) const {
        for (size_t d = 0; d < spin_dim_SU3; ++d) M_stag_arr[d] = 0.0;
        const size_t offset = lattice_size_SU2 * spin_dim_SU2;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            const size_t atom = i % N_atoms_SU3;
            const double sign = afm_sublattice_signs_SU3[atom];
            const size_t idx = offset + i * spin_dim_SU3;
            for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                    M_stag_arr[mu] += sign * sublattice_frames_SU3[atom](mu, nu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU3; ++d) M_stag_arr[d] /= double(lattice_size_SU3);
    }

    /**
     * All magnetisation observables of a flat state, as recorded by every
     * pulse-drive / MD driver (one definition for all of them):
     * SU(2): [staggered (frames x AFM signs), mean local-frame, mean global]
     * SU(3): [staggered (frames x AFM signs), mean local-frame, mean global].
     */
    Observables observe(const double* x) const {
        Observables o;
        double buf[8];
        compute_magnetization_staggered_SU2_from_flat(x, buf);
        o.first[0] = Eigen::Map<const Eigen::VectorXd>(buf, spin_dim_SU2);
        o.first[1] = SpinVector::Zero(spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i)
            o.first[1] += Eigen::Map<const Eigen::VectorXd>(x + i * spin_dim_SU2, spin_dim_SU2);
        o.first[1] /= double(lattice_size_SU2);
        compute_magnetization_global_SU2_from_flat(x, buf);
        o.first[2] = Eigen::Map<const Eigen::VectorXd>(buf, spin_dim_SU2);

        const size_t off = lattice_size_SU2 * spin_dim_SU2;
        compute_magnetization_staggered_SU3_from_flat(x, buf);
        o.second[0] = Eigen::Map<const Eigen::VectorXd>(buf, spin_dim_SU3);
        o.second[1] = SpinVector::Zero(spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i)
            o.second[1] += Eigen::Map<const Eigen::VectorXd>(x + off + i * spin_dim_SU3, spin_dim_SU3);
        o.second[1] /= double(lattice_size_SU3);
        compute_magnetization_global_SU3_from_flat(x, buf);
        o.second[2] = Eigen::Map<const Eigen::VectorXd>(buf, spin_dim_SU3);
        return o;
    }

    // ============================================================
    // FILE I/O
    // ============================================================

    /**
     * Save the spin configuration to <filename>_SU2.txt and <filename>_SU3.txt
     * (one site per line, full precision: reloads bitwise). Throws
     * std::runtime_error if a file cannot be written.
     */
    void save_spin_config(const string& filename) const {
        classical_spin::io::write_table(filename + "_SU2.txt", lattice_size_SU2, spin_dim_SU2,
                                        [&](size_t i, size_t j) { return spins_SU2[i](j); });
        classical_spin::io::write_table(filename + "_SU3.txt", lattice_size_SU3, spin_dim_SU3,
                                        [&](size_t i, size_t j) { return spins_SU3[i](j); });
    }

    /**
     * Save spin configuration to a directory with clean naming
     * Creates: <prefix>_SU2.txt and <prefix>_SU3.txt in the directory
     */
    void save_spin_config_to_dir(const string& dir, const string& prefix = "spins") const {
        save_spin_config(dir + "/" + prefix);
    }

    /**
     * Save energy information to a directory
     * Creates: energy.txt with total, SU2, and SU3 energies (both total and per-site)
     */
    void save_energy_to_dir(const string& dir, const string& prefix = "energy") const {
        double E_total = total_energy();
        double E_SU2 = total_energy_SU2();
        double E_SU3 = total_energy_SU3();
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        
        ofstream file(dir + "/" + prefix + ".txt");
        file << std::setprecision(15);
        file << "# Energy summary" << endl;
        file << "# N_SU2 = " << lattice_size_SU2 << endl;
        file << "# N_SU3 = " << lattice_size_SU3 << endl;
        file << "# N_total = " << total_sites << endl;
        file << "#" << endl;
        file << "# Total energies:" << endl;
        file << "E_total = " << E_total << endl;
        file << "E_SU2 = " << E_SU2 << endl;
        file << "E_SU3 = " << E_SU3 << endl;
        file << "#" << endl;
        file << "# Energy per site:" << endl;
        file << "E_total/N = " << E_total / total_sites << endl;
        file << "E_SU2/N_SU2 = " << E_SU2 / lattice_size_SU2 << endl;
        file << "E_SU3/N_SU3 = " << E_SU3 / lattice_size_SU3 << endl;
        file.close();
    }

    /**
     * Load a configuration written by save_spin_config from <filename>_SU2.txt
     * and <filename>_SU3.txt: exactly lattice_size_SU2 (SU3) lines of
     * spin_dim_SU2 (SU3) finite numbers each. Throws std::runtime_error naming
     * file and line on a missing file, a short file, a wrong number of
     * columns, a non-finite value, extra rows or a zero vector; both species
     * are parsed before anything is stored, so the state is unchanged on error.
     *
     * Every spin is rescaled to its length: SU(2) spins to spin_length_SU2;
     * SU(3) vectors within 1e-3 of the pure-qutrit length 2/sqrt(3) (states
     * from physicalize_SU3_state / deterministic_sweep_SU3_exact_diag) to
     * exactly 2/sqrt(3), all others to spin_length_SU3. This removes the
     * round-off of saved files without changing which manifold a state is on.
     */
    void load_spin_config(const string& filename) {
        using classical_spin::io::Table;
        const Table t2 = classical_spin::io::read_table(filename + "_SU2.txt", lattice_size_SU2, spin_dim_SU2);
        const Table t3 = classical_spin::io::read_table(filename + "_SU3.txt", lattice_size_SU3, spin_dim_SU3);
        double max_dev = 0.0;
        auto normalised = [&](const Table& t, size_t i, size_t dim, double length, bool su3,
                              const string& file) -> SpinVector {
            SpinVector s = Eigen::Map<const Eigen::VectorXd>(t.row(i), Eigen::Index(dim));
            const double n = s.norm();
            if (!(n > 0.0)) throw std::runtime_error(file + ":" + std::to_string(t.lines[i]) + ": zero spin vector");
            const double pure = 2.0 / std::sqrt(3.0);
            const double target = (su3 && dim == 8 && std::abs(n - pure) <= 1e-3 * pure) ? pure : length;
            max_dev = std::max(max_dev, std::abs(n - target) / target);
            // A spin already of the target length to round-off is kept bitwise.
            if (std::abs(n - target) <= 4.0 * std::numeric_limits<double>::epsilon() * target) return s;
            return s * (target / n);
        };
        SpinConfigSU2 s2(lattice_size_SU2);
        SpinConfigSU3 s3(lattice_size_SU3);
        for (size_t i = 0; i < lattice_size_SU2; ++i)
            s2[i] = normalised(t2, i, spin_dim_SU2, spin_length_SU2, false, filename + "_SU2.txt");
        for (size_t i = 0; i < lattice_size_SU3; ++i)
            s3[i] = normalised(t3, i, spin_dim_SU3, spin_length_SU3, true, filename + "_SU3.txt");
        spins_SU2 = std::move(s2);
        spins_SU3 = std::move(s3);
        if (max_dev > 1e-3)
            std::cerr << "Warning: " << filename << "_SU{2,3}.txt: spins rescaled to their lengths (largest "
                      << "relative change " << max_dev << ")" << std::endl;
    }

    /**
     * Save site positions (legacy naming - appends _SU2.txt/_SU3.txt to filename)
     */
    void save_positions(const string& filename) const {
        // SU(2) positions
        {
            ofstream file(filename + "_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                file << site_positions_SU2[i](0) << " "
                     << site_positions_SU2[i](1) << " "
                     << site_positions_SU2[i](2) << "\n";
            }
        }
        
        // SU(3) positions
        {
            ofstream file(filename + "_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                file << site_positions_SU3[i](0) << " "
                     << site_positions_SU3[i](1) << " "
                     << site_positions_SU3[i](2) << "\n";
            }
        }
    }

    /**
     * Save site positions to a directory with clean naming
     * Creates: positions_SU2.txt and positions_SU3.txt in the directory
     */
    void save_positions_to_dir(const string& dir) const {
        // SU(2) positions
        {
            ofstream file(dir + "/positions_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                file << site_positions_SU2[i](0) << " "
                     << site_positions_SU2[i](1) << " "
                     << site_positions_SU2[i](2) << "\n";
            }
        }
        
        // SU(3) positions
        {
            ofstream file(dir + "/positions_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                file << site_positions_SU3[i](0) << " "
                     << site_positions_SU3[i](1) << " "
                     << site_positions_SU3[i](2) << "\n";
            }
        }
    }

    /**
     * Initialize with a uniform state: SU(2) spins along direction_SU2; SU(3)
     * states along direction_SU3, which on CP^2 is replaced by the closest
     * pure state (top eigenvector of rho = 1/3 + direction.lambda / (2|direction|);
     * e.g. lambda_3 gives |1>, n = (0,0,1,0,0,0,0,1/sqrt3)) and on the sphere
     * scaled to spin_length_SU3.
     */
    void init_ferromagnetic(const SpinVector& direction_SU2, const SpinVector& direction_SU3) {
        const SpinVector dir_SU2 = direction_SU2.normalized() * spin_length_SU2;
        SpinVector dir_SU3 = direction_SU3.normalized() * spin_length_SU3;
        if (su3_on_cp2()) {
            const SpinVector unit = direction_SU3.normalized();
            classical_spin::su3::Vector8r v;
            for (int a = 0; a < 8; ++a) v(a) = unit(a);
            const auto psi = classical_spin::su3::psi_from_expectations(v);
            std::complex<double> p[3] = {psi(0), psi(1), psi(2)};
            classical_spin::su3::pure_expectations(p, dir_SU3.data());
        }

        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins_SU2[i] = dir_SU2;
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins_SU3[i] = dir_SU3;
        }
    }

    /**
     * Initialize with an uncorrelated random state: SU(2) spins uniform on
     * the sphere, SU(3) states uniform on su3_mc_manifold (Haar on CP^2).
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins_SU2[i].resize(spin_dim_SU2);
            random_point_on_sphere(spins_SU2[i].data(), spin_dim_SU2, double(spin_length_SU2));
        }
        for (size_t j = 0; j < lattice_size_SU3; ++j) {
            spins_SU3[j].resize(spin_dim_SU3);
            random_SU3_state(spins_SU3[j].data());
        }
    }

    /**
     * Dynamics under ONE pulse centred at t_B (SU(2) and SU(3) components share
     * the centre), starting from the current spins, which are left unchanged.
     * Returns observe() at every point of the exact grid
     * t_k = T_start + k step, k = 0..n (t_n <= T_end): n + 1 samples whose
     * times are computed from integers, so trajectories with different pulse
     * times are sample-by-sample comparable. `spin_state_out` (optional)
     * receives the full spin state at every sample.
     *
     * Adaptive methods (dopri5) use dense output: steps are chosen by the
     * error controller, capped inside the pulse window (|t - t_B| <= 9 width)
     * at min(step, width/4, period/4) so the pulse cannot be stepped over.
     * `pulse_window_chunking` is accepted for compatibility and ignored: the
     * exact-grid integration needs no chunking (the chunk seams of the old
     * scheme dropped steps).
     */
    PumpProbeTrajectory single_pulse_drive(
               const vector<SpinVector>& field_in_SU2, const vector<SpinVector>& field_in_SU3,
               double t_B,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method = "dopri5", bool use_gpu = false,
               vector<vector<double>>* spin_state_out = nullptr,
               bool pulse_window_chunking = true,
               double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
               double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * Dynamics under TWO pulses centred at t_B_1 and t_B_2, with independent
     * per-pulse directions; otherwise identical to single_pulse_drive.
     */
    PumpProbeTrajectory double_pulse_drive(
               const vector<SpinVector>& field_in_1_SU2, const vector<SpinVector>& field_in_1_SU3,
               double t_B_1,
               const vector<SpinVector>& field_in_2_SU2, const vector<SpinVector>& field_in_2_SU3,
               double t_B_2,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method = "dopri5", bool use_gpu = false,
               vector<vector<double>>* spin_state_out = nullptr,
               bool pulse_window_chunking = true,
               double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
               double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * Free (undriven) dynamics of the current spins over [T_start, T_end],
     * written to out_dir/trajectory.h5 on the uniform grid
     * t_k = T_start + k (save_interval dt_initial). Fixed-step methods take
     * save_interval steps of dt_initial per sample; adaptive methods use dense
     * output with dt_initial as the initial step. Diagnostics on the same grid
     * go to the /diagnostics group: energy per site and the drifts of |S_i|
     * and of the two SU(3) Casimirs |n_i|^2, d_abc n^a n^b n^c (conserved by
     * the undamped flow); a non-finite state aborts with an exception.
     * Non-positive tolerances select the method defaults (1e-6; 1e-8 for BS).
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           const string& out_dir = "", size_t save_interval = 100,
                           const string& method = "dopri5", bool use_gpu = false,
                           double abs_tol = -1.0, double rel_tol = -1.0);

    /**
     * CPU implementation of molecular_dynamics (requires HDF5 for output).
     */
    void molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           const string& out_dir = "", size_t save_interval = 100,
                           const string& method = "dopri5",
                           double abs_tol = -1.0, double rel_tol = -1.0);

    /**
     * Print lattice information
     */
    void print_info() const {
        cout << "=== Mixed Lattice Information ===" << endl;
        cout << "Dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "\nSU(2) Sublattice:" << endl;
        cout << "  Sites: " << lattice_size_SU2 << endl;
        cout << "  Spin dimension: " << spin_dim_SU2 << endl;
        cout << "  Atoms per cell: " << N_atoms_SU2 << endl;
        cout << "  Max bilinear: " << num_bi_SU2 << endl;
        cout << "  Max trilinear: " << num_tri_SU2 << endl;
        cout << "\nSU(3) Sublattice:" << endl;
        cout << "  Sites: " << lattice_size_SU3 << endl;
        cout << "  Spin dimension: " << spin_dim_SU3 << endl;
        cout << "  Atoms per cell: " << N_atoms_SU3 << endl;
        cout << "  Max bilinear: " << num_bi_SU3 << endl;
        cout << "  Max trilinear: " << num_tri_SU3 << endl;
        cout << "\nMixed Interactions:" << endl;
        cout << "  Bilinear: " << num_bi_SU2_SU3 << endl;
        cout << "  Trilinear: " << num_tri_SU2_SU3 << endl;
        cout << "=================================" << endl;
    }

    // ------------------------------------------------------------------
    // Pump-probe (2DCS) spectroscopy.
    //
    // For pulse fluences A (pump at t = 0) and B (probe at t = tau) the
    // drivers record, on ONE exact grid t_k = T_start + k T_step shared by
    // all trajectories,
    //     M0(t)        pump only,
    //     M1(tau; t)   probe only,
    //     M01(tau; t)  pump + probe,
    // so that the nonlinear signal M_NL = M01 - M0 - M1 (formed in
    // post-processing) is well defined sample by sample.
    //
    // Two exact savings are applied when their preconditions hold:
    //   W1  M1(tau; t) = M0(t - tau) for a stationary initial state of the
    //       autonomous (undriven) flow and tau on the grid; checked with the
    //       dimensionless residual relative_stationarity_residual() <=
    //       stationarity_tol. Disabled for distinct probe directions, for
    //       spin-state output and on the GPU.
    //   M01 For t < tau - W (W = probe half-support, 9 widths, where the
    //       probe envelope is < 1.6e-9 of its peak) the pump-probe drive
    //       equals the pump-only drive, so M01 is continued from the M0 state
    //       stored at that grid point and its earlier samples are those of M0
    //       (exactly zero M_NL before the probe arrives, and roughly half the
    //       M01 cost). Controlled by `reuse_m0_for_m01` (config key of the
    //       same name, default on).
    // ------------------------------------------------------------------

    /**
     * max_i |dS_i/dt|_inf of the current spins with the drive removed and
     * damping included (absolute, in inverse time units). See
     * relative_stationarity_residual() for the scale-free W1 criterion.
     */
    double max_dSdt_norm_no_drive() const;

    /**
     * M1(tau; t_k) = M0(t_k - tau) for a trajectory M0 recorded with the pump
     * at t = 0 on the grid T_start + k T_step: the sample index is shifted by
     * the integer m = tau / T_step and samples before the probe take the
     * stationary `M_ground`. Throws std::invalid_argument if tau is not a
     * multiple of T_step (to 1e-9 relative) -- rounding it, as before, put
     * M1 up to T_step/2 off its time label.
     */
    PumpProbeTrajectory synthesize_M1_from_M0(
        const PumpProbeTrajectory& M_pulse_trajectory,
        const Observables& M_ground,
        double tau,
        double T_start, double T_end, double T_step) const;

    /**
     * Pump-probe / 2DCS scan on one process (optionally OpenMP-parallel over
     * tau with `outer_omp_threads` > 1; the scan is serial on the GPU so every
     * trajectory uses one backend). Writes dir_name/pump_probe_spectroscopy.h5.
     * The ground state must be prepared beforehand (annealed or loaded).
     *   reuse_m0_for_m1   enable W1 (see above), stationarity_tol its threshold
     *   field_in_SU2_B / field_in_SU3_B  probe directions (empty: same as pump)
     *   reuse_m0_for_m01  continue M01 from stored M0 states (see above)
     * Delays: tau_start + i tau_step for i = 0..n-1 with tau_end included (to
     * round-off); tau_step must be non-zero and point towards tau_end.
     */
    void pump_probe_spectroscopy(const vector<SpinVector>& field_in_SU2,
                                 const vector<SpinVector>& field_in_SU3,
                                 double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                 double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start = 5.0, double Temp_end = 1e-3,
                                 size_t n_anneal = 1000,
                                 bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                 string dir_name = "spectroscopy_mixed",
                                 string method = "dopri5", bool use_gpu = false,
                                 bool save_spin_trajectories = false,
                                 bool reuse_m0_for_m1 = true,
                                 double stationarity_tol = 1e-6,
                                 int outer_omp_threads = 0,
                                 bool pulse_window_chunking = true,
                                 double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                 double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol,
                                 const vector<SpinVector>& field_in_SU2_B = {},
                                 const vector<SpinVector>& field_in_SU3_B = {},
                                 bool reuse_m0_for_m01 = true);

    /**
     * MPI version of pump_probe_spectroscopy with the same physics and the
     * same HDF5 file layout. Rank 0 computes M0, synthesises M1 under W1 and
     * writes the file; the other ranks pull delay indices from rank 0 one at a
     * time (dynamic load balancing) and stream each result back as soon as it
     * is computed, so no rank holds more than one delay point. With a single
     * rank it falls back to the serial driver. Errors on any rank (invalid
     * input, non-finite dynamics, I/O failure) are reported to every rank and
     * rethrown everywhere as std::runtime_error -- no rank is left blocked.
     */
    void pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in_SU2,
                                     const vector<SpinVector>& field_in_SU3,
                                     double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                     double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start = 5.0, double Temp_end = 1e-3,
                                     size_t n_anneal = 1000,
                                     bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                     string dir_name = "spectroscopy_mixed",
                                     string method = "dopri5", bool use_gpu = false,
                                     bool save_spin_trajectories = false,
                                     bool reuse_m0_for_m1 = true,
                                     double stationarity_tol = 1e-6,
                                     bool pulse_window_chunking = true,
                                     double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                     double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol,
                                     const vector<SpinVector>& field_in_SU2_B = {},
                                     const vector<SpinVector>& field_in_SU3_B = {},
                                     bool reuse_m0_for_m01 = true);

private:
    struct SpectroscopyPlan;   // validated scan description (mixed_lattice_md.cpp)
    struct DelayResult;        // M1 / partial M01 of one delay
    // M0 with the states needed to continue M01 (plan.checkpoint_indices).
    PumpProbeTrajectory run_reference_trajectory(const SpectroscopyPlan& plan, const ODEState& x_ground,
                                                 vector<vector<double>>* spin_states,
                                                 vector<ODEState>* checkpoints);
    // M1 (unless synthesised) and M01 from its start index for delay j.
    void compute_delay(const SpectroscopyPlan& plan, size_t j, const ODEState& x_ground,
                       const ODEState* checkpoint, DelayResult& out);
    SpectroscopyPlan plan_spectroscopy(const vector<SpinVector>& field_in_SU2,
                                       const vector<SpinVector>& field_in_SU3,
                                       const vector<SpinVector>& field_in_SU2_B,
                                       const vector<SpinVector>& field_in_SU3_B,
                                       double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                       double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                       double tau_start, double tau_end, double tau_step,
                                       double T_start, double T_end, double T_step,
                                       const string& method, bool use_gpu, bool save_spin_trajectories,
                                       bool reuse_m0_for_m1, double stationarity_tol,
                                       bool reuse_m0_for_m01, double abs_tol, double rel_tol) const;
public:


// =============================================================================
// GPU backend glue (opaque API of mixed_lattice_gpu_api.h; untested here: no
// CUDA toolchain in CI). Methods and tolerances follow mixed_gpu::integrate_mixed_gpu.
// =============================================================================
#ifdef CUDA_ENABLED
private:
    // Device copy of the Hamiltonian, uploaded on first GPU use. Freed by the
    // destructor; copies (e.g. the per-thread clones of the 2DCS drivers)
    // start without one instead of sharing the raw handle.
    mutable classical_spin::gpu::DeviceHandle<mixed_gpu::GPUMixedLatticeDataHandle,
                                              &mixed_gpu::destroy_gpu_mixed_lattice_data> gpu_mixed_handle_;
    
    /**
     * Flatten SU(2) sublattice data for GPU transfer
     */
    void flatten_SU2_data(
        vector<double>& flat_field,
        vector<double>& flat_onsite,
        vector<double>& flat_bilinear,
        vector<size_t>& flat_partners,
        vector<size_t>& num_bilinear_per_site
    ) const {
        flat_field.clear();
        flat_field.reserve(lattice_size_SU2 * spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t d = 0; d < spin_dim_SU2; ++d) {
                flat_field.push_back(field_SU2[i](d));
            }
        }
        
        flat_onsite.clear();
        flat_onsite.reserve(lattice_size_SU2 * spin_dim_SU2 * spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t r = 0; r < spin_dim_SU2; ++r) {
                for (size_t c = 0; c < spin_dim_SU2; ++c) {
                    flat_onsite.push_back(onsite_interaction_SU2[i](r, c));
                }
            }
        }
        
        flat_bilinear.clear();
        flat_partners.clear();
        num_bilinear_per_site.clear();
        flat_bilinear.reserve(lattice_size_SU2 * num_bi_SU2 * spin_dim_SU2 * spin_dim_SU2);
        flat_partners.reserve(lattice_size_SU2 * num_bi_SU2);
        num_bilinear_per_site.reserve(lattice_size_SU2);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners_SU2[i].size());
            for (size_t n = 0; n < num_bi_SU2; ++n) {
                if (n < bilinear_partners_SU2[i].size()) {
                    flat_partners.push_back(bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_bilinear.push_back(bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim_SU2 * spin_dim_SU2; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
    }
    
    /**
     * Flatten SU(3) sublattice data for GPU transfer
     */
    void flatten_SU3_data(
        vector<double>& flat_field,
        vector<double>& flat_onsite,
        vector<double>& flat_bilinear,
        vector<size_t>& flat_partners,
        vector<size_t>& num_bilinear_per_site
    ) const {
        flat_field.clear();
        flat_field.reserve(lattice_size_SU3 * spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t d = 0; d < spin_dim_SU3; ++d) {
                flat_field.push_back(field_SU3[i](d));
            }
        }
        
        flat_onsite.clear();
        flat_onsite.reserve(lattice_size_SU3 * spin_dim_SU3 * spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t r = 0; r < spin_dim_SU3; ++r) {
                for (size_t c = 0; c < spin_dim_SU3; ++c) {
                    flat_onsite.push_back(onsite_interaction_SU3[i](r, c));
                }
            }
        }
        
        flat_bilinear.clear();
        flat_partners.clear();
        num_bilinear_per_site.clear();
        flat_bilinear.reserve(lattice_size_SU3 * num_bi_SU3 * spin_dim_SU3 * spin_dim_SU3);
        flat_partners.reserve(lattice_size_SU3 * num_bi_SU3);
        num_bilinear_per_site.reserve(lattice_size_SU3);
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners_SU3[i].size());
            for (size_t n = 0; n < num_bi_SU3; ++n) {
                if (n < bilinear_partners_SU3[i].size()) {
                    flat_partners.push_back(bilinear_partners_SU3[i][n]);
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_bilinear.push_back(bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim_SU3 * spin_dim_SU3; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
    }
    
    /**
     * Ensure GPU mixed lattice data is initialized (lazy initialization)
     */
    void ensure_gpu_mixed_data_initialized() const {
        if (gpu_mixed_handle_) return;
        check_gpu_supported();  // the device kernels implement only part of the model
        
        // Flatten SU(2) data
        vector<double> flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2;
        vector<size_t> flat_partners_SU2, num_bi_per_site_SU2;
        flatten_SU2_data(flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2,
                        flat_partners_SU2, num_bi_per_site_SU2);
        
        // Flatten SU(3) data
        vector<double> flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3;
        vector<size_t> flat_partners_SU3, num_bi_per_site_SU3;
        flatten_SU3_data(flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3,
                        flat_partners_SU3, num_bi_per_site_SU3);
        
        // Compute max mixed bilinear neighbors
        size_t max_mixed_bi = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_mixed_bi = std::max(max_mixed_bi, mixed_bilinear_partners_SU2[i].size());
        }
        size_t max_mixed_bi_SU3 = 0;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_mixed_bi_SU3 = std::max(max_mixed_bi_SU3, mixed_bilinear_partners_SU3[i].size());
        }
        
        // Flatten mixed bilinear from SU2 perspective (3x8 matrices)
        vector<double> flat_mixed_bilinear;
        vector<size_t> flat_mixed_partners_SU2, flat_mixed_partners_SU3, num_mixed_per_site_SU2;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            num_mixed_per_site_SU2.push_back(mixed_bilinear_partners_SU2[i].size());
            for (size_t n = 0; n < max_mixed_bi; ++n) {
                if (n < mixed_bilinear_partners_SU2[i].size()) {
                    flat_mixed_partners_SU2.push_back(i);
                    flat_mixed_partners_SU3.push_back(mixed_bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_mixed_bilinear.push_back(mixed_bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                } else {
                    flat_mixed_partners_SU2.push_back(SIZE_MAX);
                    flat_mixed_partners_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU2 * spin_dim_SU3; ++j) {
                        flat_mixed_bilinear.push_back(0.0);
                    }
                }
            }
        }
        
        // Flatten mixed bilinear from SU3 perspective (8x3 matrices)
        vector<double> flat_mixed_bilinear_SU3;
        vector<size_t> flat_mixed_partners_SU2_from_SU3, num_mixed_per_site_SU3;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            num_mixed_per_site_SU3.push_back(mixed_bilinear_partners_SU3[i].size());
            for (size_t n = 0; n < max_mixed_bi_SU3; ++n) {
                if (n < mixed_bilinear_partners_SU3[i].size()) {
                    flat_mixed_partners_SU2_from_SU3.push_back(mixed_bilinear_partners_SU3[i][n]);
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_mixed_bilinear_SU3.push_back(mixed_bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                } else {
                    flat_mixed_partners_SU2_from_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU3 * spin_dim_SU2; ++j) {
                        flat_mixed_bilinear_SU3.push_back(0.0);
                    }
                }
            }
        }
        
        // Create GPU handle
        gpu_mixed_handle_.reset(mixed_gpu::create_gpu_mixed_lattice_data(
            lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
            lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
            num_bi_SU2, num_bi_SU3, max_mixed_bi, max_mixed_bi_SU3,
            flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2,
            flat_partners_SU2, num_bi_per_site_SU2,
            flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3,
            flat_partners_SU3, num_bi_per_site_SU3,
            flat_mixed_bilinear, flat_mixed_partners_SU2,
            flat_mixed_partners_SU3, num_mixed_per_site_SU2,
            flat_mixed_bilinear_SU3, flat_mixed_partners_SU2_from_SU3, num_mixed_per_site_SU3));
    }
    
    /**
     * Update GPU pulse parameters for SU(2)
     */
    void update_gpu_pulse_SU2() const {
        if (!gpu_mixed_handle_) return;
        
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms_SU2 * spin_dim_SU2);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive_SU2[p].size(); ++d) {
                flat_field_drive.push_back(field_drive_SU2[p](d));
            }
        }
        
        mixed_gpu::set_gpu_pulse_SU2(
            gpu_mixed_handle_.get(),
            flat_field_drive,
            field_drive_amp_SU2,
            field_drive_width_SU2,
            field_drive_freq_SU2,
            t_pulse_SU2[0],
            t_pulse_SU2[1]
        );
    }
    
    /**
     * Update GPU pulse parameters for SU(3)
     */
    void update_gpu_pulse_SU3() const {
        if (!gpu_mixed_handle_) return;
        
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms_SU3 * spin_dim_SU3);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive_SU3[p].size(); ++d) {
                flat_field_drive.push_back(field_drive_SU3[p](d));
            }
        }
        
        mixed_gpu::set_gpu_pulse_SU3(
            gpu_mixed_handle_.get(),
            flat_field_drive,
            field_drive_amp_SU3,
            field_drive_width_SU3,
            field_drive_freq_SU3,
            t_pulse_SU3[0],
            t_pulse_SU3[1]
        );
    }
    
    /**
     * GPU version of molecular_dynamics using opaque API
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                               const string& out_dir = "", size_t save_interval = 100,
                               const string& method = "dopri5", double abs_tol = 1e-6,
                               double rel_tol = 1e-6) {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        return;
#else
        ensure_directory_exists(out_dir);
        
        cout << "Running mixed lattice molecular dynamics with GPU acceleration (API): t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Step size: " << dt_initial << endl;
        
        // Ensure GPU data is initialized
        ensure_gpu_mixed_data_initialized();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state();
        mixed_gpu::set_gpu_mixed_spins(gpu_mixed_handle_.get(), h_state);
        
        // Create HDF5 writer
        std::unique_ptr<HDF5MixedMDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MixedMDWriter>(
                hdf5_file, 
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3, method + "_gpu_api", 
                dt_initial, T_start, T_end, save_interval, 
                spin_length_SU2, spin_length_SU3,
                &site_positions_SU2, &site_positions_SU3, 10000);
        }
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> trajectory;
        mixed_gpu::integrate_mixed_gpu(gpu_mixed_handle_.get(), T_start, T_end, dt_initial,
                                       save_interval, trajectory, method, abs_tol, rel_tol);
        
        // Write trajectory to HDF5
        size_t save_count = 0;
        for (const auto& [t, state_vec] : trajectory) {
            const Observables o = observe(state_vec.data());
            const SpinVector& M_SU2 = o.first[1];
            const SpinVector& M_SU3 = o.second[1];
            
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, o.first[0], o.first[1], o.first[2],
                                            o.second[0], o.second[1], o.second[2], state_vec.data());
                save_count++;
            }
            
            if (save_count % 10 == 0) {
                cout << "t=" << t << ", |M_SU2|=" << M_SU2.norm() << ", |M_SU3|=" << M_SU3.norm() << endl;
            }
        }
        
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "GPU molecular dynamics complete!" << endl;
#endif
    }
    
    /**
     * GPU version of single_pulse_drive using opaque API
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>
    single_pulse_drive_gpu(const vector<SpinVector>& field_in_SU2,
                           const vector<SpinVector>& field_in_SU3,
                           double t_B,
                           double pulse_amp_SU2_in, double pulse_width_SU2_in, double pulse_freq_SU2_in,
                           double pulse_amp_SU3_in, double pulse_width_SU3_in, double pulse_freq_SU3_in,
                           double T_start, double T_end, double step_size,
                           const string& method = "dopri5", double abs_tol = 1e-8,
                           double rel_tol = 1e-8) {
        
        // Set up pulses
        set_pulse_SU2(field_in_SU2, t_B, 
                     vector<SpinVector>(N_atoms_SU2, SpinVector::Zero(spin_dim_SU2)), 0.0,
                     pulse_amp_SU2_in, pulse_width_SU2_in, pulse_freq_SU2_in);
        set_pulse_SU3(field_in_SU3, t_B,
                     vector<SpinVector>(N_atoms_SU3, SpinVector::Zero(spin_dim_SU3)), 0.0,
                     pulse_amp_SU3_in, pulse_width_SU3_in, pulse_freq_SU3_in);
        
        ensure_gpu_mixed_data_initialized();
        update_gpu_pulse_SU2();
        update_gpu_pulse_SU3();
        
        // Transfer initial state
        ODEState h_state = spins_to_state();
        mixed_gpu::set_gpu_mixed_spins(gpu_mixed_handle_.get(), h_state);
        
        // Integrate
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        mixed_gpu::integrate_mixed_gpu(gpu_mixed_handle_.get(), T_start, T_end, step_size,
                                       1, raw_trajectory, method, abs_tol, rel_tol);
        
        // Same observables as the CPU drivers (MixedLattice::observe).
        PumpProbeTrajectory trajectory;
        trajectory.reserve(raw_trajectory.size());
        for (const auto& [t, state_vec] : raw_trajectory) {
            trajectory.emplace_back(t, observe(state_vec.data()));
        }
        
        reset_pulse();
        
        return trajectory;
    }
    
    /**
     * GPU version of double_pulse_drive using opaque API
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>
    double_pulse_drive_gpu(const vector<SpinVector>& field_in_1_SU2,
                           const vector<SpinVector>& field_in_1_SU3,
                           double t_B_1,
                           const vector<SpinVector>& field_in_2_SU2,
                           const vector<SpinVector>& field_in_2_SU3,
                           double t_B_2,
                           double pulse_amp_SU2_in, double pulse_width_SU2_in, double pulse_freq_SU2_in,
                           double pulse_amp_SU3_in, double pulse_width_SU3_in, double pulse_freq_SU3_in,
                           double T_start, double T_end, double step_size,
                           const string& method = "dopri5", double abs_tol = 1e-8,
                           double rel_tol = 1e-8) {
        
        // Set up two-pulse configuration
        set_pulse_SU2(field_in_1_SU2, t_B_1, field_in_2_SU2, t_B_2,
                     pulse_amp_SU2_in, pulse_width_SU2_in, pulse_freq_SU2_in);
        set_pulse_SU3(field_in_1_SU3, t_B_1, field_in_2_SU3, t_B_2,
                     pulse_amp_SU3_in, pulse_width_SU3_in, pulse_freq_SU3_in);
        
        ensure_gpu_mixed_data_initialized();
        update_gpu_pulse_SU2();
        update_gpu_pulse_SU3();
        
        // Transfer initial state
        ODEState h_state = spins_to_state();
        mixed_gpu::set_gpu_mixed_spins(gpu_mixed_handle_.get(), h_state);
        
        // Integrate
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        mixed_gpu::integrate_mixed_gpu(gpu_mixed_handle_.get(), T_start, T_end, step_size,
                                       1, raw_trajectory, method, abs_tol, rel_tol);
        
        // Same observables as the CPU drivers (MixedLattice::observe).
        PumpProbeTrajectory trajectory;
        trajectory.reserve(raw_trajectory.size());
        for (const auto& [t, state_vec] : raw_trajectory) {
            trajectory.emplace_back(t, observe(state_vec.data()));
        }
        
        reset_pulse();
        
        return trajectory;
    }
#endif // CUDA_ENABLED

};

#endif // MIXED_LATTICE_REFACTORED_H
