#ifndef MIXED_LATTICE_REFACTORED_H
#define MIXED_LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "classical_spin/core/spin_config.h"  // For should_rank_write
#include "classical_spin/core/su3_coherent_state.h"  // SU(3) coherent-state utilities
                                                     // (Zhang & Batista, PRB 104, 104409 (2021))
#include "classical_spin/mc/mc_common.h"      // Common MC structs & templates
#include "classical_spin/mc/parallel_tempering.h"  // replica-exchange engine + ladder tuning
#include "classical_spin/lattice/pulse_chunking.h"  // default pump-probe tolerances
#include "classical_spin/dynamics/grid_integrate.h"   // exact-grid ODE integration
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
 * - Monte Carlo sampling (Metropolis, overrelaxation)
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
    float spin_length_SU3;       // Magnitude of SU(3) spin vectors

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
    //     SU(2)/SU(3) MD RHS.
    // Built once in build_packed_interaction_buffers() at the end of the
    // constructor; the original SpinMatrix / SpinTensor3 storage is kept
    // for the MC code path and any rebuild operations.
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
    // LOCAL FIELD CACHING FOR OPTIMIZED MONTE CARLO
    // ============================================================
    // Cached local fields for each site (used in interleaved sweeps)
    mutable vector<SpinVector> cached_local_field_SU2;
    mutable vector<SpinVector> cached_local_field_SU3;
    mutable vector<bool> field_valid_SU2;  // Whether cached field is valid
    mutable vector<bool> field_valid_SU3;  // Whether cached field is valid
    mutable bool use_field_caching;        // Enable/disable caching mode

    // Reverse lookup: which SU3 sites are affected by changes to each SU2 site
    vector<vector<size_t>> mixed_bilinear_reverse_SU2;  // SU2[i] -> list of SU3 sites coupled to it
    // Reverse lookup: which SU2 sites are affected by changes to each SU3 site  
    vector<vector<size_t>> mixed_bilinear_reverse_SU3;  // SU3[i] -> list of SU2 sites coupled to it

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

        // Initialize local field caching infrastructure
        cached_local_field_SU2.resize(lattice_size_SU2);
        cached_local_field_SU3.resize(lattice_size_SU3);
        field_valid_SU2.resize(lattice_size_SU2, false);
        field_valid_SU3.resize(lattice_size_SU3, false);
        use_field_caching = false;  // Disabled by default

        // Initialize SU(3) Bloch damping equilibrium (default: zero = infinite temperature)
        equilibrium_SU3.resize(lattice_size_SU3, SpinVector::Zero(spin_dim_SU3));

        // Build reverse lookup tables for mixed interactions
        build_reverse_lookup_tables();

        // Build per-sublattice colour partition for the parallel coloured
        // Metropolis / over-relaxation sweeps. See header doc on
        // color_of_site_SU{2,3} for the edge-set we use.
        build_color_partition();

        // Pack {bi,tri}linear interaction tensors into row-major double[]
        // buffers used by the MD hot path (see field declarations).
        build_packed_interaction_buffers();

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
     * Apply periodic boundary condition
     */
    size_t periodic_boundary(int coord, size_t dim_size) const {
        if (coord < 0) {
            return coord + dim_size;
        } else if (coord >= (int)dim_size) {
            return coord - dim_size;
        }
        return coord;
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
                         float spin_length, size_t spin_dim, size_t N_atoms)
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
                        
                        // Generate random spin
                        spins[site_idx] = gen_random_spin(spin_length, spin_dim);
                        
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

        num_bi = *std::max_element(bi_count.begin(), bi_count.end());
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
     * contiguous row-major double[] buffers used by the MD hot path
     * (`get_local_field_*_flat_into`).
     *
     * Layout (per site, n indexes the bond):
     *   bilinear_packed_*[site][n*da*db + a*db + b]              = J^n(a,b)
     *   trilinear_packed_*[site][n*da*db*dc + (a*db + b)*dc + c] = T^n[a](b,c)
     *
     * The original `bilinear_interaction_*`, `trilinear_interaction_*`,
     * `mixed_*_interaction_*` storage is preserved (used by MC code path
     * and for any rebuilds). Only one rebuild call is required after the
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
    }

    // ============================================================
    // LOCAL FIELD CACHING INFRASTRUCTURE
    // ============================================================

    /**
     * Build reverse lookup tables for mixed bilinear interactions
     * 
     * These tables enable efficient cache invalidation:
     * - mixed_bilinear_reverse_SU2[i] = list of SU(3) sites whose fields depend on SU(2) site i
     * - mixed_bilinear_reverse_SU3[i] = list of SU(2) sites whose fields depend on SU(3) site i
     */
    void build_reverse_lookup_tables() {
        // Initialize reverse lookup tables
        mixed_bilinear_reverse_SU2.resize(lattice_size_SU2);
        mixed_bilinear_reverse_SU3.resize(lattice_size_SU3);
        
        // Build reverse lookup from SU(2) -> SU(3)
        // When SU(2) site i changes, we need to invalidate SU(3) sites that couple to it
        for (size_t su3_site = 0; su3_site < lattice_size_SU3; ++su3_site) {
            for (size_t n = 0; n < mixed_bilinear_partners_SU3[su3_site].size(); ++n) {
                size_t su2_partner = mixed_bilinear_partners_SU3[su3_site][n];
                mixed_bilinear_reverse_SU2[su2_partner].push_back(su3_site);
            }
        }
        
        // Build reverse lookup from SU(3) -> SU(2)
        // When SU(3) site i changes, we need to invalidate SU(2) sites that couple to it
        for (size_t su2_site = 0; su2_site < lattice_size_SU2; ++su2_site) {
            for (size_t n = 0; n < mixed_bilinear_partners_SU2[su2_site].size(); ++n) {
                size_t su3_partner = mixed_bilinear_partners_SU2[su2_site][n];
                mixed_bilinear_reverse_SU3[su3_partner].push_back(su2_site);
            }
        }
        
        // Remove duplicates in reverse lookup tables
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            std::sort(mixed_bilinear_reverse_SU2[i].begin(), mixed_bilinear_reverse_SU2[i].end());
            mixed_bilinear_reverse_SU2[i].erase(
                std::unique(mixed_bilinear_reverse_SU2[i].begin(), mixed_bilinear_reverse_SU2[i].end()),
                mixed_bilinear_reverse_SU2[i].end());
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            std::sort(mixed_bilinear_reverse_SU3[i].begin(), mixed_bilinear_reverse_SU3[i].end());
            mixed_bilinear_reverse_SU3[i].erase(
                std::unique(mixed_bilinear_reverse_SU3[i].begin(), mixed_bilinear_reverse_SU3[i].end()),
                mixed_bilinear_reverse_SU3[i].end());
        }
        
        // Report statistics
        size_t max_reverse_SU2 = 0, max_reverse_SU3 = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_reverse_SU2 = std::max(max_reverse_SU2, mixed_bilinear_reverse_SU2[i].size());
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_reverse_SU3 = std::max(max_reverse_SU3, mixed_bilinear_reverse_SU3[i].size());
        }
        if (max_reverse_SU2 > 0 || max_reverse_SU3 > 0) {
            cout << "Reverse lookup tables built: max SU2->SU3=" << max_reverse_SU2 
                 << ", max SU3->SU2=" << max_reverse_SU3 << endl;
        }
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

    /**
     * Enable or disable local field caching mode
     * 
     * When enabled, local fields are cached and only invalidated when
     * neighboring spins change. This is beneficial for interleaved sweeps
     * with mixed interactions.
     */
    void enable_field_caching(bool enable = true) {
        use_field_caching = enable;
        if (enable) {
            invalidate_all_fields();
        }
    }

    /**
     * Initialize field cache by computing all local fields
     */
    void init_field_cache() const {
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            if (!field_valid_SU2[i]) {
                cached_local_field_SU2[i] = get_local_field_SU2(i);
                field_valid_SU2[i] = true;
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            if (!field_valid_SU3[i]) {
                cached_local_field_SU3[i] = get_local_field_SU3(i);
                field_valid_SU3[i] = true;
            }
        }
    }

    /**
     * Invalidate all cached fields
     */
    void invalidate_all_fields() const {
        std::fill(field_valid_SU2.begin(), field_valid_SU2.end(), false);
        std::fill(field_valid_SU3.begin(), field_valid_SU3.end(), false);
    }

    /**
     * Invalidate fields affected by an SU(2) spin update
     * 
     * When SU(2) site i is updated:
     * - All SU(2) sites coupled to i via bilinear/trilinear interactions
     * - All SU(3) sites coupled to i via mixed interactions
     */
    void invalidate_fields_from_SU2_update(size_t su2_site) const {
        // Invalidate the updated site itself
        field_valid_SU2[su2_site] = false;
        
        // Invalidate SU(2) neighbors (bilinear partners)
        for (size_t partner : bilinear_partners_SU2[su2_site]) {
            field_valid_SU2[partner] = false;
        }
        
        // Invalidate SU(2) trilinear partners
        for (const auto& partners : trilinear_partners_SU2[su2_site]) {
            field_valid_SU2[partners[0]] = false;
            field_valid_SU2[partners[1]] = false;
        }
        
        // Invalidate SU(3) sites coupled via mixed bilinear
        for (size_t su3_site : mixed_bilinear_reverse_SU2[su2_site]) {
            field_valid_SU3[su3_site] = false;
        }
        
        // Invalidate SU(2) and SU(3) sites coupled via mixed trilinear
        for (const auto& partners : mixed_trilinear_partners_SU2[su2_site]) {
            field_valid_SU2[partners[0]] = false;  // SU(2) partner
            field_valid_SU3[partners[1]] = false;  // SU(3) partner
        }
    }

    /**
     * Invalidate fields affected by an SU(3) spin update
     * 
     * When SU(3) site i is updated:
     * - All SU(3) sites coupled to i via bilinear/trilinear interactions
     * - All SU(2) sites coupled to i via mixed interactions
     */
    void invalidate_fields_from_SU3_update(size_t su3_site) const {
        // Invalidate the updated site itself
        field_valid_SU3[su3_site] = false;
        
        // Invalidate SU(3) neighbors (bilinear partners)
        for (size_t partner : bilinear_partners_SU3[su3_site]) {
            field_valid_SU3[partner] = false;
        }
        
        // Invalidate SU(3) trilinear partners
        for (const auto& partners : trilinear_partners_SU3[su3_site]) {
            field_valid_SU3[partners[0]] = false;
            field_valid_SU3[partners[1]] = false;
        }
        
        // Invalidate SU(2) sites coupled via mixed bilinear
        for (size_t su2_site : mixed_bilinear_reverse_SU3[su3_site]) {
            field_valid_SU2[su2_site] = false;
        }
        
        // Invalidate SU(2) and SU(3) sites coupled via mixed trilinear (SU3-SU2-SU2)
        for (const auto& partners : mixed_trilinear_partners_SU3[su3_site]) {
            field_valid_SU2[partners[0]] = false;  // SU(2) partner 1
            field_valid_SU2[partners[1]] = false;  // SU(2) partner 2
        }
    }

    /**
     * Get cached local field for SU(2) site (computes if invalid)
     */
    SpinVector get_cached_local_field_SU2(size_t site_index) const {
        if (!field_valid_SU2[site_index]) {
            cached_local_field_SU2[site_index] = get_local_field_SU2(site_index);
            field_valid_SU2[site_index] = true;
        }
        return cached_local_field_SU2[site_index];
    }

    /**
     * Get cached local field for SU(3) site (computes if invalid)
     */
    SpinVector get_cached_local_field_SU3(size_t site_index) const {
        if (!field_valid_SU3[site_index]) {
            cached_local_field_SU3[site_index] = get_local_field_SU3(site_index);
            field_valid_SU3[site_index] = true;
        }
        return cached_local_field_SU3[site_index];
    }

    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================

    /**
     * Compute energy difference for an SU(2) spin flip
     */
    double site_energy_SU2_diff(const SpinVector& new_spin, const SpinVector& old_spin, size_t site_index) const {
        const SpinVector spin_diff = new_spin - old_spin;
        
        // Field energy
        double field_energy = -spin_diff.dot(field_SU2[site_index]);
        
        // Onsite energy
        double onsite_energy = (new_spin + old_spin).dot(onsite_interaction_SU2[site_index] * spin_diff);
        
        // Bilinear SU(2)-SU(2) interactions
        double bilinear_energy = 0.0;
        for (size_t i = 0; i < bilinear_partners_SU2[site_index].size(); ++i) {
            const size_t partner_idx = bilinear_partners_SU2[site_index][i];
            bilinear_energy += spin_diff.dot(bilinear_interaction_SU2[site_index][i] * spins_SU2[partner_idx]);
        }
        
        // Mixed bilinear SU(2)-SU(3) interactions
        double mixed_bilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_bilinear_partners_SU2[site_index].size(); ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU2[site_index][i];
            mixed_bilinear_energy += spin_diff.dot(mixed_bilinear_interaction_SU2[site_index][i] * spins_SU3[partner_idx]);
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) interactions.
        //
        // For a single-spin Metropolis move at site i, the change in any
        // trilinear term T_{abc} S^i_a S^j_b S^k_c is
        //     dE = (S_new - S_old) . V,  V[a] = sum_{bc} T[a,b,c] S^j_b S^k_c
        // when neither partner is site i. This collapses two O(d^3)
        // monomial evaluations (old and new) into a single O(d^3)
        // contraction plus an O(d) dot product -- the canonical
        // tensor-network "contract first, project later" optimization.
        // The rare self-coupling case (p1==i or p2==i) still needs the
        // explicit old/new path because S^i appears in two or three slots.
        double trilinear_energy = 0.0;
        for (size_t i = 0; i < trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            const bool p1_self = (p1_idx == site_index);
            const bool p2_self = (p2_idx == site_index);

            if (!p1_self && !p2_self) {
                // Fast path (the common case): pre-contract partners.
                const SpinVector& p1 = spins_SU2[p1_idx];
                const SpinVector& p2 = spins_SU2[p2_idx];
                double dE_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double Va = 0.0;
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        const double p1b = p1(b);
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            Va += Ta(b, c) * p1b * p2(c);
                        }
                    }
                    dE_term += spin_diff(a) * Va;
                }
                trilinear_energy += dE_term;  // multiplicity == 1
            } else {
                // Slow path: a single-spin flip changes more than one slot
                // of the trilinear monomial, so the full old/new evaluation
                // is required and a multiplicity correction restores the
                // unique-term counting expected by total_energy().
                const SpinVector& p1_old = p1_self ? old_spin : spins_SU2[p1_idx];
                const SpinVector& p1_new = p1_self ? new_spin : spins_SU2[p1_idx];
                const SpinVector& p2_old = p2_self ? old_spin : spins_SU2[p2_idx];
                const SpinVector& p2_new = p2_self ? new_spin : spins_SU2[p2_idx];

                double old_term = 0.0;
                double new_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            const double coeff = Ta(b, c);
                            old_term += coeff * old_spin(a) * p1_old(b) * p2_old(c);
                            new_term += coeff * new_spin(a) * p1_new(b) * p2_new(c);
                        }
                    }
                }
                const double multiplicity = 1.0 +
                    (p1_self ? 1.0 : 0.0) + (p2_self ? 1.0 : 0.0);
                trilinear_energy += (new_term - old_term) / multiplicity;
            }
        }

        // Mixed trilinear SU(2)-SU(2)-SU(3) interactions.
        // The SU(3) partner can never collide with an SU(2) site (different
        // species), so only the SU(2) partner p1 may collide.
        double mixed_trilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            const bool p1_self = (p1_idx == site_index);

            if (!p1_self) {
                // Fast path: pre-contract V[a] = sum_{bc} T[a,b,c] p1(b) p2(c).
                const SpinVector& p1 = spins_SU2[p1_idx];
                const SpinVector& p2 = spins_SU3[p2_idx];
                double dE_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double Va = 0.0;
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        const double p1b = p1(b);
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            Va += Ta(b, c) * p1b * p2(c);
                        }
                    }
                    dE_term += spin_diff(a) * Va;
                }
                mixed_trilinear_energy += dE_term;  // multiplicity == 1
            } else {
                const SpinVector& p1_old = old_spin;
                const SpinVector& p1_new = new_spin;
                const SpinVector& p2 = spins_SU3[p2_idx];
                double old_term = 0.0;
                double new_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            const double coeff = Ta(b, c);
                            old_term += coeff * old_spin(a) * p1_old(b) * p2(c);
                            new_term += coeff * new_spin(a) * p1_new(b) * p2(c);
                        }
                    }
                }
                mixed_trilinear_energy += (new_term - old_term) / 2.0;
            }
        }
        
        return field_energy + onsite_energy + bilinear_energy + mixed_bilinear_energy + 
               trilinear_energy + mixed_trilinear_energy;
    }

    /**
     * Compute energy difference for an SU(3) spin flip
     */
    double site_energy_SU3_diff(const SpinVector& new_spin, const SpinVector& old_spin, size_t site_index) const {
        const SpinVector spin_diff = new_spin - old_spin;
        
        // Field energy
        double field_energy = -spin_diff.dot(field_SU3[site_index]);
        
        // Onsite energy
        double onsite_energy = (new_spin + old_spin).dot(onsite_interaction_SU3[site_index] * spin_diff);
        
        // Bilinear SU(3)-SU(3) interactions
        double bilinear_energy = 0.0;
        for (size_t i = 0; i < bilinear_partners_SU3[site_index].size(); ++i) {
            const size_t partner_idx = bilinear_partners_SU3[site_index][i];
            bilinear_energy += spin_diff.dot(bilinear_interaction_SU3[site_index][i] * spins_SU3[partner_idx]);
        }
        
        // Mixed bilinear SU(3)-SU(2) interactions
        double mixed_bilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_bilinear_partners_SU3[site_index].size(); ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU3[site_index][i];
            mixed_bilinear_energy += spin_diff.dot(mixed_bilinear_interaction_SU3[site_index][i] * spins_SU2[partner_idx]);
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) interactions.
        // See site_energy_SU2_diff for the algebra; the savings here are
        // proportionally larger because spin_dim_SU3 = 8 (so a triple loop
        // is 512 multiply-adds vs 27 for SU(2)).
        double trilinear_energy = 0.0;
        for (size_t i = 0; i < trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            const bool p1_self = (p1_idx == site_index);
            const bool p2_self = (p2_idx == site_index);

            if (!p1_self && !p2_self) {
                const SpinVector& p1 = spins_SU3[p1_idx];
                const SpinVector& p2 = spins_SU3[p2_idx];
                double dE_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double Va = 0.0;
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        const double p1b = p1(b);
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            Va += Ta(b, c) * p1b * p2(c);
                        }
                    }
                    dE_term += spin_diff(a) * Va;
                }
                trilinear_energy += dE_term;  // multiplicity == 1
            } else {
                const SpinVector& p1_old = p1_self ? old_spin : spins_SU3[p1_idx];
                const SpinVector& p1_new = p1_self ? new_spin : spins_SU3[p1_idx];
                const SpinVector& p2_old = p2_self ? old_spin : spins_SU3[p2_idx];
                const SpinVector& p2_new = p2_self ? new_spin : spins_SU3[p2_idx];

                double old_term = 0.0;
                double new_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            const double coeff = Ta(b, c);
                            old_term += coeff * old_spin(a) * p1_old(b) * p2_old(c);
                            new_term += coeff * new_spin(a) * p1_new(b) * p2_new(c);
                        }
                    }
                }
                const double multiplicity = 1.0 +
                    (p1_self ? 1.0 : 0.0) + (p2_self ? 1.0 : 0.0);
                trilinear_energy += (new_term - old_term) / multiplicity;
            }
        }

        // Mixed trilinear SU(3)-SU(2)-SU(2) interactions.
        // SU(2) partners are on a different sublattice from the SU(3) site,
        // so collisions are impossible; we always take the fast path.
        double mixed_trilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            const SpinVector& p1 = spins_SU2[p1_idx];
            const SpinVector& p2 = spins_SU2[p2_idx];

            double dE_term = 0.0;
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double Va = 0.0;
                const auto& Ta = T[a];
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    const double p1b = p1(b);
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        Va += Ta(b, c) * p1b * p2(c);
                    }
                }
                dE_term += spin_diff(a) * Va;
            }
            mixed_trilinear_energy += dE_term;
        }
        
        return field_energy + onsite_energy + bilinear_energy + mixed_bilinear_energy + 
               trilinear_energy + mixed_trilinear_energy;
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
     * Includes SU2-SU2 interactions, SU2 field/onsite, and half of mixed interactions
     */
    double total_energy_SU2() const {
        double energy = 0.0;
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const auto& spin = spins_SU2[i];
            
            // Field and onsite
            energy -= spin.dot(field_SU2[i]);
            energy += spin.dot(onsite_interaction_SU2[i] * spin);
            
            // Bilinear SU2-SU2
            for (size_t j = 0; j < bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU2[i][j];
                energy += 0.5 * spin.dot(bilinear_interaction_SU2[i][j] * spins_SU2[partner]);
            }
            
            // Mixed bilinear SU2-SU3 (count half for SU2)
            for (size_t j = 0; j < mixed_bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU2[i][j];
                energy += 0.5 * spin.dot(mixed_bilinear_interaction_SU2[i][j] * spins_SU3[partner]);
            }

            // Mixed trilinear SU2-SU2-SU3
            for (size_t j = 0; j < mixed_trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU2[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU2[i][j][1];
                const auto& T = mixed_trilinear_interaction_SU2[i][j];

                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spins_SU2[p1](b) * spins_SU3[p2](c);
                        }
                    }
                    energy += (1.0 / 3.0) * spin(a) * temp;
                }
            }
            
            // Trilinear SU2-SU2-SU2
            for (size_t j = 0; j < trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU2[i][j][0];
                const size_t p2 = trilinear_partners_SU2[i][j][1];
                const auto& T = trilinear_interaction_SU2[i][j];
                
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spins_SU2[p1](b) * spins_SU2[p2](c);
                        }
                    }
                    energy += (1.0/3.0) * spin(a) * temp;
                }
            }
        }
        
        return energy;
    }

    /**
     * Compute total energy of the SU(3) sublattice only
     * Includes SU3-SU3 interactions, SU3 field/onsite, and half of mixed interactions
     */
    double total_energy_SU3() const {
        double energy = 0.0;
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            const auto& spin = spins_SU3[i];
            
            // Field and onsite
            energy -= spin.dot(field_SU3[i]);
            energy += spin.dot(onsite_interaction_SU3[i] * spin);
            
            // Bilinear SU3-SU3
            for (size_t j = 0; j < bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU3[i][j];
                energy += 0.5 * spin.dot(bilinear_interaction_SU3[i][j] * spins_SU3[partner]);
            }
            
            // Mixed bilinear SU3-SU2 (count half for SU3)
            for (size_t j = 0; j < mixed_bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU3[i][j];
                energy += 0.5 * spin.dot(mixed_bilinear_interaction_SU3[i][j] * spins_SU2[partner]);
            }

            // Mixed trilinear SU3-SU2-SU2
            for (size_t j = 0; j < mixed_trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU3[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU3[i][j][1];
                const auto& T = mixed_trilinear_interaction_SU3[i][j];

                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spins_SU2[p1](b) * spins_SU2[p2](c);
                        }
                    }
                    energy += (1.0 / 3.0) * spin(a) * temp;
                }
            }
            
            // Trilinear SU3-SU3-SU3
            for (size_t j = 0; j < trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU3[i][j][0];
                const size_t p2 = trilinear_partners_SU3[i][j][1];
                const auto& T = trilinear_interaction_SU3[i][j];
                
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spins_SU3[p1](b) * spins_SU3[p2](c);
                        }
                    }
                    energy += (1.0/3.0) * spin(a) * temp;
                }
            }
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

    /**
     * Single Metropolis sweep over both sublattices (sequential: SU2 then SU3)
     * 
     * Optimized with:
     * - Precomputed inverse temperature
     * - Batched random number generation
     * - Branchless acceptance criterion
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        
        size_t accepted = 0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const double inv_T = 1.0 / T;  // Precompute inverse temperature
        
        // Batch size for random number pre-generation
        constexpr size_t BATCH_SIZE = 64;
        vector<size_t> random_sites(BATCH_SIZE);
        vector<double> random_uniforms(BATCH_SIZE);
        
        // Sweep SU(2) sublattice
        for (size_t batch_start = 0; batch_start < lattice_size_SU2; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size_SU2);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sites[j] = random_int_lehman(lattice_size_SU2);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                const size_t i = random_sites[j];
                const double rand_uniform = random_uniforms[j];
                
                SpinVector new_spin;
                if (gaussian_move) {
                    new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                    double norm = new_spin.norm();
                    if (norm > 1e-12) new_spin *= spin_length_SU2 / norm;
                    else new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                } else {
                    new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                }
                
                const double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);

                // Acceptance: short-circuit for downhill moves (was bitwise `|`,
                // which wastefully evaluated exp() even when dE <= 0).
                const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                if (accept) {
                    spins_SU2[i] = new_spin;
                    accepted++;
                }
            }
        }

        // Sweep SU(3) sublattice
        for (size_t batch_start = 0; batch_start < lattice_size_SU3; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size_SU3);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sites[j] = random_int_lehman(lattice_size_SU3);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                const size_t i = random_sites[j];
                const double rand_uniform = random_uniforms[j];
                
                SpinVector new_spin;
                if (gaussian_move) {
                    new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                    double norm = new_spin.norm();
                    if (norm > 1e-12) new_spin *= spin_length_SU3 / norm;
                    else new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                } else {
                    new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                }
                
                const double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);

                // Acceptance: logical OR for short-circuit (was bitwise).
                const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                if (accept) {
                    spins_SU3[i] = new_spin;
                    accepted++;
                }
            }
        }

        return double(accepted) / double(total_sites);
    }

    /**
     * Coloured Metropolis sweep — parallel over SU(2) sites within each
     * SU(2) colour, then parallel over SU(3) sites within each SU(3) colour.
     *
     * Differs from `metropolis()` (random-with-replacement, coupon-collector
     * style) in that this version visits every SU(2) and every SU(3) site
     * exactly once per call, in a colour-stratified deterministic order
     * within each colour. The Markov chain is still detailed-balance
     * correct (each colour pass is a valid Metropolis sub-sweep over a
     * subset of independent single-spin moves), and the per-site sampling
     * is *better* (no missed sites).
     *
     * Race-free guarantee: within an SU(2) colour, no two sites share any
     * SU(2)-SU(2) bilinear, SU(2)-SU(2)-SU(2) trilinear, or
     * SU(2)-SU(2)-SU(3) trilinear interaction (the SU(3) read partners are
     * frozen during the SU(2) pass, hence not racy). Symmetric story for
     * SU(3). Mixed bilinear couples SU(2) ↔ SU(3) directly across the two
     * passes, and is correct because each pass treats the other species as
     * a frozen background.
     *
     * Falls back to serial `metropolis()` if no colour partition is built
     * or only one OpenMP thread is available.
     */
    double metropolis_parallel(double T, bool gaussian_move = false,
                               double sigma = 60.0) {
        if (n_colors_SU2 == 0 && n_colors_SU3 == 0)
            return metropolis(T, gaussian_move, sigma);
#ifdef _OPENMP
        if (omp_get_max_threads() <= 1) return metropolis(T, gaussian_move, sigma);
#else
        return metropolis(T, gaussian_move, sigma);
#endif
        if (T <= 0) return 0.0;

        const double inv_T = 1.0 / T;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;

#ifdef _OPENMP
        const int n_threads = omp_get_max_threads();
#else
        const int n_threads = 1;
#endif
        // Cache-line padded per-thread counters: avoid false sharing of
        // adjacent counter slots in the final reduction.
        struct alignas(64) PaddedAccept { size_t v = 0; char pad[64 - sizeof(size_t)]; };
        std::vector<PaddedAccept> per_thread_accepted(n_threads);

        // PERSISTENT OpenMP region wrapping BOTH the SU(2) and SU(3) colour
        // passes — one fork/join per sweep, with #pragma omp barrier
        // between every colour boundary. For a TmFeO3 mixed lattice this
        // collapses (n_colors_SU2 + n_colors_SU3) team-creation costs (each
        // ~few µs) into one. At small L this was the dominant per-sweep
        // overhead for the parallel kernel.
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            size_t local_accepted = 0;

            // -------------------------- SU(2) pass --------------------------
            for (size_t c = 0; c < n_colors_SU2; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU2[c];
                const size_t off_hi = sites_by_color_csr_off_SU2[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU2[off];

                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                        const double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU2 / norm;
                        else new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    }

                    const double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);
                    const double rand_uniform = random_double_lehman(0.0, 1.0);
                    const bool accept = (dE <= 0.0) ||
                                        (rand_uniform < std::exp(-dE * inv_T));
                    if (accept) {
                        spins_SU2[i] = new_spin;
                        ++local_accepted;
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            // -------------------------- SU(3) pass --------------------------
            for (size_t c = 0; c < n_colors_SU3; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU3[c];
                const size_t off_hi = sites_by_color_csr_off_SU3[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU3[off];

                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                        const double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU3 / norm;
                        else new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    }

                    const double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);
                    const double rand_uniform = random_double_lehman(0.0, 1.0);
                    const bool accept = (dE <= 0.0) ||
                                        (rand_uniform < std::exp(-dE * inv_T));
                    if (accept) {
                        spins_SU3[i] = new_spin;
                        ++local_accepted;
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            per_thread_accepted[tid].v += local_accepted;
        } // end omp parallel

        size_t accepted = 0;
        for (auto& a : per_thread_accepted) accepted += a.v;
        return double(accepted) / double(total_sites);
    }

    /**
     * Coloured over-relaxation sweep. Same race-free / parallelism story as
     * `metropolis_parallel`. Falls back to serial `overrelaxation()` if the
     * colour partition is empty or only one thread is available.
     *
     * NOTE: unlike serial `overrelaxation()`, this version does *not* call
     * `get_cached_local_field_*` even if `use_field_caching` is on. The
     * field cache uses shared `field_valid_*` / `cached_local_field_*`
     * arrays that are not safe to mutate from multiple threads. The
     * cache is much less useful in a coloured sweep anyway because each
     * site's local field would typically be invalidated by neighbour
     * updates in the previous colour pass.
     */
    void overrelaxation_parallel() {
        if (n_colors_SU2 == 0 && n_colors_SU3 == 0) { overrelaxation(); return; }
#ifdef _OPENMP
        if (omp_get_max_threads() <= 1) { overrelaxation(); return; }
#else
        overrelaxation(); return;
#endif

        // PERSISTENT OpenMP region wrapping both sublattices.
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            // -------------------------- SU(2) pass --------------------------
            for (size_t c = 0; c < n_colors_SU2; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU2[c];
                const size_t off_hi = sites_by_color_csr_off_SU2[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU2[off];
                    SpinVector local_field = get_local_field_SU2(i);
                    const double norm = local_field.squaredNorm();
                    if (norm > 1e-12) {
                        const double proj = 2.0 * spins_SU2[i].dot(local_field) / norm;
                        spins_SU2[i] = local_field * proj - spins_SU2[i];
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            // -------------------------- SU(3) pass --------------------------
            for (size_t c = 0; c < n_colors_SU3; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU3[c];
                const size_t off_hi = sites_by_color_csr_off_SU3[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU3[off];
                    SpinVector local_field = get_local_field_SU3(i);
                    const double norm = local_field.squaredNorm();
                    if (norm > 1e-12) {
                        const double proj = 2.0 * spins_SU3[i].dot(local_field) / norm;
                        spins_SU3[i] = local_field * proj - spins_SU3[i];
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }
        } // end omp parallel
    }

    /**
     * Interleaved Metropolis sweep: alternates between SU(2) and SU(3) updates
     * 
     * This improves equilibration when mixed bilinear interactions are non-zero,
     * as changes in one sublattice immediately affect the other sublattice's
     * energy landscape during the same sweep.
     * 
     * Uses local field caching with lazy invalidation for efficiency.
     * 
     * Optimized with:
     * - Precomputed inverse temperature
     * - Batched random number generation
     * - Branchless acceptance criterion
     * 
     * @param T           Temperature
     * @param gaussian_move Use Gaussian moves (true) or uniform random (false)
     * @param sigma       Width of Gaussian moves
     * @return            Acceptance rate
     */
    double metropolis_interleaved(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        
        size_t accepted = 0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const double inv_T = 1.0 / T;  // Precompute inverse temperature
        
        // Determine whether to use caching (beneficial when mixed interactions exist)
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        if (has_mixed && use_field_caching) {
            // Initialize all cached fields
            init_field_cache();
        }
        
        // Batch size for random number pre-generation
        constexpr size_t BATCH_SIZE = 64;
        vector<size_t> random_sublattice(BATCH_SIZE);  // Which sublattice to update
        vector<size_t> random_sites(BATCH_SIZE);        // Site within sublattice
        vector<double> random_uniforms(BATCH_SIZE);     // For acceptance
        
        for (size_t batch_start = 0; batch_start < total_sites; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, total_sites);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sublattice[j] = random_int_lehman(total_sites);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                // Probabilistically choose which sublattice to update
                const bool update_SU2 = (random_sublattice[j] < lattice_size_SU2);
                const double rand_uniform = random_uniforms[j];
                
                if (update_SU2) {
                    const size_t i = random_int_lehman(lattice_size_SU2);
                    
                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                        double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU2 / norm;
                        else new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    }
                    
                    const double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);

                    // Acceptance: logical OR (was bitwise).
                    const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                    if (accept) {
                        spins_SU2[i] = new_spin;
                        accepted++;

                        // Invalidate cached fields for affected sites
                        if (has_mixed && use_field_caching) {
                            invalidate_fields_from_SU2_update(i);
                        }
                    }
                } else {
                    const size_t i = random_int_lehman(lattice_size_SU3);
                    
                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                        double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU3 / norm;
                        else new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    }
                    
                    const double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);

                    // Acceptance: logical OR (was bitwise).
                    const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                    if (accept) {
                        spins_SU3[i] = new_spin;
                        accepted++;

                        // Invalidate cached fields for affected sites
                        if (has_mixed && use_field_caching) {
                            invalidate_fields_from_SU3_update(i);
                        }
                    }
                }
            }
        }
        
        return double(accepted) / double(total_sites);
    }

    /**
     * Over-relaxation sweep (microcanonical, zero acceptance rate)
     * Reflects spins across their local field direction
     */
    void overrelaxation() {
        // Over-relaxation for SU(2) spins
        for (size_t count = 0; count < lattice_size_SU2; ++count) {
            size_t i = random_int_lehman(lattice_size_SU2);
            SpinVector local_field = get_local_field_SU2(i);
            double norm = local_field.squaredNorm();
            
            if (norm > 1e-12) {
                double proj = 2.0 * spins_SU2[i].dot(local_field) / norm;
                spins_SU2[i] = local_field * proj - spins_SU2[i];
            }
        }
        
        // Over-relaxation for SU(3) spins
        for (size_t count = 0; count < lattice_size_SU3; ++count) {
            size_t i = random_int_lehman(lattice_size_SU3);
            SpinVector local_field = get_local_field_SU3(i);
            double norm = local_field.squaredNorm();
            
            if (norm > 1e-12) {
                double proj = 2.0 * spins_SU3[i].dot(local_field) / norm;
                spins_SU3[i] = local_field * proj - spins_SU3[i];
            }
        }
    }

    /**
     * Interleaved over-relaxation sweep (microcanonical)
     * 
     * Alternates between SU(2) and SU(3) updates, ensuring that changes
     * in one sublattice are immediately reflected in the local field
     * computation of the other sublattice during the same sweep.
     * 
     * Uses local field caching with lazy invalidation for efficiency
     * when mixed interactions are present.
     */
    void overrelaxation_interleaved() {
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        
        // Initialize field cache if using caching mode
        if (has_mixed && use_field_caching) {
            init_field_cache();
        }
        
        for (size_t n = 0; n < total_sites; ++n) {
            // Probabilistically choose which sublattice to update
            bool update_SU2 = (random_int_lehman(total_sites) < lattice_size_SU2);
            
            if (update_SU2) {
                size_t i = random_int_lehman(lattice_size_SU2);
                SpinVector local_field = (has_mixed && use_field_caching) ? 
                    get_cached_local_field_SU2(i) : get_local_field_SU2(i);
                double norm = local_field.squaredNorm();
                
                if (norm > 1e-12) {
                    double proj = 2.0 * spins_SU2[i].dot(local_field) / norm;
                    spins_SU2[i] = local_field * proj - spins_SU2[i];
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU2_update(i);
                    }
                }
            } else {
                size_t i = random_int_lehman(lattice_size_SU3);
                SpinVector local_field = (has_mixed && use_field_caching) ?
                    get_cached_local_field_SU3(i) : get_local_field_SU3(i);
                double norm = local_field.squaredNorm();
                
                if (norm > 1e-12) {
                    double proj = 2.0 * spins_SU3[i].dot(local_field) / norm;
                    spins_SU3[i] = local_field * proj - spins_SU3[i];
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU3_update(i);
                    }
                }
            }
        }
    }

    /**
     * Interleaved deterministic sweep with caching
     * 
     * Zero-temperature relaxation that alternates between sublattices
     * and uses local field caching for efficiency.
     */
    void deterministic_sweep_interleaved() {
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        
        // Initialize field cache if using caching mode
        if (has_mixed && use_field_caching) {
            init_field_cache();
        }
        
        for (size_t n = 0; n < total_sites; ++n) {
            // Probabilistically choose which sublattice to update
            bool update_SU2 = (random_int_lehman(total_sites) < lattice_size_SU2);
            
            if (update_SU2) {
                size_t i = random_int_lehman(lattice_size_SU2);
                SpinVector local_field = (has_mixed && use_field_caching) ? 
                    get_cached_local_field_SU2(i) : get_local_field_SU2(i);
                double norm = local_field.norm();
                
                if (norm > 1e-12) {
                    spins_SU2[i] = -local_field / norm * spin_length_SU2;
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU2_update(i);
                    }
                }
            } else {
                size_t i = random_int_lehman(lattice_size_SU3);
                SpinVector local_field = (has_mixed && use_field_caching) ?
                    get_cached_local_field_SU3(i) : get_local_field_SU3(i);
                double norm = local_field.norm();
                
                if (norm > 1e-12) {
                    spins_SU3[i] = -local_field / norm * spin_length_SU3;
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU3_update(i);
                    }
                }
            }
        }
    }

    /**
     * Deterministic sweep: align each spin antiparallel to its local field
     * This is a zero-temperature relaxation step that randomly selects sites
     */
    void deterministic_sweep() {
        // Deterministic update for SU(2) spins
        for (size_t count = 0; count < lattice_size_SU2; ++count) {
            size_t i = random_int_lehman(lattice_size_SU2);
            SpinVector local_field = get_local_field_SU2(i);
            double norm = local_field.norm();
            
            if (norm > 1e-12) {
                spins_SU2[i] = -local_field / norm * spin_length_SU2;
            }
        }
        
        // Deterministic update for SU(3) spins
        for (size_t count = 0; count < lattice_size_SU3; ++count) {
            size_t i = random_int_lehman(lattice_size_SU3);
            SpinVector local_field = get_local_field_SU3(i);
            double norm = local_field.norm();
            
            if (norm > 1e-12) {
                spins_SU3[i] = -local_field / norm * spin_length_SU3;
            }
        }
    }

    // -------------------------------------------------------------------
    // SU(3) coherent-state utilities (qutrit-only, spin_dim_SU3 == 8).
    //
    // These project the stored 8-vector Bloch parameterization onto the
    // physical CP^2 manifold of qutrit pure states, following the SU(N)
    // coherent-state formulation of Zhang & Batista, PRB 104, 104409
    // (2021) and the geometric-integrator picture of Dahlbom et al., PRB
    // 106, 054423 (2022). They are opt-in: no existing code path calls
    // them automatically. See diag_tmfeo3_su3_coherent_state for the
    // standalone validation.
    //
    // Storage convention. The stored `spins_SU3[i](a)` are interpreted as
    // the raw Gell-Mann expectation values n^a = <psi|lambda^a|psi>. The
    // qutrit density matrix is
    //   rho = (1/3) I + (1/2) sum_a n^a lambda^a,
    // whose trace is identically 1 regardless of |n|. Every pure state has
    // Tr rho^2 = 1/3 + |n|^2/2 = 1, i.e. |n|^2 = 2(N-1)/N = 4/3 and
    // |n| = 2/sqrt(3) (e.g. |E1>: n_3 = 1, n_8 = 1/sqrt(3)), and the cubic
    // Casimir d_abc n^a n^b n^c = 8/9 (su3::casimir2 / su3::casimir3).
    //
    // The `spin_length_SU3` constructor argument does NOT enter these
    // routines: for SU(N>2) it is not a free normalisation knob (no
    // continuous family of states has the same Casimir spectrum), it is
    // only used by the legacy antiparallel-alignment SA, which the new
    // `deterministic_sweep_SU3_exact_diag` below replaces.
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
     * not a physical qutrit pure state (e.g. produced by the legacy
     * antiparallel-alignment SA).
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
     * Deterministic SU(3) sweep via exact local diagonalization.
     *
     * Replaces the antiparallel-alignment rule used in the standard
     * `deterministic_sweep`, which can place the stored Bloch vector
     * outside the qutrit positive cone (and which uses a meaningless
     * `spin_length_SU3` rescaling for N > 2). For each randomly selected
     * SU(3) site this:
     *   1. computes the local Gell-Mann field h^a,
     *   2. builds the 3x3 mean-field Hamiltonian
     *        H_loc = (1/2) sum_a h^a lambda^a,
     *      with sign convention matching `cross_prod_SU3_flat`,
     *   3. finds its ground-state eigenvector psi,
     *   4. writes back the physical Bloch vector n^a = <psi|lambda^a|psi>.
     *
     * SU(2) sites are updated by the usual antiparallel rule.
     */
    void deterministic_sweep_SU3_exact_diag() {
        // SU(2) sites: standard antiparallel alignment.
        for (size_t count = 0; count < lattice_size_SU2; ++count) {
            size_t i = random_int_lehman(lattice_size_SU2);
            SpinVector local_field = get_local_field_SU2(i);
            double norm = local_field.norm();
            if (norm > 1e-12) {
                spins_SU2[i] = -local_field / norm * spin_length_SU2;
            }
        }
        if (spin_dim_SU3 != 8) {
            // Fall back to antiparallel alignment for non-qutrit dims.
            for (size_t count = 0; count < lattice_size_SU3; ++count) {
                size_t i = random_int_lehman(lattice_size_SU3);
                SpinVector local_field = get_local_field_SU3(i);
                double norm = local_field.norm();
                if (norm > 1e-12) {
                    spins_SU3[i] = -local_field / norm * spin_length_SU3;
                }
            }
            return;
        }
        for (size_t count = 0; count < lattice_size_SU3; ++count) {
            size_t i = random_int_lehman(lattice_size_SU3);
            SpinVector local_field = get_local_field_SU3(i);
            if (local_field.norm() < 1e-15) continue;
            classical_spin::su3::Vector8r h;
            for (int a = 0; a < 8; ++a) h(a) = local_field(a);
            auto H_loc = classical_spin::su3::local_hamiltonian_from_field(h);
            auto psi   = classical_spin::su3::ground_state(H_loc);
            auto n_phys = classical_spin::su3::expectations_from_psi(psi);
            for (int a = 0; a < 8; ++a) spins_SU3[i](a) = n_phys(a);
        }
    }

    /**
     * Zero-temperature greedy quench with convergence check
     */
    void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000);

    /**
     * Main simulated annealing routine
     * Matches structure and features from Lattice::simulated_annealing
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
        m.energy = total_energy();
        m.energy_SU2 = total_energy_SU2();
        m.energy_SU3 = total_energy_SU3();
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
     * Helper: Perform MC sweeps with optional overrelaxation
     * Returns sum of acceptance rates from metropolis calls
     * 
     * @param n_sweeps Number of sweeps to perform
     * @param T Temperature
     * @param gaussian_move Use Gaussian moves
     * @param sigma Gaussian move width
     * @param overrelaxation_rate Perform overrelaxation every N sweeps (0 = disabled)
     * @param interleaved Use interleaved sweeps (better for mixed interactions)
     */
    double perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate = 0,
                            bool interleaved = true);

    /**
     * Get local field for SU(2) from temporary state
     */
    SpinVector get_local_field_SU2_state(size_t site_index, 
                                         const SpinConfigSU2& curr_spins2,
                                         const SpinConfigSU3& curr_spins3) const {
        SpinVector H = -field_SU2[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU2[site_index] * curr_spins2[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU2[site_index].size(); ++i) {
            H += bilinear_interaction_SU2[site_index][i] * curr_spins2[bilinear_partners_SU2[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU2[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU2[site_index][i] * curr_spins3[mixed_bilinear_partners_SU2[site_index][i]];
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * curr_spins2[p1_idx](b) * curr_spins2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * curr_spins2[p1_idx](b) * curr_spins3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

    /**
     * Get local field for SU(3) from temporary state
     */
    SpinVector get_local_field_SU3_state(size_t site_index,
                                         const SpinConfigSU2& curr_spins2,
                                         const SpinConfigSU3& curr_spins3) const {
        SpinVector H = -field_SU3[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU3[site_index] * curr_spins3[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU3[site_index].size(); ++i) {
            H += bilinear_interaction_SU3[site_index][i] * curr_spins3[bilinear_partners_SU3[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU3[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU3[site_index][i] * curr_spins2[mixed_bilinear_partners_SU3[site_index][i]];
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) contributions
        for (size_t i = 0; i < trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * curr_spins3[p1_idx](b) * curr_spins3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * curr_spins2[p1_idx](b) * curr_spins2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

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
     * Save spin configuration
     */
    void save_spin_config(const string& filename) const {
        // SU(2) spins
        {
            ofstream file(filename + "_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    file << spins_SU2[i](j) << " ";
                }
                file << "\n";
            }
        }
        
        // SU(3) spins
        {
            ofstream file(filename + "_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    file << spins_SU3[i](j) << " ";
                }
                file << "\n";
            }
        }
    }

    /**
     * Save spin configuration to a directory with clean naming
     * Creates: spins_SU2.txt and spins_SU3.txt in the directory
     */
    void save_spin_config_to_dir(const string& dir, const string& prefix = "spins") const {
        // SU(2) spins
        {
            ofstream file(dir + "/" + prefix + "_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    file << spins_SU2[i](j) << " ";
                }
                file << "\n";
            }
        }
        
        // SU(3) spins
        {
            ofstream file(dir + "/" + prefix + "_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    file << spins_SU3[i](j) << " ";
                }
                file << "\n";
            }
        }
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
     * Load spin configuration
     */
    void load_spin_config(const string& filename) {
        // Load SU(2) spins
        {
            ifstream file(filename + "_SU2.txt");
            if (!file) {
                cerr << "Error: Cannot open " << filename << "_SU2.txt" << endl;
                return;
            }
            
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    file >> spins_SU2[i](j);
                }
            }
        }
        
        // Load SU(3) spins
        {
            ifstream file(filename + "_SU3.txt");
            if (!file) {
                cerr << "Error: Cannot open " << filename << "_SU3.txt" << endl;
                return;
            }
            
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    file >> spins_SU3[i](j);
                }
            }
        }
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
     * Initialize with ferromagnetic state
     */
    void init_ferromagnetic(const SpinVector& direction_SU2, const SpinVector& direction_SU3) {
        const SpinVector dir_SU2 = direction_SU2.normalized() * spin_length_SU2;
        const SpinVector dir_SU3 = direction_SU3.normalized() * spin_length_SU3;
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins_SU2[i] = dir_SU2;
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins_SU3[i] = dir_SU3;
        }
    }

    /**
     * Initialize with random state
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins_SU2[i] = gen_random_spin(spin_length_SU2, spin_dim_SU2);
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins_SU3[i] = gen_random_spin(spin_length_SU3, spin_dim_SU3);
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
