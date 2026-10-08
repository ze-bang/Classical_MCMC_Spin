#ifndef LATTICE_REFACTORED_H
#define LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "hdf5_io.h"
#include "classical_spin/mc/mc_common.h"      // Common MC structs & templates
#include "classical_spin/mc/parallel_tempering.h"  // replica-exchange engine + ladder tuning
#include "classical_spin/lattice/pulse_chunking.h"  // default pump-probe tolerances
#include "classical_spin/lattice/correlation_accumulator.h"  // FFT spin / dimer correlations
#include "classical_spin/dynamics/spin_integrators.h"  // geometric / Langevin spin integrators
#include "classical_spin/dynamics/drive.h"             // DriveSchedule, Pulse
#include "classical_spin/dynamics/time_grid.h"         // TimeGrid, delay_grid
#include "classical_spin/io/spin_table.h"               // spin configuration text files
#include "classical_spin/core/su3_mc.h"                 // SU(3) Monte Carlo moves on CP^2
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
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// Boost.Odeint is used only by src/core/lattice_md.cpp (integrate_on_grid);
// keeping it out of this header spares every other TU its compile cost.

#ifdef HDF5_ENABLED
#include "hdf5_io.h"
#endif

// GPU support: the opaque host API (no .cu file includes this header).
#ifdef CUDA_ENABLED
#include "lattice_gpu_api.h"
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
using mc::AutocorrelationResult;
using mc::BinningResult;
using mc::Observable;
using mc::VectorObservable;
using mc::ThermodynamicObservables;
using mc::OptimizedTempGridResult;

// Simulated annealing parameters (Lattice-specific)
struct SAParams {
    double T_start = 1.0;
    double T_end = 1e-3;
    double cooling_rate = 0.9;
    size_t sweeps_per_temp = 100;
    vector<double> probe_T;
    vector<double> probe_acc;
    vector<double> probe_tau;
};

/**
 * Lattice class: Template-free implementation using Eigen3 and std::vector
 * 
 * This class manages a periodic lattice of classical spins with:
 * - Runtime-configurable dimensions (spin_dim, N_atoms, dim1, dim2, dim3)
 * - Bilinear and trilinear interactions
 * - Twisted boundary conditions
 * - Monte Carlo sampling (Metropolis, Wolff, Swendsen-Wang)
 * - Simulated annealing with auto-tuning
 * - Parallel tempering with twist boundary conditions
 * - Molecular dynamics (Landau-Lifshitz equations)
 */
class Lattice {
public:
    // Type aliases for clarity
    using SpinConfig = vector<SpinVector>;
    using CrossProductMethod = function<SpinVector(const SpinVector&, const SpinVector&)>;
    using ODEState = vector<double>;  // Flat state vector for Boost.Odeint

    // Core lattice properties
    UnitCell unit_cell;
    std::string lattice_type;  // Lattice type identifier (e.g., "pyrochlore", "pyrochlore_non_kramer")
    size_t spin_dim;         // Dimension of spin vectors (e.g., 3 for SU(2), 8 for SU(3))
    size_t N_atoms;          // Number of atoms per unit cell
    size_t dim1, dim2, dim3; // Lattice dimensions
    size_t lattice_size;     // Total number of sites = N_atoms * dim1 * dim2 * dim3
    float spin_length;       // Magnitude of spin vectors

    // Spin configuration and positions
    SpinConfig spins;                // Current spin configuration
    vector<Eigen::Vector3d> site_positions; // Real-space positions

    // Interaction lookup tables
    SpinConfig field;                                    // External field at each site
    vector<SpinMatrix> onsite_interaction;               // On-site anisotropy
    vector<vector<SpinMatrix>> bilinear_interaction;     // Bilinear coupling
    vector<vector<SpinTensor3>> trilinear_interaction;   // Trilinear coupling
    vector<vector<size_t>> bilinear_partners;            // Partner site indices
    vector<vector<array<size_t, 2>>> trilinear_partners; // Trilinear partner pairs
    vector<SpinMatrix> sublattice_frames;                // Sublattice frame transformations
    vector<double> afm_sublattice_signs;                  // Signs for staggered magnetization per sublattice

    size_t num_bi;  // Number of bilinear neighbors per site
    size_t num_tri; // Number of trilinear interactions per site

    // ------------------------------------------------------------------
    // Flat / SoA mirror of the bilinear interaction table.
    //
    // Built once at the end of initialize() (and rebuilt by the copy
    // constructor and assignment operator). Used by the hot-loop kernels
    // -- site_energy_diff_flat, overrelaxation, get_local_field_flat --
    // to eliminate the two pointer indirections of
    // vector<vector<Eigen::Matrix3d>> and to put each site's J matrices
    // on consecutive cache lines.
    //
    // Layout (CSR-style): for site i, neighbour n in
    //   [bi_flat_offset[i], bi_flat_offset[i+1]):
    //     partner index = bi_flat_partner[n]
    //     J matrix      = &bi_flat_J[n * spin_dim * spin_dim]    (row-major)
    //     wrap          = bi_flat_wrap[n]
    //     needs_twist   = bi_flat_needs_twist[n]   (precomputed wrap != 0)
    //
    // The nested vector<vector<>> tables (bilinear_interaction,
    // bilinear_partners, bilinear_wrap_dir) are still maintained as the
    // source of truth; the SoA tables are a derived cache used only by
    // the per-site hot kernels.
    // ------------------------------------------------------------------
    vector<size_t>             bi_flat_offset;
    vector<size_t>             bi_flat_partner;
    vector<double>             bi_flat_J;
    vector<uint8_t>            bi_flat_needs_twist;   // wrapped bond of an SO(3) lattice
    vector<array<int8_t, 3>>   bi_flat_wrap;
    // Twisted boundaries are folded into the bond matrices: bi_flat_J holds
    // the effective matrix (J0 M(w) for the forward entry of a bond,
    // (J0 M(w))^T for its reverse entry, M(w) the product of twist
    // rotations crossed by the bond) and bi_flat_J0 the untwisted one.
    // refresh_twisted_bonds() recomputes the wrapped entries whenever a
    // twist angle changes, so the hot kernels never branch on twists and
    // the local field is the exact gradient of the energy.
    vector<double>             bi_flat_J0;
    vector<uint8_t>            bi_flat_forward;
    size_t                     bi_flat_D2 = 0;   // = spin_dim * spin_dim, cached

    // ------------------------------------------------------------------
    // Sublattice colouring of the bond graph (bilinear + trilinear union).
    //
    // Two sites that share *any* interaction (bilinear or trilinear) get
    // different colours. A coloured Metropolis / overrelaxation sweep then
    // updates all sites of one colour in parallel without any data race
    // (the per-site kernel only reads neighbour spins and writes spins[i]).
    //
    // Layout (CSR-style):
    //   color_of_site[i]               in [0, n_colors)
    //   sites_by_color_csr_off[c..c+1) = range of sites with colour c
    //   sites_by_color_csr[off]         = site index
    //
    // Built once at the end of initialize() by `build_color_partition()`.
    // n_colors > 0 ⇒ partition is valid; n_colors == 0 ⇒ no partition built
    // (e.g. running on legacy code paths) ⇒ parallel sweeps fall back to
    // the serial entry points.
    // ------------------------------------------------------------------
    vector<uint16_t>           color_of_site;
    vector<size_t>             sites_by_color_csr_off;   // size n_colors + 1
    vector<size_t>             sites_by_color_csr;       // size lattice_size
    size_t                     n_colors = 0;

    // Twist boundary conditions
    array<SpinMatrix, 3> twist_matrices;                 // Rotation matrices per dimension
    array<SpinVector, 3> rotation_axis;                  // Rotation axes
    array<double, 3> twist_angles;                       // Current twist angles (radians)
    vector<vector<array<int8_t, 3>>> bilinear_wrap_dir;  // Wrap direction per neighbor
    vector<vector<uint8_t>> bilinear_forward;            // 1 = forward entry of the bond, 0 = reverse
    array<vector<size_t>, 3> boundary_sites_per_dim;     // Sites near boundaries
    array<size_t, 3> boundary_thickness;                 // Layers affected by twist

    // Adaptive step-size and acceptance counters for twist Metropolis (one per dim).
    // Target acceptance ≈ 0.5; step is rescaled every twist_step_adapt_window proposals.
    array<double, 3> twist_step       = {0.5, 0.5, 0.5};
    array<size_t, 3> twist_n_attempt  = {0, 0, 0};
    array<size_t, 3> twist_n_accept   = {0, 0, 0};
    static constexpr size_t twist_step_adapt_window = 50;

    // Time-dependent drive installed by set_pulse() for the member-state API
    // (landau_lifshitz_flat, ode_system, integrate_geometric). The trajectory
    // drivers take an explicit DriveSchedule instead. field_drive .. field_drive_width
    // mirror the installed drive in the two-pulse layout uploaded by the GPU
    // paths; only set_pulse() / clear_pulse() write them.
    array<SpinVector, 2> field_drive; // Two pulse polarisations (spin frame)
    array<double, 2> t_pulse;         // Pulse center times
    double field_drive_amp;           // Pulse amplitude
    double field_drive_freq;          // Pulse frequency
    double field_drive_width;         // Pulse width (Gaussian)
    classical_spin::dynamics::DriveSchedule active_drive;

    // Gilbert damping parameter for LLG dynamics
    double alpha_gilbert = 0.0;       // 0 = undamped (pure LL)

    // Langevin bath temperature for stochastic LLG (k_B = 1). Used by the
    // geometric integrators (spherical_midpoint, depondt) when > 0 together
    // with alpha_gilbert > 0; see dynamics/spin_integrators.h for the
    // fluctuation-dissipation-consistent noise amplitude.
    double langevin_temperature = 0.0;

    // Normalisation of the damped equation of motion: Landau-Lifshitz form
    // (default, damping constant λ = alpha_gilbert) or Gilbert form (the same
    // divided by 1 + α²); see dynamics/spin_integrators.h.
    classical_spin::dynamics::DampingForm damping_form = classical_spin::dynamics::DampingForm::LandauLifshitz;

    // ------------------------------------------------------------------
    // Persistent scratch buffers for cluster-MC sweeps.
    //
    // wolff_update / swendsen_wang_sweep used to allocate fresh
    // vector<double>(N) / vector<int>(N) on every call; on a 4×L³ pyrochlore
    // with L=12 (27 648 sites) this is ~hundreds of kB of malloc/free per
    // sweep. Keeping the buffers as members reuses the same allocation
    // across sweeps (and across replicas in PT, since PT swaps `spins`
    // pointers, not Lattice objects). They are `mutable` so const-qualified
    // helpers can use them when needed.
    // ------------------------------------------------------------------
    mutable vector<double>  cluster_proj_buf;     // size lattice_size
    mutable vector<uint8_t> cluster_in_cluster;   // size lattice_size
    mutable vector<size_t>  cluster_stack_buf;    // BFS stack (Wolff)
    mutable vector<int>     uf_parent;            // Union-Find parent
    mutable vector<int>     uf_size;              // Union-Find subtree size
    mutable vector<uint8_t> uf_forbid_flip;       // SW: ghost-bonded clusters
    mutable vector<uint8_t> uf_flip_root;         // SW: per-root flip flag
    mutable vector<size_t>  cluster_members_buf;  // members of the current cluster

    // True iff some twist matrix differs from the identity. Twisted-bond
    // cold paths are taken only when this is set; keep it in sync through
    // sync_twist_state() whenever twist_matrices change.
    bool twist_active = false;

    // Cached result of cluster_embedding_is_exact() (-1 = unknown).
    mutable int8_t cluster_exact_cache = -1;

    // onsite_scalar[i] = 1 when S^T A_i S is constant on the sphere (A_i's
    // symmetric part is a multiple of the identity). Such terms exert no
    // torque; the hot kernels skip them. Rebuilt by build_flat_bilinear_tables
    // (call refresh_onsite_flags() after editing onsite_interaction directly).
    vector<uint8_t> onsite_scalar;

    void refresh_onsite_flags() {
        onsite_scalar.assign(lattice_size, 1);
        for (size_t i = 0; i < lattice_size; ++i) onsite_scalar[i] = onsite_is_scalar(i) ? 1 : 0;
        cluster_exact_cache = -1;
    }

    void sync_twist_state() {
        cluster_exact_cache = -1;
        twist_active = false;
        if (spin_dim == 3)
            for (size_t d = 0; d < 3; ++d)
                if (!twist_matrices[d].isIdentity(1e-15)) twist_active = true;
        refresh_twisted_bonds();
    }

    /// M(w) = R_2^{w_2} R_1^{w_1} R_0^{w_0} (R^{-1} = R^T): the rotation a
    /// partner spin picks up when the bond crosses the boundaries w (|w_d| > 1
    /// for bonds longer than the lattice extent along d).
    Eigen::Matrix3d twist_product(const array<int8_t, 3>& wrap) const {
        Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
        for (size_t d = 0; d < 3; ++d) {
            if (wrap[d] == 0) continue;
            const Eigen::Matrix3d R = twist_matrices[d].topLeftCorner<3, 3>();
            const Eigen::Matrix3d Rw = (wrap[d] > 0) ? R : Eigen::Matrix3d(R.transpose());
            for (int n = 0; n < std::abs(int(wrap[d])); ++n) M = Rw * M;
        }
        return M;
    }

    /// Recompute the effective matrices of all wrapped bonds from J0 and
    /// the current twists. O(#wrapped bonds).
    void refresh_twisted_bonds() {
        if (spin_dim != 3 || bi_flat_J0.empty()) return;
        for (size_t k = 0; k < bi_flat_partner.size(); ++k) {
            if (!bi_flat_needs_twist[k]) continue;
            const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J0(&bi_flat_J0[9 * k]);
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(&bi_flat_J[9 * k]);
            if (bi_flat_forward[k]) {
                J = J0 * twist_product(bi_flat_wrap[k]);
            } else {
                const array<int8_t, 3>& w = bi_flat_wrap[k];
                const array<int8_t, 3> w_fwd = {int8_t(-w[0]), int8_t(-w[1]), int8_t(-w[2])};
                J = twist_product(w_fwd).transpose() * J0;
            }
        }
    }

    /**
     * Check if lattice is a pyrochlore type (pyrochlore or pyrochlore_non_kramer)
     * Used to validate pyrochlore-specific order parameters.
     */
    bool is_pyrochlore() const {
        return lattice_type == "pyrochlore" || lattice_type == "pyrochlore_non_kramer";
    }

    /**
     * Constructor: Build a lattice from a unit cell
     * 
     * @param uc        Unit cell defining the lattice structure
     * @param dim1      Lattice size in first dimension
     * @param dim2      Lattice size in second dimension
     * @param dim3      Lattice size in third dimension
     * @param spin_l    Magnitude of spin vectors
     */
    Lattice(const UnitCell& uc, size_t dim1, size_t dim2, size_t dim3, float spin_l = 1.0)
        : unit_cell(uc), 
          spin_dim(uc.N), 
          N_atoms(uc.N_atoms),
          dim1(dim1), 
          dim2(dim2), 
          dim3(dim3),
          spin_length(spin_l)
    {
        if (dim1 == 0 || dim2 == 0 || dim3 == 0)
            throw std::invalid_argument("Lattice: every lattice dimension must be >= 1");
        if (!(spin_l > 0.0f) || !std::isfinite(spin_l))
            throw std::invalid_argument("Lattice: spin_length must be positive and finite");
        uc.validate();
        // 8-component spins of a Gell-Mann-bracket cell are qutrit pure states
        // (CP^2); see set_su3_mc_manifold.
        su3_cp2 = (uc.N == 8 && uc.poisson_bracket == 2.0);
        lattice_size = N_atoms * dim1 * dim2 * dim3;
        
        // Initialize arrays
        spins.resize(lattice_size);
        site_positions.resize(lattice_size);
        field.resize(lattice_size);
        onsite_interaction.resize(lattice_size);
        bilinear_interaction.resize(lattice_size);
        trilinear_interaction.resize(lattice_size);
        bilinear_partners.resize(lattice_size);
        trilinear_partners.resize(lattice_size);
        bilinear_wrap_dir.resize(lattice_size);
        bilinear_forward.resize(lattice_size);
        sublattice_frames.resize(N_atoms);
        
        // Copy sublattice frames from unit cell
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            sublattice_frames[atom] = uc.sublattice_frames[atom];
        }
        
        // Copy AFM sublattice signs from unit cell
        afm_sublattice_signs = uc.afm_sublattice_signs;

        // Initialize twist matrices to identity
        for (size_t d = 0; d < 3; ++d) {
            twist_matrices[d] = SpinMatrix::Identity(spin_dim, spin_dim);
            rotation_axis[d] = SpinVector::Zero(spin_dim);
            if (spin_dim >= 3) rotation_axis[d](2) = 1.0; // Default: z-axis
        }

        // Initialize time-dependent field
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        t_pulse[0] = 0.0;
        t_pulse[1] = 0.0;
        field_drive_amp = 0.0;
        field_drive_freq = 0.0;
        field_drive_width = 1.0;
        active_drive = classical_spin::dynamics::DriveSchedule(N_atoms, spin_dim);

        // The RNG is seeded once per process (config key `seed`, see
        // spin_solver.cpp); constructing a lattice must not reseed it.

        // Compute boundary thickness (max neighbor offset in each dimension)
        boundary_thickness = {0, 0, 0};
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            auto range = unit_cell.bilinear_interaction.equal_range(atom);
            for (auto it = range.first; it != range.second; ++it) {
                const auto& bi = it->second;
                boundary_thickness[0] = std::max(boundary_thickness[0], size_t(std::abs(bi.offset[0])));
                boundary_thickness[1] = std::max(boundary_thickness[1], size_t(std::abs(bi.offset[1])));
                boundary_thickness[2] = std::max(boundary_thickness[2], size_t(std::abs(bi.offset[2])));
            }
        }
        {
            // A lattice no wider than 2|offset| along a bonded direction maps
            // distinct bonds onto the same pair of sites (their couplings add).
            // That is a valid periodic Hamiltonian, but rarely the intended one.
            const array<size_t, 3> dims = {dim1, dim2, dim3};
            for (size_t d = 0; d < 3; ++d)
                if (boundary_thickness[d] > 0 && dims[d] < 2 * boundary_thickness[d] + 1)
                    cout << "Warning: lattice extent " << dims[d] << " along a" << d + 1
                         << " is below 2*" << boundary_thickness[d] << "+1; periodic images of distinct bonds "
                         << "coincide (their couplings add)" << endl;
        }

        // Build the lattice
        cout << "Initializing lattice with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "Total sites: " << lattice_size << endl;
        cout << "Spin dimension: " << spin_dim << ", Atoms per cell: " << N_atoms << endl;

        // First pass: count interactions per site for proper sizing
        vector<size_t> bi_count(lattice_size, 0);
        vector<size_t> tri_count(lattice_size, 0);
        
        size_t site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        // Count forward bilinear interactions
                        auto bi_range = unit_cell.bilinear_interaction.equal_range(atom);
                        bi_count[site_idx] = std::distance(bi_range.first, bi_range.second);
                        
                        // Count forward trilinear interactions
                        auto tri_range = unit_cell.trilinear_interaction.equal_range(atom);
                        tri_count[site_idx] = std::distance(tri_range.first, tri_range.second);
                        
                        ++site_idx;
                    }
                }
            }
        }

        // Second pass: add reverse interaction counts
        site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        // Add reverse bilinear counts
                        auto bi_range = unit_cell.bilinear_interaction.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            size_t partner_idx = flatten_index_periodic(int(i) + bi.offset[0],
                                                                        int(j) + bi.offset[1],
                                                                        int(k) + bi.offset[2], bi.partner);
                            bi_count[partner_idx]++;
                        }
                        
                        // Add reverse trilinear counts (2 permutations per interaction)
                        auto tri_range = unit_cell.trilinear_interaction.equal_range(atom);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& tri = it->second;
                            size_t p1 = flatten_index_periodic(i + tri.offset1[0], 
                                                               j + tri.offset1[1], 
                                                               k + tri.offset1[2], 
                                                               tri.partner1);
                            size_t p2 = flatten_index_periodic(i + tri.offset2[0], 
                                                               j + tri.offset2[1], 
                                                               k + tri.offset2[2], 
                                                               tri.partner2);
                            tri_count[p1]++;
                            tri_count[p2]++;
                        }
                        
                        ++site_idx;
                    }
                }
            }
        }

        // Allocate storage with correct sizes
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            bilinear_interaction[idx].reserve(bi_count[idx]);
            bilinear_partners[idx].reserve(bi_count[idx]);
            bilinear_wrap_dir[idx].reserve(bi_count[idx]);
            trilinear_interaction[idx].reserve(tri_count[idx]);
            trilinear_partners[idx].reserve(tri_count[idx]);
        }

        // Third pass: build interactions
        site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        // Compute real-space position
                        Eigen::Vector3d pos = unit_cell.lattice_pos[atom];
                        pos += i * unit_cell.lattice_vectors[0];
                        pos += j * unit_cell.lattice_vectors[1];
                        pos += k * unit_cell.lattice_vectors[2];
                        site_positions[site_idx] = pos;

                        // Uniformly random initial state (sphere or CP^2)
                        spins[site_idx].resize(spin_dim);
                        random_state_into(spins[site_idx].data());

                        // Copy field from unit cell
                        field[site_idx] = unit_cell.field[atom];

                        // Copy on-site interaction
                        onsite_interaction[site_idx] = unit_cell.onsite_interaction[atom];

                        // Build bilinear interactions (forward and reverse)
                        auto bi_range = unit_cell.bilinear_interaction.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            
                            // Partner site with periodic boundaries; wrap[d] counts the
                            // periods crossed (the twist the bond picks up).
                            array<int8_t, 3> wrap = {0, 0, 0};
                            int n_wraps[3];
                            const size_t pi = wrap_coordinate(long(i) + bi.offset[0], dim1, n_wraps[0]);
                            const size_t pj = wrap_coordinate(long(j) + bi.offset[1], dim2, n_wraps[1]);
                            const size_t pk = wrap_coordinate(long(k) + bi.offset[2], dim3, n_wraps[2]);
                            for (size_t d = 0; d < 3; ++d) {
                                if (std::abs(n_wraps[d]) > 127)
                                    throw std::invalid_argument("Lattice: bond offset exceeds 127 lattice periods");
                                wrap[d] = static_cast<int8_t>(n_wraps[d]);
                            }
                            
                            size_t partner_idx = flatten_index(pi, pj, pk, bi.partner);

                            // A bond onto the site's own periodic image (lattice
                            // extent 1 along a bonded direction) is the single-ion
                            // term S^T J S: fold it into the on-site matrix so the
                            // energy, ΔE, local field and dynamics all count it once.
                            if (partner_idx == site_idx) {
                                onsite_interaction[site_idx] += 0.5 * (bi.interaction + bi.interaction.transpose());
                                continue;
                            }

                            // Add forward interaction: site_idx -> partner_idx with J
                            bilinear_interaction[site_idx].push_back(bi.interaction);
                            bilinear_partners[site_idx].push_back(partner_idx);
                            bilinear_wrap_dir[site_idx].push_back(wrap);
                            bilinear_forward[site_idx].push_back(1);
                            
                            // Add reverse interaction: partner_idx -> site_idx with J^T
                            array<int8_t, 3> wrap_reverse = {
                                static_cast<int8_t>(-wrap[0]),
                                static_cast<int8_t>(-wrap[1]),
                                static_cast<int8_t>(-wrap[2])
                            };
                            bilinear_interaction[partner_idx].push_back(bi.interaction.transpose());
                            bilinear_partners[partner_idx].push_back(site_idx);
                            bilinear_wrap_dir[partner_idx].push_back(wrap_reverse);
                            bilinear_forward[partner_idx].push_back(0);
                        }

                        // Build trilinear interactions (forward and two permutations)
                        auto tri_range = unit_cell.trilinear_interaction.equal_range(atom);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& tri = it->second;
                            
                            size_t p1 = flatten_index_periodic(i + tri.offset1[0], 
                                                               j + tri.offset1[1], 
                                                               k + tri.offset1[2], 
                                                               tri.partner1);
                            size_t p2 = flatten_index_periodic(i + tri.offset2[0], 
                                                               j + tri.offset2[1], 
                                                               k + tri.offset2[2], 
                                                               tri.partner2);
                            
                            if (p1 == site_idx || p2 == site_idx || p1 == p2) {
                                throw std::invalid_argument(
                                    "Lattice: trilinear coupling maps two of its sites onto the "
                                    "same lattice site; enlarge the lattice along the coupled "
                                    "direction(s)");
                            }

                            // Add forward interaction: T[i,j,k] with (site_idx, p1, p2)
                            trilinear_interaction[site_idx].push_back(tri.interaction);
                            trilinear_partners[site_idx].push_back({p1, p2});
                            
                            // Add permutation 1: T[j,k,i] with (p1, p2, site_idx)
                            // transpose3D(T, N, N, N) performs cyclic permutation: T[i](j,k) -> T_new[j](k,i)
                            SpinTensor3 perm1 = transpose3D(tri.interaction, spin_dim, spin_dim, spin_dim);
                            trilinear_interaction[p1].push_back(perm1);
                            trilinear_partners[p1].push_back({p2, site_idx});
                            
                            // Add permutation 2: T[k,i,j] with (p2, site_idx, p1)
                            // Apply transpose3D twice for double cyclic permutation
                            SpinTensor3 perm2 = transpose3D(perm1, spin_dim, spin_dim, spin_dim);
                            trilinear_interaction[p2].push_back(perm2);
                            trilinear_partners[p2].push_back({site_idx, p1});
                        }

                        ++site_idx;
                    }
                }
            }
        }

        // Set num_bi and num_tri to maximum (for compatibility)
        num_bi = *std::max_element(bi_count.begin(), bi_count.end());
        num_tri = *std::max_element(tri_count.begin(), tri_count.end());

        build_boundary_sites();
        build_flat_bilinear_tables();
        build_color_partition();
        cout << "Lattice initialization complete!" << endl;
        cout << "Max bilinear interactions per site: " << num_bi << endl;
        cout << "Max trilinear interactions per site: " << num_tri << endl;
        cout << "Sublattice colouring: " << n_colors << " colour(s) for "
             << lattice_size << " sites" << endl;
    }

    /**
     * Build the SoA / flat mirror of the bilinear interaction table.
     *
     * Idempotent. Call after the nested `bilinear_interaction`,
     * `bilinear_partners`, and `bilinear_wrap_dir` tables are populated
     * (i.e. at the end of initialize() and after the copy constructor /
     * assignment op runs).
     *
     * Costs O(total_bonds * spin_dim^2) memory and time; ~480 kB for
     * a 32x32x1 honeycomb with spin_dim=3 (negligible).
     */
    void build_flat_bilinear_tables() {
        cluster_exact_cache = -1;
        bi_flat_D2 = spin_dim * spin_dim;

        bi_flat_offset.assign(lattice_size + 1, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            bi_flat_offset[i + 1] = bi_flat_offset[i] + bilinear_partners[i].size();
        }
        const size_t total_bonds = bi_flat_offset[lattice_size];

        bi_flat_partner.assign(total_bonds, 0);
        bi_flat_J.assign(total_bonds * bi_flat_D2, 0.0);
        bi_flat_needs_twist.assign(total_bonds, 0);
        bi_flat_wrap.assign(total_bonds, std::array<int8_t, 3>{0, 0, 0});
        bi_flat_forward.assign(total_bonds, 1);

        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t base = bi_flat_offset[i];
            const size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                const size_t k = base + n;
                bi_flat_partner[k] = bilinear_partners[i][n];

                const auto& J = bilinear_interaction[i][n];
                double* dst = &bi_flat_J[k * bi_flat_D2];
                for (size_t a = 0; a < spin_dim; ++a) {
                    for (size_t b = 0; b < spin_dim; ++b) {
                        dst[a * spin_dim + b] = J(a, b);
                    }
                }

                const auto& wrap = bilinear_wrap_dir[i][n];
                bi_flat_wrap[k] = wrap;
                if (n < bilinear_forward[i].size()) bi_flat_forward[k] = bilinear_forward[i][n];
                // Twists are SO(3) rotations: only spin_dim == 3 bonds can carry one.
                bi_flat_needs_twist[k] =
                    (spin_dim == 3 && (wrap[0] != 0 || wrap[1] != 0 || wrap[2] != 0))
                        ? uint8_t(1) : uint8_t(0);
            }
        }
        bi_flat_J0 = bi_flat_J;
        refresh_twisted_bonds();
        refresh_onsite_flags();
    }

    /**
     * Build a sublattice colouring of the bond graph (bilinear + trilinear
     * union) so that no two sites of the same colour share any interaction.
     *
     * Used by the coloured (parallel) Metropolis / overrelaxation sweeps:
     * for each colour in turn, all sites of that colour can be updated by
     * independent OpenMP threads with zero races, because the per-site
     * kernel only reads neighbour spins and writes spins[i].
     *
     * Algorithm: greedy first-fit colouring on sites in natural order.
     * For typical lattices (honeycomb 2-colourable, kagome 3-colourable,
     * pyrochlore 4-colourable, cubic 2-colourable) this returns the
     * chromatic number; for irregular graphs it returns at most
     * (max_degree + 1) colours, which is still fine because the parallel
     * efficiency scales as ~1/n_colors and even 6-7 colours give 4-6x on
     * 8 threads.
     *
     * Idempotent. O(N * (avg_degree + n_colors)). Negligible at init.
     */
    void build_color_partition() {
        if (lattice_size == 0) {
            n_colors = 0;
            color_of_site.clear();
            sites_by_color_csr_off.clear();
            sites_by_color_csr.clear();
            return;
        }

        color_of_site.assign(lattice_size, std::numeric_limits<uint16_t>::max());

        std::vector<uint8_t> forbidden(64, 0);
        size_t max_color_used = 0;

        for (size_t i = 0; i < lattice_size; ++i) {
            std::fill(forbidden.begin(), forbidden.end(), uint8_t(0));

            const size_t bi_base = bi_flat_offset[i];
            const size_t bi_end  = bi_flat_offset[i + 1];
            for (size_t k = bi_base; k < bi_end; ++k) {
                const size_t partner = bi_flat_partner[k];
                if (partner >= lattice_size) continue;
                const uint16_t c = color_of_site[partner];
                if (c != std::numeric_limits<uint16_t>::max()) {
                    if (size_t(c) >= forbidden.size()) forbidden.resize(size_t(c) + 1, 0);
                    forbidden[c] = 1;
                }
            }

            const size_t n_tri_i = trilinear_partners[i].size();
            for (size_t n = 0; n < n_tri_i; ++n) {
                for (int e = 0; e < 2; ++e) {
                    const size_t partner = trilinear_partners[i][n][e];
                    if (partner >= lattice_size) continue;
                    const uint16_t c = color_of_site[partner];
                    if (c != std::numeric_limits<uint16_t>::max()) {
                        if (size_t(c) >= forbidden.size()) forbidden.resize(size_t(c) + 1, 0);
                        forbidden[c] = 1;
                    }
                }
            }

            uint16_t chosen = 0;
            while (size_t(chosen) < forbidden.size() && forbidden[chosen]) ++chosen;
            color_of_site[i] = chosen;
            if (size_t(chosen) > max_color_used) max_color_used = chosen;
        }

        n_colors = max_color_used + 1;

        sites_by_color_csr_off.assign(n_colors + 1, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            sites_by_color_csr_off[color_of_site[i] + 1] += 1;
        }
        for (size_t c = 0; c < n_colors; ++c) {
            sites_by_color_csr_off[c + 1] += sites_by_color_csr_off[c];
        }
        sites_by_color_csr.assign(lattice_size, 0);
        std::vector<size_t> cursor(n_colors, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            const uint16_t c = color_of_site[i];
            sites_by_color_csr[sites_by_color_csr_off[c] + cursor[c]] = i;
            ++cursor[c];
        }
    }

    /**
     * Copy constructor
     *
     * IMPORTANT: this copies the *full* configurable state of the lattice,
     * including the time-dependent drive parameters and Gilbert damping.
     * Earlier revisions left `field_drive`, `t_pulse`, `field_drive_*`,
     * `alpha_gilbert`, `twist_angles`, and `afm_sublattice_signs`
     * default-initialised, which silently broke MD / 2DCS workflows on
     * cloned lattices (set_pulse() would touch a zero-sized Eigen vector
     * and crash, and copies forgot the user-set damping). Keeping the
     * copy faithful is also a precondition for the OpenMP τ-loop in
     * Lattice::pump_probe_spectroscopy (Ingredient XV / W2): each thread
     * needs its own fully-functional clone with its own pulse buffers.
     */
    Lattice(const Lattice& other)
        : unit_cell(other.unit_cell),
          lattice_type(other.lattice_type),
          spin_dim(other.spin_dim),
          N_atoms(other.N_atoms),
          dim1(other.dim1),
          dim2(other.dim2),
          dim3(other.dim3),
          lattice_size(other.lattice_size),
          spin_length(other.spin_length),
          spins(other.spins),
          site_positions(other.site_positions),
          field(other.field),
          onsite_interaction(other.onsite_interaction),
          bilinear_interaction(other.bilinear_interaction),
          trilinear_interaction(other.trilinear_interaction),
          bilinear_partners(other.bilinear_partners),
          trilinear_partners(other.trilinear_partners),
          sublattice_frames(other.sublattice_frames),
          afm_sublattice_signs(other.afm_sublattice_signs),
          num_bi(other.num_bi),
          num_tri(other.num_tri),
          bi_flat_offset(other.bi_flat_offset),
          bi_flat_partner(other.bi_flat_partner),
          bi_flat_J(other.bi_flat_J),
          bi_flat_needs_twist(other.bi_flat_needs_twist),
          bi_flat_wrap(other.bi_flat_wrap),
          bi_flat_J0(other.bi_flat_J0),
          bi_flat_forward(other.bi_flat_forward),
          bi_flat_D2(other.bi_flat_D2),
          color_of_site(other.color_of_site),
          sites_by_color_csr_off(other.sites_by_color_csr_off),
          sites_by_color_csr(other.sites_by_color_csr),
          n_colors(other.n_colors),
          twist_matrices(other.twist_matrices),
          rotation_axis(other.rotation_axis),
          twist_angles(other.twist_angles),
          bilinear_wrap_dir(other.bilinear_wrap_dir),
          bilinear_forward(other.bilinear_forward),
          boundary_sites_per_dim(other.boundary_sites_per_dim),
          boundary_thickness(other.boundary_thickness),
          field_drive(other.field_drive),
          t_pulse(other.t_pulse),
          field_drive_amp(other.field_drive_amp),
          field_drive_freq(other.field_drive_freq),
          field_drive_width(other.field_drive_width),
          active_drive(other.active_drive),
          alpha_gilbert(other.alpha_gilbert),
          langevin_temperature(other.langevin_temperature),
          damping_form(other.damping_form),
          twist_active(other.twist_active),
          onsite_scalar(other.onsite_scalar),
          local_update(other.local_update),
          parallel_sweep_min_sites(other.parallel_sweep_min_sites),
          su3_cp2(other.su3_cp2)
    {}

    // ============================================================
    // UTILITY METHODS
    // ============================================================

    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom) const {
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }

    /**
     * Euclidean (floor) reduction of a cell coordinate onto [0, L); `n_wraps`
     * receives the number of periods crossed (negative below 0). Correct for
     * any |coord|: the previous single-step `coord ± L` returned an
     * out-of-range index (a heap overflow in the constructor) for bonds longer
     * than the lattice, e.g. the honeycomb J3 offset (1,-2,0) with L2 = 1.
     */
    static size_t wrap_coordinate(long coord, size_t L, int& n_wraps) {
        const long l = long(L);
        long q = coord / l, r = coord % l;
        if (r < 0) { r += l; --q; }
        n_wraps = int(q);
        return size_t(r);
    }

    /**
     * Apply periodic boundary condition
     */
    size_t periodic_boundary(int coord, size_t dim_size) const {
        int n_wraps;
        return wrap_coordinate(coord, dim_size, n_wraps);
    }

    /**
     * Flatten with periodic boundaries
     */
    size_t flatten_index_periodic(int i, int j, int k, size_t atom) const {
        return flatten_index(periodic_boundary(i, dim1),
                           periodic_boundary(j, dim2),
                           periodic_boundary(k, dim3),
                           atom);
    }

    /**
     * Generate random spin on n-sphere
     */
    SpinVector gen_random_spin(float spin_l) {
        SpinVector spin(spin_dim);
        gen_random_spin_into(spin.data(), spin_l);
        return spin;
    }

    /**
     * Zero-allocation variant: write a uniformly random length-spin_l vector
     * into the caller-provided buffer (must have at least spin_dim doubles).
     *
     * Used by the hot Metropolis loop to avoid allocating an Eigen::VectorXd
     * per proposed move.
     */
    void gen_random_spin_into(double* out, float spin_l) const {
        random_point_on_sphere(out, spin_dim, double(spin_l));
    }

    /// Uniformly random local state: uniform on the sphere |S| = spin_length,
    /// or a Haar-random pure state on CP^2 (su3_cp2).
    void random_state_into(double* out) const {
        if (su3_cp2) classical_spin::su3::random_cp2(out);
        else random_point_on_sphere(out, spin_dim, double(spin_length));
    }

    /**
     * Symmetric single-site proposal shared by the serial and coloured
     * Metropolis sweeps: a uniformly random state, or a small move of width
     * sigma — S + sigma u renormalised (u uniform with |u| = spin_length) on
     * the sphere, psi + sigma z on CP^2 (classical_spin::su3::propose_cp2).
     * Returns false only in the measure-zero case S + sigma u = 0.
     */
    inline bool propose_spin(const double* old_spin, bool gaussian, double sigma, double* out) const {
        if (su3_cp2) {
            if (gaussian) classical_spin::su3::propose_cp2(old_spin, sigma, out);
            else classical_spin::su3::random_cp2(out);
            return true;
        }
        gen_random_spin_into(out, spin_length);
        if (!gaussian) return true;
        double sum_sq = 0.0;
        for (size_t d = 0; d < spin_dim; ++d) {
            out[d] = old_spin[d] + sigma * out[d];
            sum_sq += out[d] * out[d];
        }
        if (sum_sq < 1e-20) return false;
        const double inv_norm = double(spin_length) / std::sqrt(sum_sq);
        for (size_t d = 0; d < spin_dim; ++d) out[d] *= inv_norm;
        return true;
    }

    /**
     * Build boundary site lists for twist updates
     */
    void build_boundary_sites() {
        for (size_t d = 0; d < 3; ++d) {
            boundary_sites_per_dim[d].clear();
        }

        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        size_t idx = flatten_index(i, j, k, atom);
                        
                        // Check if site is near boundary in each dimension
                        if (i < boundary_thickness[0] || i >= dim1 - boundary_thickness[0]) {
                            boundary_sites_per_dim[0].push_back(idx);
                        }
                        if (j < boundary_thickness[1] || j >= dim2 - boundary_thickness[1]) {
                            boundary_sites_per_dim[1].push_back(idx);
                        }
                        if (k < boundary_thickness[2] || k >= dim3 - boundary_thickness[2]) {
                            boundary_sites_per_dim[2].push_back(idx);
                        }
                    }
                }
            }
        }
    }

    /**
     * Set twist rotation axes
     */
    void set_twist_axes(const array<SpinVector, 3>& axes) {
        rotation_axis = axes;
        // Initialize twist matrices and angles to zero rotation
        for (size_t d = 0; d < 3; ++d) {
            twist_angles[d] = 0.0;
            twist_matrices[d] = rotation_from_axis_angle(rotation_axis[d], 0.0);
        }
        sync_twist_state();
    }

    /**
     * Create rotation matrix from axis-angle representation (Rodrigues' formula)
     */
    static SpinMatrix rotation_from_axis_angle(const SpinVector& axis, double angle) {
        size_t N = axis.size();
        if (N != 3) {
            return SpinMatrix::Identity(N, N); // Only defined for 3D spins
        }
        
        SpinVector n = axis.normalized();
        double c = std::cos(angle);
        double s = std::sin(angle);
        
        SpinMatrix K = SpinMatrix::Zero(3, 3);
        K(0, 1) = -n(2);
        K(0, 2) =  n(1);
        K(1, 0) =  n(2);
        K(1, 2) = -n(0);
        K(2, 0) = -n(1);
        K(2, 1) =  n(0);
        
        return SpinMatrix::Identity(3, 3) + s * K + (1.0 - c) * K * K;
    }

    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================

    /**
     * Compute energy of a single site
     */
    double site_energy(const SpinVector& spin_here, size_t site_index) const {
        // -B·S + S^T A S + S·(Σ J_eff S_j + trilinear): every term involving
        // the site, each counted once from this site's side.
        double g[MAX_SPIN_DIM];
        linear_field(site_index, spins_view(), g);
        double E = onsite_energy(site_index, spin_here.data());
        for (size_t d = 0; d < spin_dim; ++d) E += spin_here(d) * g[d];
        return E;
    }

    /**
     * Compute energy difference for spin flip (optimized for Metropolis)
     */
    double site_energy_diff(const SpinVector& new_spin, const SpinVector& old_spin, 
                           size_t site_index) const {
        return site_energy_diff_flat(new_spin.data(), old_spin.data(), site_index);
    }

    // ------------------------------------------------------------------
    // Local-field kernel layer.
    //
    // Every local quantity — Metropolis ΔE, the overrelaxation axis, the
    // T = 0 quench, the LLG right-hand side and the total energy — is built
    // from the same two primitives below, so a coupling is either right
    // everywhere or wrong everywhere (previously five hand-copied loops had
    // drifted apart: the Eigen `get_local_field` used an ad-hoc trilinear
    // term, and the twisted-bond branches overran 3-element buffers for
    // spin_dim > 3).
    //
    // `spin_of(j)` returns a pointer to the spin_dim components of site j,
    // so the kernels serve both the live configuration (SpinsView) and the
    // flat ODE state vectors (FlatView).
    // ------------------------------------------------------------------
    static constexpr size_t MAX_SPIN_DIM = 16;

    struct SpinsView {
        const SpinConfig& s;
        const double* operator()(size_t j) const { return s[j].data(); }
    };
    struct FlatView {
        const double* x;
        size_t dim;
        const double* operator()(size_t j) const { return x + j * dim; }
    };
    SpinsView spins_view() const { return SpinsView{spins}; }
    FlatView flat_view(const double* x) const { return FlatView{x, spin_dim}; }

    /**
     * H_out += Σ_n J_n S_{p(n)} + Σ_t T_t : (S_{p1(t)} ⊗ S_{p2(t)}),
     * the gradient ∂E/∂S_i of every term LINEAR in S_i (bilinear exchange
     * with twists folded into the effective bond matrices, and trilinear
     * couplings). Zeeman (-B) and the quadratic on-site term (2 A S_i) are
     * added by the callers that need them.
     */
    // The kernels below are written once with a compile-time spin dimension
    // DC (DC = 0: runtime spin_dim); the public entry points dispatch to the
    // DC = 3 (SO(3)) and DC = 8 (SU(3)) instantiations so the hot loops are
    // fully unrolled for the production cases.
    template<int DC, typename SpinOf>
    inline void accumulate_exchange_field_d(size_t site, const SpinOf& spin_of,
                                            double* __restrict H_out) const {
        const size_t D = (DC > 0) ? size_t(DC) : spin_dim;
        const size_t D2 = D * D;
        const size_t bi_end = bi_flat_offset[site + 1];
        for (size_t k = bi_flat_offset[site]; k < bi_end; ++k) {
            const double* __restrict P = spin_of(bi_flat_partner[k]);
            const double* __restrict J = &bi_flat_J[k * D2];
            for (size_t a = 0; a < D; ++a) {
                double row = 0.0;
                for (size_t b = 0; b < D; ++b) row += J[a * D + b] * P[b];
                H_out[a] += row;
            }
        }
        const size_t n_tri = trilinear_partners[site].size();
        for (size_t n = 0; n < n_tri; ++n) {
            const double* S_j = spin_of(trilinear_partners[site][n][0]);
            const double* S_k = spin_of(trilinear_partners[site][n][1]);
            const auto& T = trilinear_interaction[site][n];
            for (size_t a = 0; a < D; ++a) {
                const auto& Ta = T[a];
                double acc = 0.0;
                for (size_t b = 0; b < D; ++b) {
                    double row = 0.0;
                    for (size_t c = 0; c < D; ++c) row += Ta(b, c) * S_k[c];
                    acc += S_j[b] * row;
                }
                H_out[a] += acc;
            }
        }
    }

    template<typename SpinOf>
    inline void accumulate_exchange_field(size_t site, const SpinOf& spin_of,
                                          double* __restrict H_out) const {
        switch (spin_dim) {
            case 3:  accumulate_exchange_field_d<3>(site, spin_of, H_out); break;
            case 8:  accumulate_exchange_field_d<8>(site, spin_of, H_out); break;
            default: accumulate_exchange_field_d<0>(site, spin_of, H_out); break;
        }
    }

    /// g = -B_i + Σ J S_j + trilinear: the S_i-independent part of ∂E/∂S_i.
    template<int DC, typename SpinOf>
    inline void linear_field_d(size_t site, const SpinOf& spin_of, double* g) const {
        const size_t D = (DC > 0) ? size_t(DC) : spin_dim;
        const double* B = field[site].data();
        for (size_t d = 0; d < D; ++d) g[d] = -B[d];
        accumulate_exchange_field_d<DC>(site, spin_of, g);
    }

    template<typename SpinOf>
    inline void linear_field(size_t site, const SpinOf& spin_of, double* g) const {
        switch (spin_dim) {
            case 3:  linear_field_d<3>(site, spin_of, g); break;
            case 8:  linear_field_d<8>(site, spin_of, g); break;
            default: linear_field_d<0>(site, spin_of, g); break;
        }
    }

    /// S^T A S for the on-site matrix of `site`.
    template<int DC>
    inline double onsite_energy_d(size_t site, const double* S) const {
        const size_t D = (DC > 0) ? size_t(DC) : spin_dim;
        const double* A = onsite_interaction[site].data();  // column-major
        double e = 0.0;
        for (size_t b = 0; b < D; ++b) {
            double col = 0.0;
            for (size_t a = 0; a < D; ++a) col += S[a] * A[b * D + a];
            e += col * S[b];
        }
        return e;
    }

    inline double onsite_energy(size_t site, const double* S) const {
        switch (spin_dim) {
            case 3:  return onsite_energy_d<3>(site, S);
            case 8:  return onsite_energy_d<8>(site, S);
            default: return onsite_energy_d<0>(site, S);
        }
    }

    /**
     * True when S^T A S is constant on the sphere, i.e. the symmetric part
     * of the on-site matrix is a multiple of the identity (this includes
     * A = 0). Such terms exert no torque and are skipped by overrelaxation
     * and the quench.
     */
    inline bool onsite_is_scalar(size_t site) const {
        const auto& A = onsite_interaction[site];
        double diag_mean = 0.0, scale = 0.0;
        for (size_t a = 0; a < spin_dim; ++a) diag_mean += A(a, a);
        diag_mean /= double(spin_dim);
        for (size_t a = 0; a < spin_dim; ++a)
            for (size_t b = 0; b < spin_dim; ++b) scale = std::max(scale, std::abs(A(a, b)));
        const double tol = 1e-13 * std::max(scale, 1.0);
        for (size_t a = 0; a < spin_dim; ++a) {
            if (std::abs(A(a, a) - diag_mean) > tol) return false;
            for (size_t b = a + 1; b < spin_dim; ++b)
                if (std::abs(A(a, b) + A(b, a)) > 2.0 * tol) return false;
        }
        return true;
    }

    /**
     * Exact minimiser of e(S) = S^T A S + g^T S on the sphere |S| = s.
     *
     * This is the trust-region subproblem. With A_sym = Q diag(a) Q^T and
     * c = Q^T g, stationary points are S = -Q (A_sym + λ)^{-1} c / 2 with
     *     Σ_k c_k² / (4 (a_k + λ)²) = s²,
     * and the global minimum is the root with λ ≥ -a_min (A_sym + λ PSD).
     * The secular equation is solved in δ = λ + a_min ∈ (0, |c|/(2s)],
     * where d_k = (a_k - a_min) + δ carries no cancellation even when |g| is
     * far below the scale of A (φ(δ) = Σ c_k²/(4 d_k²) - s² decreases
     * monotonically and is <= 0 at the upper end). In the "hard case" (no
     * weight of g on the lowest eigenspace, and the remaining components
     * fit inside the sphere) δ = 0 and the leftover norm goes into the
     * a_min eigenspace, oriented along the current spin `S_cur`.
     */
    template<int N, typename Mat>
    static void minimize_quadratic_on_sphere_impl(const Mat& A, const double* g, double s,
                                                  const double* S_cur, double* S_out, size_t n) {
        using Vec = Eigen::Matrix<double, N, 1>;
        const Mat A_sym = 0.5 * (A + A.transpose());
        Eigen::SelfAdjointEigenSolver<Mat> es(A_sym);
        const Vec a = es.eigenvalues();                          // ascending
        const Mat& Q = es.eigenvectors();
        const Vec c = Q.transpose() * Eigen::Map<const Vec>(g, n);
        const Vec cur = Q.transpose() * Eigen::Map<const Vec>(S_cur, n);
        const double a_min = a(0);
        const double deg_tol = 1e-12 * std::max({std::abs(a_min), std::abs(a(n - 1)), 1e-300});
        double c_low2 = 0.0, c_norm2 = 0.0, rest2 = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const double gap = a(k) - a_min;
            c_norm2 += c(k) * c(k);
            if (gap <= deg_tol) c_low2 += c(k) * c(k);
            else rest2 += (c(k) / (2.0 * gap)) * (c(k) / (2.0 * gap));
        }
        Vec y(n);
        if (c_low2 <= 1e-28 * c_norm2 && rest2 <= s * s) {
            // Hard case: δ = 0.
            double proj2 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double gap = a(k) - a_min;
                y(k) = (gap <= deg_tol) ? 0.0 : -c(k) / (2.0 * gap);
                if (gap <= deg_tol) proj2 += cur(k) * cur(k);
            }
            const double tau = std::sqrt(std::max(0.0, s * s - rest2));
            for (size_t k = 0; k < n; ++k) {
                if (a(k) - a_min > deg_tol) continue;
                y(k) = (proj2 > 1e-30) ? tau * cur(k) / std::sqrt(proj2) : (k == 0 ? tau : 0.0);
            }
        } else {
            auto d_of = [&](size_t k, double delta) { return (a(k) - a_min) + delta; };
            auto phi = [&](double delta) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    const double d = d_of(k, delta);
                    sum += c(k) * c(k) / (4.0 * d * d);
                }
                return sum - s * s;
            };
            double lo = 0.0, hi = std::sqrt(c_norm2) / (2.0 * s);
            double delta = hi;
            for (int it = 0; it < 200; ++it) {
                const double f = phi(delta);
                if (std::abs(f) <= 1e-15 * s * s) break;
                if (f > 0.0) lo = delta; else hi = delta;
                if (hi - lo <= 1e-16 * hi) break;
                double df = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    const double d = d_of(k, delta);
                    df -= c(k) * c(k) / (2.0 * d * d * d);
                }
                double next = (df < 0.0) ? delta - f / df : 0.0;
                if (!(next > lo && next < hi))
                    next = (lo > 0.0) ? std::sqrt(lo * hi) : 0.5 * hi;  // safeguarded bisection
                delta = next;
            }
            for (size_t k = 0; k < n; ++k) y(k) = -c(k) / (2.0 * d_of(k, delta));
        }
        const double yn = y.norm();
        if (yn > 0.0) y *= s / yn;  // remove residual root-finding error
        Eigen::Map<Vec>(S_out, n) = Q * y;
    }

    static void minimize_quadratic_on_sphere(const Eigen::MatrixXd& A, const double* g,
                                             double s, const double* S_cur, double* S_out,
                                             size_t n) {
        if (n == 3) {
            const Eigen::Matrix3d A3 = A.topLeftCorner<3, 3>();
            minimize_quadratic_on_sphere_impl<3>(A3, g, s, S_cur, S_out, 3);
        } else {
            minimize_quadratic_on_sphere_impl<Eigen::Dynamic>(A, g, s, S_cur, S_out, n);
        }
    }

    /**
     * Zero-allocation Δenergy for a proposed single-site move.
     *
     * `new_spin_buf` / `old_spin_buf` are raw pointers (stack buffers in
     * the Metropolis loop and `spins[site].data()`). Exact for every term:
     * ΔE = δ·g + (new^T A new - old^T A old) with g the linear field.
     */
    template<int DC>
    inline double site_energy_diff_d(const double* __restrict new_spin_buf,
                                     const double* __restrict old_spin_buf,
                                     size_t site_index) const {
        const size_t D = (DC > 0) ? size_t(DC) : spin_dim;
        double g[MAX_SPIN_DIM];
        linear_field_d<DC>(site_index, spins_view(), g);
        double dE = 0.0;
        for (size_t d = 0; d < D; ++d) dE += (new_spin_buf[d] - old_spin_buf[d]) * g[d];
        return dE + onsite_energy_d<DC>(site_index, new_spin_buf)
                  - onsite_energy_d<DC>(site_index, old_spin_buf);
    }

    double site_energy_diff_flat(const double* __restrict new_spin_buf,
                                 const double* __restrict old_spin_buf,
                                 size_t site_index) const {
        switch (spin_dim) {
            case 3:  return site_energy_diff_d<3>(new_spin_buf, old_spin_buf, site_index);
            case 8:  return site_energy_diff_d<8>(new_spin_buf, old_spin_buf, site_index);
            default: return site_energy_diff_d<0>(new_spin_buf, old_spin_buf, site_index);
        }
    }

    /**
     * Total energy of a configuration accessed through `spin_of`:
     * Zeeman + on-site per site, each bilinear bond once (partner > i),
     * each trilinear triple once (both partners > i). Self-bonds never
     * reach the bond tables (they are folded into the on-site term at
     * construction), so the partner > i rule is exact.
     */
    template<typename SpinOf>
    double total_energy_impl(const SpinOf& spin_of) const {
        const size_t D = spin_dim;
        double E = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const double* S_i = spin_of(i);
            const double* B = field[i].data();
            for (size_t d = 0; d < D; ++d) E -= B[d] * S_i[d];
            E += onsite_energy(i, S_i);

            const size_t bi_end = bi_flat_offset[i + 1];
            for (size_t k = bi_flat_offset[i]; k < bi_end; ++k) {
                const size_t j = bi_flat_partner[k];
                if (j <= i) continue;
                const double* P = spin_of(j);
                const double* J = &bi_flat_J[k * bi_flat_D2];
                for (size_t a = 0; a < D; ++a) {
                    double row = 0.0;
                    for (size_t b = 0; b < D; ++b) row += J[a * D + b] * P[b];
                    E += S_i[a] * row;
                }
            }

            const size_t n_tri = trilinear_partners[i].size();
            for (size_t n = 0; n < n_tri; ++n) {
                const size_t p1 = trilinear_partners[i][n][0];
                const size_t p2 = trilinear_partners[i][n][1];
                if (p1 <= i || p2 <= i) continue;
                const double* S_j = spin_of(p1);
                const double* S_k = spin_of(p2);
                const auto& T = trilinear_interaction[i][n];
                for (size_t a = 0; a < D; ++a) {
                    double acc = 0.0;
                    for (size_t b = 0; b < D; ++b) {
                        double row = 0.0;
                        for (size_t c = 0; c < D; ++c) row += T[a](b, c) * S_k[c];
                        acc += S_j[b] * row;
                    }
                    E += S_i[a] * acc;
                }
            }
        }
        return E;
    }

    /**
     * Compute total energy of configuration
     */
    double total_energy(const SpinConfig& config) const {
        return total_energy_impl(SpinsView{config});
    }

    /**
     * Total energy of current spin configuration (no-arg wrapper)
     */
    double total_energy() const {
        return total_energy_impl(spins_view());
    }

    /**
     * Energy per site
     */
    double energy_density() const {
        return total_energy() / lattice_size;
    }

    /**
     * Total energy directly from a flat state array (zero allocation).
     */
    double total_energy_flat(const double* state_flat) const {
        return total_energy_impl(flat_view(state_flat));
    }

    /**
     * Energy gradient at a site, H = ∂E/∂S_i = -B + 2 A S_i + Σ J S_j + trilinear.
     *
     * Note the sign: this is the gradient, i.e. minus the physical effective
     * field. The LLG right-hand side is dS/dt = H × S = S × B_eff.
     */
    SpinVector get_local_field(size_t site_index) const {
        SpinVector H(spin_dim);
        get_local_field_flat_impl(spins_view(), site_index, H.data());
        return H;
    }

    /**
     * Energy gradient ∂E/∂S_i from a flat state array (zero allocation; the
     * LLG hot path).
     */
    void get_local_field_flat(const double* state_flat, size_t site_index, double* H_out) const {
        get_local_field_flat_impl(flat_view(state_flat), site_index, H_out);
    }

    template<typename SpinOf>
    inline void get_local_field_flat_impl(const SpinOf& spin_of, size_t site_index,
                                          double* H_out) const {
        linear_field(site_index, spin_of, H_out);
        const auto& A = onsite_interaction[site_index];
        const double* S_i = spin_of(site_index);
        for (size_t a = 0; a < spin_dim; ++a) {
            double row = 0.0;
            for (size_t b = 0; b < spin_dim; ++b) row += (A(a, b) + A(b, a)) * S_i[b];
            H_out[a] += row;
        }
    }

    /**
     * Energy gradient of site `site_index` in an arbitrary configuration.
     */
    SpinVector get_local_field_lattice(const SpinConfig& curr_spins, size_t site_index) const {
        SpinVector H(spin_dim);
        get_local_field_flat_impl(SpinsView{curr_spins}, site_index, H.data());
        return H;
    }

    // ============================================================
    // AUTOCORRELATION ANALYSIS
    // ============================================================

    /**
     * Integrated autocorrelation time of a series (Gamma method with automatic
     * windowing) — delegates to mc::compute_autocorrelation.
     */
    AutocorrelationResult compute_autocorrelation(const vector<double>& energies,
                                                   size_t base_interval = 10) {
        return mc::compute_autocorrelation(energies, base_interval);
    }

    /// Binning analysis of a scalar observable — delegates to mc::binning_analysis.
    static BinningResult binning_analysis(const vector<double>& data) {
        return mc::binning_analysis(data);
    }

    /// Component-wise binning analysis — delegates to mc::binning_analysis_vector.
    static vector<BinningResult> binning_analysis_vector(const vector<SpinVector>& data) {
        return mc::binning_analysis_vector<SpinVector>(data);
    }

    // ============================================================
    // SUBLATTICE MAGNETIZATION
    // ============================================================

    /**
     * Compute magnetization for each sublattice separately
     * 
     * @return Vector of SpinVectors, one per sublattice (N_atoms sublattices)
     *         Each vector is averaged over all unit cells in the local sublattice frame
     */
    vector<SpinVector> magnetization_sublattice() const {
        vector<SpinVector> M_sub(N_atoms);
        size_t n_cells = dim1 * dim2 * dim3;
        double inv_n_cells = 1.0 / double(n_cells);
        
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim);
        }
        
        // Flat loop over all sites - more cache-friendly
        for (size_t site = 0; site < lattice_size; ++site) {
            size_t atom = site % N_atoms;
            M_sub[atom] += spins[site];
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub[atom] *= inv_n_cells;
        }
        
        return M_sub;
    }

    /**
     * Compute magnetization for each sublattice from flat state array
     * 
     * @param state_flat Flat spin state array [lattice_size * spin_dim]
     * @param M_sub_out Output: N_atoms SpinVectors for sublattice magnetizations (in local frame)
     */
    void magnetization_sublattice_from_flat(const double* state_flat, 
                                             vector<SpinVector>& M_sub_out) const {
        size_t n_cells = dim1 * dim2 * dim3;
        
        M_sub_out.resize(N_atoms);
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub_out[atom] = SpinVector::Zero(spin_dim);
        }
        
        // Sum over all sites (in local frame)
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            const double* spin_ptr = state_flat + i * spin_dim;
            
            for (size_t mu = 0; mu < spin_dim; ++mu) {
                M_sub_out[atom](mu) += spin_ptr[mu];
            }
        }
        
        // Normalize
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub_out[atom] /= double(n_cells);
        }
    }

    // ============================================================
    // KAGOME PLANE ORDER PARAMETERS (PYROCHLORE NON-KRAMERS)
    // For pyrochlore: sublattices 0=apex, 1,2,3=kagome base
    // 
    // From unitcell_builders.cpp build_pyrochlore_non_kramer():
    // 
    // KAGOME NN BONDS (sublattices 1,2,3):
    // ┌─────────┬─────────────────┬─────────────────┬────────┐
    // │ Bond    │ Intra-cell      │ Inter-cell      │ J type │
    // ├─────────┼─────────────────┼─────────────────┼────────┤
    // │ (1,2)   │ (0, 0, 0)       │ (-1,+1, 0)      │ Jy     │
    // │ (1,3)   │ (0, 0, 0)       │ (-1, 0,+1)      │ Jx     │
    // │ (2,3)   │ (0, 0, 0)       │ ( 0,+1,-1)      │ Jz     │
    // └─────────┴─────────────────┴─────────────────┴────────┘
    // 
    // TRIANGLES (chirality):
    //   Per unit cell: 1 "up" triangle = {1(i,j,k), 2(i,j,k), 3(i,j,k)}
    //   Uses intra-cell bonds only. The inter-cell bonds connect to
    //   different triangles (no closed "down" kagome triangles on 1,2,3).
    // 
    // DIMERS (nematic):
    //   Per unit cell: 6 bonds = 3 types × 2 (intra + inter)
    //   Each bond type has 2N_cells total bonds in the lattice.
    // 
    // Spins are stored in LOCAL FRAME (x,y,z per sublattice).
    // ============================================================

    /**
     * Pyrochlore local frame: columns of R_sub give x_hat, y_hat, z_hat in global coords.
     * S_global = S_local_x * x_hat[sub] + S_local_y * y_hat[sub] + S_local_z * z_hat[sub]
     *
     * x_hat[4][3], y_hat[4][3], z_hat[4][3]  (sublattice, global xyz)
     */
    struct PyrochloreLocalFrame {
        // z_hat = local Ising axis
        static constexpr double z_hat[4][3] = {
            { 1.0/sqrt(3.0),  1.0/sqrt(3.0),  1.0/sqrt(3.0)},
            { 1.0/sqrt(3.0), -1.0/sqrt(3.0), -1.0/sqrt(3.0)},
            {-1.0/sqrt(3.0),  1.0/sqrt(3.0), -1.0/sqrt(3.0)},
            {-1.0/sqrt(3.0), -1.0/sqrt(3.0),  1.0/sqrt(3.0)}
        };
        // y_hat
        static constexpr double y_hat[4][3] = {
            { 0.0,            -1.0/sqrt(2.0),  1.0/sqrt(2.0)},
            { 0.0,             1.0/sqrt(2.0), -1.0/sqrt(2.0)},
            { 0.0,            -1.0/sqrt(2.0), -1.0/sqrt(2.0)},
            { 0.0,             1.0/sqrt(2.0),  1.0/sqrt(2.0)}
        };
        // x_hat
        static constexpr double x_hat[4][3] = {
            {-2.0/sqrt(6.0),  1.0/sqrt(6.0),  1.0/sqrt(6.0)},
            {-2.0/sqrt(6.0), -1.0/sqrt(6.0), -1.0/sqrt(6.0)},
            { 2.0/sqrt(6.0),  1.0/sqrt(6.0), -1.0/sqrt(6.0)},
            { 2.0/sqrt(6.0), -1.0/sqrt(6.0),  1.0/sqrt(6.0)}
        };

        /** Transform a local-frame spin (Sx,Sy,Sz) on sublattice sub to global frame */
        static void to_global(int sub, double Sx_l, double Sy_l, double Sz_l,
                              double& Sx_g, double& Sy_g, double& Sz_g) {
            Sx_g = Sx_l * x_hat[sub][0] + Sy_l * y_hat[sub][0] + Sz_l * z_hat[sub][0];
            Sy_g = Sx_l * x_hat[sub][1] + Sy_l * y_hat[sub][1] + Sz_l * z_hat[sub][1];
            Sz_g = Sx_l * x_hat[sub][2] + Sy_l * y_hat[sub][2] + Sz_l * z_hat[sub][2];
        }
    };

    /**
     * Structure to hold all pyrochlore order parameters
     * Computed in a single pass for efficiency
     */
    struct PyrochloreOrderParameters {
        // --- Local-frame order parameters ---
        double scalar_chirality;           // χ = <S1·(S2×S3)>  (local frame)
        Eigen::Vector3d vector_chirality;  // κ = <S1×S2 + S2×S3 + S3×S1>  (local frame)
        Eigen::Matrix3d nematic_order;     // Q[bond_type][component] = <Si·Sj>  (local frame)
        double monopole_density;           // <Q> per tetrahedron (signed)
        Eigen::Vector4d monopole_by_sublattice;  // Monopole density by minority sublattice

        // --- Global-frame order parameters ---
        double scalar_chirality_global;          // χ_global = <S1_g·(S2_g×S3_g)>
        Eigen::Vector3d vector_chirality_global;  // κ_global (3-component, global xyz)
        Eigen::Matrix3d nematic_order_global;     // Q_global[bond_type][global_component]
        
        PyrochloreOrderParameters() 
            : scalar_chirality(0.0), 
              vector_chirality(Eigen::Vector3d::Zero()),
              nematic_order(Eigen::Matrix3d::Zero()),
              monopole_density(0.0),
              monopole_by_sublattice(Eigen::Vector4d::Zero()),
              scalar_chirality_global(0.0),
              vector_chirality_global(Eigen::Vector3d::Zero()),
              nematic_order_global(Eigen::Matrix3d::Zero()) {}
    };
    
    /**
     * Compute ALL pyrochlore order parameters in a single pass
     * This is the optimized version that avoids redundant loops over cells.
     * 
     * Combines:
     * - Scalar chirality (kagome triangles)
     * - Vector chirality (kagome triangles)  
     * - Monopole density (tetrahedra)
     * - Signed monopole density (tetrahedra)
     * - Monopole density by sublattice (tetrahedra)
     * 
     * Note: Nematic order still uses bilinear_partners for bond enumeration
     * and is computed separately (different loop structure).
     * 
     * @return PyrochloreOrderParameters struct with all computed values
     */
    PyrochloreOrderParameters compute_pyrochlore_order_parameters_fast() const {
        PyrochloreOrderParameters result;
        
        // Only valid for pyrochlore lattices
        if (!is_pyrochlore()) {
            std::cerr << "Warning: compute_pyrochlore_order_parameters_fast() is only valid for pyrochlore lattices" << std::endl;
            return result;
        }
        if (N_atoms < 4 || spin_dim < 3) {
            return result;
        }
        
        const size_t n_cells = dim1 * dim2 * dim3;
        const double inv_n_cells = 1.0 / double(n_cells);
        
        // Accumulators — local frame
        double chi_sum = 0.0;                              // Scalar chirality
        Eigen::Vector3d kappa_sum = Eigen::Vector3d::Zero();  // Vector chirality
        double Q_sum = 0.0;                                // Q monopole density (signed)
        Eigen::Vector4d monopole_sub = Eigen::Vector4d::Zero();  // By sublattice

        // Accumulators — global frame
        double chi_sum_g = 0.0;
        Eigen::Vector3d kappa_sum_g = Eigen::Vector3d::Zero();
        
        // Single pass over all unit cells
        for (size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
            // Extract i, j, k from cell_idx
            size_t k = cell_idx % dim3;
            size_t j = (cell_idx / dim3) % dim2;
            size_t i = cell_idx / (dim2 * dim3);
            
            // Get all 4 sublattice spins for this cell
            size_t idx0 = flatten_index(i, j, k, 0);
            size_t idx1 = flatten_index(i, j, k, 1);
            size_t idx2 = flatten_index(i, j, k, 2);
            size_t idx3 = flatten_index(i, j, k, 3);
            
            // Cache spin vectors (avoiding repeated VectorXd allocation)
            const double S0z = spins[idx0](2);
            const double S1x = spins[idx1](0), S1y = spins[idx1](1), S1z = spins[idx1](2);
            const double S2x = spins[idx2](0), S2y = spins[idx2](1), S2z = spins[idx2](2);
            const double S3x = spins[idx3](0), S3y = spins[idx3](1), S3z = spins[idx3](2);
            
            // ===== CHIRALITY in LOCAL FRAME (kagome triangle 1,2,3) =====
            // S2 × S3
            double cross_x = S2y * S3z - S2z * S3y;
            double cross_y = S2z * S3x - S2x * S3z;
            double cross_z = S2x * S3y - S2y * S3x;
            
            // Scalar chirality: χ = S1 · (S2 × S3)
            chi_sum += S1x * cross_x + S1y * cross_y + S1z * cross_z;
            
            // Vector chirality: κ = S1 × S2 + S2 × S3 + S3 × S1
            // S1 × S2
            double s1xs2_x = S1y * S2z - S1z * S2y;
            double s1xs2_y = S1z * S2x - S1x * S2z;
            double s1xs2_z = S1x * S2y - S1y * S2x;
            
            // S3 × S1
            double s3xs1_x = S3y * S1z - S3z * S1y;
            double s3xs1_y = S3z * S1x - S3x * S1z;
            double s3xs1_z = S3x * S1y - S3y * S1x;
            
            kappa_sum(0) += s1xs2_x + cross_x + s3xs1_x;  // Note: S2×S3 = cross
            kappa_sum(1) += s1xs2_y + cross_y + s3xs1_y;
            kappa_sum(2) += s1xs2_z + cross_z + s3xs1_z;

            // ===== CHIRALITY in GLOBAL FRAME =====
            // Transform sublattice spins 1,2,3 to global frame
            double G1x, G1y, G1z, G2x, G2y, G2z, G3x, G3y, G3z;
            PyrochloreLocalFrame::to_global(1, S1x, S1y, S1z, G1x, G1y, G1z);
            PyrochloreLocalFrame::to_global(2, S2x, S2y, S2z, G2x, G2y, G2z);
            PyrochloreLocalFrame::to_global(3, S3x, S3y, S3z, G3x, G3y, G3z);

            // G2 × G3
            double gcross_x = G2y * G3z - G2z * G3y;
            double gcross_y = G2z * G3x - G2x * G3z;
            double gcross_z = G2x * G3y - G2y * G3x;

            // Scalar chirality: χ_g = G1 · (G2 × G3)
            chi_sum_g += G1x * gcross_x + G1y * gcross_y + G1z * gcross_z;

            // Vector chirality: κ_g = G1×G2 + G2×G3 + G3×G1
            double g1xg2_x = G1y * G2z - G1z * G2y;
            double g1xg2_y = G1z * G2x - G1x * G2z;
            double g1xg2_z = G1x * G2y - G1y * G2x;

            double g3xg1_x = G3y * G1z - G3z * G1y;
            double g3xg1_y = G3z * G1x - G3x * G1z;
            double g3xg1_z = G3x * G1y - G3y * G1x;

            kappa_sum_g(0) += g1xg2_x + gcross_x + g3xg1_x;
            kappa_sum_g(1) += g1xg2_y + gcross_y + g3xg1_y;
            kappa_sum_g(2) += g1xg2_z + gcross_z + g3xg1_z;
            
            // ===== MONOPOLE (tetrahedron 0,1,2,3) =====
            // Q = Σ_μ S^z_μ (signed monopole charge)
            double Q_tet = S0z + S1z + S2z + S3z;
            Q_sum += Q_tet;
            
            // Monopole by sublattice (3-1 split only)
            std::array<double, 4> Sz = {S0z, S1z, S2z, S3z};
            int n_pos = 0, n_neg = 0;
            for (size_t mu = 0; mu < 4; ++mu) {
                if (Sz[mu] > 0) n_pos++;
                else n_neg++;
            }
            
            if (n_pos == 3 && n_neg == 1) {
                // Find the minority (negative) sublattice
                for (size_t mu = 0; mu < 4; ++mu) {
                    if (Sz[mu] <= 0) {
                        monopole_sub(mu) += 1.0;
                        break;
                    }
                }
            } else if (n_pos == 1 && n_neg == 3) {
                // Find the minority (positive) sublattice
                for (size_t mu = 0; mu < 4; ++mu) {
                    if (Sz[mu] > 0) {
                        monopole_sub(mu) += 1.0;
                        break;
                    }
                }
            }
        }
        
        // Normalize
        result.scalar_chirality = chi_sum * inv_n_cells;
        result.vector_chirality = kappa_sum * inv_n_cells;
        result.monopole_density = Q_sum * inv_n_cells;
        result.monopole_by_sublattice = monopole_sub * inv_n_cells;

        result.scalar_chirality_global = chi_sum_g * inv_n_cells;
        result.vector_chirality_global = kappa_sum_g * inv_n_cells;
        
        // Nematic order requires bond enumeration - compute separately
        result.nematic_order = compute_kagome_nematic_order();
        result.nematic_order_global = compute_kagome_nematic_order_global();
        
        return result;
    }

    /**
     * Compute scalar chirality on kagome triangles
     * χ = S1 · (S2 × S3) per intra-cell triangle
     * 
     * Triangle vertices: 1(i,j,k), 2(i,j,k), 3(i,j,k)
     * Edges: (1-2) Jy, (2-3) Jz, (3-1) Jx — all intra-cell
     * 
     * @return Average scalar chirality per triangle (1 triangle per unit cell)
     */
    double compute_kagome_scalar_chirality() const {
        return compute_pyrochlore_order_parameters_fast().scalar_chirality;
    }
    
    /**
     * Compute vector chirality on kagome triangles
     * κ = S1 × S2 + S2 × S3 + S3 × S1 per intra-cell triangle
     * 
     * Triangle vertices: 1(i,j,k), 2(i,j,k), 3(i,j,k)
     * 
     * @return Average vector chirality (3-component) per triangle
     */
    Eigen::Vector3d compute_kagome_vector_chirality() const {
        return compute_pyrochlore_order_parameters_fast().vector_chirality;
    }
    
    /**
     * Compute component-resolved nematic bond order on kagome NN bonds
     * 
     * Uses bilinear_partners to enumerate all NN bonds automatically.
     * Bond types for kagome sublattices (1,2,3):
     *   Type 0: (1-2) bonds — Jy type
     *   Type 1: (2-3) bonds — Jz type  
     *   Type 2: (1-3) bonds — Jx type
     * 
     * Returns 3×3 matrix: Q[bond_type][local_component]
     * - Rows: bond types (0, 1, 2)
     * - Cols: local spin components (x=0, y=1, z=2)
     * 
     * Q^α_μ = <S_i^α S_j^α> averaged over all bonds of type μ
     */
    Eigen::Matrix3d compute_kagome_nematic_order() const {
        // Only valid for pyrochlore lattices
        if (!is_pyrochlore()) {
            std::cerr << "Warning: compute_kagome_nematic_order() is only valid for pyrochlore lattices" << std::endl;
            return Eigen::Matrix3d::Zero();
        }
        if (N_atoms < 4 || spin_dim < 3) {
            return Eigen::Matrix3d::Zero();
        }
        
        Eigen::Matrix3d Q_sum = Eigen::Matrix3d::Zero();
        Eigen::Vector3i bond_counts = Eigen::Vector3i::Zero();  // Count bonds per type
        
        // Loop over all kagome sites (sublattices 1, 2, 3)
        for (size_t site_i = 0; site_i < lattice_size; ++site_i) {
            size_t sub_i = site_i % N_atoms;
            if (sub_i == 0) continue;  // Skip apex (sublattice 0)
            
            // Loop over NN partners from bilinear_partners
            for (size_t partner_idx = 0; partner_idx < bilinear_partners[site_i].size(); ++partner_idx) {
                size_t site_j = bilinear_partners[site_i][partner_idx];
                size_t sub_j = site_j % N_atoms;
                
                if (sub_j == 0) continue;  // Skip apex bonds
                if (site_j <= site_i) continue;  // Avoid double counting (only count i < j)
                
                // Determine bond type from sublattice pair
                int bond_type = -1;
                if ((sub_i == 1 && sub_j == 2) || (sub_i == 2 && sub_j == 1)) {
                    bond_type = 0;  // (1-2) bond
                } else if ((sub_i == 2 && sub_j == 3) || (sub_i == 3 && sub_j == 2)) {
                    bond_type = 1;  // (2-3) bond
                } else if ((sub_i == 1 && sub_j == 3) || (sub_i == 3 && sub_j == 1)) {
                    bond_type = 2;  // (1-3) bond
                }
                
                if (bond_type >= 0) {
                    // Accumulate S_i^α * S_j^α for each component
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        Q_sum(bond_type, alpha) += spins[site_i](alpha) * spins[site_j](alpha);
                    }
                    bond_counts(bond_type)++;
                }
            }
        }
        
        // Normalize by number of bonds per type
        for (int bond_type = 0; bond_type < 3; ++bond_type) {
            if (bond_counts(bond_type) > 0) {
                Q_sum.row(bond_type) /= bond_counts(bond_type);
            }
        }
        
        return Q_sum;
    }

    /**
     * Compute component-resolved nematic bond order on kagome NN bonds
     * in the GLOBAL Cartesian frame.
     * 
     * Each local-frame spin is transformed to the global frame using the
     * pyrochlore local frame (x_hat, y_hat, z_hat per sublattice) before
     * computing the product S_i^α_global * S_j^α_global.
     * 
     * Returns 3×3 matrix: Q_global[bond_type][global_component]
     * - Rows: bond types (0=1-2, 1=2-3, 2=1-3)
     * - Cols: global Cartesian components (X=0, Y=1, Z=2)
     */
    Eigen::Matrix3d compute_kagome_nematic_order_global() const {
        if (!is_pyrochlore()) {
            std::cerr << "Warning: compute_kagome_nematic_order_global() is only valid for pyrochlore lattices" << std::endl;
            return Eigen::Matrix3d::Zero();
        }
        if (N_atoms < 4 || spin_dim < 3) {
            return Eigen::Matrix3d::Zero();
        }
        
        Eigen::Matrix3d Q_sum = Eigen::Matrix3d::Zero();
        Eigen::Vector3i bond_counts = Eigen::Vector3i::Zero();
        
        for (size_t site_i = 0; site_i < lattice_size; ++site_i) {
            size_t sub_i = site_i % N_atoms;
            if (sub_i == 0) continue;
            
            for (size_t partner_idx = 0; partner_idx < bilinear_partners[site_i].size(); ++partner_idx) {
                size_t site_j = bilinear_partners[site_i][partner_idx];
                size_t sub_j = site_j % N_atoms;
                
                if (sub_j == 0) continue;
                if (site_j <= site_i) continue;
                
                int bond_type = -1;
                if ((sub_i == 1 && sub_j == 2) || (sub_i == 2 && sub_j == 1)) {
                    bond_type = 0;
                } else if ((sub_i == 2 && sub_j == 3) || (sub_i == 3 && sub_j == 2)) {
                    bond_type = 1;
                } else if ((sub_i == 1 && sub_j == 3) || (sub_i == 3 && sub_j == 1)) {
                    bond_type = 2;
                }
                
                if (bond_type >= 0) {
                    // Transform both spins to global frame
                    double Gi_x, Gi_y, Gi_z, Gj_x, Gj_y, Gj_z;
                    PyrochloreLocalFrame::to_global(static_cast<int>(sub_i),
                        spins[site_i](0), spins[site_i](1), spins[site_i](2),
                        Gi_x, Gi_y, Gi_z);
                    PyrochloreLocalFrame::to_global(static_cast<int>(sub_j),
                        spins[site_j](0), spins[site_j](1), spins[site_j](2),
                        Gj_x, Gj_y, Gj_z);
                    
                    // Accumulate Si_global^α * Sj_global^α
                    Q_sum(bond_type, 0) += Gi_x * Gj_x;
                    Q_sum(bond_type, 1) += Gi_y * Gj_y;
                    Q_sum(bond_type, 2) += Gi_z * Gj_z;
                    bond_counts(bond_type)++;
                }
            }
        }
        
        for (int bond_type = 0; bond_type < 3; ++bond_type) {
            if (bond_counts(bond_type) > 0) {
                Q_sum.row(bond_type) /= bond_counts(bond_type);
            }
        }
        
        return Q_sum;
    }
    
    /**
     * Compute monopole density decomposed by sublattice type
     * 
     * For a 3-in-1-out monopole, the "type" is determined by which sublattice μ
     * has the minority spin (the 1-out). Similarly for 1-in-3-out.
     * 
     * Returns a 4-component vector:
     *   density[μ] = fraction of tetrahedra where sublattice μ is the minority
     * 
     * For ice-rule states (2-in-2-out) or double monopoles (4-in or 4-out),
     * no sublattice is counted as minority.
     * 
     * @return Eigen::Vector4d with monopole density per sublattice type
     */
    Eigen::Vector4d compute_monopole_density_by_sublattice() const {
        return compute_pyrochlore_order_parameters_fast().monopole_by_sublattice;
    }
    
    /**
     * Compute all kagome order parameters at once
     * Uses the fast single-pass implementation internally.
     * 
     * @return Tuple of (scalar_chirality, vector_chirality, nematic_order_matrix)
     */
    std::tuple<double, Eigen::Vector3d, Eigen::Matrix3d> compute_kagome_order_parameters() const {
        auto params = compute_pyrochlore_order_parameters_fast();
        return {params.scalar_chirality, params.vector_chirality, params.nematic_order};
    }
    
    /**
     * Compute all monopole diagnostics at once
     * Uses the fast single-pass implementation internally.
     * 
     * @return Tuple of (monopole_density (signed), monopole_by_sublattice)
     */
    std::tuple<double, Eigen::Vector4d> compute_monopole_diagnostics() const {
        auto params = compute_pyrochlore_order_parameters_fast();
        return {params.monopole_density, params.monopole_by_sublattice};
    }

    // ============================================================
    // COMPREHENSIVE OBSERVABLE COLLECTION
    // ============================================================

    /**
     * Collect a single measurement of all thermodynamic observables
     * Returns: (energy, sublattice_magnetizations)
     */
    std::pair<double, vector<SpinVector>> measure_observables() const {
        double E = total_energy(spins);
        vector<SpinVector> M_sub = magnetization_sublattice();
        return {E, M_sub};
    }

    /**
     * Compute comprehensive thermodynamic observables with binning error analysis
     * 
     * @param energies Vector of energy measurements
     * @param sublattice_mags Vector of sublattice magnetization measurements
     *                        Each element is a vector of N_atoms SpinVectors
     * @param T Temperature
     * @return ThermodynamicObservables struct with all observables and uncertainties
     */
    ThermodynamicObservables compute_thermodynamic_observables(
        const vector<double>& energies,
        const vector<vector<SpinVector>>& sublattice_mags,
        double T) const {
        return mc::compute_thermodynamic_observables<SpinVector>(energies, sublattice_mags, T,
                                                                 lattice_size);
    }

    /**
     * Save comprehensive thermodynamic observables to files
     */
    void save_thermodynamic_observables(const string& out_dir,
                                         const ThermodynamicObservables& obs) const;

    /**
     * Print thermodynamic observables summary to stdout
     */
    void print_thermodynamic_observables(const ThermodynamicObservables& obs) const {
        cout << "\n=== Thermodynamic Observables at T = " << obs.temperature << " ===" << endl;
        cout << std::scientific << std::setprecision(6);
        
        cout << "<E>/N = " << obs.energy.value << " ± " << obs.energy.error << endl;
        cout << "C_V   = " << obs.specific_heat.value << " ± " << obs.specific_heat.error << endl;
        
        cout << "\nTotal magnetization <M>/N:" << endl;
        for (size_t d = 0; d < obs.magnetization.values.size(); ++d) {
            cout << "  M_" << d << " = " << obs.magnetization.values[d] 
                 << " ± " << obs.magnetization.errors[d] << endl;
        }
        
        cout << "\nSublattice magnetizations:" << endl;
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& M = obs.sublattice_magnetization[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << M.values[d] << "±" << M.errors[d];
            }
            cout << ")" << endl;
        }
        
        cout << "\nEnergy-sublattice cross correlations:" << endl;
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& cross = obs.energy_sublattice_cross[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << cross.values[d] << "±" << cross.errors[d];
            }
            cout << ")" << endl;
        }
    }

    /**
     * Save thermodynamic observables to HDF5 format
     * Single file per rank with all data organized in groups
     */
    void save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const ThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<SpinVector>& magnetizations,
                                              const vector<vector<SpinVector>>& sublattice_mags,
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

    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================

    /**
     * Metropolis sweep with local spin updates. Returns acceptance rate.
     *
     * Implementation notes (post-audit refactor):
     *  - **Sequential** site order, not random-with-replacement.  Sequential
     *    sweeps are cache-friendly (spins, partners and J-matrices stream in
     *    order), still ergodic, and standard practice in modern MC packages
     *    (ALPS/looper, ALF, SpinW). This changes the exact MC trajectory
     *    but not equilibrium averages.
     *  - **Zero heap allocations** per proposed move: the new spin is built
     *    in a stack buffer, energy diff is computed via `site_energy_diff_flat`
     *    which uses raw `double*` access into `spins[site].data()`, and on
     *    accept we `memcpy` into the existing Eigen storage.
     *  - **Logical OR** acceptance (was bitwise `|`), with `dE <= 0` short
     *    circuit to skip exp() entirely on downhill moves.
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;

        const double beta = 1.0 / T;
        size_t accepted = 0;

        constexpr size_t MAX_SPIN_DIM = 16;
        alignas(32) double new_spin_buf[MAX_SPIN_DIM];

        // Batch random uniforms: amortises the lehman_next() function-call
        // overhead and keeps the inner loop tight.
        constexpr size_t BATCH_SIZE = 64;
        double rand_uniforms[BATCH_SIZE];

        for (size_t batch_start = 0; batch_start < lattice_size; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size);
            const size_t n_in_batch = batch_end - batch_start;

            for (size_t j = 0; j < n_in_batch; ++j) {
                rand_uniforms[j] = random_double_lehman(0.0, 1.0);
            }

            for (size_t j = 0; j < n_in_batch; ++j) {
                const size_t site      = batch_start + j;
                const double rand_uni  = rand_uniforms[j];
                double*      old_spin  = spins[site].data();

                // Build the proposed spin in `new_spin_buf` with no allocation.
                if (!propose_spin(old_spin, gaussian_move, sigma, new_spin_buf)) continue;

                const double dE = site_energy_diff_flat(new_spin_buf, old_spin, site);

                // Logical OR + downhill short-circuit (was bitwise `|` which
                // forced unnecessary exp() evaluation on downhill moves).
                const bool accept = (dE <= 0.0) ||
                                    (rand_uni < std::exp(-beta * dE));
                if (accept) {
                    std::memcpy(old_spin, new_spin_buf, spin_dim * sizeof(double));
                    ++accepted;
                }
            }
        }

        return double(accepted) / double(lattice_size);
    }

    /**
     * Coloured Metropolis sweep — parallel over sites within each colour.
     *
     * Iterates over the precomputed sublattice colour partition built by
     * `build_color_partition()`: for each colour in turn, all sites of that
     * colour are updated in parallel by independent OpenMP threads. Because
     * no two sites of the same colour share any (bilinear or trilinear)
     * interaction, the per-site Metropolis kernel reads only neighbour
     * spins (untouched until the next colour pass) and writes only its
     * own spins[i] slot — fully race-free.
     *
     * RNG: each OpenMP thread has its own `thread_local` Lehman state,
     * lazy-seeded from (master, tid) on first call (see `lazy_seed_thread`
     * in `simple_linear_alg.cpp`). This means the per-site RNG sequence
     * differs from the serial `metropolis()` even at 1 thread, but the
     * Markov chain is correct and each thread's stream is statistically
     * independent.
     *
     * Falls back to the serial `metropolis()` if the colour partition is
     * empty (e.g. an old `Lattice` built before colouring landed) or if
     * only one OpenMP thread is available.
     *
     * Returns the global acceptance ratio (#accepted / lattice_size).
     */
    double metropolis_parallel(double T, bool gaussian_move = false,
                               double sigma = 60.0) {
        if (n_colors == 0) return metropolis(T, gaussian_move, sigma);
#ifdef _OPENMP
        if (omp_get_max_threads() <= 1) return metropolis(T, gaussian_move, sigma);
#else
        return metropolis(T, gaussian_move, sigma);
#endif
        if (T <= 0) return 0.0;

        const double beta = 1.0 / T;

#ifdef _OPENMP
        const int n_threads = omp_get_max_threads();
#else
        const int n_threads = 1;
#endif
        // 64-byte cache-line padded counters: prevents false sharing of the
        // per-thread accept counts in the final reduction.
        struct alignas(64) PaddedAccept { size_t v = 0; char pad[64 - sizeof(size_t)]; };
        std::vector<PaddedAccept> per_thread_accepted(n_threads);

        constexpr size_t MAX_SPIN_DIM = 16;
        constexpr size_t RNG_BATCH    = 64;

        // PERSISTENT OpenMP region across all colours: a single fork/join
        // for the entire sweep with one #pragma omp barrier between colour
        // passes. Replaces the previous "one parallel region per colour"
        // pattern that paid n_colors × team-create/teardown cost per sweep.
        // For small lattices (e.g. pyrochlore L=8 with 4 colours) the
        // per-region overhead was ~25–30 % of the sweep wall time at
        // 16 threads; persistent region collapses that to ~3 %.
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            alignas(32) double new_spin_buf[MAX_SPIN_DIM];
            alignas(64) double rng_uniforms[RNG_BATCH];
            size_t rng_pos = RNG_BATCH;  // force refill on first use
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            size_t local_accepted = 0;

            for (size_t c = 0; c < n_colors; ++c) {
                const size_t off_lo = sites_by_color_csr_off[c];
                const size_t off_hi = sites_by_color_csr_off[c + 1];

#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t site     = sites_by_color_csr[off];
                    double*      old_spin = spins[site].data();

                    if (!propose_spin(old_spin, gaussian_move, sigma, new_spin_buf)) continue;

                    const double dE = site_energy_diff_flat(new_spin_buf, old_spin, site);

                    // Pull acceptance uniform from the per-thread batch
                    // buffer; refill in 64-element chunks to amortise
                    // function-call overhead of random_double_lehman.
                    if (rng_pos >= RNG_BATCH) {
                        for (size_t r = 0; r < RNG_BATCH; ++r)
                            rng_uniforms[r] = random_double_lehman(0.0, 1.0);
                        rng_pos = 0;
                    }
                    const double rand_uni = rng_uniforms[rng_pos++];

                    const bool accept = (dE <= 0.0) ||
                                        (rand_uni < std::exp(-beta * dE));
                    if (accept) {
                        std::memcpy(old_spin, new_spin_buf, spin_dim * sizeof(double));
                        ++local_accepted;
                    }
                }
                // Synchronise threads before moving to the next colour:
                // sites of colour c+1 may read partner spins that were
                // just written by some other thread in colour c.
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            per_thread_accepted[tid].v += local_accepted;
        } // end omp parallel

        size_t accepted = 0;
        for (auto& a : per_thread_accepted) accepted += a.v;
        return double(accepted) / double(lattice_size);
    }

    // ------------------------------------------------------------------
    // Heat-bath update (Miyatake et al., J. Phys. C 19, 2539 (1986)).
    //
    // For a local energy g·S_i (all terms linear in S_i), the conditional
    // distribution is P(S_i) ∝ exp(-β g·S_i): with u = cos∠(S_i, -ĝ) and
    // b = β|g|s it is P(u) ∝ exp(b u) on [-1, 1], sampled exactly by
    //     u = 1 + log1p(ξ expm1(-2b)) / b,   φ uniform.
    // Rejection-free, and far more efficient than uniform Metropolis at low
    // T. With an anisotropic on-site term the heat-bath draw of the linear
    // part is an independence proposal; Metropolis-Hastings then accepts it
    // with min(1, exp(-β ΔE_onsite)), which is exact. Only SO(3) spins;
    // other spin dimensions fall back to Metropolis.
    // ------------------------------------------------------------------
    inline bool heat_bath_site(size_t site, double beta) {
        double g[MAX_SPIN_DIM];
        linear_field(site, spins_view(), g);
        if (su3_cp2) {
            // Exact draw from exp(-beta g.n) on CP^2 (classical_spin::su3::sample_linear_cp2).
            double n_new[8];
            classical_spin::su3::sample_linear_cp2(g, beta, n_new);
            double* n = spins[site].data();
            if (!onsite_scalar[site]) {
                const double dE = onsite_energy(site, n_new) - onsite_energy(site, n);
                if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) return false;
            }
            std::memcpy(n, n_new, sizeof(n_new));
            return true;
        }
        const double s = double(spin_length);
        const double gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        double S_new[3];
        if (gn * beta * s < 1e-12) {
            random_point_on_sphere(S_new, 3, s);
        } else {
            const double b = beta * gn * s;
            const double xi = random_double_lehman(0.0, 1.0);
            double u = 1.0 + std::log1p(xi * std::expm1(-2.0 * b)) / b;
            u = std::clamp(u, -1.0, 1.0);
            const double phi = random_double_lehman(0.0, 2.0 * M_PI);
            const double n[3] = {-g[0] / gn, -g[1] / gn, -g[2] / gn};
            // Orthonormal basis (e1, e2) of the plane normal to n.
            double e1[3];
            if (std::abs(n[0]) < 0.9) { e1[0] = 0.0; e1[1] = -n[2]; e1[2] = n[1]; }
            else                      { e1[0] = n[2]; e1[1] = 0.0; e1[2] = -n[0]; }
            const double e1n = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
            for (double& c : e1) c /= e1n;
            const double e2[3] = {n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2],
                                  n[0] * e1[1] - n[1] * e1[0]};
            const double r = std::sqrt(std::max(0.0, 1.0 - u * u));
            const double c = std::cos(phi), sn = std::sin(phi);
            for (int d = 0; d < 3; ++d) S_new[d] = s * (u * n[d] + r * (c * e1[d] + sn * e2[d]));
        }
        double* S = spins[site].data();
        if (!onsite_scalar[site]) {
            const double dE = onsite_energy(site, S_new) - onsite_energy(site, S);
            if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) return false;
        }
        S[0] = S_new[0]; S[1] = S_new[1]; S[2] = S_new[2];
        return true;
    }

    /// Heat-bath sweep in natural site order; returns the acceptance ratio
    /// (1 unless anisotropic on-site terms reject some draws).
    double heat_bath(double T) {
        if (T <= 0.0) return 0.0;
        if (!heat_bath_available()) return metropolis(T);
        const double beta = 1.0 / T;
        size_t accepted = 0;
        for (size_t site = 0; site < lattice_size; ++site) accepted += heat_bath_site(site, beta);
        return double(accepted) / double(lattice_size);
    }

    /// Coloured, race-free OpenMP heat-bath sweep (see metropolis_parallel).
    double heat_bath_parallel(double T) {
        if (T <= 0.0) return 0.0;
        if (!heat_bath_available()) return metropolis_parallel(T);
#ifdef _OPENMP
        if (n_colors == 0 || omp_get_max_threads() <= 1) return heat_bath(T);
#else
        return heat_bath(T);
#endif
        const double beta = 1.0 / T;
        size_t accepted = 0;
#ifdef _OPENMP
        #pragma omp parallel reduction(+:accepted)
#endif
        {
            for (size_t c = 0; c < n_colors; ++c) {
                const size_t off_lo = sites_by_color_csr_off[c];
                const size_t off_hi = sites_by_color_csr_off[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off)
                    accepted += heat_bath_site(sites_by_color_csr[off], beta);
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }
        }
        return double(accepted) / double(lattice_size);
    }

    // ------------------------------------------------------------------
    // Local-update policy used by the SA / PT / measurement drivers.
    // ------------------------------------------------------------------
    enum class LocalUpdate { Metropolis, Gaussian, HeatBath };

    // Kernel selection for local_sweep(): heat_bath / metropolis with uniform
    // or Gaussian (σ) proposals; the coloured OpenMP kernels are used when
    // more than one thread is available and the lattice has at least
    // parallel_sweep_min_sites sites (reproducible for a fixed thread count).
    LocalUpdate local_update = LocalUpdate::Metropolis;
    size_t parallel_sweep_min_sites = 4096;

    // State space of 8-component (SU(3), qutrit) spins in Monte Carlo. CP^2:
    // pure states n = <psi|lambda|psi> (|n|^2 = 4/3) with the Fubini-Study
    // measure — the default when the unit cell uses the Gell-Mann bracket
    // (UnitCell::poisson_bracket = 2, the E = <psi|H|psi> convention whose
    // dynamics stay on CP^2). Otherwise (legacy bracket, other spin_dim) the
    // sphere |S| = spin_length. Set with set_su3_mc_manifold.
    bool su3_cp2 = false;

    /// "cp2", "sphere", or "" / "auto" (CP^2 iff spin_dim == 8 with the Gell-Mann bracket).
    /// Switching to CP^2 projects the current spins onto it.
    void set_su3_mc_manifold(const string& name) {
        if (name.empty() || name == "auto") {
            su3_cp2 = (spin_dim == 8 && unit_cell.poisson_bracket == 2.0);
        } else if (name == "cp2" || name == "CP2") {
            if (spin_dim != 8) throw std::invalid_argument("su3_mc_manifold = cp2 needs spin_dim 8");
            su3_cp2 = true;
        } else if (name == "sphere") {
            su3_cp2 = false;
        } else {
            throw std::invalid_argument("unknown su3_mc_manifold '" + name + "' (valid: cp2, sphere)");
        }
        project_su3_states();
    }

    /// On CP^2, replace every spin by its closest pure qutrit state (exact for
    /// any positive multiple of a pure state); no-op otherwise.
    void project_su3_states() {
        if (!su3_cp2) return;
        for (auto& s : spins) classical_spin::su3::project_to_cp2(s.data());
    }

    /// Whether heat_bath() is exact for these spins (SO(3), or SU(3) on CP^2).
    bool heat_bath_available() const { return spin_dim == 3 || su3_cp2; }

    static LocalUpdate parse_local_update(const string& name) {
        if (name == "metropolis" || name == "uniform") return LocalUpdate::Metropolis;
        if (name == "gaussian" || name == "adaptive") return LocalUpdate::Gaussian;
        if (name == "heat_bath" || name == "heatbath") return LocalUpdate::HeatBath;
        throw std::invalid_argument("unknown local update '" + name +
                                    "' (valid: metropolis, gaussian, heat_bath)");
    }

    bool use_parallel_sweeps() const {
#ifdef _OPENMP
        return n_colors > 0 && lattice_size >= parallel_sweep_min_sites && omp_get_max_threads() > 1;
#else
        return false;
#endif
    }

    /**
     * One local-update sweep at temperature T with the configured policy.
     * `gaussian_move` (legacy flag) selects Gaussian proposals of width σ
     * when the policy is Metropolis. Returns the acceptance ratio.
     */
    double local_sweep(double T, bool gaussian_move, double sigma) {
        const bool par = use_parallel_sweeps();
        if (local_update == LocalUpdate::HeatBath && heat_bath_available())
            return par ? heat_bath_parallel(T) : heat_bath(T);
        const bool gauss = gaussian_move || local_update == LocalUpdate::Gaussian;
        return par ? metropolis_parallel(T, gauss, sigma) : metropolis(T, gauss, sigma);
    }

    /// Overrelaxation sweep through the same serial/parallel selection.
    void overrelaxation_sweep(double T) {
        if (use_parallel_sweeps()) overrelaxation_parallel(T);
        else overrelaxation(T);
    }

    /**
     * Gaussian move around current spin
     *
     * Backwards-compatible API. Internally allocates a temporary, so prefer
     * `metropolis()` (which has the inlined zero-allocation path) for hot
     * loops. Kept for callers in tests / external utilities.
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma) {
        SpinVector new_spin = current_spin + gen_random_spin(spin_length) * sigma;
        double norm = new_spin.norm();
        if (norm < 1e-10) return current_spin;
        return new_spin * (spin_length / norm);
    }

    /**
     * Overrelaxation move at one site: reflect S_i about its linear field g
     * (Zeeman + exchange + trilinear), S' = 2 (S·g) g / |g|² - S.
     *
     * The reflection conserves every term linear in S_i exactly, so it is a
     * valid microcanonical move whenever the on-site term is constant on the
     * sphere. An anisotropic on-site term S^T A S is NOT conserved; for such
     * sites the reflection (an involution, hence a symmetric proposal) is
     * accepted with min(1, exp(-ΔE_onsite / T)) when T > 0 and skipped when
     * T <= 0, which keeps detailed balance in both cases. (The previous
     * version reflected about g + 2 A S_i, which conserves neither and
     * biased every model with single-ion anisotropy.)
     *
     * Returns true if the spin changed.
     */
    inline bool overrelax_site(size_t site, double T) {
        double g[MAX_SPIN_DIM];
        linear_field(site, spins_view(), g);
        double norm_sq = 0.0, S_dot_g = 0.0;
        double* S = spins[site].data();
        if (su3_cp2) {
            // The R^8 reflection leaves CP^2: randomise the relative phases in
            // the eigenbasis of g.lambda instead (conserves g.n; symmetric,
            // see classical_spin::su3::randomize_phases_cp2).
            double n_new[8];
            classical_spin::su3::randomize_phases_cp2(g, S, n_new);
            if (!onsite_scalar[site]) {
                if (T <= 0.0) return false;
                const double dE = onsite_energy(site, n_new) - onsite_energy(site, S);
                if (dE > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-dE / T)) return false;
            }
            std::memcpy(S, n_new, sizeof(n_new));
            return true;
        }
        for (size_t d = 0; d < spin_dim; ++d) {
            norm_sq += g[d] * g[d];
            S_dot_g += S[d] * g[d];
        }
        if (norm_sq <= 0.0) return false;
        const double k = 2.0 * S_dot_g / norm_sq;
        if (onsite_scalar[site]) {
            for (size_t d = 0; d < spin_dim; ++d) S[d] = k * g[d] - S[d];
            return true;
        }
        if (T <= 0.0) return false;
        double S_new[MAX_SPIN_DIM];
        for (size_t d = 0; d < spin_dim; ++d) S_new[d] = k * g[d] - S[d];
        const double dE = onsite_energy(site, S_new) - onsite_energy(site, S);
        if (dE <= 0.0 || random_double_lehman(0.0, 1.0) < std::exp(-dE / T)) {
            std::memcpy(S, S_new, spin_dim * sizeof(double));
            return true;
        }
        return false;
    }

    /**
     * Overrelaxation sweep over all sites in natural order.
     *
     * Microcanonical (zero rejection) for models without single-ion
     * anisotropy. With anisotropic on-site terms pass the simulation
     * temperature so that those sites receive a Metropolis-corrected
     * reflection; with the default T = 0 they are left untouched.
     */
    void overrelaxation(double T = 0.0) {
        for (size_t site = 0; site < lattice_size; ++site) overrelax_site(site, T);
    }

    /**
     * Coloured over-relaxation sweep — parallel over sites within each colour.
     *
     * Same race-free guarantee as `metropolis_parallel`: the per-site reflect
     * reads only `spins[partner]` for partners of the current site, and writes
     * only its own `spins[site]`. Within a colour all reads point to a
     * different colour, so OpenMP threads can update them concurrently with
     * no data race.
     *
     * Falls back to the serial `overrelaxation()` if no colour partition
     * is built or only one thread is available.
     */
    void overrelaxation_parallel(double T = 0.0) {
        if (n_colors == 0) { overrelaxation(T); return; }
#ifdef _OPENMP
        if (omp_get_max_threads() <= 1) { overrelaxation(T); return; }
#else
        overrelaxation(T); return;
#endif

        // PERSISTENT OpenMP region: one fork/join for the whole sweep,
        // with #pragma omp barrier between colour passes. See the
        // comment on metropolis_parallel above for the rationale.
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            for (size_t c = 0; c < n_colors; ++c) {
                const size_t off_lo = sites_by_color_csr_off[c];
                const size_t off_hi = sites_by_color_csr_off[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off)
                    overrelax_site(sites_by_color_csr[off], T);
                // Synchronise threads before moving to the next colour.
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }
        }
    }

    // ------------------------------------------------------------------
    // Cluster moves for continuous spins (Wolff 1989, Swendsen-Wang 1987).
    //
    // A random unit vector r defines the reflection R = 1 - 2 r r^T.
    // Writing S_i = S_i^⊥ + s_i r with s_i = S_i·r, a bond energy
    // S_i^T J S_j contains the embedded Ising term x_ij = K_ij s_i s_j with
    // K_ij = r^T J r. Remember E = +S^T J S here (J > 0 antiferromagnetic),
    // so a bond is *satisfied* when x_ij < 0, and the Fortuin-Kasteleyn
    // construction activates it with probability
    //     p_ij = 1 - exp(-2 β |x_ij|)   if x_ij < 0,   0 otherwise.
    // (The previous implementation activated *unsatisfied* bonds, i.e. had
    // the sign of the coupling reversed — no clusters at all for
    // ferromagnets, wrong ones for antiferromagnets — and used exp(-β|x|)
    // for the ghost bond instead of exp(-2β|x|).)
    //
    // Everything that is not of the embedded form is exact-corrected: the
    // residual ΔE_res = ΔE_true - ΔE_embedded of a cluster flip (anisotropic
    // or DM parts of J, anisotropic on-site terms, trilinear couplings,
    // twisted bonds, and the Zeeman term when no ghost spin is used) enters
    // a Metropolis filter min(1, exp(-β ΔE_res)). Flipping a whole FK
    // cluster leaves the joint bond-spin weight invariant, so the filtered
    // move satisfies detailed balance for any Hamiltonian; for isotropic
    // Heisenberg exchange ΔE_res ≡ 0 and the filter is skipped entirely.
    // ------------------------------------------------------------------

    /// K = r^T J r for bond slot k, or 0 for bonds that cannot be embedded
    /// exactly (twisted boundary bonds when a twist is active).
    inline double embedded_coupling(size_t k, const double* r) const {
        const double* J = &bi_flat_J[k * bi_flat_D2];
        double K = 0.0;
        for (size_t a = 0; a < spin_dim; ++a) {
            double row = 0.0;
            for (size_t b = 0; b < spin_dim; ++b) row += J[a * spin_dim + b] * r[b];
            K += r[a] * row;
        }
        return K;
    }

    /// True if every coupling is invariant under any reflection: scalar
    /// (isotropic, DM-free) exchange, scalar on-site terms, no trilinear
    /// couplings and no active twist. The Zeeman term is checked separately.
    bool cluster_embedding_is_exact() const {
        if (cluster_exact_cache < 0) cluster_exact_cache = compute_cluster_embedding_is_exact() ? 1 : 0;
        return cluster_exact_cache == 1;
    }

    bool compute_cluster_embedding_is_exact() const {
        if (twist_active) return false;
        for (size_t i = 0; i < lattice_size; ++i) {
            if (!trilinear_partners[i].empty() || !onsite_scalar[i]) return false;
        }
        const size_t D = spin_dim;
        for (size_t k = 0; k < bi_flat_partner.size(); ++k) {
            const double* J = &bi_flat_J[k * bi_flat_D2];
            const double c = J[0];
            const double tol = 1e-13 * std::max(1.0, std::abs(c));
            for (size_t a = 0; a < D; ++a)
                for (size_t b = 0; b < D; ++b)
                    if (std::abs(J[a * D + b] - (a == b ? c : 0.0)) > tol) return false;
        }
        return true;
    }

    bool has_nonzero_field() const {
        for (const auto& B : field)
            if (B.squaredNorm() > 0.0) return true;
        return false;
    }

    /**
     * Residual energy ΔE_true - ΔE_embedded of reflecting every site of the
     * cluster `members` (flagged in `in_cluster`) about r. `proj` holds the
     * pre-flip projections s_i = S_i·r. Spins are NOT modified.
     */
    double cluster_residual_energy(const vector<size_t>& members, const uint8_t* in_cluster,
                                   const double* r, const double* proj, bool ghost) const {
        const size_t D = spin_dim;
        auto reflected = [&](size_t j, double* out) {
            const double* S = spins[j].data();
            for (size_t d = 0; d < D; ++d) out[d] = S[d] - 2.0 * proj[j] * r[d];
        };
        double dE_true = 0.0, dE_emb = 0.0;
        double Si_new[MAX_SPIN_DIM], Pj[MAX_SPIN_DIM], Pj_new[MAX_SPIN_DIM];
        for (size_t i : members) {
            const double* Si = spins[i].data();
            reflected(i, Si_new);
            const double* B = field[i].data();
            double B_r = 0.0;
            for (size_t d = 0; d < D; ++d) B_r += B[d] * r[d];
            dE_true += 2.0 * proj[i] * B_r;                       // Zeeman
            if (ghost) dE_emb += 2.0 * proj[i] * B_r;
            dE_true += onsite_energy(i, Si_new) - onsite_energy(i, Si);

            const size_t bi_end = bi_flat_offset[i + 1];
            for (size_t k = bi_flat_offset[i]; k < bi_end; ++k) {
                const size_t j = bi_flat_partner[k];
                const double* J = &bi_flat_J[k * bi_flat_D2];
                std::memcpy(Pj, spins[j].data(), D * sizeof(double));
                if (!in_cluster[j]) {
                    // Boundary bond: (S_i' - S_i)^T J P_j = -2 s_i r^T J P_j.
                    double rJP = 0.0, sj_t = 0.0;
                    for (size_t a = 0; a < D; ++a) {
                        double row = 0.0;
                        for (size_t b = 0; b < D; ++b) row += J[a * D + b] * Pj[b];
                        rJP += r[a] * row;
                        sj_t += r[a] * Pj[a];
                    }
                    dE_true += -2.0 * proj[i] * rJP;
                    dE_emb += -2.0 * embedded_coupling(k, r) * proj[i] * sj_t;
                } else {
                    // Internal bond, visited from both ends: half each time.
                    reflected(j, Pj_new);
                    double e_new = 0.0, e_old = 0.0;
                    for (size_t a = 0; a < D; ++a) {
                        double rn = 0.0, ro = 0.0;
                        for (size_t b = 0; b < D; ++b) {
                            rn += J[a * D + b] * Pj_new[b];
                            ro += J[a * D + b] * Pj[b];
                        }
                        e_new += Si_new[a] * rn;
                        e_old += Si[a] * ro;
                    }
                    dE_true += 0.5 * (e_new - e_old);
                }
            }

            const size_t n_tri = trilinear_partners[i].size();
            for (size_t n = 0; n < n_tri; ++n) {
                const size_t p1 = trilinear_partners[i][n][0];
                const size_t p2 = trilinear_partners[i][n][1];
                const double m = 1.0 + (in_cluster[p1] ? 1.0 : 0.0) + (in_cluster[p2] ? 1.0 : 0.0);
                double S1n[MAX_SPIN_DIM], S2n[MAX_SPIN_DIM];
                const double* S1 = spins[p1].data();
                const double* S2 = spins[p2].data();
                if (in_cluster[p1]) reflected(p1, S1n); else std::memcpy(S1n, S1, D * sizeof(double));
                if (in_cluster[p2]) reflected(p2, S2n); else std::memcpy(S2n, S2, D * sizeof(double));
                const auto& T = trilinear_interaction[i][n];
                double e_new = 0.0, e_old = 0.0;
                for (size_t a = 0; a < D; ++a)
                    for (size_t b = 0; b < D; ++b)
                        for (size_t c = 0; c < D; ++c) {
                            e_new += T[a](b, c) * Si_new[a] * S1n[b] * S2n[c];
                            e_old += T[a](b, c) * Si[a] * S1[b] * S2[c];
                        }
                dE_true += (e_new - e_old) / m;
            }
        }
        return dE_true - dE_emb;
    }

    /**
     * Wolff single-cluster update. Returns the number of spins flipped
     * (0 if the cluster touched the ghost spin or was rejected by the
     * residual filter).
     *
     * @param use_ghost_field  embed the Zeeman term as a coupling to a fixed
     *        ghost spin (clusters bonded to it are not flipped); otherwise
     *        the field enters the residual filter.
     */
    size_t wolff_update(double T, bool use_ghost_field = false) {
        if (T <= 0) return 0;
        const double beta = 1.0 / T;

        const size_t seed = random_int_lehman(lattice_size);
        SpinVector r = random_unit_vector();
        const double* r_data = r.data();

        if (cluster_proj_buf.size()   < lattice_size) cluster_proj_buf.resize(lattice_size);
        if (cluster_in_cluster.size() < lattice_size) cluster_in_cluster.assign(lattice_size, 0);
        else std::fill(cluster_in_cluster.begin(), cluster_in_cluster.begin() + lattice_size, 0);
        cluster_stack_buf.clear();
        cluster_members_buf.clear();

        for (size_t i = 0; i < lattice_size; ++i) {
            const double* S = spins[i].data();
            double acc = 0.0;
            for (size_t d = 0; d < spin_dim; ++d) acc += S[d] * r_data[d];
            cluster_proj_buf[i] = acc;
        }

        bool attached_to_ghost = false;
        cluster_in_cluster[seed] = 1;
        cluster_stack_buf.push_back(seed);
        cluster_members_buf.push_back(seed);

        while (!cluster_stack_buf.empty()) {
            const size_t i = cluster_stack_buf.back();
            cluster_stack_buf.pop_back();
            const double s_i = cluster_proj_buf[i];

            const size_t bi_end = bi_flat_offset[i + 1];
            for (size_t k = bi_flat_offset[i]; k < bi_end; ++k) {
                const size_t j = bi_flat_partner[k];
                if (cluster_in_cluster[j]) continue;
                const double x = embedded_coupling(k, r_data) * s_i * cluster_proj_buf[j];
                if (x >= 0.0) continue;  // unsatisfied (or non-embedded) bond
                if (random_double_lehman(0.0, 1.0) < 1.0 - std::exp(2.0 * beta * x)) {
                    cluster_in_cluster[j] = 1;
                    cluster_stack_buf.push_back(j);
                    cluster_members_buf.push_back(j);
                }
            }

            if (use_ghost_field && !attached_to_ghost) {
                const double* B = field[i].data();
                double B_r = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) B_r += B[d] * r_data[d];
                const double x_g = -B_r * s_i;   // Zeeman energy of the embedded spin
                if (x_g < 0.0 && random_double_lehman(0.0, 1.0) < 1.0 - std::exp(2.0 * beta * x_g))
                    attached_to_ghost = true;
            }
        }
        if (attached_to_ghost) return 0;

        double dE_res = 0.0;
        if (!cluster_embedding_is_exact()) {
            dE_res = cluster_residual_energy(cluster_members_buf, cluster_in_cluster.data(),
                                             r_data, cluster_proj_buf.data(), use_ghost_field);
        } else if (!use_ghost_field) {
            // Exact embedding: only the Zeeman term is left, O(|C|).
            for (size_t i : cluster_members_buf) {
                const double* B = field[i].data();
                double B_r = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) B_r += B[d] * r_data[d];
                dE_res += 2.0 * cluster_proj_buf[i] * B_r;
            }
        }
        if (dE_res > 0.0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE_res)) return 0;

        for (size_t i : cluster_members_buf) {
            double* S = spins[i].data();
            const double two_proj = 2.0 * cluster_proj_buf[i];
            for (size_t d = 0; d < spin_dim; ++d) S[d] -= two_proj * r_data[d];
        }
        return cluster_members_buf.size();
    }

    /**
     * Generate random unit vector
     */
    SpinVector random_unit_vector() const {
        SpinVector v(spin_dim);
        gen_random_spin_into(v.data(), 1.0f);
        return v;
    }

    /**
     * Swendsen-Wang sweep: build all FK clusters of the embedded Ising model
     * and propose flipping each with probability 1/2 (clusters bonded to the
     * ghost spin are frozen). For non-embeddable Hamiltonians each proposed
     * flip is filtered sequentially with min(1, exp(-β ΔE_res)), which is
     * exact because a cluster flip leaves the FK bond weights invariant.
     * Returns the number of clusters flipped.
     */
    size_t swendsen_wang_sweep(double T, bool use_ghost_field = false) {
        if (T <= 0) return 0;

        const double beta = 1.0 / T;
        SpinVector r = random_unit_vector();
        const double* r_data = r.data();

        if (cluster_proj_buf.size() < lattice_size) cluster_proj_buf.resize(lattice_size);
        if (uf_parent.size()        < lattice_size) uf_parent.resize(lattice_size);
        if (uf_size.size()          < lattice_size) uf_size.resize(lattice_size);
        if (uf_forbid_flip.size()   < lattice_size) uf_forbid_flip.assign(lattice_size, 0);
        else std::fill(uf_forbid_flip.begin(), uf_forbid_flip.begin() + lattice_size, 0);
        if (uf_flip_root.size()     < lattice_size) uf_flip_root.assign(lattice_size, 0);
        else std::fill(uf_flip_root.begin(), uf_flip_root.begin() + lattice_size, 0);

        std::iota(uf_parent.begin(), uf_parent.begin() + lattice_size, 0);
        std::fill(uf_size.begin(), uf_size.begin() + lattice_size, 1);

        for (size_t i = 0; i < lattice_size; ++i) {
            const double* S = spins[i].data();
            double acc = 0.0;
            for (size_t d = 0; d < spin_dim; ++d) acc += S[d] * r_data[d];
            cluster_proj_buf[i] = acc;
        }

        auto find_root = [&](int x) noexcept -> int {
            int root = x;
            while (uf_parent[root] != root) root = uf_parent[root];
            while (uf_parent[x] != root) {
                int next = uf_parent[x];
                uf_parent[x] = root;
                x = next;
            }
            return root;
        };
        auto unite = [&](int a, int b) noexcept {
            a = find_root(a);
            b = find_root(b);
            if (a == b) return;
            if (uf_size[a] < uf_size[b]) std::swap(a, b);
            uf_parent[b] = a;
            uf_size[a] += uf_size[b];
        };

        // Bond percolation: each bond once (j > i), satisfied bonds only.
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t bi_end = bi_flat_offset[i + 1];
            for (size_t k = bi_flat_offset[i]; k < bi_end; ++k) {
                const size_t j = bi_flat_partner[k];
                if (j <= i) continue;
                const double x = embedded_coupling(k, r_data) * cluster_proj_buf[i] * cluster_proj_buf[j];
                if (x >= 0.0) continue;
                if (random_double_lehman(0.0, 1.0) < 1.0 - std::exp(2.0 * beta * x))
                    unite(static_cast<int>(i), static_cast<int>(j));
            }
        }

        if (use_ghost_field) {
            for (size_t i = 0; i < lattice_size; ++i) {
                const double* B = field[i].data();
                double B_r = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) B_r += B[d] * r_data[d];
                const double x_g = -B_r * cluster_proj_buf[i];
                if (x_g < 0.0 && random_double_lehman(0.0, 1.0) < 1.0 - std::exp(2.0 * beta * x_g))
                    uf_forbid_flip[find_root(static_cast<int>(i))] = 1;
            }
        }

        for (size_t i = 0; i < lattice_size; ++i) {
            const int root = find_root(static_cast<int>(i));
            if (static_cast<int>(i) == root && !uf_forbid_flip[root])
                uf_flip_root[root] = (random_double_lehman(0.0, 1.0) < 0.5) ? 1 : 0;
        }

        const bool exact = cluster_embedding_is_exact() && (use_ghost_field || !has_nonzero_field());
        if (!exact) {
            // Group members by root (counting sort), then filter each
            // proposed flip against the current configuration.
            if (cluster_in_cluster.size() < lattice_size) cluster_in_cluster.assign(lattice_size, 0);
            else std::fill(cluster_in_cluster.begin(), cluster_in_cluster.begin() + lattice_size, 0);
            vector<size_t> offset(lattice_size + 1, 0), order(lattice_size);
            for (size_t i = 0; i < lattice_size; ++i) ++offset[find_root(static_cast<int>(i)) + 1];
            for (size_t i = 0; i < lattice_size; ++i) offset[i + 1] += offset[i];
            vector<size_t> cursor(offset.begin(), offset.end() - 1);
            for (size_t i = 0; i < lattice_size; ++i) order[cursor[find_root(static_cast<int>(i))]++] = i;

            size_t flipped = 0;
            for (size_t root = 0; root < lattice_size; ++root) {
                if (!uf_flip_root[root]) continue;
                cluster_members_buf.assign(order.begin() + offset[root], order.begin() + offset[root + 1]);
                for (size_t i : cluster_members_buf) cluster_in_cluster[i] = 1;
                // Projections of the current (partially updated) configuration.
                for (size_t i : cluster_members_buf) {
                    const double* S = spins[i].data();
                    double acc = 0.0;
                    for (size_t d = 0; d < spin_dim; ++d) acc += S[d] * r_data[d];
                    cluster_proj_buf[i] = acc;
                }
                const double dE_res = cluster_residual_energy(cluster_members_buf, cluster_in_cluster.data(),
                                                              r_data, cluster_proj_buf.data(), use_ghost_field);
                const bool accept = dE_res <= 0.0 || random_double_lehman(0.0, 1.0) < std::exp(-beta * dE_res);
                for (size_t i : cluster_members_buf) {
                    cluster_in_cluster[i] = 0;
                    if (!accept) continue;
                    double* S = spins[i].data();
                    const double two_proj = 2.0 * cluster_proj_buf[i];
                    for (size_t d = 0; d < spin_dim; ++d) S[d] -= two_proj * r_data[d];
                }
                if (accept) ++flipped;
            }
            return flipped;
        }

        size_t flipped_clusters = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const int root = find_root(static_cast<int>(i));
            if (uf_flip_root[root]) {
                double* S = spins[i].data();
                const double two_proj = 2.0 * cluster_proj_buf[i];
                for (size_t d = 0; d < spin_dim; ++d) S[d] -= two_proj * r_data[d];
                if (static_cast<int>(i) == root) ++flipped_clusters;
            }
        }
        return flipped_clusters;
    }

    /**
     * Convenience: multiple Wolff updates
     */
    size_t wolff_sweep(double T, size_t k = 1, bool use_ghost_field = false) {
        size_t total = 0;
        for (size_t c = 0; c < k; ++c) {
            total += wolff_update(T, use_ghost_field);
        }
        return total;
    }

    /**
     * Zero-temperature block-coordinate descent: visit every site in order
     * and replace S_i by the exact minimiser of E with all other spins held
     * fixed. For a linear local energy that is S_i = -s g/|g|; with an
     * anisotropic on-site term S^T A S it is the trust-region solution of
     * `minimize_quadratic_on_sphere`. Each step can only lower the energy,
     * so repeated sweeps converge monotonically to a local minimum.
     *
     * (The previous version set S_i = -s H/|H| with H including 2 A S_i,
     * a fixed-point iteration that can climb in energy — it drove an
     * easy-plane model from E/N = -0.62 to +0.73 — and drew sites with
     * replacement, looping forever if every local field vanished.)
     *
     * Returns the largest change |ΔS_i| of the last sweep.
     */
    double deterministic_sweep(size_t num_sweeps = 1) {
        const double s = double(spin_length);
        double max_change = 0.0;
        double g[MAX_SPIN_DIM], S_new[MAX_SPIN_DIM];
        for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
            max_change = 0.0;
            for (size_t i = 0; i < lattice_size; ++i) {
                linear_field(i, spins_view(), g);
                double* S = spins[i].data();
                if (su3_cp2) {
                    if (onsite_scalar[i]) {
                        classical_spin::su3::ground_state_cp2(g, S_new);
                    } else {
                        // Ground state of the on-site term linearised at S, kept
                        // only if it lowers the local energy (monotone descent).
                        double h[8];
                        const Eigen::Map<const Eigen::VectorXd> Sv(S, 8);
                        const Eigen::VectorXd AS = onsite_interaction[i] * Sv;
                        for (int a = 0; a < 8; ++a) h[a] = g[a] + 2.0 * AS(a);
                        classical_spin::su3::ground_state_cp2(h, S_new);
                        double e_new = onsite_energy(i, S_new), e_old = onsite_energy(i, S);
                        for (int a = 0; a < 8; ++a) { e_new += g[a] * S_new[a]; e_old += g[a] * S[a]; }
                        if (!(e_new < e_old)) continue;
                    }
                } else if (onsite_scalar[i]) {
                    double norm = 0.0;
                    for (size_t d = 0; d < spin_dim; ++d) norm += g[d] * g[d];
                    norm = std::sqrt(norm);
                    if (norm < 1e-300) continue;  // no torque: leave the spin
                    for (size_t d = 0; d < spin_dim; ++d) S_new[d] = -s * g[d] / norm;
                } else {
                    minimize_quadratic_on_sphere(onsite_interaction[i], g, s, S, S_new, spin_dim);
                }
                double change = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) change += (S_new[d] - S[d]) * (S_new[d] - S[d]);
                max_change = std::max(max_change, std::sqrt(change));
                std::memcpy(S, S_new, spin_dim * sizeof(double));
            }
        }
        return max_change;
    }

    /**
     * Zero-temperature quench to a local minimum: repeat descent sweeps
     * until the energy change per sweep falls below rel_tol·|E| and no spin
     * moves by more than sqrt(rel_tol)·s.
     */
    void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000) {
        double E_prev = total_energy();
        for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
            const double max_change = deterministic_sweep(1);
            const double E_curr = total_energy();
            if (std::abs(E_curr - E_prev) <= rel_tol * (std::abs(E_prev) + 1e-18) &&
                max_change <= std::sqrt(rel_tol) * double(spin_length)) {
                break;
            }
            E_prev = E_curr;
        }
    }

    /**
     * Metropolis update for twist boundary matrices
     * Returns acceptance count (number of accepted moves)
     * 
     * Optimizes the rotation angle around a fixed z-axis.
     * Uses hybrid strategy:
     *   - At high T: larger steps + occasional global moves for exploration
     *   - At low T: small incremental steps for fine-tuning
     *   - Occasional random global moves (10% probability) to escape local minima
     */
    size_t metropolis_twist_sweep(double T);

    /**
     * Deterministic relaxation of the twist angles at T = 0.
     * For each dimension d with Ld > 1, performs a 1-D golden-section line
     * search in the twist angle θ_d that minimizes the boundary-bond energy.
     * The bracket is initialized around the current angle with width ±π and
     * iteratively shrunk; runs `n_passes` sweeps over the dimensions.
     *
     * This is the SOTA companion to `metropolis_twist_sweep` for the
     * deterministic (T = 0) phase: at zero temperature the boundary energy
     * is a smooth 1-D function of θ_d, so line minimization converges in
     * O(log(1/tol)) evaluations and resolves incommensurate ordering
     * wave-vectors that would be locked to the discrete grid n/L by
     * standard PBC.
     */
    void relax_twist_angles(size_t n_passes = 4, double tol = 1e-10);
    
    /**
     * Extract axis-angle representation from rotation matrix
     * Uses the formula: angle = arccos((trace(R) - 1) / 2)
     * Axis is extracted from the antisymmetric part of R
     */
    static void extract_axis_angle_from_rotation(const SpinMatrix& R, SpinVector& axis, double& angle) {
        if (R.rows() != 3 || R.cols() != 3) {
            axis = SpinVector::Zero(R.rows());
            if (axis.size() >= 3) axis(2) = 1.0;
            angle = 0.0;
            return;
        }
        
        double trace = R(0, 0) + R(1, 1) + R(2, 2);
        double cos_angle = (trace - 1.0) / 2.0;
        cos_angle = std::clamp(cos_angle, -1.0, 1.0);
        angle = std::acos(cos_angle);
        
        // Handle special cases
        if (std::abs(angle) < 1e-10) {
            // Identity rotation
            axis = SpinVector::Zero(3);
            axis(2) = 1.0;
            angle = 0.0;
            return;
        }
        
        if (std::abs(angle - M_PI) < 1e-10) {
            // 180 degree rotation - extract axis from diagonal
            axis = SpinVector::Zero(3);
            axis(0) = std::sqrt(std::max(0.0, (R(0, 0) + 1.0) / 2.0));
            axis(1) = std::sqrt(std::max(0.0, (R(1, 1) + 1.0) / 2.0));
            axis(2) = std::sqrt(std::max(0.0, (R(2, 2) + 1.0) / 2.0));
            // Determine signs from off-diagonal elements
            if (R(0, 1) < 0) axis(1) = -axis(1);
            if (R(0, 2) < 0) axis(2) = -axis(2);
            return;
        }
        
        // General case: extract axis from antisymmetric part
        double sin_angle = std::sin(angle);
        axis = SpinVector::Zero(3);
        axis(0) = (R(2, 1) - R(1, 2)) / (2.0 * sin_angle);
        axis(1) = (R(0, 2) - R(2, 0)) / (2.0 * sin_angle);
        axis(2) = (R(1, 0) - R(0, 1)) / (2.0 * sin_angle);
        
        // Normalize (should already be unit, but ensure numerical stability)
        double norm = axis.norm();
        if (norm > 1e-10) {
            axis /= norm;
        } else {
            axis(2) = 1.0;
        }
    }

    
    // ============================================================
    // SIMULATED ANNEALING AUTO-TUNING
    // ============================================================

    /**
     * Auto-tune simulated annealing parameters
     */
    SAParams tune_simulated_annealing(double Tmin_guess = 0.0,
                                      double Tmax_guess = 0.0,
                                      bool gaussian_move = false,
                                      size_t overrelaxation_rate = 0,
                                      size_t pilot_sweeps = 300,
                                      double acc_hi_target = 0.7,
                                      double acc_lo_target = 0.02) {
        SAParams params;
        
        // Backup current state
        SpinConfig spins_backup = spins;
        
        // Lambda: probe acceptance and autocorrelation at temperature T
        auto probe_once = [&](double T, size_t sweeps, double base_interval,
                             double& acc_out, double& tau_out) {
            double sigma = 1000.0;
            double acc_sum = 0.0;
            vector<double> energies;
            energies.reserve(sweeps / size_t(base_interval) + 1);
            
            for (size_t i = 0; i < sweeps; ++i) {
                acc_sum += metropolis(T, gaussian_move, sigma);
                
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    overrelaxation();
                }
                
                if (i % size_t(base_interval) == 0) {
                    energies.push_back(total_energy(spins));
                }
            }
            
            acc_out = acc_sum / double(sweeps);
            
            if (energies.size() > 10) {
                auto acf = compute_autocorrelation(energies, size_t(base_interval));
                tau_out = acf.tau_int;
            } else {
                tau_out = 1.0;
            }
        };
        
        // Calibrate T_start (high acceptance)
        {
            double T = (Tmax_guess > 0.0) ? Tmax_guess : 1.0;
            double acc = 0.0, tau = 1.0;
            
            // Expand up if needed
            for (size_t iter = 0; iter < 25; ++iter) {
                probe_once(T, pilot_sweeps, 10, acc, tau);
                params.probe_T.push_back(T);
                params.probe_acc.push_back(acc);
                params.probe_tau.push_back(tau);
                
                if (acc >= acc_hi_target) break;
                T *= 2.0;
            }
            
            // Binary search to center in target band
            double Thigh = params.probe_T.back();
            double Tlow = Thigh / 100.0;
            for (size_t k = 0; k < 20; ++k) {
                double Tmid = 0.5 * (Tlow + Thigh);
                probe_once(Tmid, pilot_sweeps, 10, acc, tau);
                params.probe_T.push_back(Tmid);
                params.probe_acc.push_back(acc);
                params.probe_tau.push_back(tau);
                
                if (acc > acc_hi_target) {
                    Thigh = Tmid;
                } else {
                    Tlow = Tmid;
                }
            }
            
            params.T_start = Thigh;
        }
        
        // Calibrate T_end (low acceptance, energy converged)
        {
            double T = (Tmin_guess > 0.0 && Tmin_guess < params.T_start) ? 
                       Tmin_guess : params.T_start * 1e-3;
            T = max(T, params.T_start * 1e-6);
            
            double acc = 1.0, tau = 1.0;
            double cur = params.T_start;
            
            for (size_t iter = 0; iter < 40 && cur > T; ++iter) {
                cur *= 0.5;
                probe_once(cur, pilot_sweeps, 10, acc, tau);
                params.probe_T.push_back(cur);
                params.probe_acc.push_back(acc);
                params.probe_tau.push_back(tau);
                
                if (acc < acc_lo_target) break;
            }
            
            params.T_end = max(1e-12, min(cur, params.T_start / 1e3));
        }
        
        // Choose sweeps_per_temp from autocorrelation time
        double tau_max = 1.0;
        for (double t : params.probe_tau) {
            tau_max = std::max(tau_max, t);
        }
        params.sweeps_per_temp = std::max<size_t>(100, size_t(10.0 * tau_max));
        
        // Number of temperature steps
        size_t K = std::max<size_t>(50, size_t(10.0 * std::sqrt(tau_max)));
        K = std::min<size_t>(2000, K);
        params.cooling_rate = std::pow(params.T_end / params.T_start, 1.0 / double(K));
        params.cooling_rate = std::clamp(params.cooling_rate, 0.85, 0.995);
        
        // Restore original spins
        spins = spins_backup;
        
        cout << "Auto-tuned SA parameters:" << endl;
        cout << "  T_start = " << params.T_start << endl;
        cout << "  T_end = " << params.T_end << endl;
        cout << "  cooling_rate = " << params.cooling_rate << endl;
        cout << "  sweeps_per_temp = " << params.sweeps_per_temp << endl;
        
        return params;
    }

    // ============================================================
    // SIMULATED ANNEALING
    // ============================================================

    /**
     * Main simulated annealing routine
     * 
     * @param T_start              Starting temperature
     * @param T_end                Final temperature (cooling stops here)
     * @param n_anneal             Number of MC sweeps per temperature step
     * @param overrelaxation_rate  Overrelaxation frequency (0 = disabled)
     * @param boundary_update      Enable twist boundary condition updates
     * @param gaussian_move        Use Gaussian moves instead of uniform
     * @param cooling_rate         Temperature reduction factor (default: 0.9)
     * @param out_dir              Output directory for configurations
     * @param save_observables     Save energy/magnetization trajectories
     * @param T_zero               Perform zero-temperature deterministic sweeps
     * @param n_deterministics     Number of T=0 sweeps (if T_zero=true)
     * @param twist_sweep_count    Twist BC sweeps per MC sweep (default: 100)
     */
    void simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            size_t overrelaxation_rate = 0,
                            bool boundary_update = false,
                            bool gaussian_move = false,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = false,
                            bool T_zero = false,
                            size_t n_deterministics = 1000,
                            size_t twist_sweep_count = 100);

    /**
     * Perform detailed measurements at final temperature
     * Computes: energy, specific heat, sublattice magnetizations, and cross-correlations
     * All with binning analysis for error estimation
     */
    void perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   size_t overrelaxation_rate, const string& out_dir);

    /**
     * Save sublattice magnetization time series to file
     */
    void save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<vector<SpinVector>>& sublattice_mags) const;

    /**
     * Compute and save thermodynamic observables
     */
    void compute_and_save_observables(const vector<double>& energies,
                                     const vector<SpinVector>& magnetizations,
                                     double T, const string& out_dir);

    /**
     * Save autocorrelation results
     */
    void save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf);

    /**
     * Cluster-based annealing (Wolff/SW)
     */
    void cluster_annealing(double T_start, double T_end, size_t n_anneal,
                          size_t wolff_per_temp, bool use_sw = false,
                          bool use_ghost_field = false, double cooling_rate = 0.9,
                          string out_dir = "");

    // ============================================================
    // PARALLEL TEMPERING
    // ============================================================

    /**
     * Parallel tempering (one replica per rank of `comm`) on the shared engine
     * mc::run_parallel_tempering: deterministic even/odd replica exchange with
     * labelled replicas (round trips, f(T), per-edge acceptance), proposal
     * width adapted per temperature while equilibrating and frozen while
     * measuring, Gamma-method / jackknife statistics.
     *
     * @param temp              Temperature ladder, one per rank, strictly increasing
     * @param n_anneal          Equilibration MC steps
     * @param n_measure         Measurement MC steps
     * @param overrelaxation_rate k > 0: one overrelaxation sweep per step and a local
     *                          sweep every k-th step (as perform_mc_sweeps); 0: local sweeps only
     * @param swap_rate         MC steps between exchange rounds (0 = no exchanges)
     * @param probe_rate        MC steps between measurements
     * @param dir_name          Output directory ("" = no files)
     * @param rank_to_write     Ranks writing per-temperature files (-1 = all)
     * @param gaussian_move     Adaptive Gaussian proposals (also selected by local_update)
     * @param comm              MPI communicator
     * @param verbose           Unused (kept for source compatibility)
     * @param accumulate_correlations  Accumulate spin and nearest-shell dimer correlations
     *                          (create_correlation_accumulator) into rank_dir/correlations_T*.h5
     * @param n_bond_types      Ignored (bond classes come from the unit cell)
     * @return per-rank and whole-ladder statistics (see mc::PTResult)
     */
    mc::PTResult parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false, bool accumulate_correlations = false,
                           size_t n_bond_types = 3);
    
    /**
     * Save kagome order parameters to HDF5 file (pyrochlore patch)
     * Nematic order is now component-resolved: [bond_type x spin_component] 3x3 matrix
     * Includes monopole density (signed) and monopole by sublattice
     * Includes global-frame chirality and nematic order parameters
     */
    void save_kagome_order_parameters(const string& rank_dir, double temperature,
                                       const vector<double>& scalar_chi,
                                       const vector<Eigen::Vector3d>& vector_chi,
                                       const vector<Eigen::Matrix3d>& nematic,
                                       const vector<double>& monopole,
                                       const vector<Eigen::Vector4d>& monopole_sub,
                                       const vector<double>& scalar_chi_global,
                                       const vector<Eigen::Vector3d>& vector_chi_global,
                                       const vector<Eigen::Matrix3d>& nematic_global) const;
    
    /**
     * One sample of the current configuration into both channels of `acc`
     * (spin correlations, and dimer correlations when it has bond classes).
     */
    void accumulate_correlations_internal(RealSpaceCorrelationAccumulator& acc) const {
        acc.add_sample(spins);
    }

    /**
     * Tune the temperature ladder with the replica chain (one replica per rank,
     * collective): rounds of doubling length measure per-edge rejection (and
     * the label flow f(T)) and move the interior temperatures — by default to
     * equal rejection (non-reversible PT schedule, Syed et al., JRSS-B 84, 321
     * (2022)), or by the Katzgraber et al. flow feedback. The MC step follows
     * the same schedule as parallel_tempering(), and each rank ends with a
     * replica equilibrated near its tuned temperature.
     */
    mc::LadderTuningResult tune_temperature_ladder(const mc::LadderTuningOptions& options,
                                                   size_t overrelaxation_rate, bool gaussian_move,
                                                   MPI_Comm comm = MPI_COMM_WORLD);

    /// Geometric ladder T_i = T_min (T_max/T_min)^(i/(R-1)) (delegates to mc::).
    static vector<double> generate_geometric_temperature_ladder(double Tmin, double Tmax, size_t R) {
        return mc::generate_geometric_temperature_ladder(Tmin, Tmax, R);
    }

    // ============================================================
    // SPIN DYNAMICS  (implementation: src/core/lattice_md.cpp)
    //
    // Equation of motion (γ = ħ = 1):
    //
    //   dS_i/dt = g [S_i × B_i - (α/|S_i|) S_i × (S_i × B_i)],
    //   B_i     = -∂E/∂S_i + B_drive(t, i),
    //
    // with g = 1 (damping_form Landau-Lifshitz, λ = alpha_gilbert, default) or
    // g = 1/(1 + α²) (Gilbert form). For
    // spin_dim = 8 (SU(3)) × is the structure-constant product
    // (a × b)_i = f_ijk a_j b_k; the damping term keeps its double-bracket
    // form and still dissipates, dE/dt = -(α/|S|) |∂E/∂S × S|² <= 0, while
    // conserving |S|. No other spin dimension has a Lie-algebra cross
    // product, so the dynamics drivers reject them.
    //
    // Every driver samples on an exact grid t_k = t0 + k dt
    // (dynamics/time_grid.h) and takes the drive as an explicit, immutable
    // DriveSchedule (dynamics/drive.h): the lattice itself is never mutated
    // by a trajectory, so concurrent trajectories can share one instance.
    // ============================================================

    using DriveSchedule = classical_spin::dynamics::DriveSchedule;
    using Pulse = classical_spin::dynamics::Pulse;
    using TimeGrid = classical_spin::dynamics::TimeGrid;

    /** One (t, [M_staggered_global, M_local, M_global]) sample per grid point. */
    using PumpProbeTrajectory = vector<pair<double, array<SpinVector, 3>>>;

    /** Integrator choice and accuracy of one trajectory. */
    struct DynamicsSettings {
        string method = "dopri5";  ///< any name accepted by parse_ode_method (dynamics/ode_method.h)
        double dt = 0.01;          ///< step (fixed-step and geometric methods) or initial step (adaptive)
        double abs_tol = 1e-8;     ///< error-controlled methods only
        double rel_tol = 1e-8;
        double max_dt = 0.0;       ///< error-controlled step cap, 0 = none
    };

    /** Observer of integrate_on_grid: configuration x at sample k, t = grid[k]. */
    using GridObserver = std::function<void(const double* x, size_t k, double t)>;

    /**
     * Right-hand side of the equation of motion for the flat state x
     * (lattice_size * spin_dim values) under an explicit drive.
     */
    void landau_lifshitz_rhs(const double* x, double* dxdt, double t, const DriveSchedule& drive) const;

    /** landau_lifshitz_rhs with the drive installed by set_pulse(). */
    void landau_lifshitz_flat(const double* state_flat, double* dsdt_flat, double t) const {
        landau_lifshitz_rhs(state_flat, dsdt_flat, t, active_drive);
    }

    /** Drive field (spin frame) of the installed drive at a site. */
    SpinVector drive_field_at_time(double t, size_t site_index) const;

    /**
     * Spin-frame polarisation of a field given per sublattice in the GLOBAL
     * frame. The sublattice frame F_a maps spin components to global ones,
     * S_global = F_a S (the convention of every magnetisation observable and
     * of UnitCell::set_sublattice_frame), so the Zeeman energy
     * -B_g · S_global = -(F_a^T B_g) · S: the field acting on the spin
     * variables is F_a^T B_g. (set_pulse used to apply F_a, which drives the
     * wrong components for any non-symmetric frame, e.g. the Kitaev frame.)
     */
    vector<double> local_polarisation(const vector<SpinVector>& field_global) const;

    /** An empty drive for this lattice; add pulses with add_pulse(). */
    DriveSchedule make_drive() const { return DriveSchedule(N_atoms, spin_dim); }

    /** Append a pulse whose polarisation is given per sublattice in the global frame. */
    void add_pulse(DriveSchedule& drive, const vector<SpinVector>& field_global, const Pulse& p) const {
        drive.add(p, local_polarisation(field_global));
    }

    /**
     * Install the two-pulse drive used by landau_lifshitz_flat, ode_system,
     * integrate_geometric and the GPU paths (equal amplitude, width and
     * frequency; polarisations in the global frame).
     */
    void set_pulse(const vector<SpinVector>& field_in1, double t_B1,
                   const vector<SpinVector>& field_in2, double t_B2,
                   double pulse_amp, double pulse_width, double pulse_freq);

    /** Remove the installed drive. */
    void clear_pulse();

    /**
     * Convert SpinConfig to flat state vector
     */
    ODEState spins_to_state(const SpinConfig& spins_vec) const {
        ODEState state(lattice_size * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t j = 0; j < spin_dim; ++j) {
                state[i * spin_dim + j] = spins_vec[i](j);
            }
        }
        return state;
    }

    /**
     * Convert flat state vector back to SpinConfig
     */
    SpinConfig state_to_spins(const ODEState& state) const {
        SpinConfig spins_vec(lattice_size);
        for (size_t i = 0; i < lattice_size; ++i) {
            spins_vec[i] = SpinVector(spin_dim);
            for (size_t j = 0; j < spin_dim; ++j) {
                spins_vec[i](j) = state[i * spin_dim + j];
            }
        }
        return spins_vec;
    }

    /**
     * The three magnetisation channels of every dynamics output, in one pass:
     *   out[0 .. D)   M_antiferro = Σ_i s_a(i) F_a(i) S_i / N   (staggered, global frame,
     *                                                         signs afm_sublattice_signs)
     *   out[D .. 2D)  M_local     = Σ_i S_i / N
     *   out[2D .. 3D) M_global    = Σ_i F_a(i) S_i / N
     * Sublattice sums P_a = Σ_{i∈a} S_i are accumulated first (O(N D)), the
     * frames applied once per sublattice (O(N_atoms D²)). `scratch` holds
     * N_atoms * spin_dim doubles, so the call does not allocate. This is the
     * ONLY definition of these channels: observers and ground-state
     * baselines must use it (a different "antiferro" baseline produced a
     * step artefact in the synthesised M1 of 2DCS scans).
     */
    void measure_magnetizations(const double* x, double* out, double* scratch) const;
    array<SpinVector, 3> measure_magnetizations(const double* x) const;

    /**
     * Integrate `state` across `grid`; `observer` is called exactly once per
     * sample, k = 0 .. grid.n - 1, with t = grid[k], and `state` ends at
     * grid.t_end().
     *   - geometric and fixed-step methods take m = max(1, round(grid.dt /
     *     settings.dt)) steps of exactly grid.dt / m per sample interval;
     *   - dopri5 / bulirsch_stoer use dense output (steps chosen by the error
     *     controller, samples interpolated at the exact grid times);
     *     cash_karp54 / rkf78 step exactly onto each sample time.
     * Langevin noise (langevin_temperature > 0) needs a geometric method.
     * Throws std::invalid_argument for an unknown method, spin_dim not in
     * {3, 8} or inconsistent settings.
     */
    void integrate_on_grid(ODEState& state, const TimeGrid& grid, const DriveSchedule& drive,
                           const DynamicsSettings& settings, const GridObserver& observer) const;

    /** Magnetisation trajectory of the current spins under `drive` on `grid`. */
    PumpProbeTrajectory drive_trajectory(const DriveSchedule& drive, const TimeGrid& grid,
                                         const DynamicsSettings& settings) const;

    /**
     * Adapter exposing the lattice to the geometric integrators of
     * dynamics/spin_integrators.h: effective fields B_i = -∂E/∂S_i + B_drive
     * and the sublattice colouring. `drive` = nullptr uses the installed drive.
     */
    struct DynamicsModel {
        const Lattice& lat;
        const DriveSchedule* drive = nullptr;
        mutable double f[DriveSchedule::kMaxPulses] = {};   // envelopes cached by set_time

        const DriveSchedule& schedule() const { return drive ? *drive : lat.active_drive; }
        size_t n_sites() const { return lat.lattice_size; }
        double spin_length() const { return double(lat.spin_length); }
        void set_time(double t) const { schedule().envelopes(t, f); }
        void field_site(const double* x, double, size_t i, double* B) const {
            // A scalar on-site matrix (A = cI, e.g. a folded isotropic self-
            // bond) contributes 2c S_i to the gradient: parallel to S_i, no
            // torque. It is dropped because the discrete schemes — colour
            // splitting in particular, which freezes B_i over a sub-step —
            // would otherwise rotate about a tilted axis.
            double H[MAX_SPIN_DIM];
            if (lat.onsite_scalar[i]) lat.linear_field(i, lat.flat_view(x), H);
            else lat.get_local_field_flat(x, i, H);
            for (size_t d = 0; d < 3; ++d) B[d] = -H[d];
            const DriveSchedule& s = schedule();
            if (!s.empty()) s.accumulate(i % lat.N_atoms, f, B);
        }
        void field_all(const double* x, double t, double* B) const {
            set_time(t);
            const size_t N = lat.lattice_size;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(N >= 512)
#endif
            for (size_t i = 0; i < N; ++i) field_site(x, t, i, B + 3 * i);
        }
        size_t n_colors() const { return lat.n_colors; }
        const size_t* color_sites(size_t c, size_t& count) const {
            count = lat.sites_by_color_csr_off[c + 1] - lat.sites_by_color_csr_off[c];
            return lat.sites_by_color_csr.data() + lat.sites_by_color_csr_off[c];
        }
        bool energy_linear_in_each_spin() const {
            for (size_t i = 0; i < lat.lattice_size; ++i)
                if (!lat.onsite_scalar[i]) return false;
            return true;
        }
    };

    /**
     * Fixed-step integration with a norm-preserving geometric method
     * (spherical_midpoint, depondt, color_split, color_split4) under the
     * installed drive, stochastic when langevin_temperature > 0. Same
     * contract as odeint::integrate_const: steps of exactly dt at t0 + k dt,
     * the observer called at t0 and after every step, stopping at the last
     * grid point not beyond t1.
     */
    template<typename Observer>
    void integrate_geometric(ODEState& state, double t0, double t1, double dt,
                             Observer observer, const string& method) const {
        if (spin_dim != 3) {
            throw std::invalid_argument("geometric spin integrators need spin_dim == 3 (got " +
                                        std::to_string(spin_dim) + ")");
        }
        if (!(dt > 0.0)) throw std::invalid_argument("integrate_geometric: dt must be positive");
        DynamicsModel model{*this};
        classical_spin::dynamics::SpinIntegrator<DynamicsModel> integrator(
            model, classical_spin::dynamics::parse_geometric_method(method),
            {alpha_gilbert, langevin_temperature, damping_form});
        const long n_steps = std::max(0L, long(std::floor((t1 - t0) / dt + 1e-9)));
        observer(state, t0);
        for (long k = 0; k < n_steps; ++k) {
            integrator.step(state.data(), t0 + double(k) * dt, dt);
            observer(state, t0 + double(k + 1) * dt);
        }
    }

    /**
     * Evolve the lattice's own spins in place from t0 to t1 (fixed step dt)
     * with a geometric integrator; `observer(spins_flat, t)` is optional.
     */
    template<typename Observer>
    void evolve_spins(double t0, double t1, double dt, const string& method, Observer observer) {
        ODEState state = spins_to_state(spins);
        integrate_geometric(state, t0, t1, dt, observer, method);
        // Reach t1 exactly with a final partial step (no observer call).
        const double t_reached = t0 + std::floor((t1 - t0) / dt + 1e-9) * dt;
        if (t1 - t_reached > 1e-12 * std::max(1.0, std::abs(t1))) {
            integrate_geometric(state, t_reached, t1, t1 - t_reached,
                                [](const ODEState&, double) {}, method);
        }
        for (size_t i = 0; i < lattice_size; ++i)
            for (size_t d = 0; d < spin_dim; ++d) spins[i](d) = state[i * spin_dim + d];
    }
    void evolve_spins(double t0, double t1, double dt, const string& method) {
        evolve_spins(t0, t1, dt, method, [](const ODEState&, double) {});
    }

    /**
     * ODE system function (installed drive): dx/dt = f(x, t)
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);

    /**
     * Molecular dynamics of the current spins from T_start to T_end, written
     * to out_dir/trajectory.h5 (HDF5 builds) on the uniform grid
     * t_k = T_start + k dt_save, dt_save = save_interval * dt_initial:
     * /trajectory/{times, magnetization_*, spins} plus the diagnostics
     * /trajectory/energy_density and /trajectory/max_norm_error
     * (max_i ||S_i| - spin_length|) and the attribute /metadata/dt_save.
     * The final configuration goes to out_dir/final_spins.txt; Lattice::spins
     * is left unchanged.
     *
     * Geometric and fixed-step methods step with dt_initial (save_interval
     * steps per sample); dopri5 / bulirsch_stoer adapt their step and are
     * sampled by dense output. alpha_gilbert and langevin_temperature are
     * honoured (finite-temperature Langevin MD needs a geometric method).
     *
     * @param abs_tol, rel_tol  error-controlled methods; <= 0 selects 1e-6
     *                          (1e-8 for bulirsch_stoer)
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5", bool use_gpu = false,
                           double abs_tol = -1.0, double rel_tol = -1.0);

    // ------------------------------------------------------------
    // Dynamical structure factor S^{ab}(q, ω) (dynamics/structure_factor.h)
    // ------------------------------------------------------------

    /** Settings of dynamical_structure_factor(). */
    struct DSSFSettings {
        vector<Eigen::Vector3d> q_points;   ///< wave vectors (Cartesian, 1 / length of site_positions)
        double temperature = 0.0;           ///< sampling temperature (k_B = 1); 0: the current state only
        size_t n_samples = 1;               ///< independent thermal samples
        double t_equilibrate = 50.0;        ///< Langevin time before the first sample
        double t_decorrelate = 10.0;        ///< Langevin time between samples
        double alpha_sampling = 0.1;        ///< damping of the sampling thermostat
        double t_max = 100.0;               ///< length of each energy-conserving trajectory
        double dt = 0.05;                   ///< integration step
        size_t save_every = 1;              ///< sample spacing = save_every * dt
        string method = "spherical_midpoint";  ///< integrator of the measured trajectories
        bool hann_window = true;            ///< Hann window (else none)
    };

    /** S^{ab}(q, ω) on the frequencies omega (increasing), global frame. */
    struct DSSFResult {
        vector<Eigen::Vector3d> q;
        vector<double> omega;
        size_t n_samples = 0;
        double temperature = 0.0, dt_sample = 0.0, t_max = 0.0;
        vector<std::complex<double>> S;         ///< [q][omega][a][b], sample mean
        vector<double> S_err;                   ///< standard error of Re S, same layout
        vector<std::complex<double>> S_static;  ///< [q][a][b], equal-time <A^a_q A^b_q*>
        std::complex<double> at(size_t iq, size_t iw, int a, int b) const {
            return S[((iq * omega.size() + iw) * 3 + a) * 3 + b];
        }
    };

    /**
     * Classical dynamical structure factor by thermal sampling and
     * energy-conserving dynamics: the current spins are equilibrated with
     * stochastic LLG at `temperature` (spherical_midpoint, alpha_sampling),
     * then for each sample a deterministic trajectory (alpha = T = 0, the
     * chosen method) of length t_max is recorded as A_q(t) =
     * N^{-1/2} Σ_i e^{-i q·r_i} F_a S_i(t) and transformed (estimator and sum
     * rule in dynamics/structure_factor.h). Lattice::spins ends in the last
     * sampled state; damping settings are restored. spin_dim must be 3.
     */
    DSSFResult dynamical_structure_factor(const DSSFSettings& settings);

    /** Reciprocal vectors b_i (a_i · b_j = 2π δ_ij) of the unit cell. */
    array<Eigen::Vector3d, 3> reciprocal_vectors() const;

    /** Write a DSSFResult to /dssf in an HDF5 file (overwrites the file). */
    static void write_dssf(const DSSFResult& result, const string& file);

private:
    /// Flat magnetisation series (grid.n x 3 x spin_dim) of a trajectory from x0.
    vector<double> record_magnetizations(ODEState x0, const TimeGrid& grid, const DriveSchedule& drive,
                                         const DynamicsSettings& settings,
                                         const std::function<void()>& on_sample = nullptr) const;
    /// Flat series -> PumpProbeTrajectory on `grid`.
    PumpProbeTrajectory to_trajectory(const vector<double>& flat, const TimeGrid& grid) const;
    /// Magnetisation trajectory from raw (t, flat state) snapshots (GPU paths).
    PumpProbeTrajectory trajectory_from_states(
        const std::vector<std::pair<double, std::vector<double>>>& raw) const;

public:

    // ============================================================
    // OBSERVABLES
    // ============================================================

    /**
     * Global-frame magnetization per site, M = (1/N) Σ_i F_{s(i)} S_i: the
     * sublattice sums first (O(N d)), then one frame rotation per sublattice.
     */
    SpinVector magnetization_global() const {
        vector<SpinVector> sum(N_atoms, SpinVector::Zero(spin_dim));
        for (size_t i = 0; i < lattice_size; ++i) sum[i % N_atoms] += spins[i];
        SpinVector M = SpinVector::Zero(spin_dim);
        for (size_t a = 0; a < N_atoms; ++a) M += sublattice_frames[a] * sum[a];
        return M / double(lattice_size);
    }

    /**
     * Structure factor of the current configuration, per site and in the
     * global frame (S_global = F_s S_local):
     *
     *     S^{αβ}(q) = (1/N) Σ_ij S_i^α S_j^β e^{-i q·(r_i - r_j)}
     *               = (1/N) A^α(q) A^β(q)^*,   A(q) = Σ_i S_i e^{-i q·r_i},
     *
     * a Hermitian spin_dim x spin_dim matrix (direct sum over sites, any q).
     * For thermal averages use the correlation accumulator, which is
     * normalised per unit cell (N_atoms times larger).
     */
    Eigen::MatrixXcd structure_factor_matrix(const Eigen::Vector3d& q) const {
        Eigen::VectorXcd A = Eigen::VectorXcd::Zero(spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            const double phase = q.dot(site_positions[i]);
            const std::complex<double> e(std::cos(phase), -std::sin(phase));
            A += (sublattice_frames[i % N_atoms] * spins[i]).cast<std::complex<double>>() * e;
        }
        return A * A.adjoint() / double(lattice_size);
    }

    /** Tr S(q) = (1/N) |Σ_i S_i e^{-i q·r_i}|^2 (global frame, all components). */
    double structure_factor(const Eigen::Vector3d& q) const {
        return structure_factor_matrix(q).trace().real();
    }

    /** Real (symmetric, neutron) part of structure_factor_matrix(q); spin_dim == 3. */
    Eigen::Matrix3d structure_factor_tensor(const Eigen::Vector3d& q) const {
        if (spin_dim != 3) throw std::invalid_argument("structure_factor_tensor: needs spin_dim == 3");
        return structure_factor_matrix(q).real();
    }

    // ============================================================
    // CORRELATION ACCUMULATOR (correlation_accumulator.h)
    // ============================================================

    /**
     * Geometry of this lattice for RealSpaceCorrelationAccumulator: cell grid,
     * sublattice positions and frames, and the bond classes of the unit cell's
     * bilinear couplings in neighbour shell `dimer_shell` (1 = shortest coupled
     * bonds, 0 = all coupled bonds; no bond classes = no dimer channel).
     */
    RealSpaceCorrelationAccumulator::Geometry correlation_geometry(size_t dimer_shell = 1) const {
        RealSpaceCorrelationAccumulator::Geometry g;
        g.dim1 = dim1;
        g.dim2 = dim2;
        g.dim3 = dim3;
        g.n_sublattices = N_atoms;
        g.spin_dim = spin_dim;
        g.lattice_vectors = {unit_cell.lattice_vectors[0], unit_cell.lattice_vectors[1],
                             unit_cell.lattice_vectors[2]};
        g.positions.assign(unit_cell.lattice_pos.begin(), unit_cell.lattice_pos.begin() + N_atoms);
        g.frames.assign(sublattice_frames.begin(), sublattice_frames.end());
        g.bond_classes = RealSpaceCorrelationAccumulator::bond_classes_from_unit_cell(unit_cell, dimer_shell);
        return g;
    }

    /**
     * Accumulator for this lattice (correlation_geometry(1), default options).
     * `n_bond_types` is ignored: bond classes are the geometrically distinct
     * nearest-shell bonds of the unit cell.
     */
    RealSpaceCorrelationAccumulator create_correlation_accumulator(size_t /*n_bond_types*/ = 0) const {
        RealSpaceCorrelationAccumulator acc;
        acc.initialize(correlation_geometry(1));
        return acc;
    }

    RealSpaceCorrelationAccumulator create_correlation_accumulator(
        const RealSpaceCorrelationAccumulator::Options& options, size_t dimer_shell) const {
        RealSpaceCorrelationAccumulator acc;
        acc.initialize(correlation_geometry(dimer_shell), options);
        return acc;
    }

    /** One sample of the spin channel (call every probe_rate sweeps while measuring). */
    void accumulate_correlations(RealSpaceCorrelationAccumulator& acc) const {
        acc.accumulate_spin_correlations(spins);
    }

    /**
     * One sample of the dimer channel: D^α_c(R) = S^α_i S^α_j over the bond
     * classes stored in `acc` (RealSpaceCorrelationAccumulator::
     * bond_classes_from_unit_cell is the only bond -> class map). A no-op when
     * `acc` has no bond classes.
     */
    void accumulate_dimer_correlations(RealSpaceCorrelationAccumulator& acc) const {
        if (!acc.bond_classes().empty()) acc.accumulate_dimer_correlations(spins);
    }

    // ============================================================
    // FILE I/O
    // ============================================================

    /**
     * Save the spin configuration: lattice_size lines of spin_dim values at
     * full precision (reloads bitwise). Throws std::runtime_error if the file
     * cannot be written.
     */
    void save_spin_config(const string& filename) const {
        classical_spin::io::write_table(filename, lattice_size, spin_dim,
                                        [&](size_t i, size_t j) { return spins[i](j); });
    }

    /**
     * Load a spin configuration written by save_spin_config: exactly
     * lattice_size lines of spin_dim finite numbers ('#' comments allowed).
     * Every spin is rescaled to spin_length (SU(3) spins on CP^2: projected
     * onto the closest pure state unless already pure to round-off). Throws std::runtime_error naming
     * the file and line on a missing file, a short file, a wrong number of
     * columns, a non-finite value, extra rows or a zero vector; the current
     * spins are left unchanged on error.
     */
    void load_spin_config(const string& filename) {
        const classical_spin::io::Table t = classical_spin::io::read_table(filename, lattice_size, spin_dim);
        SpinConfig loaded(lattice_size);
        double max_dev = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const SpinVector s = Eigen::Map<const Eigen::VectorXd>(t.row(i), Eigen::Index(spin_dim));
            const double n = s.norm();
            if (!(n > 0.0))
                throw std::runtime_error(filename + ":" + std::to_string(t.lines[i]) + ": zero spin vector");
            if (su3_cp2) {
                // Qutrit pure states: kept bitwise when |n|^2 = 4/3 and the cubic
                // Casimir is 8/9 (together they force rho's spectrum to {1, 0, 0}),
                // otherwise replaced by the closest pure state.
                loaded[i] = s;
                if (std::abs(classical_spin::su3::casimir2(s.data()) - 4.0 / 3.0) > 1e-12 ||
                    std::abs(classical_spin::su3::casimir3(s.data()) - 8.0 / 9.0) > 1e-12)
                    classical_spin::su3::project_to_cp2(loaded[i].data());
                continue;
            }
            max_dev = std::max(max_dev, std::abs(n - spin_length) / spin_length);
            // A spin already of length spin_length to round-off is kept bitwise.
            loaded[i] = (std::abs(n - spin_length) <= 4.0 * std::numeric_limits<double>::epsilon() * spin_length)
                            ? s : SpinVector(s * (spin_length / n));
        }
        spins = std::move(loaded);
        if (su3_cp2) return;
        if (max_dev > 1e-3)
            std::cerr << "Warning: " << filename << ": spins rescaled to spin_length = " << spin_length
                      << " (largest relative change " << max_dev << ")" << std::endl;
    }

    /**
     * Save twist boundary condition data to file
     * Includes both axis-angle representation and full SO(3) matrices
     */
    void save_twist_angles(const string& filename) const {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << std::scientific << std::setprecision(16);
        file << "# Twist boundary condition data for each dimension\n";
        file << "# Section 1: Axis-angle representation\n";
        file << "# Format: dimension axis_x axis_y axis_z angle(rad)\n";
        
        for (size_t d = 0; d < 3; ++d) {
            file << d << " ";
            for (Eigen::Index i = 0; i < rotation_axis[d].size(); ++i) {
                file << rotation_axis[d](i) << " ";
            }
            file << twist_angles[d] << "\n";
        }
        
        file << "\n# Section 2: Full SO(3) rotation matrices\n";
        file << "# Format: dimension followed by 3x3 matrix (row-major)\n";
        
        for (size_t d = 0; d < 3; ++d) {
            file << "# Dimension " << d << " twist matrix:\n";
            file << d << "\n";
            for (Eigen::Index row = 0; row < twist_matrices[d].rows(); ++row) {
                for (Eigen::Index col = 0; col < twist_matrices[d].cols(); ++col) {
                    file << twist_matrices[d](row, col);
                    if (col < twist_matrices[d].cols() - 1) file << " ";
                }
                file << "\n";
            }
        }
        
        file.close();
    }

    /**
     * Save site positions to file
     */
    void save_positions(const string& filename) const {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << std::scientific << std::setprecision(16);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            file << site_positions[i](0) << " " 
                 << site_positions[i](1) << " "
                 << site_positions[i](2) << "\n";
        }
        
        file.close();
    }

    /**
     * Initialize spins from file
     */
    void read_spins_from_file(const string& filename) {
        load_spin_config(filename);
    }

    /**
     * Set a specific spin
     */
    void set_spin(size_t site_index, const SpinVector& spin_in) {
        if (site_index < lattice_size) {
            spins[site_index] = spin_in;
        }
    }

    /**
     * Get a specific spin
     */
    const SpinVector& get_spin(size_t site_index) const {
        return spins[site_index];
    }

    /**
     * Initialize with ferromagnetic configuration
     */
    void init_ferromagnetic(const SpinVector& direction) {
        SpinVector spin_aligned = direction.normalized() * spin_length;
        if (su3_cp2) classical_spin::su3::project_to_cp2(spin_aligned.data());   // closest pure state
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = spin_aligned;
        }
    }

    /**
     * Initialize with Néel (antiferromagnetic) configuration
     */
    void init_neel(const SpinVector& direction) {
        SpinVector spin_up = direction.normalized() * spin_length;
        SpinVector spin_down = -spin_up;
        
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            size_t i = idx / (N_atoms * dim2 * dim3);
            size_t j = (idx / (N_atoms * dim3)) % dim2;
            size_t k = (idx / N_atoms) % dim3;
            
            spins[idx] = ((i + j + k) % 2 == 0) ? spin_up : spin_down;
        }
    }

    /**
     * Initialize with random spins
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size; ++i) random_state_into(spins[i].data());
    }

    /**
     * Print lattice information
     */
    void print_info() const {
        cout << "=== Lattice Information ===" << endl;
        cout << "Dimensions: " << dim1 << " × " << dim2 << " × " << dim3 << endl;
        cout << "Atoms per cell: " << N_atoms << endl;
        cout << "Total sites: " << lattice_size << endl;
        cout << "Spin dimension: " << spin_dim << endl;
        cout << "Spin length: " << spin_length << endl;
        cout << "Max bilinear neighbors per site: " << num_bi << endl;
        cout << "Max trilinear interactions per site: " << num_tri << endl;
        cout << "Current energy density: " << energy_density() << endl;
        cout << "Current magnetization: |M| = " << magnetization_global().norm() << endl;
    }

    // ============================================================
    // ADDITIONAL MAGNETIZATION OBSERVABLES
    // ============================================================

    /**
     * Compute local magnetization (simple average, no frame transformation)
     */
    SpinVector magnetization_local() const {
        SpinVector M = SpinVector::Zero(spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            M += spins[i];
        }
        return M / double(lattice_size);
    }

private:
    /**
     * Helper: Perform MC sweeps with optional overrelaxation
     * Returns sum of acceptance rates from metropolis calls
     * 
     * @param n_sweeps             Number of MC sweeps
     * @param T                    Temperature
     * @param gaussian_move        Use Gaussian moves
     * @param sigma                Gaussian width (modified in-place)
     * @param overrelaxation_rate  Overrelaxation frequency (0 = disabled)
     * @param boundary_update      Enable twist boundary updates
     * @param twist_sweep_count    Number of twist sweeps per MC sweep (default: 100)
     * @param twist_acc_ptr        Optional pointer to store cumulative twist acceptance count
     */
    double perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate = 0,
                            bool boundary_update = false,
                            size_t twist_sweep_count = 100,
                            size_t* twist_acc_ptr = nullptr);

    /**
     * Helper: Collect energy samples with regular MC sweeps
     */
    vector<double> collect_energy_samples(size_t n_samples, size_t interval,
                                         double T, bool gaussian_move, double& sigma,
                                         size_t overrelaxation_rate = 0);

    /**
     * Helper: Safely create directories if path is non-empty
     */
    static void ensure_directory_exists(const string& dir_path) {
        if (!dir_path.empty()) {
            std::filesystem::create_directories(dir_path);
        }
    }

    /**
     * Helper: Save energy and magnetization time series to files
     */
    void save_observables(const string& dir_path,
                         const vector<double>& energies,
                         const vector<SpinVector>& magnetizations);


public:
    /**
     * Magnetisation trajectory under a single Gaussian pulse centred at t_B
     * (field_in: one global-frame direction per sublattice), sampled at the
     * exact grid t_k = T_start + k step_size. Lattice::spins is the initial
     * state and is left unchanged.
     *
     * @param method  any name accepted by parse_ode_method (dynamics/ode_method.h)
     * @param pulse_window_chunking  ignored (kept for source compatibility):
     *        the trajectory is integrated in one pass on the global grid,
     *        which removed the seam bug of the old segment-by-segment scheme.
     */
    PumpProbeTrajectory single_pulse_drive(
               const vector<SpinVector>& field_in, double t_B,
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method = "dopri5", bool use_gpu = false,
               bool pulse_window_chunking = true,
               double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
               double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * As single_pulse_drive, with two pulses of equal amplitude, width and
     * frequency centred at t_B_1 and t_B_2. Pulses with different parameters
     * are built with make_pulse() and run through drive_trajectory().
     */
    PumpProbeTrajectory double_pulse_drive(
                   const vector<SpinVector>& field_in_1, double t_B_1,
                   const vector<SpinVector>& field_in_2, double t_B_2,
                   double pulse_amp, double pulse_width, double pulse_freq,
                   double T_start, double T_end, double step_size,
                   string method = "dopri5", bool use_gpu = false,
                   bool pulse_window_chunking = true,
                   double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                   double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    // ============================================================
    // 2DCS / pump-probe delay scans
    // ============================================================

    /**
     * Drive-free stationarity of the current configuration:
     * max_i |S_i × H_i| / max_i (|S_i| |H_i|), the sine of the largest torque
     * angle (0 for an exact equilibrium, scale-free in J and S). The W1 delay
     * scan reuses the reference trajectory only below `stationarity_tol`.
     */
    double stationarity_residual() const;

    /**
     * Largest component of dS/dt in the current configuration with the drive
     * switched off (absolute units of |H||S|).
     */
    double max_dSdt_norm_no_drive() const;

    /**
     * Complete pump-probe non-linear spectroscopy scan on the current
     * configuration (taken as the ground state):
     *   M0(t)      pump at t = 0,
     *   M1(t, τ)   probe at t = τ,
     *   M01(t, τ)  pump at 0 + probe at τ,
     * every trajectory on the same grid t_k = T_start + k T_step, so the
     * non-linear signal M_NL = M01 - M0 - M1 is defined sample by sample.
     * Delays are tau_start + i tau_step covering [tau_start, tau_end].
     *
     * W1 (reuse_m0_for_m1): when the configuration is stationary
     * (stationarity_residual() <= stationarity_tol), every τ is a multiple of
     * T_step and the probe window starts after T_start (T_start <= τ - 9 w),
     * M1(t, τ) is the τ-translate of one reference single-pulse trajectory
     * instead of a fresh integration. Delays that do not qualify are
     * integrated, so the result never depends on the optimisation.
     *
     * W2 (outer_omp_threads): OpenMP threads over τ (0 = all). The lattice is
     * shared read-only; no per-thread copies are made.
     *
     * Temp_start .. quench_sweeps are written as metadata only.
     * pulse_window_chunking is ignored (see single_pulse_drive).
     */
    void pump_probe_spectroscopy(const vector<SpinVector>& field_in,
                                 double pulse_amp, double pulse_width, double pulse_freq,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start = 5.0, double Temp_end = 1e-3,
                                 size_t n_anneal = 1000,
                                 bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                 string dir_name = "spectroscopy", string method = "dopri5",
                                 bool use_gpu = false,
                                 bool reuse_m0_for_m1 = true,
                                 double stationarity_tol = 1e-6,
                                 int outer_omp_threads = 0,
                                 bool pulse_window_chunking = true,
                                 double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                 double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * MPI-parallel version of pump_probe_spectroscopy over the ranks of
     * `comm`. Rank 0's configuration is the ground state (broadcast on
     * entry). Delays are handed out dynamically: every rank, rank 0 included,
     * computes; results stream to rank 0, which writes them as they arrive
     * (memory O(one trajectory) per rank). An exception on any rank is
     * propagated: every rank throws std::runtime_error after the scan, so no
     * rank is left blocked in a collective.
     */
    void pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start = 5.0, double Temp_end = 1e-3,
                                     size_t n_anneal = 1000,
                                     bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                     string dir_name = "spectroscopy", string method = "dopri5",
                                     bool use_gpu = false,
                                     bool reuse_m0_for_m1 = true,
                                     double stationarity_tol = 1e-6,
                                     bool pulse_window_chunking = true,
                                     double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                     double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol,
                                     MPI_Comm comm = MPI_COMM_WORLD);

private:
    /// Everything a delay scan needs, validated by make_scan_spec().
    struct PumpProbeScanSpec {
        vector<SpinVector> field;          // pulse polarisation, global frame, per sublattice
        double amp = 0, width = 1, freq = 0;
        vector<double> taus;
        double tau_start = 0, tau_end = 0, tau_step = 1;
        TimeGrid grid;
        DynamicsSettings settings;
        bool reuse_m0_for_m1 = true;
        double stationarity_tol = 1e-6;
        // metadata only
        double T_end_requested = 0, Temp_start = 0, Temp_end = 0;
        size_t n_anneal = 0, quench_sweeps = 0;
        bool T_zero_quench = false;
        string dir_name;
    };
    struct PumpProbeScan;  // per-scan state and the W1 reference (lattice_md.cpp)

    PumpProbeScanSpec make_scan_spec(const vector<SpinVector>& field_in,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start, double Temp_end, size_t n_anneal,
                                     bool T_zero_quench, size_t quench_sweeps,
                                     const string& dir_name, const string& method,
                                     bool reuse_m0_for_m1, double stationarity_tol,
                                     double abs_tol, double rel_tol) const;
    void run_pump_probe_scan(const PumpProbeScanSpec& spec, MPI_Comm comm, int outer_omp_threads) const;

    /// Copy root's dynamical state (spins, twist matrices and angles, damping,
    /// bath temperature) to every rank of comm, so all ranks integrate the
    /// same Hamiltonian from the same state.
    void broadcast_dynamical_state(MPI_Comm comm, int root);

    /**
     * W1 time translation of a reference single-pulse response R (flat, 3 *
     * spin_dim values per sample, sample j at t0 + (ref_j0 + j) dt of the scan
     * grid): out[k] = R(t_k - τ), τ = shift * dt, for k < n; samples before
     * the reference starts are the ground-state `baseline`. The caller
     * guarantees that R started from the ground state before the pulse acted
     * and extends to t_{n-1} - τ.
     */
    void synthesize_probe_response(const vector<double>& ref, long ref_j0, const double* baseline,
                                   long shift, size_t n, double* out) const;
#ifdef HDF5_ENABLED
    std::unique_ptr<HDF5PumpProbeWriter> open_scan_writer(const PumpProbeScanSpec& spec, double E_ground,
                                                          const SpinVector& M_ground) const;
#endif

public:

    // Note: GPU-accelerated methods use the modular GPU implementation in lattice_gpu.cuh/cu
    // For C++ compilation, use_gpu parameter will automatically fallback to CPU implementation.

    /**
     * Return true if any site has a trilinear coupling configured.
     *
     * The GPU code path in `src/gpu/lattice_gpu.cu::compute_local_field_device`
     * currently does not include a trilinear term, and
     * `create_gpu_lattice_data_internal` never uploads the trilinear tables.
     * Callers therefore must refuse to run on GPU when any trilinear coupling
     * is present, otherwise the simulation silently drops real physics.
     */
    bool has_trilinear_interactions() const {
        for (size_t i = 0; i < trilinear_partners.size(); ++i) {
            if (!trilinear_partners[i].empty()) return true;
        }
        return false;
    }

    /**
     * Whether the GPU right-hand side implements the current model. The GPU
     * kernels have no trilinear term, no damping, no Langevin noise and no
     * twisted boundaries, and handle spin_dim 3 and 8 (Gell-Mann bracket
     * only); the dynamics
     * drivers fall back to the CPU (with a warning naming `reason`) instead of
     * silently integrating a different equation of motion.
     */
    bool gpu_supports_model(string& reason) const {
        if (spin_dim != 3 && spin_dim != 8) reason = "spin_dim " + std::to_string(spin_dim);
        else if (has_trilinear_interactions()) reason = "trilinear couplings";
        else if (alpha_gilbert != 0.0) reason = "Gilbert damping";
        else if (langevin_temperature != 0.0) reason = "Langevin bath";
        else if (twist_active) reason = "twisted boundaries";
        // The device RHS hard-codes the Gell-Mann bracket c = 2.
        else if (spin_dim == 8 && unit_cell.poisson_bracket != 2.0) reason = "SU(3) legacy bracket convention";
        else return true;
        return false;
    }


// =============================================================================
// GPU backend glue (opaque API of lattice_gpu_api.h; untested here: no CUDA
// toolchain in CI). Methods and tolerances follow gpu::integrate_gpu.
// =============================================================================
#ifdef CUDA_ENABLED
private:
    // Device copy of the Hamiltonian, uploaded on first GPU use. Freed by the
    // destructor; a copied Lattice starts without one (DeviceHandle never
    // shares a handle between objects).
    mutable classical_spin::gpu::DeviceHandle<gpu::GPULatticeDataHandle, &gpu::destroy_gpu_lattice_data> gpu_handle_;
    
    /**
     * Ensure GPU lattice data is initialized (lazy initialization)
     * Uses the opaque API from lattice_gpu_api.h
     */
    void ensure_gpu_data_initialized() const {
        if (gpu_handle_) return;

        if (has_trilinear_interactions()) {
            throw std::runtime_error(
                "Lattice::ensure_gpu_data_initialized: this lattice has "
                "trilinear couplings, but the GPU code path in "
                "src/gpu/lattice_gpu.cu does not implement them yet. "
                "Set use_gpu=false, or run on a Hamiltonian without "
                "trilinear terms. (See audit item T3.)");
        }

        // Flatten field data
        vector<double> flat_field;
        flat_field.reserve(lattice_size * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                flat_field.push_back(field[i](d));
            }
        }
        
        // Flatten onsite interaction matrices
        vector<double> flat_onsite;
        flat_onsite.reserve(lattice_size * spin_dim * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t r = 0; r < spin_dim; ++r) {
                for (size_t c = 0; c < spin_dim; ++c) {
                    flat_onsite.push_back(onsite_interaction[i](r, c));
                }
            }
        }
        
        // Flatten bilinear interaction data
        vector<double> flat_bilinear;
        vector<size_t> flat_partners;
        vector<size_t> num_bilinear_per_site;
        
        flat_bilinear.reserve(lattice_size * num_bi * spin_dim * spin_dim);
        flat_partners.reserve(lattice_size * num_bi);
        num_bilinear_per_site.reserve(lattice_size);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners[i].size());
            for (size_t n = 0; n < num_bi; ++n) {
                if (n < bilinear_partners[i].size()) {
                    flat_partners.push_back(bilinear_partners[i][n]);
                    for (size_t r = 0; r < spin_dim; ++r) {
                        for (size_t c = 0; c < spin_dim; ++c) {
                            flat_bilinear.push_back(bilinear_interaction[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim * spin_dim; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
        
        // Create GPU data using opaque API
        gpu_handle_.reset(gpu::create_gpu_lattice_data(
            lattice_size, spin_dim, N_atoms, num_bi,
            flat_field, flat_onsite, flat_bilinear,
            flat_partners, num_bilinear_per_site));
    }
    
    /**
     * Update GPU pulse parameters
     */
    void update_gpu_pulse() const {
        if (!gpu_handle_) return;
        
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms * spin_dim);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive[p].size(); ++d) {
                flat_field_drive.push_back(field_drive[p](d));
            }
        }
        
        gpu::set_gpu_pulse(
            gpu_handle_.get(),
            flat_field_drive,
            field_drive_amp,
            field_drive_width,
            field_drive_freq,
            t_pulse[0],
            t_pulse[1]
        );
    }
    
    /**
     * GPU version of molecular_dynamics using opaque API
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5", double abs_tol = 1e-6, double rel_tol = 1e-6) {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        return;
#else
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        cout << "Running molecular dynamics with GPU acceleration: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << " (GPU via API)" << endl;
        cout << "Step size: " << dt_initial << endl;
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::set_gpu_spins(gpu_handle_.get(), h_state);
        
        // Create HDF5 writer
        std::unique_ptr<HDF5MDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MDWriter>(
                hdf5_file, lattice_size, spin_dim, N_atoms, 
                dim1, dim2, dim3, method + "_gpu_api", 
                dt_initial, T_start, T_end, save_interval, spin_length, 
                &site_positions, 10000);
        }
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> trajectory;
        gpu::integrate_gpu(gpu_handle_.get(), T_start, T_end, dt_initial,
                          save_interval, trajectory, method, abs_tol, rel_tol);
        
        // Write trajectory to HDF5 (post-processing on CPU)
        size_t save_count = 0;
        for (const auto& [t, state_vec] : trajectory) {
            const array<SpinVector, 3> M = measure_magnetizations(state_vec.data());
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, M[0], M[1], M[2], state_vec.data());
                save_count++;
            }
            if (save_count % 10 == 0) {
                double E = total_energy_flat(state_vec.data()) / lattice_size;
                cout << "t=" << t << ", E/N=" << E << ", |M|=" << M[1].norm() << endl;
            }
        }
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "GPU molecular dynamics complete!" << endl;
#endif
    }

    /**
     * GPU batched τ-scan for 2DCS (untested: no CUDA toolchain in CI).
     *
     * Runs 1 + n_τ replicas in one batched GPU launch on an identical time
     * grid: replica 0 is M0 (second pulse pushed past the window), replica
     * b + 1 is M01 at τ_b. M1 is synthesised from M0 by time translation,
     * which is exact only when every delay qualifies for W1 with M0 itself as
     * the reference: stationary ground state, τ_b a multiple of T_step,
     * τ_b >= 0 and the probe window inside the run (T_start <= τ_b - 9 w).
     * Anything else is refused with std::invalid_argument (use the CPU path).
     * Output: the HDF5PumpProbeWriter schema of the CPU drivers.
     */
    void pump_probe_spectroscopy_gpu_batched(const PumpProbeScanSpec& spec) {
#ifndef HDF5_ENABLED
        (void) spec;
        throw std::runtime_error("pump_probe_spectroscopy_gpu_batched: HDF5 support is required");
#else
        if (has_trilinear_interactions()) {
            throw std::invalid_argument(
                "pump_probe_spectroscopy_gpu_batched: lattice has trilinear "
                "couplings which are not supported on GPU.");
        }
        const TimeGrid& grid = spec.grid;
        const size_t n_tau = spec.taus.size();
        const size_t D3 = 3 * spin_dim;
        if (!spec.reuse_m0_for_m1) {
            throw std::invalid_argument("pump_probe_spectroscopy_gpu_batched requires reuse_m0_for_m1 = true "
                                        "(M1 is synthesised from M0); use the CPU path otherwise");
        }
        const double residual = stationarity_residual();
        if (residual > spec.stationarity_tol) {
            throw std::invalid_argument("pump_probe_spectroscopy_gpu_batched: ground state not stationary "
                                        "(residual " + std::to_string(residual) + " > stationarity_tol " +
                                        std::to_string(spec.stationarity_tol) + "); use the CPU path");
        }
        const double half = classical_spin::dynamics::kPulseSupportWidths * spec.width;
        vector<long> shift(n_tau);
        for (size_t b = 0; b < n_tau; ++b) {
            const double r = spec.taus[b] / grid.dt;
            shift[b] = std::lround(r);
            if (std::abs(r - double(shift[b])) > 1e-6 || spec.taus[b] < 0.0 ||
                grid.t0 > spec.taus[b] - half) {
                throw std::invalid_argument("pump_probe_spectroscopy_gpu_batched: delay " +
                                            std::to_string(spec.taus[b]) + " cannot be synthesised from M0 "
                                            "(needs tau >= 0, tau a multiple of T_step and T_start <= tau - 9 w); "
                                            "use the CPU path");
            }
        }

        std::filesystem::create_directories(spec.dir_name);
        save_positions(spec.dir_name + "/positions.txt");
        save_spin_config(spec.dir_name + "/initial_spins.txt");
        const ODEState ground = spins_to_state(spins);
        vector<double> baseline(D3), scratch(N_atoms * spin_dim);
        measure_magnetizations(ground.data(), baseline.data(), scratch.data());

        std::cout << "[GPU batched 2DCS] " << (n_tau + 1) << " replicas (1 reference + "
                  << n_tau << " delays), " << spec.settings.method << std::endl;
        const double t_pulse2_disabled = grid.t_end() + 100.0 * std::max(spec.width, 1.0);
        std::vector<double> batch_tau2(n_tau + 1);
        batch_tau2[0] = t_pulse2_disabled;
        for (size_t b = 0; b < n_tau; ++b) batch_tau2[b + 1] = spec.taus[b];

        set_pulse(spec.field, 0.0, spec.field, 0.0 /*per-replica*/, spec.amp, spec.width, spec.freq);
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        std::vector<double> flat_init(ground.begin(), ground.end());
        std::vector<double> flat_afm(afm_sublattice_signs.begin(), afm_sublattice_signs.end());
        std::vector<double> flat_frames(N_atoms * spin_dim * spin_dim);
        for (size_t a = 0; a < N_atoms; ++a)
            for (size_t r = 0; r < spin_dim; ++r)
                for (size_t c = 0; c < spin_dim; ++c)
                    flat_frames[(a * spin_dim + r) * spin_dim + c] = sublattice_frames[a](r, c);
        gpu::BatchedMagResult batched = gpu::integrate_gpu_batched(
            gpu_handle_.get(), flat_init, batch_tau2, flat_afm, flat_frames,
            grid.t0, grid.t_end(), grid.dt, /*save_interval=*/1, spec.settings.method,
            spec.settings.abs_tol, spec.settings.rel_tol);
        clear_pulse();
        if (batched.n_time_points != grid.n || batched.B != n_tau + 1) {
            throw std::runtime_error("pump_probe_spectroscopy_gpu_batched: GPU returned " +
                                     std::to_string(batched.n_time_points) + " samples x " +
                                     std::to_string(batched.B) + " replicas, expected " +
                                     std::to_string(grid.n) + " x " + std::to_string(n_tau + 1));
        }
        auto replica = [&](size_t b) {
            vector<double> flat(grid.n * D3);
            for (size_t k = 0; k < grid.n; ++k)
                std::copy_n(batched.mag_data.data() + (k * batched.B + b) * D3, D3, flat.data() + k * D3);
            return flat;
        };
        const vector<double> M0 = replica(0);
        auto writer = open_scan_writer(spec, energy_density(), magnetization_local());
        writer->write_reference_trajectory(to_trajectory(M0, grid));
        vector<double> M1(grid.n * D3);
        for (size_t b = 0; b < n_tau; ++b) {
            synthesize_probe_response(M0, 0, baseline.data(), shift[b], grid.n, M1.data());
            writer->write_tau_trajectory(int(b), spec.taus[b], to_trajectory(M1, grid),
                                         to_trajectory(replica(b + 1), grid));
        }
        writer->close();
        std::cout << "[GPU batched 2DCS] written to " << spec.dir_name << "/pump_probe_spectroscopy.h5" << std::endl;
#endif  // HDF5_ENABLED
    }

    /**
     * GPU version of single_pulse_drive using opaque API
     */
    vector<pair<double, array<SpinVector, 3>>> single_pulse_drive_gpu(
               const vector<SpinVector>& field_in, double t_B,
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method = "dopri5", double abs_tol = 1e-8, double rel_tol = 1e-8) {
        
        // Set up pulse
        set_pulse(field_in, t_B, vector<SpinVector>(N_atoms, SpinVector::Zero(spin_dim)), 
                 0.0, pulse_amp, pulse_width, pulse_freq);
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::set_gpu_spins(gpu_handle_.get(), h_state);
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        gpu::integrate_gpu(gpu_handle_.get(), T_start, T_end, step_size,
                          1, raw_trajectory, method, abs_tol, rel_tol);
        
        // Convert raw snapshots to the magnetisation trajectory
        PumpProbeTrajectory trajectory = trajectory_from_states(raw_trajectory);
        clear_pulse();
        return trajectory;
    }
    
    /**
     * GPU version of double_pulse_drive using opaque API
     */
    vector<pair<double, array<SpinVector, 3>>> double_pulse_drive_gpu(
                   const vector<SpinVector>& field_in_1, double t_B_1,
                   const vector<SpinVector>& field_in_2, double t_B_2,
                   double pulse_amp, double pulse_width, double pulse_freq,
                   double T_start, double T_end, double step_size,
                   string method = "dopri5", double abs_tol = 1e-8, double rel_tol = 1e-8) {
        
        // Set up two-pulse configuration
        set_pulse(field_in_1, t_B_1, field_in_2, t_B_2, 
                 pulse_amp, pulse_width, pulse_freq);
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::set_gpu_spins(gpu_handle_.get(), h_state);
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        gpu::integrate_gpu(gpu_handle_.get(), T_start, T_end, step_size,
                          1, raw_trajectory, method, abs_tol, rel_tol);
        
        // Convert raw snapshots to the magnetisation trajectory
        PumpProbeTrajectory trajectory = trajectory_from_states(raw_trajectory);
        clear_pulse();
        return trajectory;
    }
#endif // CUDA_ENABLED

};

#endif // LATTICE_REFACTORED_H
