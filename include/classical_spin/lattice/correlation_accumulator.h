#pragma once
/**
 * correlation_accumulator.h — equal-time spin and dimer correlations of a
 * periodic lattice, accumulated in Fourier space.
 *
 * Conventions (cell R = n1 a1 + n2 a2 + n3 a3, n_i in [0, L_i); site (R, s)
 * at r = R + τ_s; N_c = L1 L2 L3 cells; spins in the GLOBAL frame,
 * S_global = F_s S_local with F_s the sublattice frame):
 *
 *   F_{sα}(k)        = Σ_R S^α_{R,s} e^{-i k·R},       k = Σ_i (m_i / L_i) b_i,
 *   X_{ss'}^{αβ}(k)  = F_{sα}(k) F_{s'β}(k)^*          (one sample; averaged),
 *   S^{αβ}(q)        = (1/N_c) Σ_{ij} <S_i^α S_j^β> e^{-i q·(r_i - r_j)}
 *                    = (1/N_c) Σ_{ss'} e^{-i q·(τ_s - τ_s')} <X_{ss'}^{αβ}(k)>,
 *
 * for every commensurate q = k + G (G a reciprocal lattice vector): the
 * physical (neutron) structure factor per unit cell, Hermitian in (α, β); its
 * imaginary antisymmetric part is the chiral term. Off-grid q is rejected:
 * under periodic boundaries only commensurate q are defined.
 *
 * Sum rule (Parseval): Σ_k Σ_s Σ_α <X_{ss}^{αα}(k)> / N_c = N_c N_atoms S²,
 * i.e. N_atoms S² per cell; on a Bravais lattice this is Σ_q S(q) over the
 * N_c grid points of one Brillouin zone.
 *
 * Cost per sample: one FFT of the cell grid per (sublattice, component) — two
 * real fields packed into one complex transform — plus the cross spectra,
 * O(N log N + n_pairs d² N); no O(N_c²) tables. The cross spectra of real
 * fields obey X(-k) = X(k)^*, so only half of the k grid is stored (cut along
 * the last axis with L > 1). All sublattice pairs s <= s' and all component
 * pairs (α, β) are kept, complex: nothing is symmetrised away, and the s > s'
 * blocks follow from X_{s's}^{βα} = (X_{ss'}^{αβ})^*.
 *
 * Dimers. A bond class is one geometrically distinct bond of the unit cell,
 * (source, partner, offset) in the canonical orientation of UnitCell::validate
 * (source < partner, or equal with the offset lexicographically positive);
 * its bond in cell R joins (R, source) and (R + offset, partner) and is centred
 * at R + ρ_c, ρ_c = τ_source + (bond vector)/2. The dimer field
 * D^α_c(R) = S^α_i S^α_j (global frame) is accumulated like the spins:
 *
 *   S_D^{α}_{cc'}(q) = (1/N_c) e^{-i q·(ρ_c - ρ_c')} <G_{cα}(k) G_{c'α}(k)^*>.
 *
 * bond_classes_from_unit_cell() is the ONLY map from bonds to classes.
 *
 * Error bars. Samples go into n_bins bins; when every bin is full, adjacent
 * pairs are merged and the bin size doubles, so the bins stay contiguous
 * blocks of the Markov chain whose length grows with the run (n_bins/2 to
 * n_bins complete blocks). Errors are delete-one-bin jackknife estimates
 * (unequal bin sizes weighted, Busing et al. 1999), valid for nonlinear
 * functions such as connected correlators.
 */

#include "classical_spin/core/fft.h"
#include "classical_spin/core/unitcell.h"

#include <Eigen/Dense>
#include <mpi.h>

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class RealSpaceCorrelationAccumulator {
public:
    using cplx = std::complex<double>;

    /** One geometrically distinct bond of the unit cell (see the file comment). */
    struct BondClass {
        std::size_t source = 0;
        std::size_t partner = 0;
        Eigen::Vector3i offset = Eigen::Vector3i::Zero();  ///< partner cell - source cell (lattice coordinates)
        Eigen::Vector3d center = Eigen::Vector3d::Zero();  ///< bond centre in the home cell
        double length = 0.0;
    };

    /** Lattice geometry the accumulator works on. */
    struct Geometry {
        std::size_t dim1 = 1, dim2 = 1, dim3 = 1;
        std::size_t n_sublattices = 1;
        std::size_t spin_dim = 3;
        std::array<Eigen::Vector3d, 3> lattice_vectors = {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                                                          Eigen::Vector3d::UnitZ()};
        std::vector<Eigen::Vector3d> positions;   ///< τ_s, one per sublattice
        std::vector<Eigen::MatrixXd> frames;      ///< F_s (spin_dim x spin_dim); empty = identity
        std::vector<BondClass> bond_classes;      ///< dimer bonds; empty = no dimer channel
    };

    struct Options {
        /// Bins for the jackknife errors (1 = no error bars; otherwise even, >= 2).
        std::size_t n_bins = 16;
    };

    /** Mean and jackknife error (real and imaginary parts separately) of a complex matrix. */
    struct Estimate {
        Eigen::MatrixXcd mean;
        Eigen::MatrixXcd error;   ///< NaN when fewer than two bins hold samples
        std::size_t n_bins = 0;   ///< bins that entered the jackknife
    };

    // ------------------------------------------------------------------
    // Bond classes (the single bond -> class map)
    // ------------------------------------------------------------------

    /**
     * Canonical orientation of the bond (source, R) -> (partner, R + offset):
     * swapped (and the offset negated) unless source < partner, or
     * source == partner with the offset lexicographically positive. Throws
     * std::invalid_argument for the zero bond (source == partner, offset 0).
     */
    static BondClass canonical_bond(std::size_t source, std::size_t partner, const Eigen::Vector3i& offset,
                                    const std::vector<Eigen::Vector3d>& positions,
                                    const std::array<Eigen::Vector3d, 3>& lattice_vectors);

    /**
     * Bond classes of the unit cell's bilinear couplings, each bond once, in
     * neighbour shell `shell` of the coupled bonds (1 = the shortest coupled
     * bonds, 2 = the next length, ...; 0 = every coupled bond). Sorted by
     * (length, source, partner, offset).
     */
    static std::vector<BondClass> bond_classes_from_unit_cell(const UnitCell& uc, std::size_t shell = 1);

    // ------------------------------------------------------------------
    // Set-up
    // ------------------------------------------------------------------

    RealSpaceCorrelationAccumulator() = default;

    /** Validates the geometry (std::invalid_argument naming the field) and clears all data. */
    void initialize(const Geometry& geometry, const Options& options);
    void initialize(const Geometry& geometry) { initialize(geometry, Options()); }

    /**
     * Legacy set-up: identity frames, no dimer channel. `n_bonds` is ignored
     * (bond classes now come from the unit cell, see bond_classes_from_unit_cell).
     */
    void initialize(std::size_t d1, std::size_t d2, std::size_t d3, std::size_t n_sub, std::size_t n_bonds,
                    std::size_t sdim, const std::array<Eigen::Vector3d, 3>& lattice_vectors,
                    const std::vector<Eigen::Vector3d>& sublattice_positions);

    bool initialized() const { return initialized_; }

    // ------------------------------------------------------------------
    // Sampling (spins in the LOCAL frame, site index ((n1 L2 + n2) L3 + n3) N_atoms + s)
    // ------------------------------------------------------------------

    /** Add one sample to the spin channel. */
    void accumulate_spin_correlations(const std::vector<Eigen::VectorXd>& spins);
    /** Add one sample to the dimer channel (throws if the accumulator has no bond classes). */
    void accumulate_dimer_correlations(const std::vector<Eigen::VectorXd>& spins);
    /** Both channels (the dimer channel only when bond classes exist). */
    void add_sample(const std::vector<Eigen::VectorXd>& spins);

    // ------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------

    std::size_t n_samples() const { return spin_.n_samples; }
    std::size_t n_dimer_samples() const { return dimer_.n_samples; }
    std::size_t n_cells() const { return n_cells_; }
    std::size_t n_sublattices() const { return n_sub_; }
    std::size_t spin_dim() const { return n_comp_; }
    std::array<std::size_t, 3> dims() const { return dims_; }
    std::size_t n_bins() const { return options_.n_bins; }
    const std::vector<BondClass>& bond_classes() const { return classes_; }
    const std::array<Eigen::Vector3d, 3>& reciprocal_vectors() const { return recip_; }

    /** Wave vector of grid index (m1, m2, m3): Σ_i (m_i / L_i) b_i (any integers, zone extensions included). */
    Eigen::Vector3d grid_wavevector(long m1, long m2, long m3) const;
    /**
     * Grid index of a commensurate q, m_i = q·a_i L_i / 2π (not reduced).
     * Throws std::invalid_argument when q is not on the grid (to 1e-8).
     */
    std::array<long, 3> grid_index(const Eigen::Vector3d& q) const;

    /**
     * S^{αβ}(q) per unit cell (spin_dim x spin_dim, Hermitian), global frame.
     * connected: subtract the disconnected part N_c Σ_{ss'} e^{-iq·(τ_s-τ_s')} <m_s^α><m_s'^β>
     * (present only at q = G), m_s the sublattice magnetisation per cell.
     */
    Eigen::MatrixXcd structure_factor(const Eigen::Vector3d& q, bool connected = false) const;
    Estimate structure_factor_estimate(const Eigen::Vector3d& q, bool connected = false) const;

    /** Legacy: the real (symmetric, neutron) part of S^{αβ}(q); needs spin_dim == 3. */
    Eigen::Matrix3d compute_Sq(const Eigen::Vector3d& q, bool connected = true) const;
    /** Legacy: real part of S(q) and its jackknife error. */
    std::pair<Eigen::Matrix3d, Eigen::Matrix3d> compute_Sq_with_error(const Eigen::Vector3d& q,
                                                                      bool connected = true) const;

    /**
     * C_{ss'}^{αβ}(d) = (1/N_c) Σ_R <S^α_{R,s} S^β_{R+d,s'}> for every cell
     * displacement d (row-major over (d1, d2, d3) in [0, L_i)), by inverse FFT
     * of the averaged cross spectra. connected subtracts <m_s^α><m_s'^β>.
     */
    std::vector<double> real_space_correlation(std::size_t s, std::size_t s2, std::size_t alpha,
                                               std::size_t beta, bool connected = false) const;

    /** Mean global-frame magnetisation per cell of sublattice s. */
    Eigen::VectorXd sublattice_magnetization(std::size_t s) const;
    /** Mean dimer <D^α_c>. */
    double dimer_mean(std::size_t bond_class, std::size_t alpha) const;

    /**
     * Dimer structure factor, one n_classes x n_classes Hermitian matrix per
     * spin component α. connected subtracts N_c e^{-iq·(ρ_c-ρ_c')} <D^α_c><D^α_c'> at q = G.
     */
    std::vector<Eigen::MatrixXcd> dimer_structure_factor(const Eigen::Vector3d& q, bool connected = true) const;
    /** Jackknife estimate of dimer_structure_factor() (one Estimate per component). */
    std::vector<Estimate> dimer_structure_factor_estimate(const Eigen::Vector3d& q, bool connected = true) const;
    /** Legacy: real parts of dimer_structure_factor(); needs spin_dim == 3. */
    std::array<Eigen::MatrixXd, 3> compute_Sq_dimer(const Eigen::Vector3d& q, bool connected = true) const;
    /** Legacy: Σ_α of compute_Sq_dimer(). */
    Eigen::MatrixXd compute_Sq_dimer_total(const Eigen::Vector3d& q, bool connected = true) const;

    // ------------------------------------------------------------------
    // Bookkeeping
    // ------------------------------------------------------------------

    /** Clear all samples (geometry kept). */
    void reset();

    /**
     * Add another accumulator's samples bin by bin (independent runs at the
     * SAME temperature, e.g. other seeds). Every mean and spectrum is merged.
     * An uninitialised *this becomes a copy of `other`; an uninitialised
     * `other` adds nothing; any geometry mismatch throws std::invalid_argument.
     */
    void merge(const RealSpaceCorrelationAccumulator& other);

    /**
     * Collective over `comm`: sum every rank's samples onto rank 0 (MPI_Reduce
     * of one packed double buffer). Ranks must hold the same geometry —
     * checked collectively first, every rank throws std::invalid_argument on a
     * mismatch. Only combine replicas at the same temperature. Ranks other
     * than 0 keep their own data.
     */
    void mpi_reduce(MPI_Comm comm = MPI_COMM_WORLD);

    /** Bytes held by every buffer of the accumulator (capacities). */
    std::size_t storage_bytes() const;

#ifdef HDF5_ENABLED
    /**
     * Write geometry, conventions, the mean and per-bin cross spectra, the
     * structure factor on the first-zone grid with errors and the dimer data
     * to `filename` (truncated). Throws std::runtime_error on HDF5 failure.
     */
    void save_hdf5(const std::string& filename, const std::string& group_name = "/correlations") const;
#endif

    /**
     * Text table of S(q) on q = q1 b1 + q2 b2 + q3 b3 for n_q points per
     * range (end points included). Every q must be commensurate (q_i L_i
     * integer): std::invalid_argument otherwise. Columns: q1 q2 q3 qx qy qz
     * S_total Re S_xx Re S_yy Re S_zz Re S_xy Re S_xz Re S_yz (spin_dim 3), or
     * q1 q2 q3 qx qy qz S_total for other spin dimensions.
     */
    void save_structure_factor_grid(const std::string& filename, std::pair<double, double> q1_range,
                                    std::pair<double, double> q2_range, std::pair<double, double> q3_range,
                                    std::size_t n_q1, std::size_t n_q2, std::size_t n_q3,
                                    const Eigen::Vector3d& b1, const Eigen::Vector3d& b2,
                                    const Eigen::Vector3d& b3, bool connected = true) const;

private:
    /// Binned accumulation of cross spectra (complex, half k grid) and means.
    struct Channel {
        std::size_t spectrum_size = 0;   ///< complex entries per bin
        std::size_t mean_size = 0;       ///< real entries per bin
        std::size_t bin_size = 1;        ///< samples per complete bin
        std::size_t current = 0;         ///< bin being filled
        std::size_t n_samples = 0;
        std::vector<std::vector<cplx>> spectra;
        std::vector<std::vector<double>> means;
        std::vector<std::size_t> counts;

        void allocate(std::size_t n_bins, std::size_t spectrum, std::size_t mean);
        void clear();
        void finish_sample(std::size_t n_bins);   ///< count the sample, rebin when full
        std::size_t bytes() const;
    };

    struct HalfIndex {
        std::size_t index;   ///< into the half grid
        bool conjugate;      ///< X(k) = conj(X(-k)) with -k stored
    };

    void require_initialized(const char* what) const;
    void check_spins(const std::vector<Eigen::VectorXd>& spins, const char* what) const;
    void to_global(const std::vector<Eigen::VectorXd>& spins);
    void transform_fields(std::size_t n_fields, const double* fields);   ///< fields [n_fields][N_c] -> F_half_
    HalfIndex half_index(const std::array<long, 3>& m) const;
    std::array<long, 3> reduce(const std::array<long, 3>& m) const;
    std::size_t pair_index(std::size_t s, std::size_t s2) const;   ///< s <= s2
    std::size_t class_pair_index(std::size_t c, std::size_t c2) const;   ///< c <= c2

    /// Spin cross-spectrum block X_{ss'}^{αβ}(k) at half index h from summed
    /// spectrum `spec` (any s, s'; conjugated for -k).
    cplx spin_block(const cplx* spec, const HalfIndex& h, std::size_t s, std::size_t s2, std::size_t a,
                    std::size_t b) const;
    /// S(q) from summed spectrum and means over `n` samples.
    Eigen::MatrixXcd spin_sq_from(const std::vector<cplx>& spec, const std::vector<double>& mean, double n,
                                  const Eigen::Vector3d& q, const HalfIndex& h, bool zone_center,
                                  bool connected) const;
    std::vector<Eigen::MatrixXcd> dimer_sq_from(const std::vector<cplx>& spec, const std::vector<double>& mean,
                                                double n, const Eigen::Vector3d& q, const HalfIndex& h,
                                                bool zone_center, bool connected) const;
    /// Jackknife over the bins of `ch` of a matrix-valued estimator f(spec, mean, n).
    template <class F>
    std::vector<Estimate> jackknife(const Channel& ch, std::size_t n_out, F&& f) const;
    void totals(const Channel& ch, std::vector<cplx>& spec, std::vector<double>& mean) const;
    std::vector<double> geometry_signature() const;

    // Geometry
    bool initialized_ = false;
    Options options_;
    std::array<std::size_t, 3> dims_ = {1, 1, 1};
    std::array<std::size_t, 3> half_dims_ = {1, 1, 1};
    int half_axis_ = -1;            ///< axis cut in half (-1: none, N_c == 1)
    std::size_t n_cells_ = 0, n_half_ = 0, n_sub_ = 0, n_comp_ = 0, n_sites_ = 0, n_pairs_ = 0;
    std::array<Eigen::Vector3d, 3> lattice_vectors_;
    std::array<Eigen::Vector3d, 3> recip_;
    std::vector<Eigen::Vector3d> positions_;
    std::vector<Eigen::MatrixXd> frames_;
    bool identity_frames_ = true;
    std::vector<BondClass> classes_;
    std::size_t n_class_pairs_ = 0;
    std::vector<std::size_t> partner_site_;   ///< [class][cell] -> partner site
    std::vector<std::size_t> half_to_full_;   ///< half index -> full cell index of k
    std::vector<std::size_t> neg_full_;       ///< full index of k -> full index of -k

    // Data
    Channel spin_, dimer_;

    // Scratch (reused across samples)
    classical_spin::fft::Plan3D plan_;
    std::vector<double> global_;     ///< [site][component]
    std::vector<double> fields_;     ///< [field][cell]
    std::vector<cplx> work_;         ///< N_c
    std::vector<cplx> F_half_;       ///< [field][half k]
};
