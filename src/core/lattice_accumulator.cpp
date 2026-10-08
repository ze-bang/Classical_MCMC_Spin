/**
 * @file lattice_accumulator.cpp
 * @brief RealSpaceCorrelationAccumulator: FFT-based equal-time spin and dimer
 *        correlations (conventions in correlation_accumulator.h).
 *
 * The previous real-space kernels cost O(N_c^2) per sample (an N_c x N_c
 * displaced-cell table and a sum over all cell pairs per displacement: 1.9 s
 * per sample on an L = 12 pyrochlore, 1.5 GB of table at L = 24), correlated
 * local-frame spins, symmetrised the (α, β) and (s, s') blocks away and folded
 * displacements into [0, L), which made S(q) wrong off the commensurate grid.
 * Accumulating cross spectra of the per-sublattice FFTs gives the same
 * information exactly in O(N log N), in the global frame, without losing the
 * antisymmetric (chiral) part.
 */

#include "classical_spin/lattice/correlation_accumulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace {

using cplx = std::complex<double>;

bool lexicographically_positive(const Eigen::Vector3i& n) {
    for (int d = 0; d < 3; ++d)
        if (n[d] != 0) return n[d] > 0;
    return false;
}

long euclid_mod(long a, long L) {
    const long r = a % L;
    return r < 0 ? r + L : r;
}

cplx phase(double angle) { return cplx(std::cos(angle), std::sin(angle)); }

}  // namespace

// ============================================================================
// Bond classes
// ============================================================================

RealSpaceCorrelationAccumulator::BondClass RealSpaceCorrelationAccumulator::canonical_bond(
    std::size_t source, std::size_t partner, const Eigen::Vector3i& offset,
    const std::vector<Eigen::Vector3d>& positions, const std::array<Eigen::Vector3d, 3>& lattice_vectors) {
    if (source >= positions.size() || partner >= positions.size())
        throw std::invalid_argument("canonical_bond: sublattice index out of range");
    if (source == partner && offset.isZero())
        throw std::invalid_argument("canonical_bond: a bond from a site to itself (zero offset) is not a bond");
    BondClass c;
    c.source = source;
    c.partner = partner;
    c.offset = offset;
    if (source > partner || (source == partner && !lexicographically_positive(offset))) {
        std::swap(c.source, c.partner);
        c.offset = -offset;
    }
    const Eigen::Vector3d bond = double(c.offset[0]) * lattice_vectors[0] + double(c.offset[1]) * lattice_vectors[1] +
                                 double(c.offset[2]) * lattice_vectors[2] + positions[c.partner] -
                                 positions[c.source];
    c.center = positions[c.source] + 0.5 * bond;
    c.length = bond.norm();
    return c;
}

std::vector<RealSpaceCorrelationAccumulator::BondClass> RealSpaceCorrelationAccumulator::bond_classes_from_unit_cell(
    const UnitCell& uc, std::size_t shell) {
    const std::array<Eigen::Vector3d, 3> lv = {uc.lattice_vectors[0], uc.lattice_vectors[1], uc.lattice_vectors[2]};
    // Several couplings on one bond (or one bond declared from either end)
    // are one class.
    std::map<std::tuple<std::size_t, std::size_t, int, int, int>, BondClass> unique;
    for (const auto& [src, bi] : uc.bilinear_interaction) {
        if (std::size_t(src) == bi.partner && bi.offset.isZero()) continue;  // not a bond
        const BondClass c = canonical_bond(std::size_t(src), bi.partner, bi.offset, uc.lattice_pos, lv);
        unique.emplace(std::make_tuple(c.source, c.partner, c.offset[0], c.offset[1], c.offset[2]), c);
    }
    std::vector<BondClass> all;
    for (const auto& kv : unique) all.push_back(kv.second);
    const double tol = 1e-8;
    auto same_length = [tol](double a, double b) { return std::abs(a - b) <= tol * std::max(1.0, a); };
    // Stable: equal lengths keep the (source, partner, offset) order of the map.
    std::stable_sort(all.begin(), all.end(), [&](const BondClass& a, const BondClass& b) {
        return !same_length(a.length, b.length) && a.length < b.length;
    });
    if (shell == 0) return all;
    std::vector<double> shells;
    for (const BondClass& c : all)
        if (shells.empty() || !same_length(c.length, shells.back())) shells.push_back(c.length);
    if (shell > shells.size()) return {};
    std::vector<BondClass> out;
    for (const BondClass& c : all)
        if (same_length(c.length, shells[shell - 1])) out.push_back(c);
    return out;
}

// ============================================================================
// Channel (binned accumulation)
// ============================================================================

void RealSpaceCorrelationAccumulator::Channel::allocate(std::size_t n_bins, std::size_t spectrum, std::size_t mean) {
    spectrum_size = spectrum;
    mean_size = mean;
    spectra.assign(n_bins, std::vector<cplx>(spectrum, cplx(0.0)));
    means.assign(n_bins, std::vector<double>(mean, 0.0));
    counts.assign(n_bins, 0);
    bin_size = 1;
    current = 0;
    n_samples = 0;
}

void RealSpaceCorrelationAccumulator::Channel::clear() {
    for (auto& s : spectra) std::fill(s.begin(), s.end(), cplx(0.0));
    for (auto& m : means) std::fill(m.begin(), m.end(), 0.0);
    std::fill(counts.begin(), counts.end(), std::size_t(0));
    bin_size = 1;
    current = 0;
    n_samples = 0;
}

void RealSpaceCorrelationAccumulator::Channel::finish_sample(std::size_t n_bins) {
    ++counts[current];
    ++n_samples;
    if (n_bins < 2 || counts[current] < bin_size) return;
    if (++current < n_bins) return;
    // Every bin is full: merge neighbours, so the bins stay contiguous blocks
    // of the chain, each now holding 2 * bin_size samples.
    const std::size_t half = n_bins / 2;
    for (std::size_t i = 0; i < half; ++i) {
        for (std::size_t j = 0; j < spectrum_size; ++j) spectra[i][j] = spectra[2 * i][j] + spectra[2 * i + 1][j];
        for (std::size_t j = 0; j < mean_size; ++j) means[i][j] = means[2 * i][j] + means[2 * i + 1][j];
        counts[i] = counts[2 * i] + counts[2 * i + 1];
    }
    for (std::size_t i = half; i < n_bins; ++i) {
        std::fill(spectra[i].begin(), spectra[i].end(), cplx(0.0));
        std::fill(means[i].begin(), means[i].end(), 0.0);
        counts[i] = 0;
    }
    current = half;
    bin_size *= 2;
}

std::size_t RealSpaceCorrelationAccumulator::Channel::bytes() const {
    std::size_t b = counts.capacity() * sizeof(std::size_t);
    for (const auto& s : spectra) b += s.capacity() * sizeof(cplx);
    for (const auto& m : means) b += m.capacity() * sizeof(double);
    return b;
}

// ============================================================================
// Set-up
// ============================================================================

void RealSpaceCorrelationAccumulator::initialize(const Geometry& g, const Options& options) {
    const std::string who = "RealSpaceCorrelationAccumulator: ";
    if (g.dim1 == 0 || g.dim2 == 0 || g.dim3 == 0) throw std::invalid_argument(who + "dimensions must be >= 1");
    if (g.n_sublattices == 0) throw std::invalid_argument(who + "n_sublattices must be >= 1");
    if (g.spin_dim == 0) throw std::invalid_argument(who + "spin_dim must be >= 1");
    if (g.positions.size() != g.n_sublattices)
        throw std::invalid_argument(who + "positions must hold one entry per sublattice (" +
                                    std::to_string(g.positions.size()) + " given, n_sublattices = " +
                                    std::to_string(g.n_sublattices) + ")");
    for (const auto& p : g.positions)
        if (!p.allFinite()) throw std::invalid_argument(who + "non-finite sublattice position");
    if (!g.frames.empty()) {
        if (g.frames.size() != g.n_sublattices)
            throw std::invalid_argument(who + "frames must be empty (identity) or one per sublattice");
        for (const auto& F : g.frames)
            if (std::size_t(F.rows()) != g.spin_dim || std::size_t(F.cols()) != g.spin_dim || !F.allFinite())
                throw std::invalid_argument(who + "each frame must be a finite spin_dim x spin_dim matrix");
    }
    Eigen::Matrix3d A;
    for (int d = 0; d < 3; ++d) A.col(d) = g.lattice_vectors[d];
    if (!A.allFinite() || !(std::abs(A.determinant()) > 1e-12))
        throw std::invalid_argument(who + "lattice vectors are degenerate");
    if (options.n_bins == 0 || (options.n_bins > 1 && options.n_bins % 2 != 0))
        throw std::invalid_argument(who + "n_bins must be 1 or an even number >= 2");
    {
        std::map<std::tuple<std::size_t, std::size_t, int, int, int>, int> seen;
        for (const BondClass& c : g.bond_classes) {
            const BondClass k = canonical_bond(c.source, c.partner, c.offset, g.positions, g.lattice_vectors);
            if (k.source != c.source || k.partner != c.partner || k.offset != c.offset)
                throw std::invalid_argument(who + "bond class not in canonical orientation "
                                                  "(use canonical_bond / bond_classes_from_unit_cell)");
            if (!seen.emplace(std::make_tuple(c.source, c.partner, c.offset[0], c.offset[1], c.offset[2]), 0).second)
                throw std::invalid_argument(who + "duplicate bond class");
        }
    }

    options_ = options;
    dims_ = {g.dim1, g.dim2, g.dim3};
    n_cells_ = g.dim1 * g.dim2 * g.dim3;
    n_sub_ = g.n_sublattices;
    n_comp_ = g.spin_dim;
    n_sites_ = n_cells_ * n_sub_;
    n_pairs_ = n_sub_ * (n_sub_ + 1) / 2;
    lattice_vectors_ = g.lattice_vectors;
    const Eigen::Matrix3d Ainv = A.inverse();
    for (int d = 0; d < 3; ++d) recip_[d] = 2.0 * M_PI * Ainv.row(d).transpose();   // b_i . a_j = 2π δ_ij
    positions_ = g.positions;
    identity_frames_ = true;
    frames_.assign(n_sub_, Eigen::MatrixXd::Identity(n_comp_, n_comp_));
    if (!g.frames.empty()) {
        frames_ = g.frames;
        for (const auto& F : frames_)
            if (!F.isIdentity(0.0)) identity_frames_ = false;
    }
    // Centres and lengths always follow from the geometry (one definition).
    classes_.clear();
    for (const BondClass& c : g.bond_classes)
        classes_.push_back(canonical_bond(c.source, c.partner, c.offset, positions_, lattice_vectors_));
    n_class_pairs_ = classes_.size() * (classes_.size() + 1) / 2;

    // Half grid: cut the last axis that has more than one cell.
    half_axis_ = -1;
    for (int d = 2; d >= 0; --d)
        if (dims_[d] > 1) {
            half_axis_ = d;
            break;
        }
    half_dims_ = dims_;
    if (half_axis_ >= 0) half_dims_[half_axis_] = dims_[half_axis_] / 2 + 1;
    n_half_ = half_dims_[0] * half_dims_[1] * half_dims_[2];
    half_to_full_.assign(n_half_, 0);
    for (std::size_t a = 0; a < half_dims_[0]; ++a)
        for (std::size_t b = 0; b < half_dims_[1]; ++b)
            for (std::size_t c = 0; c < half_dims_[2]; ++c)
                half_to_full_[(a * half_dims_[1] + b) * half_dims_[2] + c] = (a * dims_[1] + b) * dims_[2] + c;
    neg_full_.assign(n_cells_, 0);
    for (std::size_t a = 0; a < dims_[0]; ++a)
        for (std::size_t b = 0; b < dims_[1]; ++b)
            for (std::size_t c = 0; c < dims_[2]; ++c) {
                const std::size_t na = (dims_[0] - a) % dims_[0], nb = (dims_[1] - b) % dims_[1],
                                  nc = (dims_[2] - c) % dims_[2];
                neg_full_[(a * dims_[1] + b) * dims_[2] + c] = (na * dims_[1] + nb) * dims_[2] + nc;
            }

    partner_site_.assign(classes_.size() * n_cells_, 0);
    for (std::size_t k = 0; k < classes_.size(); ++k) {
        const BondClass& c = classes_[k];
        for (std::size_t a = 0; a < dims_[0]; ++a)
            for (std::size_t b = 0; b < dims_[1]; ++b)
                for (std::size_t e = 0; e < dims_[2]; ++e) {
                    const long pa = euclid_mod(long(a) + c.offset[0], long(dims_[0]));
                    const long pb = euclid_mod(long(b) + c.offset[1], long(dims_[1]));
                    const long pe = euclid_mod(long(e) + c.offset[2], long(dims_[2]));
                    const std::size_t cell = (a * dims_[1] + b) * dims_[2] + e;
                    const std::size_t pcell = (std::size_t(pa) * dims_[1] + std::size_t(pb)) * dims_[2] + std::size_t(pe);
                    partner_site_[k * n_cells_ + cell] = pcell * n_sub_ + c.partner;
                }
    }

    spin_.allocate(options_.n_bins, n_half_ * n_pairs_ * n_comp_ * n_comp_, n_sub_ * n_comp_);
    if (classes_.empty())
        dimer_ = Channel();
    else
        dimer_.allocate(options_.n_bins, n_half_ * n_class_pairs_ * n_comp_, classes_.size() * n_comp_);

    plan_ = classical_spin::fft::Plan3D(dims_[0], dims_[1], dims_[2]);
    global_.assign(n_sites_ * n_comp_, 0.0);
    fields_.clear();
    work_.assign(n_cells_, cplx(0.0));
    F_half_.clear();
    initialized_ = true;
}

void RealSpaceCorrelationAccumulator::initialize(std::size_t d1, std::size_t d2, std::size_t d3, std::size_t n_sub,
                                                 std::size_t /*n_bonds*/, std::size_t sdim,
                                                 const std::array<Eigen::Vector3d, 3>& lattice_vectors,
                                                 const std::vector<Eigen::Vector3d>& sublattice_positions) {
    Geometry g;
    g.dim1 = d1;
    g.dim2 = d2;
    g.dim3 = d3;
    g.n_sublattices = n_sub;
    g.spin_dim = sdim;
    g.lattice_vectors = lattice_vectors;
    g.positions = sublattice_positions;
    initialize(g);
}

// ============================================================================
// Sampling
// ============================================================================

void RealSpaceCorrelationAccumulator::require_initialized(const char* what) const {
    if (!initialized_)
        throw std::logic_error(std::string("RealSpaceCorrelationAccumulator::") + what +
                               ": accumulator not initialized");
}

void RealSpaceCorrelationAccumulator::check_spins(const std::vector<Eigen::VectorXd>& spins, const char* what) const {
    if (spins.size() != n_sites_)
        throw std::invalid_argument(std::string("RealSpaceCorrelationAccumulator::") + what + ": got " +
                                    std::to_string(spins.size()) + " spins, expected " + std::to_string(n_sites_));
    for (const auto& s : spins)
        if (std::size_t(s.size()) != n_comp_)
            throw std::invalid_argument(std::string("RealSpaceCorrelationAccumulator::") + what + ": spin has " +
                                        std::to_string(s.size()) + " components, expected " +
                                        std::to_string(n_comp_));
}

void RealSpaceCorrelationAccumulator::to_global(const std::vector<Eigen::VectorXd>& spins) {
    const std::size_t d = n_comp_;
    for (std::size_t i = 0; i < n_sites_; ++i) {
        double* g = global_.data() + i * d;
        if (identity_frames_) {
            for (std::size_t a = 0; a < d; ++a) g[a] = spins[i][a];
        } else {
            const Eigen::MatrixXd& F = frames_[i % n_sub_];
            for (std::size_t a = 0; a < d; ++a) {
                double v = 0.0;
                for (std::size_t b = 0; b < d; ++b) v += F(a, b) * spins[i][b];
                g[a] = v;
            }
        }
    }
}

void RealSpaceCorrelationAccumulator::transform_fields(std::size_t n_fields, const double* fields) {
    F_half_.resize(n_fields * n_half_);
    const std::size_t N = n_cells_;
    for (std::size_t f = 0; f < n_fields; f += 2) {
        const bool pair = (f + 1 < n_fields);
        const double* x = fields + f * N;
        const double* y = pair ? fields + (f + 1) * N : nullptr;
        for (std::size_t c = 0; c < N; ++c) work_[c] = cplx(x[c], pair ? y[c] : 0.0);
        plan_.execute(work_.data(), -1);
        cplx* Fx = F_half_.data() + f * n_half_;
        cplx* Fy = pair ? F_half_.data() + (f + 1) * n_half_ : nullptr;
        for (std::size_t h = 0; h < n_half_; ++h) {
            const std::size_t k = half_to_full_[h];
            const cplx Z = work_[k];
            if (!pair) {
                Fx[h] = Z;
                continue;
            }
            // Two real fields per transform: Z = X + iY, X(-k) = X(k)^*, Y(-k) = Y(k)^*.
            const cplx Zm = std::conj(work_[neg_full_[k]]);
            Fx[h] = 0.5 * (Z + Zm);
            Fy[h] = cplx(0.0, -0.5) * (Z - Zm);
        }
    }
}

void RealSpaceCorrelationAccumulator::accumulate_spin_correlations(const std::vector<Eigen::VectorXd>& spins) {
    require_initialized("accumulate_spin_correlations");
    check_spins(spins, "accumulate_spin_correlations");
    to_global(spins);
    const std::size_t d = n_comp_, N = n_cells_, n_fields = n_sub_ * d;
    fields_.resize(n_fields * N);
    std::vector<double>& mean = spin_.means[spin_.current];
    for (std::size_t s = 0; s < n_sub_; ++s)
        for (std::size_t a = 0; a < d; ++a) {
            double* f = fields_.data() + (s * d + a) * N;
            double sum = 0.0;
            for (std::size_t c = 0; c < N; ++c) sum += (f[c] = global_[(c * n_sub_ + s) * d + a]);
            mean[s * d + a] += sum / double(N);
        }
    transform_fields(n_fields, fields_.data());
    cplx* spec = spin_.spectra[spin_.current].data();
    const std::size_t per_k = n_pairs_ * d * d;
    for (std::size_t h = 0; h < n_half_; ++h) {
        cplx* out = spec + h * per_k;
        std::size_t p = 0;
        for (std::size_t s = 0; s < n_sub_; ++s)
            for (std::size_t s2 = s; s2 < n_sub_; ++s2, ++p)
                for (std::size_t a = 0; a < d; ++a) {
                    const cplx Fa = F_half_[(s * d + a) * n_half_ + h];
                    for (std::size_t b = 0; b < d; ++b)
                        out[(p * d + a) * d + b] += Fa * std::conj(F_half_[(s2 * d + b) * n_half_ + h]);
                }
    }
    spin_.finish_sample(options_.n_bins);
}

void RealSpaceCorrelationAccumulator::accumulate_dimer_correlations(const std::vector<Eigen::VectorXd>& spins) {
    require_initialized("accumulate_dimer_correlations");
    if (classes_.empty())
        throw std::logic_error("RealSpaceCorrelationAccumulator::accumulate_dimer_correlations: no bond classes "
                               "(initialize with Geometry::bond_classes)");
    check_spins(spins, "accumulate_dimer_correlations");
    to_global(spins);
    const std::size_t d = n_comp_, N = n_cells_, n_cl = classes_.size(), n_fields = n_cl * d;
    fields_.resize(n_fields * N);
    std::vector<double>& mean = dimer_.means[dimer_.current];
    for (std::size_t k = 0; k < n_cl; ++k) {
        const std::size_t src = classes_[k].source;
        for (std::size_t a = 0; a < d; ++a) {
            double* f = fields_.data() + (k * d + a) * N;
            double sum = 0.0;
            for (std::size_t c = 0; c < N; ++c) {
                const std::size_t i = c * n_sub_ + src, j = partner_site_[k * N + c];
                sum += (f[c] = global_[i * d + a] * global_[j * d + a]);
            }
            mean[k * d + a] += sum / double(N);
        }
    }
    transform_fields(n_fields, fields_.data());
    cplx* spec = dimer_.spectra[dimer_.current].data();
    const std::size_t per_k = n_class_pairs_ * d;
    for (std::size_t h = 0; h < n_half_; ++h) {
        cplx* out = spec + h * per_k;
        std::size_t p = 0;
        for (std::size_t k = 0; k < n_cl; ++k)
            for (std::size_t k2 = k; k2 < n_cl; ++k2, ++p)
                for (std::size_t a = 0; a < d; ++a)
                    out[p * d + a] += F_half_[(k * d + a) * n_half_ + h] *
                                      std::conj(F_half_[(k2 * d + a) * n_half_ + h]);
    }
    dimer_.finish_sample(options_.n_bins);
}

void RealSpaceCorrelationAccumulator::add_sample(const std::vector<Eigen::VectorXd>& spins) {
    accumulate_spin_correlations(spins);
    if (!classes_.empty()) accumulate_dimer_correlations(spins);
}

// ============================================================================
// Indexing helpers
// ============================================================================

std::size_t RealSpaceCorrelationAccumulator::pair_index(std::size_t s, std::size_t s2) const {
    return s * (2 * n_sub_ - s - 1) / 2 + s2;
}

std::size_t RealSpaceCorrelationAccumulator::class_pair_index(std::size_t c, std::size_t c2) const {
    const std::size_t n = classes_.size();
    return c * (2 * n - c - 1) / 2 + c2;
}

Eigen::Vector3d RealSpaceCorrelationAccumulator::grid_wavevector(long m1, long m2, long m3) const {
    require_initialized("grid_wavevector");
    return (double(m1) / double(dims_[0])) * recip_[0] + (double(m2) / double(dims_[1])) * recip_[1] +
           (double(m3) / double(dims_[2])) * recip_[2];
}

std::array<long, 3> RealSpaceCorrelationAccumulator::grid_index(const Eigen::Vector3d& q) const {
    require_initialized("grid_index");
    if (!q.allFinite()) throw std::invalid_argument("RealSpaceCorrelationAccumulator: non-finite wave vector");
    std::array<long, 3> m{};
    for (int d = 0; d < 3; ++d) {
        const double x = q.dot(lattice_vectors_[d]) * double(dims_[d]) / (2.0 * M_PI);
        const double r = std::round(x);
        if (std::abs(x - r) > 1e-8 * std::max(1.0, std::abs(x))) {
            std::ostringstream os;
            os << "RealSpaceCorrelationAccumulator: q = (" << q.transpose() << ") is not on the commensurate grid of the "
               << dims_[0] << "x" << dims_[1] << "x" << dims_[2] << " lattice (q.a_" << d + 1 << " L_" << d + 1
               << " / 2pi = " << std::setprecision(12) << x
               << " is not an integer); S(q) is only defined there under periodic boundaries";
            throw std::invalid_argument(os.str());
        }
        m[d] = long(r);
    }
    return m;
}

std::array<long, 3> RealSpaceCorrelationAccumulator::reduce(const std::array<long, 3>& m) const {
    return {euclid_mod(m[0], long(dims_[0])), euclid_mod(m[1], long(dims_[1])), euclid_mod(m[2], long(dims_[2]))};
}

RealSpaceCorrelationAccumulator::HalfIndex RealSpaceCorrelationAccumulator::half_index(
    const std::array<long, 3>& m_in) const {
    std::array<long, 3> m = reduce(m_in);
    bool conj = false;
    if (half_axis_ >= 0 && m[half_axis_] > long(dims_[half_axis_] / 2)) {
        m = reduce({-m[0], -m[1], -m[2]});
        conj = true;
    }
    const std::size_t idx = (std::size_t(m[0]) * half_dims_[1] + std::size_t(m[1])) * half_dims_[2] + std::size_t(m[2]);
    return {idx, conj};
}

RealSpaceCorrelationAccumulator::cplx RealSpaceCorrelationAccumulator::spin_block(
    const cplx* slice, const HalfIndex& h, std::size_t s, std::size_t s2, std::size_t a, std::size_t b) const {
    const std::size_t d = n_comp_;
    // X_{s's}^{βα}(k) = X_{ss'}^{αβ}(k)^*, and X(-k) = X(k)^* for real fields.
    const cplx v = (s <= s2) ? slice[(pair_index(s, s2) * d + a) * d + b]
                             : std::conj(slice[(pair_index(s2, s) * d + b) * d + a]);
    return h.conjugate ? std::conj(v) : v;
}

void RealSpaceCorrelationAccumulator::totals(const Channel& ch, std::vector<cplx>& spec,
                                             std::vector<double>& mean) const {
    spec.assign(ch.spectrum_size, cplx(0.0));
    mean.assign(ch.mean_size, 0.0);
    for (std::size_t b = 0; b < ch.counts.size(); ++b) {
        if (ch.counts[b] == 0) continue;
        for (std::size_t j = 0; j < ch.spectrum_size; ++j) spec[j] += ch.spectra[b][j];
        for (std::size_t j = 0; j < ch.mean_size; ++j) mean[j] += ch.means[b][j];
    }
}

// ============================================================================
// Estimators
// ============================================================================

template <class F>
std::vector<RealSpaceCorrelationAccumulator::Estimate> RealSpaceCorrelationAccumulator::jackknife(
    const Channel& ch, std::size_t n_out, F&& f) const {
    // f(skip) evaluates the estimator on all bins except `skip` (-1: all bins).
    std::vector<Estimate> out(n_out);
    std::vector<std::size_t> used;
    for (std::size_t b = 0; b < ch.counts.size(); ++b)
        if (ch.counts[b] > 0) used.push_back(b);
    const std::vector<Eigen::MatrixXcd> full = f(-1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i = 0; i < n_out; ++i) {
        out[i].mean = full[i];
        out[i].n_bins = used.size();
        out[i].error = Eigen::MatrixXcd::Constant(full[i].rows(), full[i].cols(), cplx(nan, nan));
    }
    if (used.size() < 2) return out;
    const double N = double(ch.n_samples);
    std::vector<Eigen::MatrixXd> var_re(n_out), var_im(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
        var_re[i] = Eigen::MatrixXd::Zero(full[i].rows(), full[i].cols());
        var_im[i] = var_re[i];
    }
    for (std::size_t b : used) {
        const std::vector<Eigen::MatrixXcd> th = f(long(b));
        // Delete-a-group jackknife with unequal groups: weight 1 - n_b / n.
        const double w = 1.0 - double(ch.counts[b]) / N;
        for (std::size_t i = 0; i < n_out; ++i) {
            const Eigen::MatrixXcd diff = th[i] - full[i];
            var_re[i] += w * diff.real().cwiseAbs2();
            var_im[i] += w * diff.imag().cwiseAbs2();
        }
    }
    for (std::size_t i = 0; i < n_out; ++i) {
        out[i].error = Eigen::MatrixXcd(full[i].rows(), full[i].cols());
        out[i].error.real() = var_re[i].cwiseSqrt();
        out[i].error.imag() = var_im[i].cwiseSqrt();
    }
    return out;
}

Eigen::MatrixXcd RealSpaceCorrelationAccumulator::spin_sq_from(const std::vector<cplx>& slice,
                                                               const std::vector<double>& mean, double n,
                                                               const Eigen::Vector3d& q, const HalfIndex& h,
                                                               bool zone_center, bool connected) const {
    const std::size_t d = n_comp_;
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(d, d);
    if (!(n > 0.0)) return S;
    const double Nc = double(n_cells_);
    for (std::size_t s = 0; s < n_sub_; ++s)
        for (std::size_t s2 = 0; s2 < n_sub_; ++s2) {
            const cplx ph = phase(-q.dot(positions_[s] - positions_[s2]));
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b) {
                    cplx v = ph * spin_block(slice.data(), h, s, s2, a, b) / (n * Nc);
                    if (connected && zone_center) v -= ph * Nc * (mean[s * d + a] / n) * (mean[s2 * d + b] / n);
                    S(a, b) += v;
                }
        }
    return S;
}

std::vector<Eigen::MatrixXcd> RealSpaceCorrelationAccumulator::dimer_sq_from(
    const std::vector<cplx>& slice, const std::vector<double>& mean, double n, const Eigen::Vector3d& q,
    const HalfIndex& h, bool zone_center, bool connected) const {
    const std::size_t d = n_comp_, n_cl = classes_.size();
    std::vector<Eigen::MatrixXcd> S(d, Eigen::MatrixXcd::Zero(n_cl, n_cl));
    if (!(n > 0.0)) return S;
    const double Nc = double(n_cells_);
    for (std::size_t c = 0; c < n_cl; ++c)
        for (std::size_t c2 = 0; c2 < n_cl; ++c2) {
            const cplx ph = phase(-q.dot(classes_[c].center - classes_[c2].center));
            for (std::size_t a = 0; a < d; ++a) {
                cplx v = (c <= c2) ? slice[class_pair_index(c, c2) * d + a]
                                   : std::conj(slice[class_pair_index(c2, c) * d + a]);
                if (h.conjugate) v = std::conj(v);
                v = ph * v / (n * Nc);
                if (connected && zone_center) v -= ph * Nc * (mean[c * d + a] / n) * (mean[c2 * d + a] / n);
                S[a](c, c2) = v;
            }
        }
    return S;
}

RealSpaceCorrelationAccumulator::Estimate RealSpaceCorrelationAccumulator::structure_factor_estimate(
    const Eigen::Vector3d& q, bool connected) const {
    require_initialized("structure_factor_estimate");
    const std::array<long, 3> m = reduce(grid_index(q));
    const HalfIndex h = half_index(m);
    const bool zone_center = (m[0] == 0 && m[1] == 0 && m[2] == 0);
    const std::size_t per_k = n_pairs_ * n_comp_ * n_comp_;
    const Channel& ch = spin_;
    auto f = [&](long skip) {
        std::vector<cplx> slice(per_k, cplx(0.0));
        std::vector<double> mean(ch.mean_size, 0.0);
        double n = 0.0;
        for (std::size_t b = 0; b < ch.counts.size(); ++b) {
            if (long(b) == skip || ch.counts[b] == 0) continue;
            const cplx* src = ch.spectra[b].data() + h.index * per_k;
            for (std::size_t j = 0; j < per_k; ++j) slice[j] += src[j];
            for (std::size_t j = 0; j < ch.mean_size; ++j) mean[j] += ch.means[b][j];
            n += double(ch.counts[b]);
        }
        return std::vector<Eigen::MatrixXcd>{spin_sq_from(slice, mean, n, q, h, zone_center, connected)};
    };
    return jackknife(ch, 1, f)[0];
}

Eigen::MatrixXcd RealSpaceCorrelationAccumulator::structure_factor(const Eigen::Vector3d& q, bool connected) const {
    require_initialized("structure_factor");
    const std::array<long, 3> m = reduce(grid_index(q));
    const HalfIndex h = half_index(m);
    const bool zone_center = (m[0] == 0 && m[1] == 0 && m[2] == 0);
    const std::size_t per_k = n_pairs_ * n_comp_ * n_comp_;
    std::vector<cplx> slice(per_k, cplx(0.0));
    std::vector<double> mean(spin_.mean_size, 0.0);
    for (std::size_t b = 0; b < spin_.counts.size(); ++b) {
        if (spin_.counts[b] == 0) continue;
        const cplx* src = spin_.spectra[b].data() + h.index * per_k;
        for (std::size_t j = 0; j < per_k; ++j) slice[j] += src[j];
        for (std::size_t j = 0; j < spin_.mean_size; ++j) mean[j] += spin_.means[b][j];
    }
    return spin_sq_from(slice, mean, double(spin_.n_samples), q, h, zone_center, connected);
}

Eigen::Matrix3d RealSpaceCorrelationAccumulator::compute_Sq(const Eigen::Vector3d& q, bool connected) const {
    if (n_comp_ != 3) throw std::invalid_argument("compute_Sq: needs spin_dim == 3 (use structure_factor)");
    return structure_factor(q, connected).real();
}

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> RealSpaceCorrelationAccumulator::compute_Sq_with_error(
    const Eigen::Vector3d& q, bool connected) const {
    if (n_comp_ != 3) throw std::invalid_argument("compute_Sq_with_error: needs spin_dim == 3");
    const Estimate e = structure_factor_estimate(q, connected);
    return {e.mean.real(), e.error.real()};
}

std::vector<double> RealSpaceCorrelationAccumulator::real_space_correlation(std::size_t s, std::size_t s2,
                                                                            std::size_t alpha, std::size_t beta,
                                                                            bool connected) const {
    require_initialized("real_space_correlation");
    if (s >= n_sub_ || s2 >= n_sub_ || alpha >= n_comp_ || beta >= n_comp_)
        throw std::invalid_argument("real_space_correlation: sublattice or component index out of range");
    std::vector<double> C(n_cells_, 0.0);
    if (spin_.n_samples == 0) return C;
    std::vector<cplx> spec;
    std::vector<double> mean;
    totals(spin_, spec, mean);
    const double n = double(spin_.n_samples), Nc = double(n_cells_);
    const std::size_t per_k = n_pairs_ * n_comp_ * n_comp_;
    std::vector<cplx> X(n_cells_);
    for (std::size_t a = 0; a < dims_[0]; ++a)
        for (std::size_t b = 0; b < dims_[1]; ++b)
            for (std::size_t c = 0; c < dims_[2]; ++c) {
                const HalfIndex h = half_index({long(a), long(b), long(c)});
                X[(a * dims_[1] + b) * dims_[2] + c] =
                    spin_block(spec.data() + h.index * per_k, h, s, s2, alpha, beta);
            }
    // C(d) = (1/N_c^2) Σ_k <X(k)> e^{-i k·d}.
    plan_.execute(X.data(), -1);
    const double disc = connected ? (mean[s * n_comp_ + alpha] / n) * (mean[s2 * n_comp_ + beta] / n) : 0.0;
    for (std::size_t i = 0; i < n_cells_; ++i) C[i] = X[i].real() / (n * Nc * Nc) - disc;
    return C;
}

Eigen::VectorXd RealSpaceCorrelationAccumulator::sublattice_magnetization(std::size_t s) const {
    require_initialized("sublattice_magnetization");
    if (s >= n_sub_) throw std::invalid_argument("sublattice_magnetization: sublattice index out of range");
    Eigen::VectorXd m = Eigen::VectorXd::Zero(n_comp_);
    if (spin_.n_samples == 0) return m;
    for (std::size_t b = 0; b < spin_.counts.size(); ++b)
        for (std::size_t a = 0; a < n_comp_; ++a) m[a] += spin_.means[b][s * n_comp_ + a];
    return m / double(spin_.n_samples);
}

double RealSpaceCorrelationAccumulator::dimer_mean(std::size_t bond_class, std::size_t alpha) const {
    require_initialized("dimer_mean");
    if (bond_class >= classes_.size() || alpha >= n_comp_)
        throw std::invalid_argument("dimer_mean: bond class or component index out of range");
    if (dimer_.n_samples == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t b = 0; b < dimer_.counts.size(); ++b) sum += dimer_.means[b][bond_class * n_comp_ + alpha];
    return sum / double(dimer_.n_samples);
}

std::vector<RealSpaceCorrelationAccumulator::Estimate> RealSpaceCorrelationAccumulator::dimer_structure_factor_estimate(
    const Eigen::Vector3d& q, bool connected) const {
    require_initialized("dimer_structure_factor_estimate");
    if (classes_.empty()) throw std::logic_error("dimer_structure_factor: the accumulator has no bond classes");
    const std::array<long, 3> m = reduce(grid_index(q));
    const HalfIndex h = half_index(m);
    const bool zone_center = (m[0] == 0 && m[1] == 0 && m[2] == 0);
    const std::size_t per_k = n_class_pairs_ * n_comp_;
    const Channel& ch = dimer_;
    auto f = [&](long skip) {
        std::vector<cplx> slice(per_k, cplx(0.0));
        std::vector<double> mean(ch.mean_size, 0.0);
        double n = 0.0;
        for (std::size_t b = 0; b < ch.counts.size(); ++b) {
            if (long(b) == skip || ch.counts[b] == 0) continue;
            const cplx* src = ch.spectra[b].data() + h.index * per_k;
            for (std::size_t j = 0; j < per_k; ++j) slice[j] += src[j];
            for (std::size_t j = 0; j < ch.mean_size; ++j) mean[j] += ch.means[b][j];
            n += double(ch.counts[b]);
        }
        return dimer_sq_from(slice, mean, n, q, h, zone_center, connected);
    };
    return jackknife(ch, n_comp_, f);
}

std::vector<Eigen::MatrixXcd> RealSpaceCorrelationAccumulator::dimer_structure_factor(const Eigen::Vector3d& q,
                                                                                     bool connected) const {
    std::vector<Eigen::MatrixXcd> out;
    for (const Estimate& e : dimer_structure_factor_estimate(q, connected)) out.push_back(e.mean);
    return out;
}

std::array<Eigen::MatrixXd, 3> RealSpaceCorrelationAccumulator::compute_Sq_dimer(const Eigen::Vector3d& q,
                                                                                bool connected) const {
    if (n_comp_ != 3) throw std::invalid_argument("compute_Sq_dimer: needs spin_dim == 3");
    const std::vector<Eigen::MatrixXcd> S = dimer_structure_factor(q, connected);
    return {S[0].real(), S[1].real(), S[2].real()};
}

Eigen::MatrixXd RealSpaceCorrelationAccumulator::compute_Sq_dimer_total(const Eigen::Vector3d& q,
                                                                       bool connected) const {
    const std::array<Eigen::MatrixXd, 3> S = compute_Sq_dimer(q, connected);
    return S[0] + S[1] + S[2];
}

// ============================================================================
// Bookkeeping
// ============================================================================

void RealSpaceCorrelationAccumulator::reset() {
    spin_.clear();
    dimer_.clear();
}

std::vector<double> RealSpaceCorrelationAccumulator::geometry_signature() const {
    std::vector<double> v = {initialized_ ? 1.0 : 0.0, double(dims_[0]), double(dims_[1]), double(dims_[2]),
                             double(n_sub_), double(n_comp_), double(options_.n_bins), double(classes_.size())};
    if (!initialized_) return v;
    for (const auto& a : lattice_vectors_) v.insert(v.end(), a.data(), a.data() + 3);
    for (const auto& p : positions_) v.insert(v.end(), p.data(), p.data() + 3);
    for (const auto& F : frames_) v.insert(v.end(), F.data(), F.data() + F.size());
    for (const BondClass& c : classes_) {
        v.push_back(double(c.source));
        v.push_back(double(c.partner));
        for (int d = 0; d < 3; ++d) v.push_back(double(c.offset[d]));
    }
    return v;
}

void RealSpaceCorrelationAccumulator::merge(const RealSpaceCorrelationAccumulator& other) {
    if (!other.initialized_) return;
    if (!initialized_) {
        *this = other;
        return;
    }
    const std::vector<double> a = geometry_signature(), b = other.geometry_signature();
    bool same = (a.size() == b.size());
    for (std::size_t i = 0; same && i < a.size(); ++i)
        same = std::abs(a[i] - b[i]) <= 1e-12 * std::max(1.0, std::abs(a[i]));
    if (!same)
        throw std::invalid_argument("RealSpaceCorrelationAccumulator::merge: accumulators differ in geometry "
                                    "(dimensions, sublattices, frames, bond classes or n_bins)");
    auto merge_channel = [](Channel& dst, const Channel& src) {
        for (std::size_t bin = 0; bin < dst.counts.size(); ++bin) {
            for (std::size_t j = 0; j < dst.spectrum_size; ++j) dst.spectra[bin][j] += src.spectra[bin][j];
            for (std::size_t j = 0; j < dst.mean_size; ++j) dst.means[bin][j] += src.means[bin][j];
            dst.counts[bin] += src.counts[bin];
        }
        dst.n_samples += src.n_samples;
        dst.bin_size = std::max(dst.bin_size, src.bin_size);
        dst.current = std::max(dst.current, src.current);
    };
    merge_channel(spin_, other.spin_);
    merge_channel(dimer_, other.dimer_);
}

void RealSpaceCorrelationAccumulator::mpi_reduce(MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1) return;

    // Collective geometry check: an FNV-1a hash of the signature must agree.
    const std::vector<double> sig = geometry_signature();
    std::uint64_t hash = 1469598103934665603ULL;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(sig.data());
    for (std::size_t i = 0; i < sig.size() * sizeof(double); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    std::uint64_t lo = hash, hi = hash;
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_UINT64_T, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_UINT64_T, MPI_MAX, comm);
    if (lo != hi)
        throw std::invalid_argument("RealSpaceCorrelationAccumulator::mpi_reduce: ranks hold accumulators of "
                                    "different geometry (or not all are initialized)");
    if (!initialized_) return;

    std::uint64_t shape[4] = {spin_.bin_size, spin_.current, dimer_.bin_size, dimer_.current};
    MPI_Allreduce(MPI_IN_PLACE, shape, 4, MPI_UINT64_T, MPI_MAX, comm);

    // One packed buffer: per channel and bin, the spectrum (re, im), the means, the count.
    std::vector<double> buf;
    auto pack = [&](const Channel& ch) {
        for (std::size_t b = 0; b < ch.counts.size(); ++b) {
            for (const cplx& z : ch.spectra[b]) {
                buf.push_back(z.real());
                buf.push_back(z.imag());
            }
            buf.insert(buf.end(), ch.means[b].begin(), ch.means[b].end());
            buf.push_back(double(ch.counts[b]));
        }
    };
    pack(spin_);
    pack(dimer_);
    const std::size_t chunk = std::size_t(1) << 28;   // keeps each MPI count within int
    for (std::size_t off = 0; off < buf.size(); off += chunk) {
        const int n = int(std::min(chunk, buf.size() - off));
        if (rank == 0)
            MPI_Reduce(MPI_IN_PLACE, buf.data() + off, n, MPI_DOUBLE, MPI_SUM, 0, comm);
        else
            MPI_Reduce(buf.data() + off, nullptr, n, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    if (rank != 0) return;
    std::size_t pos = 0;
    auto unpack = [&](Channel& ch, std::uint64_t bin_size, std::uint64_t current) {
        if (ch.counts.empty()) return;
        ch.n_samples = 0;
        for (std::size_t b = 0; b < ch.counts.size(); ++b) {
            for (cplx& z : ch.spectra[b]) {
                z = cplx(buf[pos], buf[pos + 1]);
                pos += 2;
            }
            for (double& m : ch.means[b]) m = buf[pos++];
            ch.counts[b] = std::size_t(std::llround(buf[pos++]));
            ch.n_samples += ch.counts[b];
        }
        ch.bin_size = std::size_t(bin_size);
        ch.current = std::size_t(current);
    };
    unpack(spin_, shape[0], shape[1]);
    unpack(dimer_, shape[2], shape[3]);
}

std::size_t RealSpaceCorrelationAccumulator::storage_bytes() const {
    std::size_t b = spin_.bytes() + dimer_.bytes();
    b += positions_.capacity() * sizeof(Eigen::Vector3d);
    for (const auto& F : frames_) b += std::size_t(F.size()) * sizeof(double);
    b += classes_.capacity() * sizeof(BondClass);
    b += (partner_site_.capacity() + half_to_full_.capacity() + neg_full_.capacity()) * sizeof(std::size_t);
    b += (global_.capacity() + fields_.capacity()) * sizeof(double);
    b += (work_.capacity() + F_half_.capacity()) * sizeof(cplx);
    return b;
}

// ============================================================================
// Output
// ============================================================================

#ifdef HDF5_ENABLED
void RealSpaceCorrelationAccumulator::save_hdf5(const std::string& filename, const std::string& group_name) const {
    require_initialized("save_hdf5");
    H5::Exception::dontPrint();
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::Group group = (group_name.empty() || group_name == "/") ? file.openGroup("/")
                                                                     : file.createGroup(group_name);
        auto attr_u = [](H5::Group& g, const std::string& name, unsigned long long v) {
            H5::Attribute a = g.createAttribute(name, H5::PredType::NATIVE_ULLONG, H5::DataSpace(H5S_SCALAR));
            a.write(H5::PredType::NATIVE_ULLONG, &v);
        };
        auto attr_s = [](H5::Group& g, const std::string& name, const std::string& v) {
            H5::StrType t(H5::PredType::C_S1, v.size() + 1);
            H5::Attribute a = g.createAttribute(name, t, H5::DataSpace(H5S_SCALAR));
            a.write(t, v.c_str());
        };
        auto write_d = [](H5::Group& g, const std::string& name, const std::vector<double>& v,
                          std::vector<hsize_t> shape) {
            H5::DataSpace space(int(shape.size()), shape.data());
            H5::DataSet ds = g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
            if (!v.empty()) ds.write(v.data(), H5::PredType::NATIVE_DOUBLE);
        };
        auto write_i = [](H5::Group& g, const std::string& name, const std::vector<long long>& v,
                          std::vector<hsize_t> shape) {
            H5::DataSpace space(int(shape.size()), shape.data());
            H5::DataSet ds = g.createDataSet(name, H5::PredType::NATIVE_LLONG, space);
            if (!v.empty()) ds.write(v.data(), H5::PredType::NATIVE_LLONG);
        };
        const std::size_t d = n_comp_;
        const hsize_t hd = hsize_t(d);

        attr_u(group, "n_samples", spin_.n_samples);
        attr_u(group, "n_dimer_samples", dimer_.n_samples);
        attr_u(group, "dim1", dims_[0]);
        attr_u(group, "dim2", dims_[1]);
        attr_u(group, "dim3", dims_[2]);
        attr_u(group, "n_sublattices", n_sub_);
        attr_u(group, "spin_dim", d);
        attr_u(group, "n_bins", options_.n_bins);
        attr_u(group, "bin_size", spin_.bin_size);
        attr_s(group, "frame", "global (S_global = F_s S_local)");
        attr_s(group, "normalization", "per_unit_cell");
        attr_s(group, "fft_convention", "F_s(k) = sum_R S_s(R) exp(-i k.R), k = sum_i (m_i/L_i) b_i");
        attr_s(group, "cross_spectrum",
               "X_ss'^ab(k) = <F_sa(k) conj(F_s'b(k))>, s <= s', half k grid (X(-k) = conj X(k))");
        attr_s(group, "structure_factor",
               "S^ab(q) = (1/N_c) sum_ss' exp(-i q.(tau_s - tau_s')) X_ss'^ab(k), q = k + G");

        std::vector<double> lv, rv;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                lv.push_back(lattice_vectors_[i][j]);
                rv.push_back(recip_[i][j]);
            }
        write_d(group, "lattice_vectors", lv, {3, 3});
        write_d(group, "reciprocal_vectors", rv, {3, 3});
        std::vector<double> pos, fr;
        for (const auto& p : positions_) pos.insert(pos.end(), p.data(), p.data() + 3);
        for (const auto& F : frames_)
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b) fr.push_back(F(a, b));
        write_d(group, "sublattice_positions", pos, {hsize_t(n_sub_), 3});
        write_d(group, "sublattice_frames", fr, {hsize_t(n_sub_), hd, hd});
        std::vector<long long> pairs, kidx;
        for (std::size_t s = 0; s < n_sub_; ++s)
            for (std::size_t s2 = s; s2 < n_sub_; ++s2) {
                pairs.push_back((long long)s);
                pairs.push_back((long long)s2);
            }
        write_i(group, "sublattice_pairs", pairs, {hsize_t(n_pairs_), 2});
        for (std::size_t h = 0; h < n_half_; ++h) {
            const std::size_t k = half_to_full_[h];
            kidx.push_back((long long)(k / (dims_[1] * dims_[2])));
            kidx.push_back((long long)((k / dims_[2]) % dims_[1]));
            kidx.push_back((long long)(k % dims_[2]));
        }
        write_i(group, "k_index", kidx, {hsize_t(n_half_), 3});

        auto write_channel = [&](H5::Group& g, const Channel& ch, const std::string& prefix,
                                 const std::vector<hsize_t>& spec_shape, const std::vector<hsize_t>& mean_shape) {
            std::vector<cplx> spec;
            std::vector<double> mean;
            totals(ch, spec, mean);
            const double n = ch.n_samples > 0 ? double(ch.n_samples) : 1.0;
            std::vector<double> re_im(2 * spec.size());
            for (std::size_t j = 0; j < spec.size(); ++j) {
                re_im[2 * j] = spec[j].real() / n;
                re_im[2 * j + 1] = spec[j].imag() / n;
            }
            for (double& m : mean) m /= n;
            std::vector<hsize_t> s1 = spec_shape;
            s1.push_back(2);
            write_d(g, prefix + "cross_spectrum", re_im, s1);
            write_d(g, prefix + "mean", mean, mean_shape);
            // Per-bin means (empty bins hold zeros) for offline error analysis.
            const std::size_t nb = ch.counts.size();
            std::vector<double> bins(nb * 2 * ch.spectrum_size), bin_means(nb * ch.mean_size);
            std::vector<long long> counts(nb);
            for (std::size_t b = 0; b < nb; ++b) {
                counts[b] = (long long)ch.counts[b];
                const double c = ch.counts[b] > 0 ? double(ch.counts[b]) : 1.0;
                for (std::size_t j = 0; j < ch.spectrum_size; ++j) {
                    bins[(b * ch.spectrum_size + j) * 2] = ch.spectra[b][j].real() / c;
                    bins[(b * ch.spectrum_size + j) * 2 + 1] = ch.spectra[b][j].imag() / c;
                }
                for (std::size_t j = 0; j < ch.mean_size; ++j) bin_means[b * ch.mean_size + j] = ch.means[b][j] / c;
            }
            std::vector<hsize_t> s2 = {hsize_t(nb)};
            s2.insert(s2.end(), s1.begin(), s1.end());
            write_d(g, prefix + "cross_spectrum_bins", bins, s2);
            std::vector<hsize_t> s3 = {hsize_t(nb)};
            s3.insert(s3.end(), mean_shape.begin(), mean_shape.end());
            write_d(g, prefix + "mean_bins", bin_means, s3);
            write_i(g, prefix + "bin_counts", counts, {hsize_t(nb)});
        };
        write_channel(group, spin_, "spin_", {hsize_t(n_half_), hsize_t(n_pairs_), hd, hd}, {hsize_t(n_sub_), hd});

        // S(q) with jackknife errors on the first-zone grid q = k.
        if (spin_.n_samples > 0) {
            std::vector<double> S(n_cells_ * d * d * 2), E(n_cells_ * d * d * 2);
            for (std::size_t a = 0; a < dims_[0]; ++a)
                for (std::size_t b = 0; b < dims_[1]; ++b)
                    for (std::size_t c = 0; c < dims_[2]; ++c) {
                        const Estimate e = structure_factor_estimate(grid_wavevector(long(a), long(b), long(c)));
                        const std::size_t base = ((a * dims_[1] + b) * dims_[2] + c) * d * d;
                        for (std::size_t i = 0; i < d; ++i)
                            for (std::size_t j = 0; j < d; ++j) {
                                const std::size_t idx = (base + i * d + j) * 2;
                                S[idx] = e.mean(i, j).real();
                                S[idx + 1] = e.mean(i, j).imag();
                                E[idx] = e.error(i, j).real();
                                E[idx + 1] = e.error(i, j).imag();
                            }
                    }
            const std::vector<hsize_t> shape = {hsize_t(dims_[0]), hsize_t(dims_[1]), hsize_t(dims_[2]), hd, hd, 2};
            write_d(group, "structure_factor", S, shape);
            write_d(group, "structure_factor_error", E, shape);
        }

        if (!classes_.empty()) {
            H5::Group dg = group.createGroup("dimers");
            std::vector<long long> src, prt, off;
            std::vector<double> ctr, len;
            for (const BondClass& c : classes_) {
                src.push_back((long long)c.source);
                prt.push_back((long long)c.partner);
                for (int i = 0; i < 3; ++i) {
                    off.push_back(c.offset[i]);
                    ctr.push_back(c.center[i]);
                }
                len.push_back(c.length);
            }
            const hsize_t nc = hsize_t(classes_.size());
            write_i(dg, "bond_class_source", src, {nc});
            write_i(dg, "bond_class_partner", prt, {nc});
            write_i(dg, "bond_class_offset", off, {nc, 3});
            write_d(dg, "bond_class_center", ctr, {nc, 3});
            write_d(dg, "bond_class_length", len, {nc});
            attr_u(dg, "n_samples", dimer_.n_samples);
            attr_s(dg, "dimer", "D^a_c(R) = S^a_(R,source) S^a_(R+offset,partner), global frame");
            attr_s(dg, "structure_factor",
                   "S_D^a_cc'(q) = (1/N_c) exp(-i q.(rho_c - rho_c')) <G_ca(k) conj(G_c'a(k))>, c <= c' stored");
            write_channel(dg, dimer_, "dimer_", {hsize_t(n_half_), hsize_t(n_class_pairs_), hd}, {nc, hd});
        }
        file.close();
    } catch (const H5::Exception& e) {
        throw std::runtime_error("RealSpaceCorrelationAccumulator::save_hdf5(" + filename + "): " + e.getDetailMsg());
    }
}
#endif

void RealSpaceCorrelationAccumulator::save_structure_factor_grid(
    const std::string& filename, std::pair<double, double> q1_range, std::pair<double, double> q2_range,
    std::pair<double, double> q3_range, std::size_t n_q1, std::size_t n_q2, std::size_t n_q3,
    const Eigen::Vector3d& b1, const Eigen::Vector3d& b2, const Eigen::Vector3d& b3, bool connected) const {
    require_initialized("save_structure_factor_grid");
    if (n_q1 == 0 || n_q2 == 0 || n_q3 == 0)
        throw std::invalid_argument("save_structure_factor_grid: every n_q must be >= 1");
    auto value = [](std::pair<double, double> r, std::size_t i, std::size_t n) {
        return (n > 1) ? r.first + (r.second - r.first) * double(i) / double(n - 1) : 0.5 * (r.first + r.second);
    };
    // Validate every q before writing anything.
    std::vector<Eigen::Vector3d> qs;
    std::vector<std::array<double, 3>> hs;
    for (std::size_t i1 = 0; i1 < n_q1; ++i1)
        for (std::size_t i2 = 0; i2 < n_q2; ++i2)
            for (std::size_t i3 = 0; i3 < n_q3; ++i3) {
                const std::array<double, 3> h = {value(q1_range, i1, n_q1), value(q2_range, i2, n_q2),
                                                 value(q3_range, i3, n_q3)};
                const Eigen::Vector3d q = h[0] * b1 + h[1] * b2 + h[2] * b3;
                (void)grid_index(q);
                qs.push_back(q);
                hs.push_back(h);
            }
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("save_structure_factor_grid: cannot open " + filename);
    file << "# Spin structure factor S(q) per unit cell (global frame), FFT correlation accumulator\n";
    file << "# n_samples = " << spin_.n_samples << ", lattice " << dims_[0] << " x " << dims_[1] << " x " << dims_[2]
         << " x " << n_sub_ << " sublattices, connected = " << (connected ? "yes" : "no") << "\n";
    if (n_comp_ == 3)
        file << "# q1 q2 q3 qx qy qz S_total ReS_xx ReS_yy ReS_zz ReS_xy ReS_xz ReS_yz\n";
    else
        file << "# q1 q2 q3 qx qy qz S_total\n";
    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        const Eigen::MatrixXcd S = structure_factor(qs[i], connected);
        file << hs[i][0] << ' ' << hs[i][1] << ' ' << hs[i][2] << ' ' << qs[i][0] << ' ' << qs[i][1] << ' '
             << qs[i][2] << ' ' << S.trace().real();
        if (n_comp_ == 3)
            file << ' ' << S(0, 0).real() << ' ' << S(1, 1).real() << ' ' << S(2, 2).real() << ' ' << S(0, 1).real()
                 << ' ' << S(0, 2).real() << ' ' << S(1, 2).real();
        file << '\n';
    }
    if (!file) throw std::runtime_error("save_structure_factor_grid: write to " + filename + " failed");
}
