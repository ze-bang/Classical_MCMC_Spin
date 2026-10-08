// test_observables.cpp — FFT correlation accumulator, structure factor and
// dimer correlations against results known independently of the code:
//
//   - the FFT against the defining O(n^2) DFT sum (every length 1..40 and
//     some large primes / powers of two, 1D and 3D);
//   - S(q) from the accumulator against the direct double sum
//     (1/N_c) Σ_ij S_i S_j e^{-iq(r_i - r_j)} on random configurations, for q
//     in several Brillouin zones, to 1e-12 (pyrochlore with its local frames,
//     so the global-frame rotation is exercised);
//   - closed forms: ferromagnet (Bragg peaks N_c |Σ_s e^{-iG·τ_s}|² S²), Néel
//     state (N_c S² at Q = (π,π,π)), spiral (chiral Im S^{xy}(Q) = N S²/4) and
//     the Parseval sum rule (N_atoms S² per cell);
//   - dimers: geometry of the bond classes (every coupled bond in exactly one
//     class, centres at bond midpoints), the ↑↓↓↑ chain (dimer covering:
//     S_D(π) = N_c, S_D(0) = 0), the columnar honeycomb covering, and FFT ==
//     direct sum;
//   - jackknife errors equal the textbook standard error when every bin holds
//     one sample; merge() equals accumulating everything in one object;
//     off-grid q, wrong spin sizes and mismatched geometries are rejected.
#include "physics_test_util.h"

#include "classical_spin/core/fft.h"
#include "classical_spin/lattice/correlation_accumulator.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <set>
#include <stdexcept>
#include <unistd.h>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

using namespace phys_test;
using cplx = std::complex<double>;
using Acc = RealSpaceCorrelationAccumulator;

namespace {

std::mt19937_64& rng() {
    static std::mt19937_64 g(20261008);
    return g;
}

Eigen::VectorXd random_unit(size_t d, double length) {
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::VectorXd v(d);
    for (size_t a = 0; a < d; ++a) v[a] = n(rng());
    return length * v / v.norm();
}

void randomize(Lattice& lat) {
    for (auto& s : lat.spins) s = random_unit(lat.spin_dim, lat.spin_length);
}

// Global-frame spin of site i.
Eigen::VectorXd global_spin(const Lattice& lat, size_t i) { return lat.sublattice_frames[i % lat.N_atoms] * lat.spins[i]; }

// Direct (1/N_c) A(q) A(q)^*, A(q) = Σ_i S_i e^{-iq·r_i} (global frame).
Eigen::MatrixXcd direct_sq(const Lattice& lat, const Eigen::Vector3d& q) {
    Eigen::VectorXcd A = Eigen::VectorXcd::Zero(lat.spin_dim);
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        const double ph = q.dot(lat.site_positions[i]);
        A += global_spin(lat, i).cast<cplx>() * cplx(std::cos(ph), -std::sin(ph));
    }
    const double Nc = double(lat.dim1 * lat.dim2 * lat.dim3);
    return A * A.adjoint() / Nc;
}

double max_abs(const Eigen::MatrixXcd& M) { return M.cwiseAbs().maxCoeff(); }

// Nearest-neighbour honeycomb from the geometric shell.
UnitCell honeycomb_nn_cell() {
    HoneyComb uc(3);
    for (const auto& b : uc.bonds_at_distance(1.0 / std::sqrt(3.0)))
        uc.set_bilinear_interaction(Eigen::Matrix3d::Identity(), b.source, b.partner, b.offset);
    return uc;
}

// ---------------------------------------------------------------------------
void test_fft() {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double worst = 0.0;
    std::vector<size_t> lengths;
    for (size_t n = 1; n <= 40; ++n) lengths.push_back(n);
    for (size_t n : {64, 97, 128, 131, 210}) lengths.push_back(n);
    for (size_t n : lengths) {
        classical_spin::fft::Plan1D plan(n);
        for (int sign : {-1, +1}) {
            std::vector<cplx> x(n), y(n), ref(n);
            for (auto& v : x) v = cplx(u(rng()), u(rng()));
            for (size_t m = 0; m < n; ++m) {
                cplx acc(0.0);
                for (size_t j = 0; j < n; ++j) {
                    const double ph = sign * 2.0 * M_PI * double((j * m) % n) / double(n);
                    acc += x[j] * cplx(std::cos(ph), std::sin(ph));
                }
                ref[m] = acc;
            }
            // Strided execution (stride 3) as used by the 3D transform.
            std::vector<cplx> strided(3 * n);
            for (size_t j = 0; j < n; ++j) strided[3 * j] = x[j];
            plan.execute(strided.data(), 3, sign);
            for (size_t m = 0; m < n; ++m) worst = std::max(worst, std::abs(strided[3 * m] - ref[m]) / std::sqrt(double(n)));
            y = x;
            plan.execute(y.data(), 1, sign);
            for (size_t m = 0; m < n; ++m) worst = std::max(worst, std::abs(y[m] - ref[m]) / std::sqrt(double(n)));
        }
    }
    check(worst < 1e-12, "FFT == direct DFT for n = 1..40, 64, 97, 128, 131, 210 (max err " + std::to_string(worst) + ")");

    for (auto dims : {std::array<size_t, 3>{3, 4, 5}, std::array<size_t, 3>{6, 1, 7}, std::array<size_t, 3>{1, 1, 17}}) {
        const size_t n1 = dims[0], n2 = dims[1], n3 = dims[2], N = n1 * n2 * n3;
        std::vector<cplx> x(N), y;
        for (auto& v : x) v = cplx(u(rng()), u(rng()));
        y = x;
        classical_spin::fft::Plan3D(n1, n2, n3).execute(y.data(), -1);
        double err = 0.0;
        for (size_t a = 0; a < n1; ++a)
            for (size_t b = 0; b < n2; ++b)
                for (size_t c = 0; c < n3; ++c) {
                    cplx acc(0.0);
                    for (size_t i = 0; i < n1; ++i)
                        for (size_t j = 0; j < n2; ++j)
                            for (size_t k = 0; k < n3; ++k) {
                                const double ph = -2.0 * M_PI * (double(a * i) / n1 + double(b * j) / n2 + double(c * k) / n3);
                                acc += x[(i * n2 + j) * n3 + k] * cplx(std::cos(ph), std::sin(ph));
                            }
                    err = std::max(err, std::abs(acc - y[(a * n2 + b) * n3 + c]));
                }
        check(err < 1e-12, "3D FFT == direct sum on " + std::to_string(n1) + "x" + std::to_string(n2) + "x" +
                               std::to_string(n3) + " (err " + std::to_string(err) + ")");
    }
}

// ---------------------------------------------------------------------------
void test_fft_equals_direct_pyrochlore() {
    for (size_t L : {3, 4}) {
        Lattice lat(Pyrochlore(3), L, L, L, 1.0);
        Acc acc = lat.create_correlation_accumulator();
        const int n_samples = 3;
        std::vector<Eigen::Vector3d> qs;
        for (long m1 : {0L, 1L, -2L, long(L), long(2 * L + 1)})
            for (long m2 : {0L, 2L, long(L) - 1})
                for (long m3 : {0L, 1L, -long(L) - 1}) qs.push_back(acc.grid_wavevector(m1, m2, m3));
        std::vector<Eigen::MatrixXcd> ref(qs.size(), Eigen::MatrixXcd::Zero(3, 3));
        double single_site_err = 0.0;
        for (int smp = 0; smp < n_samples; ++smp) {
            randomize(lat);
            acc.add_sample(lat.spins);
            for (size_t i = 0; i < qs.size(); ++i) ref[i] += direct_sq(lat, qs[i]) / double(n_samples);
            // Single-configuration helper: per site, the same global-frame tensor.
            const Eigen::MatrixXcd per_site = lat.structure_factor_matrix(qs[1]);
            single_site_err = std::max(single_site_err,
                                       max_abs(per_site * double(lat.N_atoms) - direct_sq(lat, qs[1])) /
                                           std::max(1.0, max_abs(per_site)));
        }
        double err = 0.0, scale = 0.0;
        for (size_t i = 0; i < qs.size(); ++i) {
            err = std::max(err, max_abs(acc.structure_factor(qs[i]) - ref[i]));
            scale = std::max(scale, max_abs(ref[i]));
        }
        check(err <= 1e-12 * std::max(1.0, scale),
              "pyrochlore L=" + std::to_string(L) + ": FFT S(q) == direct sum over 3 random samples, q in several zones "
              "(rel err " + std::to_string(err / std::max(1.0, scale)) + ")");
        check(single_site_err < 1e-12, "Lattice::structure_factor_matrix (per site) == direct global-frame sum / N_atoms");
        const Eigen::Vector3d q = qs[7];
        check_close(lat.structure_factor(q), lat.structure_factor_matrix(q).trace().real(), 1e-12,
                    "Lattice::structure_factor = Tr S(q) (all components, global frame)");
        // Hermiticity: S^{ab}(q)^* = S^{ba}(q).
        const Eigen::MatrixXcd S = acc.structure_factor(qs[5]);
        check(max_abs(S - S.adjoint()) < 1e-12 * std::max(1.0, max_abs(S)), "S(q) is Hermitian");

        // Real-space correlation C_{ss'}^{ab}(d) = (1/N_c) Σ_R <S^a_{R,s} S^b_{R+d,s'}> (one fresh sample).
        Acc one = lat.create_correlation_accumulator();
        randomize(lat);
        one.add_sample(lat.spins);
        double cerr = 0.0;
        const size_t s = 1, s2 = 3, a = 0, b = 2;
        const std::vector<double> C = one.real_space_correlation(s, s2, a, b);
        for (size_t d1 = 0; d1 < L; ++d1)
            for (size_t d2 = 0; d2 < L; ++d2)
                for (size_t d3 = 0; d3 < L; ++d3) {
                    double sum = 0.0;
                    for (size_t n1 = 0; n1 < L; ++n1)
                        for (size_t n2 = 0; n2 < L; ++n2)
                            for (size_t n3 = 0; n3 < L; ++n3) {
                                const size_t i = lat.flatten_index(n1, n2, n3, s);
                                const size_t j = lat.flatten_index((n1 + d1) % L, (n2 + d2) % L, (n3 + d3) % L, s2);
                                sum += global_spin(lat, i)[a] * global_spin(lat, j)[b];
                            }
                    cerr = std::max(cerr, std::abs(sum / double(L * L * L) - C[(d1 * L + d2) * L + d3]));
                }
        check(cerr < 1e-12, "real_space_correlation == direct (1/N_c) Σ_R S_{R,s} S_{R+d,s'} (err " +
                                std::to_string(cerr) + ")");
    }
}

// ---------------------------------------------------------------------------
void test_closed_forms() {
    // Ferromagnet along n in the GLOBAL frame on the pyrochlore (local spins F_s^T n S).
    {
        const size_t L = 3;
        const double S0 = 1.5;
        Lattice lat(Pyrochlore(3), L, L, L, S0);
        const Eigen::Vector3d n = Eigen::Vector3d(1, 2, -2) / 3.0;
        for (size_t i = 0; i < lat.lattice_size; ++i) lat.spins[i] = lat.sublattice_frames[i % 4].transpose() * n * S0;
        Acc acc = lat.create_correlation_accumulator();
        acc.add_sample(lat.spins);
        const double Nc = double(L * L * L);
        const Eigen::Matrix3d nn = n * n.transpose();
        double err = 0.0;
        for (auto m : {std::array<long, 3>{0, 0, 0}, std::array<long, 3>{long(L), 0, 0},
                       std::array<long, 3>{long(L), -long(L), 2 * long(L)}}) {
            const Eigen::Vector3d G = acc.grid_wavevector(m[0], m[1], m[2]);
            cplx basis(0.0);
            for (size_t s = 0; s < 4; ++s) basis += std::exp(cplx(0.0, -G.dot(lat.unit_cell.lattice_pos[s])));
            const Eigen::MatrixXcd expect = (Nc * std::norm(basis) * S0 * S0 * nn).cast<cplx>();
            err = std::max(err, max_abs(acc.structure_factor(G) - expect));
        }
        check(err < 1e-10, "ferromagnet: S(G) = N_c |Σ_s e^{-iG·τ_s}|^2 S^2 n n^T at Bragg points (err " +
                               std::to_string(err) + ")");
        const Eigen::Vector3d q1 = acc.grid_wavevector(1, 0, 2);
        check(max_abs(acc.structure_factor(q1)) < 1e-10, "ferromagnet: S(q) = 0 away from Bragg points");
        check(max_abs(acc.structure_factor(Eigen::Vector3d::Zero(), true)) < 1e-10,
              "ferromagnet: connected S(0) = 0 (no fluctuations)");
        check((acc.sublattice_magnetization(2) - S0 * n).norm() < 1e-12,
              "sublattice magnetisation reported in the global frame");
    }
    // Néel state on the simple cubic lattice: S^{zz}(π,π,π) = N_c S^2.
    {
        const size_t L = 4;
        Lattice lat(simple_cubic_cell(1), L, L, L, 1.0);
        for (size_t a = 0; a < L; ++a)
            for (size_t b = 0; b < L; ++b)
                for (size_t c = 0; c < L; ++c)
                    lat.spins[lat.flatten_index(a, b, c, 0)] = Eigen::Vector3d(0, 0, ((a + b + c) % 2) ? -1.0 : 1.0);
        Acc acc = lat.create_correlation_accumulator();
        acc.add_sample(lat.spins);
        const Eigen::Vector3d Q(M_PI, M_PI, M_PI);
        const Eigen::MatrixXcd S = acc.structure_factor(Q);
        check_close(S(2, 2).real(), double(L * L * L), 1e-10, "Neel: S^zz(pi,pi,pi) = N_c S^2");
        double rest = 0.0;
        for (long m1 = 0; m1 < long(L); ++m1)
            for (long m2 = 0; m2 < long(L); ++m2)
                for (long m3 = 0; m3 < long(L); ++m3)
                    if (!(m1 == 2 && m2 == 2 && m3 == 2))
                        rest = std::max(rest, max_abs(acc.structure_factor(acc.grid_wavevector(m1, m2, m3))));
        check(rest < 1e-10, "Neel: S(q) = 0 at every other grid point");
    }
    // Planar spiral on a chain: S_n = S (cos Qn, sin Qn, 0) gives the chiral
    // (antisymmetric imaginary) part Im S^{xy}(Q) = N S^2 / 4.
    {
        const size_t L = 12;
        const long m0 = 5;
        const double S0 = 2.0;
        Lattice lat(chain_cell(Eigen::Matrix3d::Identity()), L, 1, 1, S0);
        const double Q = 2.0 * M_PI * double(m0) / double(L);
        for (size_t n = 0; n < L; ++n) lat.spins[n] = S0 * Eigen::Vector3d(std::cos(Q * n), std::sin(Q * n), 0.0);
        Acc acc = lat.create_correlation_accumulator();
        acc.add_sample(lat.spins);
        const Eigen::MatrixXcd S = acc.structure_factor(acc.grid_wavevector(m0, 0, 0));
        check_close(S(0, 1).imag(), double(L) * S0 * S0 / 4.0, 1e-10, "spiral: Im S^xy(Q) = N S^2/4 (chiral part kept)");
        check_close(S(1, 0).imag(), -double(L) * S0 * S0 / 4.0, 1e-10, "spiral: Im S^yx(Q) = -N S^2/4");
        check_close(S(0, 0).real(), double(L) * S0 * S0 / 4.0, 1e-10, "spiral: S^xx(Q) = N S^2/4");
        const Eigen::MatrixXcd Sm = acc.structure_factor(acc.grid_wavevector(-m0, 0, 0));
        check_close(Sm(0, 1).imag(), -double(L) * S0 * S0 / 4.0, 1e-10, "spiral: chirality flips at -Q (half-grid conjugation)");
    }
    // Sum rules on random configurations.
    {
        // Bravais lattice: Σ_{q in one zone} Tr S(q) = N_c S^2.
        const size_t L1 = 5, L2 = 4;
        const double S0 = 1.5;
        Lattice tri(triangular_heisenberg_cell(1.0), L1, L2, 1, S0);
        Acc acc = tri.create_correlation_accumulator();
        for (int k = 0; k < 4; ++k) {
            randomize(tri);
            acc.add_sample(tri.spins);
        }
        double sum = 0.0;
        for (long m1 = 0; m1 < long(L1); ++m1)
            for (long m2 = 0; m2 < long(L2); ++m2) sum += acc.structure_factor(acc.grid_wavevector(m1, m2, 0)).trace().real();
        check_close(sum / double(L1 * L2), S0 * S0, 1e-12, "sum rule, triangular: (1/N_c) Σ_q Tr S(q) = S^2");

        // Pyrochlore: τ_s - τ_s' are half lattice vectors, so the 2x2x2 extended
        // zone averages the sublattice interference away: <Tr S(q)> = N_atoms S^2.
        const size_t L = 3;
        Lattice pyro(Pyrochlore(3), L, L, L, S0);
        Acc pa = pyro.create_correlation_accumulator();
        for (int k = 0; k < 3; ++k) {
            randomize(pyro);
            pa.add_sample(pyro.spins);
        }
        double ext = 0.0;
        for (long m1 = 0; m1 < 2 * long(L); ++m1)
            for (long m2 = 0; m2 < 2 * long(L); ++m2)
                for (long m3 = 0; m3 < 2 * long(L); ++m3)
                    ext += pa.structure_factor(pa.grid_wavevector(m1, m2, m3)).trace().real();
        check_close(ext / (8.0 * L * L * L), 4.0 * S0 * S0, 1e-11,
                    "sum rule, pyrochlore: average of Tr S(q) over the extended zone = N_atoms S^2 per cell");
        double c0 = 0.0;
        for (size_t s = 0; s < 4; ++s)
            for (size_t a = 0; a < 3; ++a) c0 += pa.real_space_correlation(s, s, a, a)[0];
        check_close(c0, 4.0 * S0 * S0, 1e-12, "sum rule, pyrochlore: Σ_s Tr C_ss(0) = N_atoms S^2 (Parseval)");
    }
}

// ---------------------------------------------------------------------------
void test_dimers() {
    // Bond classes of the pyrochlore: 12 geometrically distinct NN bonds (up and
    // down tetrahedra), every coupled bond of the lattice in exactly one class.
    {
        const size_t L = 3;
        Lattice lat(pyrochlore_heisenberg_cell(1.0), L, L, L, 1.0);
        Acc acc = lat.create_correlation_accumulator();
        const auto& cl = acc.bond_classes();
        check(cl.size() == 12, "pyrochlore: 12 nearest-neighbour bond classes (got " + std::to_string(cl.size()) + ")");
        std::set<std::pair<size_t, size_t>> lattice_bonds, class_bonds;
        for (size_t i = 0; i < lat.lattice_size; ++i)
            for (size_t j : lat.bilinear_partners[i]) lattice_bonds.insert({std::min(i, j), std::max(i, j)});
        bool geometry_ok = true;
        size_t n_class_bonds = 0;
        for (const auto& c : cl) {
            geometry_ok = geometry_ok && std::abs(c.length - std::sqrt(2.0) / 4.0) < 1e-12;
            const Eigen::Vector3d bond = lat.unit_cell.bond_vector(c.source, c.partner, c.offset);
            geometry_ok = geometry_ok && (c.center - (lat.unit_cell.lattice_pos[c.source] + 0.5 * bond)).norm() < 1e-12;
            for (size_t n1 = 0; n1 < L; ++n1)
                for (size_t n2 = 0; n2 < L; ++n2)
                    for (size_t n3 = 0; n3 < L; ++n3) {
                        const size_t i = lat.flatten_index(n1, n2, n3, c.source);
                        const size_t j = lat.flatten_index_periodic(int(n1) + c.offset[0], int(n2) + c.offset[1],
                                                                    int(n3) + c.offset[2], c.partner);
                        class_bonds.insert({std::min(i, j), std::max(i, j)});
                        ++n_class_bonds;
                    }
        }
        check(geometry_ok, "pyrochlore bond classes: length r_NN, centre = τ_source + bond/2");
        check(n_class_bonds == lattice_bonds.size() && class_bonds == lattice_bonds,
              "pyrochlore: the classes cover every coupled bond exactly once (" + std::to_string(n_class_bonds) +
                  " class bonds, " + std::to_string(lattice_bonds.size()) + " lattice bonds)");
        // The same unordered bond declared from either end is one class.
        UnitCell twice = honeycomb_nn_cell();
        twice.set_bilinear_interaction(0.5 * Eigen::Matrix3d::Identity(), 0, 1, Eigen::Vector3i(0, 0, 0));
        check(Acc::bond_classes_from_unit_cell(twice).size() == 3,
              "several couplings on one bond form one class (honeycomb: 3 classes)");
    }

    // Perfect dimer covering of a chain, ↑↓↓↑ ↑↓↓↑ ...: D(n) = S_n·S_{n+1} = -1, +1, -1, +1, ...
    // so C_D(d) = (-1)^d, S_D(π) = N_c and S_D(0) = 0 (one bond class).
    {
        const size_t L = 8;
        Lattice lat(chain_cell(Eigen::Matrix3d::Identity()), L, 1, 1, 1.0);
        const int pattern[4] = {1, -1, -1, 1};
        for (size_t n = 0; n < L; ++n) lat.spins[n] = Eigen::Vector3d(0, 0, pattern[n % 4]);
        Acc acc = lat.create_correlation_accumulator();
        acc.add_sample(lat.spins);
        check(acc.bond_classes().size() == 1, "chain: one bond class");
        check_close(acc.bond_classes()[0].center[0], 0.5, 1e-15, "chain: bond centre at the bond midpoint");
        const auto Sd_pi = acc.dimer_structure_factor(acc.grid_wavevector(long(L) / 2, 0, 0), false);
        const auto Sd_0 = acc.dimer_structure_factor(Eigen::Vector3d::Zero(), false);
        check_close(Sd_pi[2](0, 0).real(), double(L), 1e-12, "dimer covering: S_D^z(pi) = N_c");
        check_close(std::abs(Sd_0[2](0, 0)), 0.0, 1e-12, "dimer covering: S_D^z(0) = 0");
        check_close(acc.dimer_mean(0, 2), 0.0, 1e-15, "dimer covering: <D^z> = 0");
        check_close(std::abs(Sd_pi[0](0, 0)) + std::abs(Sd_pi[1](0, 0)), 0.0, 1e-15, "dimer covering: x, y channels empty");
        // Same through the Lattice wrappers (separate spin / dimer channels).
        Acc w = lat.create_correlation_accumulator();
        lat.accumulate_correlations(w);
        lat.accumulate_dimer_correlations(w);
        check(w.n_samples() == 1 && w.n_dimer_samples() == 1, "Lattice wrappers feed one sample to each channel");
    }

    // Columnar covering of the honeycomb: the intra-cell bond (offset 0) is the
    // dimer. S_A(R) = s(R) z, S_B(R) = -s(R) z with s = (-1)^{n2} makes it the only
    // antiparallel bond: <D^z> = -1 on that class, +1 on the two others.
    {
        const size_t L = 4;
        Lattice lat(honeycomb_nn_cell(), L, L, 1, 1.0);
        for (size_t a = 0; a < L; ++a)
            for (size_t b = 0; b < L; ++b) {
                const double s = (b % 2) ? -1.0 : 1.0;
                lat.spins[lat.flatten_index(a, b, 0, 0)] = Eigen::Vector3d(0, 0, s);
                lat.spins[lat.flatten_index(a, b, 0, 1)] = Eigen::Vector3d(0, 0, -s);
            }
        Acc acc = lat.create_correlation_accumulator();
        acc.add_sample(lat.spins);
        const auto& cl = acc.bond_classes();
        bool means_ok = cl.size() == 3;
        size_t dimer_class = 99;
        for (size_t c = 0; c < cl.size(); ++c) {
            const double expect = cl[c].offset.isZero() ? -1.0 : 1.0;
            if (cl[c].offset.isZero()) dimer_class = c;
            means_ok = means_ok && std::abs(acc.dimer_mean(c, 2) - expect) < 1e-14;
        }
        check(means_ok && dimer_class < 3, "honeycomb columnar covering: <D^z> = -1 on the dimer class, +1 elsewhere");
        check((cl[dimer_class].center - Eigen::Vector3d(0, 0.5 / std::sqrt(3.0), 0)).norm() < 1e-14,
              "honeycomb: dimer class centred at the A-B midpoint");
        // At a reciprocal lattice vector G: S_D^z_{cc'}(G) = N_c D_c D_c' e^{-iG(ρ_c - ρ_c')}.
        const Eigen::Vector3d G = acc.grid_wavevector(long(L), 2 * long(L), 0);
        const auto SG = acc.dimer_structure_factor(G, false);
        double err = 0.0;
        for (size_t c = 0; c < 3; ++c)
            for (size_t c2 = 0; c2 < 3; ++c2) {
                const cplx expect = double(L * L) * acc.dimer_mean(c, 2) * acc.dimer_mean(c2, 2) *
                                    std::exp(cplx(0.0, -G.dot(cl[c].center - cl[c2].center)));
                err = std::max(err, std::abs(SG[2](c, c2) - expect));
            }
        check(err < 1e-11, "honeycomb covering: S_D(G) carries the bond-centre phases (err " + std::to_string(err) + ")");
        check(max_abs(acc.dimer_structure_factor(acc.grid_wavevector(1, 0, 0), false)[2]) < 1e-11,
              "honeycomb covering: S_D(q) = 0 away from G (uniform per class)");
    }

    // FFT dimer S(q) == direct double sum over bonds (pyrochlore, random spins).
    {
        const size_t L = 3;
        Lattice lat(pyrochlore_heisenberg_cell(1.0), L, L, L, 1.0);
        randomize(lat);
        Acc acc = lat.create_correlation_accumulator();
        acc.add_sample(lat.spins);
        const auto& cl = acc.bond_classes();
        const Eigen::Vector3d q = acc.grid_wavevector(1, -2, long(L) + 1);
        const auto S = acc.dimer_structure_factor(q, false);
        double err = 0.0, scale = 0.0;
        for (size_t a = 0; a < 3; ++a) {
            std::vector<cplx> G(cl.size(), cplx(0.0));
            for (size_t c = 0; c < cl.size(); ++c)
                for (size_t n1 = 0; n1 < L; ++n1)
                    for (size_t n2 = 0; n2 < L; ++n2)
                        for (size_t n3 = 0; n3 < L; ++n3) {
                            const size_t i = lat.flatten_index(n1, n2, n3, cl[c].source);
                            const size_t j = lat.flatten_index_periodic(int(n1) + cl[c].offset[0], int(n2) + cl[c].offset[1],
                                                                        int(n3) + cl[c].offset[2], cl[c].partner);
                            const Eigen::Vector3d r = double(n1) * lat.unit_cell.lattice_vectors[0] +
                                                      double(n2) * lat.unit_cell.lattice_vectors[1] +
                                                      double(n3) * lat.unit_cell.lattice_vectors[2] + cl[c].center;
                            G[c] += global_spin(lat, i)[a] * global_spin(lat, j)[a] * std::exp(cplx(0.0, -q.dot(r)));
                        }
            for (size_t c = 0; c < cl.size(); ++c)
                for (size_t c2 = 0; c2 < cl.size(); ++c2) {
                    const cplx ref = G[c] * std::conj(G[c2]) / double(L * L * L);
                    err = std::max(err, std::abs(S[a](c, c2) - ref));
                    scale = std::max(scale, std::abs(ref));
                }
        }
        check(err < 1e-12 * std::max(1.0, scale), "dimer S_D(q): FFT == direct sum over bond pairs (err " +
                                                      std::to_string(err) + ")");
    }
}

// ---------------------------------------------------------------------------
void test_errors_merge_validation() {
    const size_t L = 3;
    Lattice lat(pyrochlore_heisenberg_cell(1.0), L, L, L, 1.0);
    const Eigen::Vector3d q = Eigen::Vector3d::Zero();
    // n_bins = 16 and 12 samples: one sample per bin, so the jackknife error is
    // the standard error of the mean, sqrt(Σ (x_i - x̄)^2 / (n (n-1))).
    {
        Acc acc = lat.create_correlation_accumulator(Acc::Options{16}, 1);
        std::vector<double> x;
        const Eigen::Vector3d qq = acc.grid_wavevector(1, 0, 0);
        for (int k = 0; k < 12; ++k) {
            randomize(lat);
            acc.add_sample(lat.spins);
            x.push_back(direct_sq(lat, qq)(0, 0).real());
        }
        double mean = 0.0;
        for (double v : x) mean += v / x.size();
        double ss = 0.0;
        for (double v : x) ss += (v - mean) * (v - mean);
        const double se = std::sqrt(ss / (x.size() * (x.size() - 1.0)));
        const Acc::Estimate e = acc.structure_factor_estimate(qq);
        check_close(e.mean(0, 0).real(), mean, 1e-12, "estimate mean = sample mean");
        check_close(e.error(0, 0).real(), se, 1e-12 * std::max(1.0, se), "jackknife error = standard error (one sample per bin)");
        check(e.n_bins == 12, "12 bins used");
    }
    // Rebinning keeps the mean exact; merge() == one accumulator fed with everything.
    {
        Acc all = lat.create_correlation_accumulator(Acc::Options{4}, 1);
        Acc a = lat.create_correlation_accumulator(Acc::Options{4}, 1);
        Acc b = lat.create_correlation_accumulator(Acc::Options{4}, 1);
        Eigen::MatrixXcd ref = Eigen::MatrixXcd::Zero(3, 3);
        const Eigen::Vector3d qq = all.grid_wavevector(0, 1, 2);
        const int n = 37;
        for (int k = 0; k < n; ++k) {
            randomize(lat);
            all.add_sample(lat.spins);
            (k < 20 ? a : b).add_sample(lat.spins);
            ref += direct_sq(lat, qq) / double(n);
        }
        check(max_abs(all.structure_factor(qq) - ref) < 1e-12 * std::max(1.0, max_abs(ref)),
              "rebinned accumulator (n_bins = 4, 37 samples): mean exact");
        const Acc::Estimate e = all.structure_factor_estimate(qq);
        check(e.n_bins >= 2 && e.n_bins <= 4 && std::isfinite(e.error(0, 0).real()), "rebinned: 2..4 bins, finite errors");
        Acc merged;               // uninitialised: becomes a copy
        merged.merge(a);
        merged.merge(b);
        check(merged.n_samples() == size_t(n) && merged.n_dimer_samples() == size_t(n), "merge: sample counts add up");
        check(max_abs(merged.structure_factor(qq) - all.structure_factor(qq)) < 1e-12 * std::max(1.0, max_abs(ref)),
              "merge: S(q) equals the single accumulator");
        check(max_abs(merged.structure_factor(q, true) - all.structure_factor(q, true)) < 1e-11,
              "merge: connected S(0) (uses the merged means)");
        double dmax = 0.0;
        for (size_t c = 0; c < all.bond_classes().size(); ++c)
            for (size_t al = 0; al < 3; ++al) dmax = std::max(dmax, std::abs(merged.dimer_mean(c, al) - all.dimer_mean(c, al)));
        check(dmax < 1e-13, "merge: every dimer mean (all components) merged");
        const auto Dm = merged.dimer_structure_factor(qq, true), Da = all.dimer_structure_factor(qq, true);
        check(max_abs(Dm[1] - Da[1]) < 1e-11, "merge: dimer S(q) equals the single accumulator");
        Acc other(lat.create_correlation_accumulator(Acc::Options{8}, 1));
        bool threw = false;
        try { merged.merge(other); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "merge rejects a different n_bins");
        Lattice big(pyrochlore_heisenberg_cell(1.0), 4, 4, 4, 1.0);
        Acc wrong = big.create_correlation_accumulator(Acc::Options{4}, 1);
        threw = false;
        try { merged.merge(wrong); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "merge rejects different dimensions");
        all.reset();
        check(all.n_samples() == 0 && max_abs(all.structure_factor(qq)) == 0.0, "reset clears the samples");
    }
    // Input validation.
    {
        Acc acc = lat.create_correlation_accumulator();
        bool threw = false;
        try { acc.structure_factor(Eigen::Vector3d(0.1, 0.2, 0.3)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "off-grid q is rejected (std::invalid_argument)");
        threw = false;
        try {
            acc.save_structure_factor_grid("/dev/null", {0.0, 0.5}, {0.0, 0.0}, {0.0, 0.0}, 4, 1, 1,
                                           acc.reciprocal_vectors()[0], acc.reciprocal_vectors()[1],
                                           acc.reciprocal_vectors()[2]);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "save_structure_factor_grid rejects a q range off the commensurate grid");
        std::vector<Eigen::VectorXd> bad(lat.spins.begin(), lat.spins.end());
        bad[5] = Eigen::Vector2d(1, 0);
        threw = false;
        try { acc.add_sample(bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a spin of the wrong dimension is rejected");
        bad.pop_back();
        threw = false;
        try { acc.add_sample(bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a configuration of the wrong size is rejected");
        Acc::Geometry g = lat.correlation_geometry();
        g.frames[1] = Eigen::Matrix2d::Identity();
        threw = false;
        try { Acc x; x.initialize(g); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "frames of the wrong size are rejected");
        g = lat.correlation_geometry();
        g.bond_classes[0].offset = -g.bond_classes[0].offset;
        std::swap(g.bond_classes[0].source, g.bond_classes[0].partner);
        threw = false;
        try { Acc x; x.initialize(g); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a bond class in non-canonical orientation is rejected");
        threw = false;
        try { Acc x; x.initialize(lat.correlation_geometry(), Acc::Options{3}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "odd n_bins is rejected");
    }
    // Generic spin dimension (2 components): FFT == direct.
    {
        Acc::Geometry g;
        g.dim1 = 5;
        g.dim2 = 3;
        g.dim3 = 1;
        g.n_sublattices = 2;
        g.spin_dim = 2;
        g.positions = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.3, 0.1, 0)};
        Acc acc;
        acc.initialize(g);
        std::vector<Eigen::VectorXd> s(30);
        for (auto& v : s) v = random_unit(2, 1.0);
        acc.add_sample(s);
        const Eigen::Vector3d q = acc.grid_wavevector(2, -1, 0);
        Eigen::VectorXcd A = Eigen::VectorXcd::Zero(2);
        for (size_t c = 0; c < 15; ++c)
            for (size_t sub = 0; sub < 2; ++sub) {
                const Eigen::Vector3d r = double(c / 3) * g.lattice_vectors[0] + double(c % 3) * g.lattice_vectors[1] + g.positions[sub];
                A += s[c * 2 + sub].cast<cplx>() * std::exp(cplx(0.0, -q.dot(r)));
            }
        check(max_abs(acc.structure_factor(q) - A * A.adjoint() / 15.0) < 1e-12, "spin_dim = 2: FFT == direct sum");
    }
}

// ---------------------------------------------------------------------------
void test_cost_and_io() {
    // O(N log N): one pyrochlore sample at L = 12 (6912 sites, spins + 12 dimer
    // classes) used to take ~2 s with the O(N_c^2) kernels.
    {
        Lattice lat(pyrochlore_heisenberg_cell(1.0), 12, 12, 12, 1.0);
        Acc acc = lat.create_correlation_accumulator();
        randomize(lat);
        const auto t0 = std::chrono::steady_clock::now();
        const int n = 4;
        for (int k = 0; k < n; ++k) acc.add_sample(lat.spins);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n;
        std::printf("    L=12 pyrochlore: %.2f ms per sample, %.1f MB held\n", 1e3 * dt, acc.storage_bytes() / 1e6);
        check(dt < 1.0, "L = 12 pyrochlore sample (spins + dimers) in < 1 s");
    }
#ifdef HDF5_ENABLED
    {
        const size_t L = 3;
        Lattice lat(pyrochlore_heisenberg_cell(1.0), L, L, L, 1.0);
        Acc acc = lat.create_correlation_accumulator(Acc::Options{4}, 1);
        for (int k = 0; k < 5; ++k) {
            randomize(lat);
            acc.add_sample(lat.spins);
        }
        const std::string file = (std::filesystem::temp_directory_path() /
                                  ("test_observables_" + std::to_string(::getpid()) + ".h5")).string();
        acc.save_hdf5(file);
        H5::H5File f(file, H5F_ACC_RDONLY);
        H5::DataSet ds = f.openDataSet("/correlations/structure_factor");
        hsize_t dims[6];
        ds.getSpace().getSimpleExtentDims(dims);
        std::vector<double> S(dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * dims[5]);
        ds.read(S.data(), H5::PredType::NATIVE_DOUBLE);
        const Eigen::MatrixXcd ref = acc.structure_factor(acc.grid_wavevector(1, 2, 0));
        const size_t base = ((1 * L + 2) * L + 0) * 9;
        double err = 0.0;
        for (size_t a = 0; a < 3; ++a)
            for (size_t b = 0; b < 3; ++b)
                err = std::max(err, std::abs(cplx(S[(base + a * 3 + b) * 2], S[(base + a * 3 + b) * 2 + 1]) - ref(a, b)));
        check(dims[0] == L && dims[3] == 3 && dims[5] == 2 && err < 1e-13, "HDF5: structure_factor dataset round-trips");
        std::vector<long long> counts(4);
        f.openDataSet("/correlations/spin_bin_counts").read(counts.data(), H5::PredType::NATIVE_LLONG);
        check(counts[0] + counts[1] + counts[2] + counts[3] == 5, "HDF5: bin counts add up to n_samples");
        H5::DataSet bc = f.openDataSet("/correlations/dimers/bond_class_offset");
        hsize_t bdims[2];
        bc.getSpace().getSimpleExtentDims(bdims);
        check(bdims[0] == 12 && bdims[1] == 3, "HDF5: 12 bond classes with offsets");
        f.close();
        std::filesystem::remove(file);
    }
#endif
}

}  // namespace

int main() {
    test_fft();
    test_fft_equals_direct_pyrochlore();
    test_closed_forms();
    test_dimers();
    test_errors_merge_validation();
    test_cost_and_io();
    return finish("test_observables");
}
