// physics_test_util.h — shared helpers for the physics validation suite.
//
// Every test in tests/ compares a simulation kernel against a result that is
// known independently of the code under test: closed-form statistical
// mechanics (free spins, the 1D Heisenberg ring), deterministic quadrature of
// the Boltzmann integral for tiny clusters, analytic spin precession and
// linear spin-wave frequencies, and exactly known classical ground-state
// energies. Statistical comparisons use a blocking (batch-means) error bar,
// so a correct sampler passes with overwhelming probability at the chosen
// tolerance while a detailed-balance violation of a few percent fails.
#pragma once

#include "classical_spin/core/unitcell.h"
#include "classical_spin/lattice/lattice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace phys_test {

// --------------------------------------------------------------------------
// Minimal check harness
// --------------------------------------------------------------------------
inline int& failures() { static int n = 0; return n; }

inline void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
    if (!ok) ++failures();
}

// |measured - expected| <= n_sigma * err + abs_floor
inline void check_stat(double measured, double err, double expected,
                       const std::string& what, double n_sigma = 5.0,
                       double abs_floor = 0.0) {
    const double dev = std::abs(measured - expected);
    const double tol = n_sigma * err + abs_floor;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s: measured %.6f +- %.6f, exact %.6f (dev %.2f sigma)",
                  what.c_str(), measured, err, expected,
                  err > 0 ? dev / err : (dev > 0 ? 1e9 : 0.0));
    check(dev <= tol, buf);
}

inline void check_close(double measured, double expected, double tol,
                        const std::string& what) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s: got %.10g, expected %.10g (tol %.1e)",
                  what.c_str(), measured, expected, tol);
    check(std::abs(measured - expected) <= tol, buf);
}

inline int finish(const char* suite) {
    if (failures() == 0) {
        std::printf("\n%s: all checks passed\n", suite);
        return 0;
    }
    std::printf("\n%s: %d check(s) FAILED\n", suite, failures());
    return 1;
}

// --------------------------------------------------------------------------
// Statistics: batch-means error of a (correlated) time series.
// --------------------------------------------------------------------------
struct MeanErr { double mean; double err; };

inline MeanErr batch_means(const std::vector<double>& x, size_t n_batches = 32) {
    const size_t n = x.size();
    if (n < 2 * n_batches) n_batches = std::max<size_t>(2, n / 2);
    const size_t b = n / n_batches;
    double total = 0.0;
    std::vector<double> means(n_batches, 0.0);
    for (size_t k = 0; k < n_batches; ++k) {
        for (size_t i = 0; i < b; ++i) means[k] += x[k * b + i];
        means[k] /= double(b);
        total += means[k];
    }
    const double mean = total / double(n_batches);
    double var = 0.0;
    for (double m : means) var += (m - mean) * (m - mean);
    var /= double(n_batches - 1);
    return {mean, std::sqrt(var / double(n_batches))};
}

// --------------------------------------------------------------------------
// Closed forms
// --------------------------------------------------------------------------
// Langevin function L(x) = coth x - 1/x (series near 0 to avoid cancellation).
inline double langevin(double x) {
    if (std::abs(x) < 1e-4) return x / 3.0 - x * x * x / 45.0;
    return 1.0 / std::tanh(x) - 1.0 / x;
}

// d L / dx = 1/x^2 - 1/sinh^2 x
inline double langevin_prime(double x) {
    if (std::abs(x) < 1e-3) return 1.0 / 3.0 - x * x / 15.0;
    const double s = std::sinh(x);
    return 1.0 / (x * x) - 1.0 / (s * s);
}

// --------------------------------------------------------------------------
// Gauss-Legendre quadrature on [-1, 1] (Newton iteration on P_n).
// --------------------------------------------------------------------------
inline void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    for (int i = 0; i < (n + 1) / 2; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double pp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            const double z1 = z;
            z = z1 - p1 / pp;
            if (std::abs(z - z1) < 1e-15) break;
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
}

// Product quadrature on the unit 2-sphere: GL in cos(theta) x trapezoid in phi
// (spectrally accurate for smooth integrands). Weights sum to 4*pi.
struct SphereRule {
    std::vector<Eigen::Vector3d> pts;
    std::vector<double> wts;
};

inline SphereRule sphere_rule(int n_theta, int n_phi) {
    std::vector<double> x, w;
    gauss_legendre(n_theta, x, w);
    SphereRule r;
    for (int a = 0; a < n_theta; ++a) {
        const double ct = x[a], st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
        for (int b = 0; b < n_phi; ++b) {
            const double ph = 2.0 * M_PI * (b + 0.5) / n_phi;
            r.pts.emplace_back(st * std::cos(ph), st * std::sin(ph), ct);
            r.wts.push_back(w[a] * 2.0 * M_PI / n_phi);
        }
    }
    return r;
}

// --------------------------------------------------------------------------
// Model builders (all spin_dim = 3)
// --------------------------------------------------------------------------
inline UnitCell simple_cubic_cell(size_t n_atoms = 1) {
    std::vector<Eigen::Vector3d> pos(n_atoms, Eigen::Vector3d::Zero());
    for (size_t a = 0; a < n_atoms; ++a) pos[a] = Eigen::Vector3d(0.1 * a, 0, 0);
    return UnitCell(3, n_atoms, pos,
                    {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
                     Eigen::Vector3d(0, 0, 1)});
}

// Nearest-neighbour chain along a1: E = sum_i S_i^T J S_{i+1}.
inline UnitCell chain_cell(const Eigen::Matrix3d& J,
                           const Eigen::Vector3d& field = Eigen::Vector3d::Zero(),
                           const Eigen::Matrix3d& onsite = Eigen::Matrix3d::Zero()) {
    UnitCell uc = simple_cubic_cell(1);
    uc.set_bilinear_interaction(J, 0, 0, Eigen::Vector3i(1, 0, 0));
    uc.set_field(field, 0);
    uc.set_onsite_interaction(onsite, 0);
    return uc;
}

// Isotropic triangular lattice (one site per cell, three NN bond directions).
inline UnitCell triangular_heisenberg_cell(double J) {
    Triangular uc(3);
    const Eigen::Matrix3d Jm = J * Eigen::Matrix3d::Identity();
    uc.set_bilinear_interaction(Jm, 0, 0, Eigen::Vector3i(1, 0, 0));
    uc.set_bilinear_interaction(Jm, 0, 0, Eigen::Vector3i(0, 1, 0));
    uc.set_bilinear_interaction(Jm, 0, 0, Eigen::Vector3i(-1, 1, 0));
    return uc;
}

// Isotropic nearest-neighbour pyrochlore. Bond list: every pair of the four
// sublattices inside the "up" tetrahedron (offset 0) plus the corresponding
// "down" tetrahedron bond (offset -/+ a_k), generated by brute-force distance.
inline UnitCell pyrochlore_heisenberg_cell(double J) {
    Pyrochlore uc(3);
    for (size_t a = 0; a < 4; ++a) uc.set_sublattice_frame(Eigen::Matrix3d::Identity(), a);
    const Eigen::Matrix3d Jm = J * Eigen::Matrix3d::Identity();
    const double nn = std::sqrt(2.0) / 4.0;  // NN distance for these vectors
    for (size_t a = 0; a < 4; ++a) {
        for (size_t b = a + 1; b < 4; ++b) {
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    for (int k = -1; k <= 1; ++k) {
                        Eigen::Vector3d d = uc.lattice_pos[b] - uc.lattice_pos[a] +
                                            i * uc.lattice_vectors[0] +
                                            j * uc.lattice_vectors[1] +
                                            k * uc.lattice_vectors[2];
                        if (std::abs(d.norm() - nn) < 1e-9)
                            uc.set_bilinear_interaction(Jm, a, b, Eigen::Vector3i(i, j, k));
                    }
        }
    }
    return uc;
}

// Energy per site sampled along a Markov chain driven by `sweep`.
inline MeanErr sample_energy_density(Lattice& lat, size_t n_therm, size_t n_meas,
                                     const std::function<void()>& sweep) {
    for (size_t i = 0; i < n_therm; ++i) sweep();
    std::vector<double> e;
    e.reserve(n_meas);
    for (size_t i = 0; i < n_meas; ++i) {
        sweep();
        e.push_back(lat.total_energy() / double(lat.lattice_size));
    }
    return batch_means(e);
}

}  // namespace phys_test
