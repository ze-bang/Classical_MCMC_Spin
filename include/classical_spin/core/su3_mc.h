#pragma once
/**
 * su3_mc.h — Monte Carlo moves for SU(3) (qutrit) coherent states, shared by
 * Lattice (spin_dim 8) and MixedLattice.
 *
 * A site stores the Gell-Mann expectations n^a = <psi|lambda^a|psi> of a pure
 * state psi in C^3 (|n|^2 = 4/3, see su3_coherent_state.h), so the state space
 * is CP^2 and the invariant measure is the Fubini–Study (Haar) measure. Every
 * move below maps CP^2 onto itself and satisfies detailed balance with respect
 * to that measure:
 *
 *  - random_cp2      : Haar-uniform psi (normalised complex Gaussian vector);
 *  - propose_cp2     : psi' = normalise(psi + sigma z), z complex Gaussian. The
 *                      kernel is invariant under the stabiliser of [psi], so its
 *                      density depends only on |<psi|psi'>| and is symmetric;
 *  - sample_linear_cp2: exact draw from exp(-beta h.n) (heat bath): in the
 *                      eigenbasis of H = h.lambda the populations p_k =
 *                      |<v_k|psi>|^2 are uniform on the simplex under the
 *                      Fubini–Study measure (moment map of the torus action,
 *                      Duistermaat–Heckman) and the energy is sum_k p_k e_k;
 *  - randomize_phases_cp2: overrelaxation — random relative phases in the
 *                      eigenbasis of H, which conserve every p_k and hence
 *                      h.n; a random unitary from an inversion-symmetric
 *                      distribution, so the kernel is symmetric;
 *  - ground_state_cp2: argmin over CP^2 of h.n (lowest eigenvector of H);
 *  - project_to_cp2  : closest pure state (top eigenvector of rho).
 */

#include "classical_spin/core/simple_linear_alg.h"   // random_normal_lehman, random_double_lehman
#include "classical_spin/core/su3_coherent_state.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace classical_spin {
namespace su3 {

/// Complex Gaussian with E|z|^2 = 1/3: psi + sigma z then moves a normalised
/// qutrit state by the same typical geodesic angle as an SU(2) Gaussian move
/// of width sigma (E|z_perp|^2 = 2/3 in both cases).
inline Complex complex_normal_sixth() {
    constexpr double s = 0.40824829046386301637;   // 1/sqrt(6)
    const double re = random_normal_lehman();
    return {s * re, s * random_normal_lehman()};
}

/// Normalise psi (in place) and write its expectations to n; false if psi ~ 0.
inline bool normalise_to_expectations(Complex* psi, double* n) {
    const double nrm2 = std::norm(psi[0]) + std::norm(psi[1]) + std::norm(psi[2]);
    if (!(nrm2 > 1e-300)) return false;
    const double inv = 1.0 / std::sqrt(nrm2);
    for (int k = 0; k < 3; ++k) psi[k] *= inv;
    pure_expectations(psi, n);
    return true;
}

/// Haar-uniform pure state, written to n[0..7].
inline void random_cp2(double* n) {
    Complex psi[3];
    do {
        for (int k = 0; k < 3; ++k) psi[k] = complex_normal_sixth();
    } while (!normalise_to_expectations(psi, n));
}

/// Symmetric small move of width sigma from the pure state n; out may alias n.
inline void propose_cp2(const double* n, double sigma, double* out) {
    Complex psi[3];
    psi_from_pure_expectations(n, psi);
    for (int k = 0; k < 3; ++k) psi[k] += sigma * complex_normal_sixth();
    double tmp[8];
    if (normalise_to_expectations(psi, tmp)) std::copy_n(tmp, 8, out);
    else std::copy_n(n, 8, out);
}

/**
 * Exact draw from P(psi) ∝ exp(-beta <psi|H|psi>), H = h.lambda, on CP^2.
 * With eigenvalues e_0 <= e_1 <= e_2, a = beta (e_1 - e_0), b = beta (e_2 - e_0):
 * p_2 has the marginal ∝ e^{-b p_2} (1 - e^{-a (1 - p_2)}), drawn from the
 * truncated exponential e^{-b p_2} and accepted with (1 - e^{-a(1-p_2)}) /
 * (1 - e^{-a}) (>= 1/2 on average); then p_1 | p_2 is a truncated exponential
 * on [0, 1 - p_2]; phases uniform. Writes n[0..7].
 */
inline void sample_linear_cp2(const double* h, double beta, double* n) {
    Eigen::Vector3d ev;
    Matrix3c V;
    eigen_hermitian3(gell_mann_sum(h), ev, V);
    const double a = beta * (ev(1) - ev(0)), b = beta * (ev(2) - ev(0));
    double p2 = 0.0;
    for (;;) {
        const double u = random_double_lehman(0.0, 1.0);
        p2 = (b > 1e-12) ? -std::log1p(u * std::expm1(-b)) / b : u;
        p2 = std::clamp(p2, 0.0, 1.0);
        const double acc = (a > 1e-12) ? std::expm1(-a * (1.0 - p2)) / std::expm1(-a) : 1.0 - p2;
        if (random_double_lehman(0.0, 1.0) < acc) break;
    }
    const double w = 1.0 - p2;
    const double u = random_double_lehman(0.0, 1.0);
    double p1 = (a * w > 1e-12) ? -std::log1p(u * std::expm1(-a * w)) / a : u * w;
    p1 = std::clamp(p1, 0.0, w);
    const double p0 = std::max(0.0, 1.0 - p1 - p2);
    const Complex c0(std::sqrt(p0), 0.0);
    const Complex c1 = std::polar(std::sqrt(p1), random_double_lehman(0.0, 2.0 * M_PI));
    const Complex c2 = std::polar(std::sqrt(p2), random_double_lehman(0.0, 2.0 * M_PI));
    Complex psi[3];
    for (int k = 0; k < 3; ++k) psi[k] = c0 * V(k, 0) + c1 * V(k, 1) + c2 * V(k, 2);
    normalise_to_expectations(psi, n);
}

/// Overrelaxation about h: random relative phases in the eigenbasis of h.lambda (h.n conserved).
inline void randomize_phases_cp2(const double* h, const double* n, double* out) {
    Eigen::Vector3d ev;
    Matrix3c V;
    eigen_hermitian3(gell_mann_sum(h), ev, V);
    Complex psi[3];
    psi_from_pure_expectations(n, psi);
    Complex c[3];
    for (int k = 0; k < 3; ++k)
        c[k] = std::conj(V(0, k)) * psi[0] + std::conj(V(1, k)) * psi[1] + std::conj(V(2, k)) * psi[2];
    c[1] *= std::polar(1.0, random_double_lehman(0.0, 2.0 * M_PI));
    c[2] *= std::polar(1.0, random_double_lehman(0.0, 2.0 * M_PI));
    for (int k = 0; k < 3; ++k) psi[k] = V(k, 0) * c[0] + V(k, 1) * c[1] + V(k, 2) * c[2];
    double tmp[8];
    if (normalise_to_expectations(psi, tmp)) std::copy_n(tmp, 8, out);
    else std::copy_n(n, 8, out);
}

/// argmin_{n in CP^2} h.n: the expectations of the lowest eigenvector of h.lambda.
inline void ground_state_cp2(const double* h, double* n) {
    Eigen::Vector3d ev;
    Matrix3c V;
    eigen_hermitian3(gell_mann_sum(h), ev, V);
    Complex psi[3] = {V(0, 0), V(1, 0), V(2, 0)};
    if (!normalise_to_expectations(psi, n)) {
        psi[0] = 1.0; psi[1] = 0.0; psi[2] = 0.0;
        pure_expectations(psi, n);
    }
}

/// Replace n by the closest pure state (top eigenvector of rho = 1/3 + n.lambda/2;
/// exact for any positive multiple of a pure state).
inline void project_to_cp2(double* n) {
    Vector8r v;
    for (int a = 0; a < 8; ++a) v(a) = n[a];
    const Vector3c psi = psi_from_expectations(v);
    Complex p[3] = {psi(0), psi(1), psi(2)};
    normalise_to_expectations(p, n);
}

}  // namespace su3
}  // namespace classical_spin
