// =============================================================================
// Exercise 06 -- elliptic solvers, spectral methods, and ETDRK4
// =============================================================================
// Three ideas, each of which turns an O(N^2)-ish algorithm into an O(N)-ish one:
//
//   MULTIGRID.  Relaxation (Jacobi/GS) kills high-frequency error fast and
//   low-frequency error not at all -- the error becomes smooth after a few
//   sweeps, and a smooth function is exactly what a coarse grid represents
//   well. Recurse. Result: residual reduced by a fixed factor per V-cycle,
//   independent of N. Optimal, O(N).
//
//   SPECTRAL.  For periodic smooth data, differentiation is multiplication by
//   i*k in Fourier space, with error decaying faster than any power of dx
//   ("spectral accuracy"). 64 points can beat 10^6 finite-difference points.
//
//   EXPONENTIAL TIME DIFFERENCING.  For a stiff SEMILINEAR problem
//   u_t = L u + N(u) with L diagonal in Fourier space, integrate L exactly
//   (as in ex03's exp_euler_step) and only approximate N. ETDRK4 is 4th order
//   with no stability limit from L at all. This is the state of the art for
//   Kuramoto-Sivashinsky / Allen-Cahn / NLS-type problems.
//
// WHAT TO IMPLEMENT
//   1. poisson_residual, gauss_seidel_sweep, sor_sweep
//   2. restrict_full_weighting, prolong_linear, mg_vcycle
//   3. fft (radix-2, in place)
//   4. spectral_derivative, poisson_fft
//   5. Etdrk4::init and Etdrk4::step
//
// GOTCHAS
//   * The optimal SOR parameter for the 1-D Poisson problem on n interior
//     points is omega* = 2/(1 + sin(pi/(n+1))). Off by a little and you lose
//     most of the speed-up; that fragility is why multigrid won.
//   * The ETDRK4 coefficient functions like (e^z - 1 - z)/z^2 suffer
//     catastrophic cancellation as z -> 0 (the modes you care about most).
//     Evaluating them naively in double precision loses ~8 digits. The fix
//     (Kassam & Trefethen 2005) is to average the analytic expression over a
//     small circle in the complex plane around z -- a Cauchy integral, which
//     is perfectly conditioned. That trick is provided below as `phi_contour`;
//     understanding WHY it is needed is the point.
// =============================================================================
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex06 {

using np::Cplx;
using np::Vec;

// ============================ 1. relaxation ================================
// Model problem: -u'' = f on (0,1), u(0) = u(1) = 0, n points, h = 1/(n-1).
// Discretised: (-u_{j-1} + 2 u_j - u_{j+1}) / h^2 = f_j

// r_j = f_j - (-u_{j-1} + 2u_j - u_{j+1})/h^2   (0 at the boundaries)
inline void poisson_residual(const Vec& u, const Vec& f, double h, Vec& r) {
    (void)u; (void)f; (void)h; (void)r;
    NP_TODO("ex06::poisson_residual");
}

// One in-place Gauss-Seidel sweep: u_j <- (u_{j-1} + u_{j+1} + h^2 f_j) / 2,
// sweeping j = 1..n-2 in order and using already-updated values.
inline void gauss_seidel_sweep(Vec& u, const Vec& f, double h) {
    (void)u; (void)f; (void)h;
    NP_TODO("ex06::gauss_seidel_sweep");
}

// SOR: u_j <- (1-omega) u_j + omega * (u_{j-1} + u_{j+1} + h^2 f_j) / 2
inline void sor_sweep(Vec& u, const Vec& f, double h, double omega) {
    (void)u; (void)f; (void)h; (void)omega;
    NP_TODO("ex06::sor_sweep");
}

// ============================ 2. multigrid =================================
// Grids have n = 2^k + 1 points. Coarse grid: nc = (n+1)/2.

// Full weighting: rc[i] = ( r[2i-1] + 2 r[2i] + r[2i+1] ) / 4 on the interior,
// 0 on the coarse boundaries.
inline void restrict_full_weighting(const Vec& r, Vec& rc) {
    (void)r; (void)rc;
    NP_TODO("ex06::restrict_full_weighting");
}

// Linear interpolation: e[2i] = ec[i],  e[2i+1] = (ec[i] + ec[i+1]) / 2.
inline void prolong_linear(const Vec& ec, Vec& e) {
    (void)ec; (void)e;
    NP_TODO("ex06::prolong_linear");
}

// Recursive V-cycle:
//   nu1 pre-smoothing sweeps
//   r = residual;  restrict to rc
//   ec = 0;  recurse on the coarse grid with spacing 2h
//   prolong ec and ADD to u  (coarse-grid CORRECTION, not a coarse solution)
//   nu2 post-smoothing sweeps
// Base case n <= 3: one interior point, solve exactly: u_1 = h^2 f_1 / 2.
inline void mg_vcycle(Vec& u, const Vec& f, double h, int nu1 = 2, int nu2 = 2) {
    (void)u; (void)f; (void)h; (void)nu1; (void)nu2;
    NP_TODO("ex06::mg_vcycle");
}

// ============================ 3. FFT =======================================
// In-place radix-2 Cooley-Tukey. a.size() must be a power of two.
// Forward:  A_k = sum_j a_j exp(-2 pi i j k / N)
// Inverse:  a_j = (1/N) sum_k A_k exp(+2 pi i j k / N)
// Steps: bit-reversal permutation, then log2(N) butterfly stages.
inline void fft(std::vector<Cplx>& a, bool inverse) {
    (void)a; (void)inverse;
    NP_TODO("ex06::fft");
}

// ============================ 4. spectral ==================================
// Periodic domain [0, L) with n = 2^k samples u_j = u(j L / n).
// Wavenumbers: k_m = 2 pi m / L for m = 0..n/2-1, then m - n for m >= n/2.
// The Nyquist mode m = n/2 must be zeroed for an odd-order derivative
// (it is not resolvable and produces a spurious imaginary part).
inline void spectral_derivative(const Vec& u, double L, Vec& du) {
    (void)u; (void)L; (void)du;
    NP_TODO("ex06::spectral_derivative");
}

// Solve -u'' = f on a periodic domain: uhat_m = fhat_m / k_m^2, mean mode 0.
inline void poisson_fft(const Vec& f, double L, Vec& u) {
    (void)f; (void)L; (void)u;
    NP_TODO("ex06::poisson_fft");
}

// ============================ 5. ETDRK4 ====================================
// Provided: contour-integral evaluation of a coefficient function.
// Averages g(z + r e^{i theta}) over M equispaced points on a circle of radius
// `radius`, which is a Cauchy integral for g(z) and is numerically stable even
// where the closed form cancels.
template <class G>
inline Cplx phi_contour(G&& g, Cplx z, int M = 32, double radius = 1.0) {
    Cplx s = 0.0;
    for (int j = 0; j < M; ++j) {
        double th = 2.0 * pi * (double(j) + 0.5) / double(M);
        s += g(z + radius * Cplx(std::cos(th), std::sin(th)));
    }
    return s / double(M);
}

// Kassam-Trefethen ETDRK4 for  v' = lam*v + Nhat(v)  mode by mode.
struct Etdrk4 {
    int n = 0;
    double h = 0;
    std::vector<Cplx> E, E2, Q, f1, f2, f3;

    // lam[m] = the linear symbol of mode m (e.g. -eps*k^2 + 1 for Allen-Cahn).
    //
    //   E  [m] = exp(h lam)
    //   E2 [m] = exp(h lam / 2)
    //   with z = h*lam, evaluated by contour average:
    //   Q  [m] = h * < (e^{z/2} - 1) / z >
    //   f1 [m] = h * < (-4 - z + e^z (4 - 3z + z^2)) / z^3 >
    //   f2 [m] = h * < ( 2 + z + e^z (-2 + z)      ) / z^3 >
    //   f3 [m] = h * < (-4 - 3z - z^2 + e^z (4 - z)) / z^3 >
    void init(const std::vector<double>& lam, double dt, int M = 32) {
        (void)lam; (void)dt; (void)M;
        NP_TODO("ex06::Etdrk4::init");
    }

    // One step. `N` maps Fourier coefficients to the Fourier coefficients of
    // the nonlinear term: N(vhat, out).
    //
    //   Nv = N(v)
    //   a  = E2*v + Q*Nv                 ; Na = N(a)
    //   b  = E2*v + Q*Na                 ; Nb = N(b)
    //   c  = E2*a + Q*(2*Nb - Nv)        ; Nc = N(c)
    //   v  = E*v + Nv*f1 + 2*(Na + Nb)*f2 + Nc*f3
    template <class NL>
    void step(std::vector<Cplx>& v, NL&& N) const {
        (void)v; (void)N;
        NP_TODO("ex06::Etdrk4::step");
    }
};

}  // namespace np::ex06
