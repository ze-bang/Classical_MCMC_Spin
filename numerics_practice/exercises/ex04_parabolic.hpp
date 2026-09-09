// =============================================================================
// Exercise 04 -- parabolic PDEs:  u_t = alpha * Laplacian(u)
// =============================================================================
// Discretise space first (method of lines) and you are back in ex01/ex03: the
// heat equation becomes a linear ODE system  u' = A u  whose stiffness is
//      |lambda_max| = 4 alpha / dx^2
// and therefore grows without bound as you refine the mesh. Every explicit
// method inherits a dt ~ dx^2 shackle. That single fact is why implicit and
// operator-splitting methods exist.
//
// Grid convention (1-D): n points x_j = j*dx, j = 0..n-1, dx = 1/(n-1),
// homogeneous Dirichlet u[0] = u[n-1] = 0 held fixed.
//
// WHAT TO IMPLEMENT
//   1. laplacian_1d            -- second difference, the building block
//   2. heat_ftcs_step          -- forward-in-time centred-in-space, dt <= dx^2/(2a)
//   3. heat_crank_nicolson_step-- trapezoid in time, unconditionally stable
//   4. heat2d_adi_step         -- Peaceman-Rachford ADI: 2-D at 1-D cost
//
// HOW THE TESTS SEPARATE SPACE ERROR FROM TIME ERROR
// This is a technique worth stealing. If you compare a numerical solution to
// the PDE's analytic solution, the O(dx^2) space error swamps the time error
// and the fitted "time order" is garbage. Instead compare against the *exact
// solution of the semi-discrete system*: for u_j = sin(pi x_j), the discrete
// Laplacian has the exact eigenvalue
//      lambda_h = -alpha * (2 - 2 cos(pi dx)) / dx^2
// so the semi-discrete solution is exactly sin(pi x_j) exp(lambda_h t). Measure
// against that on a FIXED grid and you see the time discretisation alone.
//
// GOTCHAS
//   * r = alpha*dt/dx^2 > 1/2 makes FTCS blow up, and it blows up in the
//     highest wavenumber -- a sawtooth, not a smooth error. The test checks this.
//   * Crank-Nicolson is A-stable but NOT L-stable (same story as ex03's
//     trapezoid). With a discontinuous initial condition and a big dt it rings:
//     the classic "CN oscillations near a step". BDF2 or backward Euler fixes it.
//   * ADI is only unconditionally stable in 2-D. The naive 3-D generalisation
//     of Peaceman-Rachford is conditionally stable; use Douglas-Gunn instead.
// =============================================================================
#pragma once
#include <cmath>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex04 {

using np::Vec;

// -----------------------------------------------------------------------------
// 1. out[j] = (u[j-1] - 2 u[j] + u[j+1]) / dx^2 on the interior; out = 0 at the
//    two Dirichlet boundaries.
// -----------------------------------------------------------------------------
inline void laplacian_1d(const Vec& u, double dx, Vec& out) {
    (void)u; (void)dx; (void)out;
    NP_TODO("ex04::laplacian_1d");
}

// -----------------------------------------------------------------------------
// 2. FTCS:  u_j^{n+1} = u_j^n + r (u_{j-1}^n - 2u_j^n + u_{j+1}^n),
//    r = alpha dt / dx^2.  Boundaries stay 0.
// -----------------------------------------------------------------------------
inline void heat_ftcs_step(Vec& u, double alpha, double dx, double dt) {
    (void)u; (void)alpha; (void)dx; (void)dt;
    NP_TODO("ex04::heat_ftcs_step");
}

// -----------------------------------------------------------------------------
// 3. Crank-Nicolson:
//        (I - (r/2) D) u^{n+1} = (I + (r/2) D) u^n,   D = second-difference
//    i.e. for interior j:
//        -r/2 u_{j-1}^{n+1} + (1+r) u_j^{n+1} - r/2 u_{j+1}^{n+1}
//          =  r/2 u_{j-1}^n   + (1-r) u_j^n   + r/2 u_{j+1}^n
//    Solve the tridiagonal system with np::thomas (O(n), no matrix assembly).
//    Only the interior unknowns j = 1..n-2 are in the system.
// -----------------------------------------------------------------------------
inline void heat_crank_nicolson_step(Vec& u, double alpha, double dx,
                                     double dt) {
    (void)u; (void)alpha; (void)dx; (void)dt;
    NP_TODO("ex04::heat_crank_nicolson_step");
}

// -----------------------------------------------------------------------------
// 4. Peaceman-Rachford ADI on an nx-by-ny grid, u[j*nx + i], dx = dy,
//    homogeneous Dirichlet on all four edges. Half-step implicit in x, then
//    half-step implicit in y:
//
//        (I - (dt/2) alpha Dxx) u*     = (I + (dt/2) alpha Dyy) u^n
//        (I - (dt/2) alpha Dyy) u^{n+1}= (I + (dt/2) alpha Dxx) u*
//
//    Each half-step is a batch of independent tridiagonal solves -- ny of them
//    along rows, then nx of them along columns. Cost O(N) per step in 2-D, with
//    no stability restriction. This "solve one direction at a time" idea is the
//    ancestor of every modern operator-splitting scheme, including the BAOAB
//    splitting you will build in ex09.
// -----------------------------------------------------------------------------
inline void heat2d_adi_step(Vec& u, int nx, int ny, double alpha, double dx,
                            double dt) {
    (void)u; (void)nx; (void)ny; (void)alpha; (void)dx; (void)dt;
    NP_TODO("ex04::heat2d_adi_step");
}

// ---- provided helpers -------------------------------------------------------

// Exact eigenvalue of the discrete 1-D Laplacian for mode sin(k pi x).
inline double discrete_laplacian_eigenvalue(double k, double dx) {
    double s = std::sin(0.5 * k * pi * dx);
    return -4.0 * s * s / (dx * dx);
}

}  // namespace np::ex04
