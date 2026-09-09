// =============================================================================
// Exercise 05 -- hyperbolic conservation laws:  u_t + f(u)_x = 0
// =============================================================================
// Parabolic problems forgive you; hyperbolic ones do not. There is no diffusion
// to erase your mistakes, information travels at finite speed, and solutions
// develop discontinuities from smooth data in finite time. Three hard-won facts
// drive everything here:
//
//   (a) Godunov's theorem: a LINEAR monotone scheme is at most 1st order.
//       So every 2nd-order non-oscillatory scheme must be nonlinear -- hence
//       "limiters", which switch the scheme's own coefficients on the data.
//   (b) Lax-Wendroff theorem: if a CONSERVATIVE scheme converges, it converges
//       to a weak solution with the right shock speeds. Write your update as a
//       flux difference u_j -= (dt/dx)(F_{j+1/2} - F_{j-1/2}) and shock speeds
//       come out right for free. Discretise u_t + u u_x = 0 non-conservatively
//       and you will get a confidently wrong shock speed.
//   (c) CFL: the numerical domain of dependence must contain the physical one.
//
// Domain: [0,1) with n cells, dx = 1/n, PERIODIC. u[j] is the cell average on
// [j dx, (j+1) dx).
//
// WHAT TO IMPLEMENT
//   1. advect_upwind_step      -- order 1, monotone, very diffusive
//   2. advect_lax_wendroff_step-- order 2, oscillatory at discontinuities
//   3. minmod + advect_muscl_step -- order 2 AND non-oscillatory (TVD)
//   4. burgers_rusanov_step    -- nonlinear flux, shock capturing
//
// GOTCHAS
//   * Use the OLD time level everywhere in one sweep. Updating u in place turns
//     upwind into an implicit-ish scheme and quietly changes the answer.
//   * Periodic indexing: (j + n) % n on both sides.
//   * The MUSCL half-step term (1 - a dt/dx) is what makes it 2nd order in TIME
//     as well as space. Drop it and you get an order-1-in-time scheme with an
//     order-2 spatial stencil -- the fitted order will read ~1.
// =============================================================================
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex05 {

using np::Vec;

inline int wrap(int j, int n) { return (j % n + n) % n; }

// -----------------------------------------------------------------------------
// 1. First-order upwind for u_t + a u_x = 0, a > 0:
//        u_j^{n+1} = u_j^n - (a dt/dx) (u_j^n - u_{j-1}^n)
// Its modified equation is u_t + a u_x = (a dx/2)(1 - C) u_xx with C = a dt/dx:
// the scheme is really solving an ADVECTION-DIFFUSION equation. That numerical
// viscosity is what keeps it monotone, and what smears every front.
// (Handle a < 0 too if you like: take the difference from the other side.)
// -----------------------------------------------------------------------------
inline void advect_upwind_step(Vec& u, double a, double dx, double dt) {
    (void)u; (void)a; (void)dx; (void)dt;
    NP_TODO("ex05::advect_upwind_step");
}

// -----------------------------------------------------------------------------
// 2. Lax-Wendroff (2nd order in space AND time, from a Taylor expansion using
//    u_tt = a^2 u_xx):
//        u_j^{n+1} = u_j - (C/2)(u_{j+1} - u_{j-1}) + (C^2/2)(u_{j+1} - 2u_j + u_{j-1})
//    with C = a dt/dx. The leading error is now DISPERSIVE (u_xxx), which is why
//    it produces trailing oscillations behind a jump instead of smearing it.
// -----------------------------------------------------------------------------
inline void advect_lax_wendroff_step(Vec& u, double a, double dx, double dt) {
    (void)u; (void)a; (void)dx; (void)dt;
    NP_TODO("ex05::advect_lax_wendroff_step");
}

// -----------------------------------------------------------------------------
// 3a. minmod(a, b) = a if |a| < |b| and ab > 0
//                  = b if |b| <= |a| and ab > 0
//                  = 0 if ab <= 0            (opposite signs -> extremum -> flatten)
// -----------------------------------------------------------------------------
inline double minmod(double a, double b) {
    (void)a; (void)b;
    NP_TODO("ex05::minmod");
}

// -----------------------------------------------------------------------------
// 3b. MUSCL with a minmod slope limiter, a > 0:
//        sigma_j        = minmod( (u_j - u_{j-1})/dx, (u_{j+1} - u_j)/dx )
//        uL_{j+1/2}     = u_j + (dx/2)(1 - a dt/dx) sigma_j
//        F_{j+1/2}      = a * uL_{j+1/2}
//        u_j^{n+1}      = u_j - (dt/dx)(F_{j+1/2} - F_{j-1/2})
// Second order where the solution is smooth, first order (and monotone) exactly
// at extrema and discontinuities -- the nonlinearity demanded by Godunov.
// -----------------------------------------------------------------------------
inline void advect_muscl_step(Vec& u, double a, double dx, double dt) {
    (void)u; (void)a; (void)dx; (void)dt;
    NP_TODO("ex05::advect_muscl_step");
}

// -----------------------------------------------------------------------------
// 4. Inviscid Burgers, u_t + (u^2/2)_x = 0, with the Rusanov (local Lax-
//    Friedrichs) flux:
//        F_{j+1/2} = 1/2 [ f(u_j) + f(u_{j+1}) ]
//                    - 1/2 * max(|u_j|, |u_{j+1}|) * (u_{j+1} - u_j)
//        u_j^{n+1} = u_j - (dt/dx)(F_{j+1/2} - F_{j-1/2})
//    The second term is an upwind-strength artificial viscosity built from the
//    local maximum wave speed. Conservative by construction, so the Lax-Wendroff
//    theorem guarantees the correct shock speed s = (u_L + u_R)/2.
// -----------------------------------------------------------------------------
inline void burgers_rusanov_step(Vec& u, double dx, double dt) {
    (void)u; (void)dx; (void)dt;
    NP_TODO("ex05::burgers_rusanov_step");
}

// ---- provided helpers -------------------------------------------------------
inline double total_variation(const Vec& u) {
    double tv = 0;
    const int n = int(u.size());
    for (int j = 0; j < n; ++j) tv += std::fabs(u[wrap(j + 1, n)] - u[j]);
    return tv;
}
inline double integral(const Vec& u, double dx) {
    double s = 0;
    for (double v : u) s += v;
    return s * dx;
}

}  // namespace np::ex05
