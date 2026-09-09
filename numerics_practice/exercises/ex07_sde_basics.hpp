// =============================================================================
// Exercise 07 -- stochastic differential equations: the two convergence orders
// =============================================================================
//        dX = a(t, X) dt + b(t, X) dW        (Ito)
//        dX = a(t, X) dt + b(t, X) o dW      (Stratonovich)
//
// Two things change relative to the deterministic world, and both bite:
//
//  (A) THERE ARE TWO CONVERGENCE ORDERS.
//      strong order p:  E| X_N - X(T) |        <= C h^p   (pathwise accuracy)
//      weak order q:    | E g(X_N) - E g(X(T)) | <= C h^q (accuracy of averages)
//      Euler-Maruyama is strong-0.5 / weak-1.0. If all you want is <A>, the
//      strong order is irrelevant -- and almost everything you actually compute
//      in statistical physics is an average. Knowing which order you need is
//      the single most useful idea in this file.
//
//  (B) THE STOCHASTIC INTEGRAL IS AMBIGUOUS.
//      Ito evaluates b at the left endpoint; Stratonovich at the midpoint. For
//      multiplicative noise (b depending on X) they give genuinely DIFFERENT
//      processes, related by the drift shift a_Strat = a_Ito - (1/2) b b'.
//      Physics (a real noise with a short correlation time -- Wong-Zakai) gives
//      Stratonovich; mathematics and most SDE literature default to Ito. The
//      stochastic Landau-Lifshitz equation in ex10 is Stratonovich, and getting
//      this wrong silently changes the equilibrium magnetisation.
//
// All steps take the Brownian increment dW EXPLICITLY rather than drawing it.
// That is deliberate: the strong-error test drives every step size with the
// SAME realised Brownian path (coarse increments = sums of fine ones), which is
// the only way to measure a strong order without drowning in Monte Carlo noise.
//
// WHAT TO IMPLEMENT
//   1. euler_maruyama_step      -- strong 0.5, weak 1.0
//   2. milstein_step            -- strong 1.0 (needs db/dx)
//   3. stratonovich_heun_step   -- strong 1.0 for Stratonovich, derivative-free
//   4. brownian_path            -- utility used by the tests
//
// GOTCHAS
//   * dW ~ N(0, dt), i.e. sqrt(dt) * normal(). Writing dt * normal() is the
//     classic bug and it silently turns your noise off as dt -> 0.
//   * Milstein's extra term is (1/2) b b' (dW^2 - dt). The "- dt" is what makes
//     it an Ito correction; drop it and you bias the drift by (1/2) b b' dt.
//   * Heun applied with an ITO drift converges to the STRATONOVICH solution.
//     That is not a bug in Heun -- it is what Heun is for.
// =============================================================================
#pragma once
#include <cmath>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex07 {

// -----------------------------------------------------------------------------
// 1. Euler-Maruyama (Ito):   X += a(t,X) dt + b(t,X) dW
// -----------------------------------------------------------------------------
template <class A, class B>
void euler_maruyama_step(A&& a, B&& b, double t, double& x, double dt,
                         double dW) {
    x += a(t, x) * dt + b(t, x) * dW;

}

// -----------------------------------------------------------------------------
// 2. Milstein (Ito):
//        X += a dt + b dW + (1/2) b * (db/dx) * (dW^2 - dt)
// The extra term comes from expanding b(X_s) inside the stochastic integral to
// one more order -- it is exactly the Ito-Taylor term of strong order 1.
// `dbdx` has the same signature as b and returns db/dx.
// -----------------------------------------------------------------------------
template <class A, class B, class Bp>
void milstein_step(A&& a, B&& b, Bp&& dbdx, double t, double& x, double dt,
                   double dW) {
    const double bx = b(t, x);
    x += a(t, x) * dt + bx * dW + 0.5 * bx * dbdx(t, x) * (dW * dW - dt);
}

// -----------------------------------------------------------------------------
// 3. Stochastic Heun -- solves the STRATONOVICH equation, derivative-free,
//    strong order 1. Exactly the ex01 Heun shape with dW in place of dt for the
//    noise term:
//        xbar = x + a(t,x) dt + b(t,x) dW
//        x   += (dt/2) [ a(t,x) + a(t+dt,xbar) ] + (dW/2) [ b(t,x) + b(t+dt,xbar) ]
// This is the scheme almost every spin-dynamics code uses, for the good reason
// that it needs no derivative of the (matrix-valued, geometry-dependent) noise
// coefficient. You will build the LLG version of it in ex10.
// -----------------------------------------------------------------------------
template <class A, class B>
void stratonovich_heun_step(A&& a, B&& b, double t, double& x, double dt,
                            double dW) {
    (void)a; (void)b; (void)t; (void)x; (void)dt; (void)dW;
    NP_TODO("ex07::stratonovich_heun_step");
}

// -----------------------------------------------------------------------------
// 4. Utility: n independent increments of a Brownian path with step dt,
//    i.e. dW[i] = sqrt(dt) * N(0,1).
// -----------------------------------------------------------------------------
inline std::vector<double> brownian_path(int n, double dt, Rng& g) {
    (void)n; (void)dt; (void)g;
    NP_TODO("ex07::brownian_path");
}

// ---- provided ---------------------------------------------------------------
// Coarsen a fine increment array by summing groups of `factor` increments.
// Brownian increments are independent Gaussians, so the sum over a coarse
// interval IS the coarse increment -- the two resolutions see one path.
inline std::vector<double> coarsen(const std::vector<double>& dW, int factor) {
    std::vector<double> out(dW.size() / size_t(factor), 0.0);
    for (size_t i = 0; i < out.size(); ++i)
        for (int k = 0; k < factor; ++k) out[i] += dW[i * size_t(factor) + size_t(k)];
    return out;
}

}  // namespace np::ex07
