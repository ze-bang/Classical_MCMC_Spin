// =============================================================================
// Exercise 02 -- symplectic / geometric integrators for separable Hamiltonians
//     H(q,p) = p^2 / 2m + U(q),      qdot = p/m,   pdot = F(q) = -dU/dq
// =============================================================================
// This is the tier that matters most for your day job: spin dynamics, MD, and
// every Langevin scheme in ex09 are built by *splitting* a Hamiltonian into
// exactly-solvable pieces and composing the flows.
//
// The point of a symplectic method is NOT accuracy -- RK4 is more accurate per
// step.  The point is that a symplectic method is the *exact* flow of a nearby
// "shadow" Hamiltonian H~ = H + O(h^p).  Consequences you will measure:
//   * energy error stays bounded forever instead of drifting secularly;
//   * quadratic invariants of a central force (angular momentum) are exact;
//   * the map is time-reversible to machine precision.
//
// Convention: `force(q, f)` writes F(q) = -dU/dq into f.
//
// WHAT TO IMPLEMENT
//   1. symplectic_euler_step   -- order 1, the elementary A-then-B split
//   2. velocity_verlet_step    -- order 2, the workhorse (BAB / "kick-drift-kick")
//   3. leapfrog_step           -- order 2, ABA / "drift-kick-drift"
//   4. yoshida4_step           -- order 4 by triple composition
//
// GOTCHAS
//   * Velocity Verlet must reuse the force: exactly ONE force evaluation per
//     step. That is why the caller passes `f` in and gets it back updated --
//     `f` must hold F(q) on entry and F(q_new) on exit. If your version needs
//     two force calls per step you built it wrong (in MD that is a 2x slowdown).
//   * "Symplectic with adaptive step size" is a contradiction: varying h breaks
//     the shadow Hamiltonian and the energy drifts again. Fixed h only.
// =============================================================================
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex02 {

using np::Vec;

// -----------------------------------------------------------------------------
// 1. Symplectic Euler (variant "B then A"):
//        p_{n+1} = p_n + h F(q_n)
//        q_{n+1} = q_n + h p_{n+1} / m        <-- note: the NEW p
// Using the old p instead gives plain explicit Euler, which is not symplectic.
// That one-character difference is the whole exercise.
// -----------------------------------------------------------------------------
template <class Force>
void symplectic_euler_step(Force&& force, Vec& q, Vec& p, double m, double h) {
    (void)force; (void)q; (void)p; (void)m; (void)h;
    NP_TODO("ex02::symplectic_euler_step");
}

// -----------------------------------------------------------------------------
// 2. Velocity Verlet (BAB): half kick, full drift, half kick.
//        p       += (h/2) f                  (f = F(q) on entry)
//        q       += h p / m
//        f        = F(q)                     (the single force evaluation)
//        p       += (h/2) f
// On exit `f` holds F(q_new), ready for the next step.
// -----------------------------------------------------------------------------
template <class Force>
void velocity_verlet_step(Force&& force, Vec& q, Vec& p, Vec& f, double m,
                          double h) {
    (void)force; (void)q; (void)p; (void)f; (void)m; (void)h;
    NP_TODO("ex02::velocity_verlet_step");
}

// -----------------------------------------------------------------------------
// 3. Position Verlet / leapfrog (ABA): half drift, full kick, half drift.
//        q += (h/2) p/m ;  p += h F(q) ;  q += (h/2) p/m
// Same order, different shadow Hamiltonian, different error constant. Worth
// implementing because ex09's BAOAB is exactly this pattern with an extra
// stochastic "O" letter inserted in the middle.
// -----------------------------------------------------------------------------
template <class Force>
void leapfrog_step(Force&& force, Vec& q, Vec& p, double m, double h) {
    (void)force; (void)q; (void)p; (void)m; (void)h;
    NP_TODO("ex02::leapfrog_step");
}

// -----------------------------------------------------------------------------
// 4. Yoshida's 4th-order composition. A symmetric 2nd-order method S(h)
// composed as
//        S(w1 h) . S(w0 h) . S(w1 h)
// is 4th order provided  2 w1 + w0 = 1  and  2 w1^3 + w0^3 = 0, i.e.
//        w1 = 1 / (2 - 2^(1/3)),        w0 = -2^(1/3) / (2 - 2^(1/3)).
// Note w0 < 0: a *negative time* substep. That is unavoidable -- no symplectic
// composition of order > 2 has all-positive coefficients, which is exactly why
// 4th-order splitting fails for diffusion-type (irreversible) problems.
// Build it out of leapfrog_step (self-contained, no force cache to thread).
// -----------------------------------------------------------------------------
template <class Force>
void yoshida4_step(Force&& force, Vec& q, Vec& p, double m, double h) {
    (void)force; (void)q; (void)p; (void)m; (void)h;
    NP_TODO("ex02::yoshida4_step");
}

}  // namespace np::ex02
