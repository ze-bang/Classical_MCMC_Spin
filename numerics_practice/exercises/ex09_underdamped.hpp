// =============================================================================
// Exercise 09 -- underdamped Langevin: splitting thermostats (BAOAB & friends)
// =============================================================================
//        dq = (p/m) dt
//        dp = F(q) dt - gamma p dt + sqrt(2 gamma m kT) dW
//
// This is the equation behind every Langevin thermostat in every MD code, and
// the state of the art is not a Runge-Kutta method -- it is an OPERATOR
// SPLITTING. Write the generator as L = L_A + L_B + L_O:
//
//     A  (drift)   q += (p/m) h            exact flow of the kinetic part
//     B  (kick)    p += F(q) h             exact flow of the potential part
//     O  (thermal) p  = c p + sqrt(m kT (1-c^2)) xi,   c = exp(-gamma h)
//                                          EXACT Ornstein-Uhlenbeck propagator
//
// Each piece is solved exactly; the only error is in the non-commutativity.
// Different orderings of the same three letters are different integrators with
// dramatically different sampling accuracy:
//
//     ABOBA, BAOAB, OBABO, OABAO, ...
//
// The punchline (Leimkuhler & Matthews 2013): BAOAB has by far the smallest
// error in CONFIGURATIONAL averages -- for a quadratic potential it is exact,
// and in the high-friction limit it gains two extra orders. Since almost every
// thermodynamic observable you care about is configurational, that makes the
// letter order a free 10-100x accuracy win. Same cost, one force evaluation.
//
// The Gronbech-Jensen-Farago (GJF) scheme reaches the same place from a
// different direction: it is constructed so that the discrete-time position
// statistics and the diffusion constant are exact for a harmonic well at any
// step size.
//
// WHAT TO IMPLEMENT
//   1. ou_step, drift, kick   -- the three exact sub-propagators
//   2. langevin_euler_step    -- the naive scheme, for contrast
//   3. baoab_step, aboba_step, obabo_step
//   4. gjf_step
//
// HOW THE TESTS MEASURE IT
// Same Lyapunov trick as ex08 (np::stationary_covariance on the 2-D state
// (q,p)), which turns each scheme's exact sampling bias into a printable
// number. You will see, with zero statistical noise:
//     OBABO         <p^2> exact,  <q^2> = kT/k / (1 - h^2 w^2/4)
//     BAOAB / GJF   <q^2> exact,  <p^2> = m kT (1 - h^2 w^2/4)
//     Euler         both wrong, and unstable once gamma*h > 2
// Then an anharmonic (double-well) run shows the same ordering survives
// nonlinearity, which is the part that actually matters.
//
// GOTCHAS
//   * The O step is NOT "p -= gamma p h + noise". It is the exact exponential
//     e^{-gamma h}; using the linearised version reintroduces a step-size
//     stability limit (gamma h < 2) that the splitting does not have.
//   * The fluctuation-dissipation prefactor is sqrt(m kT (1 - e^{-2 gamma h})).
//     In the small-h limit that is sqrt(2 gamma m kT h) -- check your code
//     reduces to that, it is the fastest way to catch a factor of 2.
//   * BAOAB must evaluate the force ONCE per step (after the second A). Doing
//     it twice doubles your cost for nothing.
// =============================================================================
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex09 {

using np::Vec;

struct Thermostat {
    double gamma = 1.0;
    double kT = 1.0;
};

// -----------------------------------------------------------------------------
// 1. The three exact sub-propagators.
//    O: c = exp(-gamma h);  p <- c p + sqrt(m kT (1 - c^2)) * N(0,1)
// -----------------------------------------------------------------------------
template <class Rng>
void ou_step(Vec& p, double m, double h, const Thermostat& th, Rng& g) {
    (void)p; (void)m; (void)h; (void)th; (void)g;
    NP_TODO("ex09::ou_step");
}

inline void drift(Vec& q, const Vec& p, double m, double h) {
    (void)q; (void)p; (void)m; (void)h;
    NP_TODO("ex09::drift");
}

inline void kick(Vec& p, const Vec& f, double h) {
    (void)p; (void)f; (void)h;
    NP_TODO("ex09::kick");
}

// -----------------------------------------------------------------------------
// 2. Naive Euler-Maruyama on the Langevin system (what you write first, and
//    what you should stop writing):
//        p += F(q) h - gamma p h + sqrt(2 gamma m kT h) N(0,1)
//        q += (p/m) h
//        f  = F(q)
// -----------------------------------------------------------------------------
template <class Force, class Rng>
void langevin_euler_step(Force&& force, Vec& q, Vec& p, Vec& f, double m,
                         double h, const Thermostat& th, Rng& g) {
    (void)force; (void)q; (void)p; (void)f; (void)m; (void)h; (void)th; (void)g;
    NP_TODO("ex09::langevin_euler_step");
}

// -----------------------------------------------------------------------------
// 3. The splittings. `f` holds F(q) on entry and F(q_new) on exit (one force
//    evaluation per step).
//
//    BAOAB:  B(h/2) A(h/2) O(h) A(h/2) [force] B(h/2)
//    ABOBA:  A(h/2) [force] B(h/2) O(h) B(h/2) A(h/2) [force]
//    OBABO:  O(h/2) B(h/2) A(h) [force] B(h/2) O(h/2)
// -----------------------------------------------------------------------------
template <class Force, class Rng>
void baoab_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
                const Thermostat& th, Rng& g) {
    (void)force; (void)q; (void)p; (void)f; (void)m; (void)h; (void)th; (void)g;
    NP_TODO("ex09::baoab_step");
}

template <class Force, class Rng>
void aboba_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
                const Thermostat& th, Rng& g) {
    (void)force; (void)q; (void)p; (void)f; (void)m; (void)h; (void)th; (void)g;
    NP_TODO("ex09::aboba_step");
}

template <class Force, class Rng>
void obabo_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
                const Thermostat& th, Rng& g) {
    (void)force; (void)q; (void)p; (void)f; (void)m; (void)h; (void)th; (void)g;
    NP_TODO("ex09::obabo_step");
}

// -----------------------------------------------------------------------------
// 4. Gronbech-Jensen-Farago (2013). With gh2 = gamma h / 2:
//        b     = 1 / (1 + gh2)
//        a     = (1 - gh2) / (1 + gh2)
//        beta  = sqrt(2 gamma m kT h) * N(0,1)          (one draw per step)
//        q_new = q + b h p/m + b h^2 f_old /(2m) + b h beta /(2m)
//        f_new = F(q_new)
//        p_new = a p + (h/2)(a f_old + f_new) + b beta
//    Note the structure: it is velocity Verlet with every occurrence of h
//    rescaled by b, plus one noise term entering both q and p. Save f_old
//    BEFORE overwriting f.
// -----------------------------------------------------------------------------
template <class Force, class Rng>
void gjf_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
              const Thermostat& th, Rng& g) {
    (void)force; (void)q; (void)p; (void)f; (void)m; (void)h; (void)th; (void)g;
    NP_TODO("ex09::gjf_step");
}

}  // namespace np::ex09
