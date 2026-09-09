// =============================================================================
// Exercise 08 -- overdamped Langevin (Brownian dynamics) and its sampling bias
// =============================================================================
//        gamma xdot = F(x) + sqrt(2 gamma kT) eta(t)
//        dx = (F(x)/gamma) dt + sqrt(2 kT/gamma) dW
//
// The invariant density is exactly Boltzmann, rho(x) ~ exp(-U(x)/kT), for ANY
// gamma. Note what that means: if you only want equilibrium averages, the
// dynamics is a free choice and you should pick whichever integrator samples
// rho most accurately per unit of CPU time. THAT is the design criterion for a
// thermostat -- not trajectory accuracy.
//
// So the quantity to measure is not the pathwise error of ex07 but the bias of
// the INVARIANT MEASURE: how far the stationary distribution of your discrete
// chain sits from exp(-U/kT). Three answers, in increasing order of cleverness:
//
//   Euler-Maruyama            bias O(dt)     -- and can be *transient* (blow up)
//   Leimkuhler-Matthews       bias O(dt^2)   -- one extra line of code
//   MALA (Metropolis-adjusted) bias 0        -- at the price of a reject step
//
// WHAT TO IMPLEMENT
//   1. bd_euler_step
//   2. ou_exact_step                 -- exact propagator for F = -k x
//   3. bd_leimkuhler_matthews_step
//   4. mala_step
//
// HOW THE TESTS MEASURE BIAS WITHOUT MONTE CARLO NOISE
// For a harmonic potential every one of these schemes is an affine-Gaussian map
//      z_{n+1} = M z_n + b + L xi,
// whose stationary covariance solves the discrete Lyapunov equation
//      Sigma = M Sigma M^T + L L^T.
// np::stationary_covariance() recovers M and L by calling YOUR step function
// with scripted noise, then solves that equation. The result is the exact
// sampling bias as a deterministic number -- no sampling, no error bars. Steal
// this trick; it turns "run 10^9 steps and squint" into a unit test.
//
// GOTCHAS
//   * sqrt(2 kT dt / gamma) -- three ways to get the prefactor wrong. If your
//     measured <x^2> is off by a constant factor rather than by O(dt), it is
//     this.
//   * Euler-Maruyama on a quartic potential is not merely biased: for drifts
//     growing faster than linearly the chain can be transient and escape to
//     infinity at large dt (Roberts & Tweedie 1996). The test shows it.
//   * MALA's acceptance ratio needs the PROPOSAL densities, and the proposal is
//     not symmetric (its mean is drift-shifted). Forgetting q(x|y)/q(y|x) gives
//     a chain that looks fine and samples the wrong distribution.
// =============================================================================
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex08 {

struct Params {
    double gamma = 1.0;
    double kT = 1.0;
};

// -----------------------------------------------------------------------------
// 1. Euler-Maruyama:  x += (F(x)/gamma) dt + sqrt(2 kT dt / gamma) * N(0,1)
// -----------------------------------------------------------------------------
template <class Force, class Rng>
void bd_euler_step(Force&& F, double& x, double dt, const Params& p, Rng& g) {
    x += (F(x) / p.gamma) * dt + std::sqrt(2.0 * p.kT * dt / p.gamma) * g.normal();
}

// -----------------------------------------------------------------------------
// 2. Exact Ornstein-Uhlenbeck propagator for F(x) = -k x. The OU process is
//    Gaussian, so its transition density is known in closed form and we can
//    take an EXACT step of any size:
//        c = exp(-k dt / gamma)
//        x <- c x + sqrt( (kT/k) (1 - c^2) ) * N(0,1)
//    Check the two limits: dt -> 0 recovers Euler-Maruyama; dt -> infinity
//    gives a fresh draw from N(0, kT/k), the exact Boltzmann distribution.
//    This one-liner is the "O" (Ornstein-Uhlenbeck) piece of every splitting
//    thermostat in ex09.
// -----------------------------------------------------------------------------
template <class Rng>
void ou_exact_step(double& x, double dt, double k, const Params& p, Rng& g) {
    (void)x; (void)dt; (void)k; (void)p; (void)g;
    NP_TODO("ex08::ou_exact_step");
}

// -----------------------------------------------------------------------------
// 3. Leimkuhler-Matthews. Identical to Euler-Maruyama except that the noise is
//    the AVERAGE of the current and previous Gaussian draws:
//        x += (F(x)/gamma) dt + sqrt(2 kT dt/gamma) * (xi_prev + xi_new)/2
//    Same cost, same one random number per step, and the invariant-measure
//    error drops from O(dt) to O(dt^2). The averaged noise is a (mildly
//    correlated) approximation of the noise at the midpoint of the step, which
//    cancels the leading term of the Fokker-Planck discretisation error.
//    Remember to store xi_new into st.xi_prev for the next step.
// -----------------------------------------------------------------------------
struct LmState {
    double x = 0.0;
    double xi_prev = 0.0;
};

template <class Force, class Rng>
void bd_leimkuhler_matthews_step(Force&& F, LmState& st, double dt,
                                 const Params& p, Rng& g) {
    (void)F; (void)st; (void)dt; (void)p; (void)g;
    NP_TODO("ex08::bd_leimkuhler_matthews_step");
}

// -----------------------------------------------------------------------------
// 4. MALA = Euler-Maruyama proposal + Metropolis-Hastings accept/reject.
//    With D = kT/gamma and proposal variance s2 = 2 D dt:
//        mu_fwd = x + (F(x)/gamma) dt
//        y      = mu_fwd + sqrt(s2) N(0,1)
//        mu_bwd = y + (F(y)/gamma) dt
//        log alpha = -(U(y) - U(x))/kT
//                    - (x - mu_bwd)^2/(2 s2) + (y - mu_fwd)^2/(2 s2)
//        accept if log(uniform) < log alpha
//    Return true on acceptance. The reject step makes the chain EXACTLY
//    Boltzmann for any dt -- the discretisation error is converted from a bias
//    into a rejection rate, which is a much better trade.
// -----------------------------------------------------------------------------
template <class U, class Force, class Rng>
bool mala_step(U&& energy, Force&& F, double& x, double dt, const Params& p,
               Rng& g) {
    (void)energy; (void)F; (void)x; (void)dt; (void)p; (void)g;
    NP_TODO("ex08::mala_step");
}

// ---- provided helpers -------------------------------------------------------
// <f> under exp(-U/kT) on [xmin, xmax] by Simpson's rule.
template <class U, class F>
double boltzmann_average(U&& energy, F&& f, double xmin, double xmax, double kT,
                         int n = 200000) {
    double num = 0, den = 0;
    const double h = (xmax - xmin) / n;
    for (int i = 0; i <= n; ++i) {
        double x = xmin + i * h;
        double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        double p = std::exp(-energy(x) / kT);
        num += w * p * f(x);
        den += w * p;
    }
    return num / den;
}

}  // namespace np::ex08
