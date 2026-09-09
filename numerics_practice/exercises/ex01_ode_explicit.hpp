// =============================================================================
// Exercise 01 -- explicit one-step methods for  y' = f(t, y)
// =============================================================================
// Everything downstream (PDE method-of-lines, spin dynamics, Langevin) is built
// out of these.  Get the bookkeeping reflexes right here.
//
// Convention used everywhere in this package:
//     f(t, y, dydt)   writes dy/dt into `dydt` (already correctly sized)
//
// WHAT TO IMPLEMENT (in order):
//   1. euler_step        -- order 1
//   2. midpoint_step     -- order 2 (explicit midpoint / RK2)
//   3. heun_step         -- order 2 (explicit trapezoid)
//   4. rk4_step          -- order 4 (the classical one)
//   5. dopri5_step       -- Dormand-Prince 5(4) embedded pair
//   6. integrate_adaptive-- PI step-size controller on top of dopri5
//
// THE TESTS CHECK: fitted convergence order from a log-log fit of global error
// vs h, and that the adaptive driver actually delivers the requested tolerance.
//
// GOTCHAS
//   * Global order = local order - 1.  A method whose *local* truncation error
//     is O(h^{p+1}) has *global* error O(h^p) because you take T/h steps.
//   * Every stage of an RK method evaluates f at (t + c_i h, y + h*sum a_ij k_j).
//     Forgetting the `h` in the stage argument silently drops you an order.
//   * Do not alias: k2 must be computed from a *temporary* state, not from y.
// =============================================================================
#pragma once
#include <algorithm>
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex01 {

using np::Vec;

// -----------------------------------------------------------------------------
// 1. Forward Euler:  y_{n+1} = y_n + h f(t_n, y_n)
// -----------------------------------------------------------------------------
template <class F>
void euler_step(F&& f, double t, Vec& y, double h) {
    static thread_local Vec k; // what is this for?  It is a temporary storage for the derivative f(t, y)
    //what is static? static means that the variable k will retain its value between function calls, and thread_local means that each thread will have its own instance of k. This is useful in a multi-threaded context to avoid data races.
    k.assign(y.size(), 0.0);
    f(t, y, k);
    for (size_t i = 0; i < y.size(); ++i) y[i] += h * k[i];
}

// -----------------------------------------------------------------------------
// 2. Explicit midpoint:
//        k1 = f(t, y)
//        k2 = f(t + h/2, y + (h/2) k1)
//        y_{n+1} = y_n + h k2
// -----------------------------------------------------------------------------
template <class F>
void midpoint_step(F&& f, double t, Vec& y, double h) {
    static thread_local Vec k1, k2, ytmp;
    k1.assign(y.size(), 0.0); k2.assign(y.size(), 0.0); ytmp.assign(y.size(), 0.0);
    f(t, y, k1);
    for (size_t i = 0; i < y.size(); ++i) ytmp[i] = y[i] + 0.5 * h * k1[i];
    f(t + 0.5 * h, ytmp, k2);
    for (size_t i = 0; i < y.size(); ++i) y[i] += h * k2[i];    
}

// -----------------------------------------------------------------------------
// 3. Heun (explicit trapezoid):
//        k1 = f(t, y);  k2 = f(t + h, y + h k1)
//        y_{n+1} = y_n + (h/2)(k1 + k2)
// This "predictor then average the slopes" shape reappears verbatim as the
// stochastic Heun scheme in ex07/ex10 -- it is how you integrate a Stratonovich
// SDE, and it is the standard workhorse for the stochastic LLG equation.
// -----------------------------------------------------------------------------
template <class F>
void heun_step(F&& f, double t, Vec& y, double h) {
    static thread_local Vec k1, k2, ytmp;
    k1.assign(y.size(), 0.0); k2.assign(y.size(), 0.0); ytmp.assign(y.size(), 0.0);
    f(t, y, k1);
    for (size_t i = 0; i < y.size(); ++i) ytmp[i] = y[i] + h * k1[i];
    f(t + h, ytmp, k2);
    for (size_t i = 0; i < y.size(); ++i) y[i] += 0.5 * h * (k1[i] + k2[i]);
}

// -----------------------------------------------------------------------------
// 4. Classical RK4:
//        k1 = f(t,       y)
//        k2 = f(t + h/2, y + (h/2) k1)
//        k3 = f(t + h/2, y + (h/2) k2)
//        k4 = f(t + h,   y + h k3)
//        y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
// -----------------------------------------------------------------------------
template <class F>
void rk4_step(F&& f, double t, Vec& y, double h) {
    (void)f; (void)t; (void)y; (void)h;
    NP_TODO("ex01::rk4_step");
}

// -----------------------------------------------------------------------------
// 5. Dormand-Prince 5(4).  Seven stages; the 5th-order solution is what you
// propagate, the 4th-order one exists only to estimate the error.
//
// Fill y5 (5th order) and y4 (4th order).  The Butcher tableau:
//
//   c  = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
//   a21= 1/5
//   a31= 3/40        a32= 9/40
//   a41= 44/45       a42= -56/15      a43= 32/9
//   a51= 19372/6561  a52= -25360/2187 a53= 64448/6561  a54= -212/729
//   a61= 9017/3168   a62= -355/33     a63= 46732/5247  a64= 49/176   a65= -5103/18656
//   a71= 35/384      a72= 0           a73= 500/1113    a74= 125/192  a75= -2187/6784  a76= 11/84
//   b5 = a7 row (so stage 7 IS y5 -- "first same as last")
//   b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
//
// FSAL: k7 = f(t+h, y5) can be reused as the next step's k1.  Skip that
// optimisation on the first pass; add it once the order test is green.
// -----------------------------------------------------------------------------
template <class F>
void dopri5_step(F&& f, double t, const Vec& y, double h, Vec& y5, Vec& y4) {
    (void)f; (void)t; (void)y; (void)h; (void)y5; (void)y4;
    NP_TODO("ex01::dopri5_step");
}

struct AdaptiveResult {
    double t = 0;       // time actually reached
    long naccepted = 0;
    long nrejected = 0;
    double h_last = 0;
};

// -----------------------------------------------------------------------------
// 6. Adaptive driver with a PI controller.
//
//   scaled error:  err = sqrt( mean_i [ (y5_i - y4_i) / sc_i ]^2 )
//                  sc_i = atol + rtol * max(|y_i|, |y5_i|)
//   accept iff err <= 1.
//   step update (PI, Gustafsson):
//       fac = 0.9 * err^(-alpha) * err_prev^(beta),  alpha = 0.7/5, beta = 0.4/5
//       h  *= clamp(fac, 0.2, 5.0)
//   On rejection, do NOT use the I term (set fac = 0.9*err^(-1/5)) and clamp
//   the growth to <= 1.
//   Never step past t1: clip h, and stop when t reaches t1.
//
// WHY THE PI TERM: a pure "err^(-1/5)" controller oscillates (accept-reject-
// accept-reject) on problems with rapidly changing scales.  The integral term
// damps that; it is what every production code (dopri5, LSODA, CVODE) uses.
// -----------------------------------------------------------------------------
template <class F>
AdaptiveResult integrate_adaptive(F&& f, double t0, double t1, Vec& y,
                                  double h0, double atol, double rtol) {
    (void)f; (void)t0; (void)t1; (void)y; (void)h0; (void)atol; (void)rtol;
    NP_TODO("ex01::integrate_adaptive");
}

// Convenience: fixed-step driver. Provided.
template <class Stepper, class F>
void integrate_fixed(Stepper&& step, F&& f, double t0, double t1, Vec& y,
                     long nsteps) {
    double h = (t1 - t0) / double(nsteps);
    for (long n = 0; n < nsteps; ++n) step(f, t0 + double(n) * h, y, h);
}

}  // namespace np::ex01
