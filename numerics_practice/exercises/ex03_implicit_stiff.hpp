// =============================================================================
// Exercise 03 -- implicit and linearly-implicit methods for stiff systems
// =============================================================================
// A system is *stiff* when the step size you need for STABILITY is far smaller
// than the one you need for ACCURACY. Explicit methods then waste 10^3-10^6
// steps resolving a transient that decayed at t = 0. The cure is to build the
// Jacobian into the step.
//
// Where this bites you in practice: Landau-Lifshitz with large damping, spin-
// lattice coupling with a stiff phonon bath, reaction-diffusion, and every
// method-of-lines discretisation of a diffusion operator (ex04) whose stiffness
// is ~ alpha/dx^2 and therefore unbounded as you refine the mesh.
//
// Conventions:
//     f(t, y, dydt)     right-hand side
//     jac(t, y, J)      writes the Jacobian df_i/dy_j into an n x n Mat
//                       (J arrives zeroed -- you may assume that)
//
// WHAT TO IMPLEMENT
//   1. backward_euler_step  -- order 1, A- and L-stable; Newton inside
//   2. trapezoid_step       -- order 2, A-stable but NOT L-stable (see the test)
//   3. bdf2_step            -- order 2, L-stable, 2-step (needs y_{n-1})
//   4. rosenbrock2_step     -- order 2, linearly implicit: two linear solves,
//                              zero Newton iterations, no convergence failures
//   5. exp_euler_step       -- exponential integrator: treat the linear part
//                              exactly with e^{hA}
//
// KEY CONCEPTS THE TESTS PROBE
//   * A-stability: the stability region contains the whole left half plane, so
//     h can be chosen for accuracy alone.
//   * L-stability: additionally R(z) -> 0 as z -> -infinity, so infinitely stiff
//     modes are *annihilated* rather than merely bounded. The trapezoidal rule
//     is A-stable but has R(-inf) = -1: stiff transients ring forever with
//     alternating sign. The test measures exactly this.
//
// GOTCHAS
//   * Newton solves G(y) = y - y_n - h f(t_{n+1}, y) = 0, so the Newton matrix
//     is  I - h J,  NOT J. Forgetting the identity is the classic bug.
//   * A good initial guess (y_n, or an explicit-Euler predictor) is worth more
//     than a tighter tolerance.
//   * Rosenbrock methods use J once per step and never iterate; their order
//     survives even with an *approximate* J (that is the "-W" in Rosenbrock-W),
//     which makes them the pragmatic choice when the exact Jacobian is painful.
// =============================================================================
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex03 {

using np::Mat;
using np::Vec;

// -----------------------------------------------------------------------------
// 1. Backward Euler:  y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})
//
// Newton iteration on G(y) = y - y_n - h f(t+h, y):
//        solve (I - h J(t+h, y^k)) dy = -G(y^k);   y^{k+1} = y^k + dy
// Stop when norm_inf(dy) <= tol (or norm_inf(G) <= tol). Use np::lu_solve.
// -----------------------------------------------------------------------------
template <class F, class J>
void backward_euler_step(F&& f, J&& jac, double t, Vec& y, double h,
                         double tol = 1e-12, int maxit = 50) {
    (void)f; (void)jac; (void)t; (void)y; (void)h; (void)tol; (void)maxit;
    NP_TODO("ex03::backward_euler_step");
}

// -----------------------------------------------------------------------------
// 2. Trapezoidal rule (= Crank-Nicolson when f is the spatial operator):
//        y_{n+1} = y_n + (h/2) [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]
// Newton matrix: I - (h/2) J.
// -----------------------------------------------------------------------------
template <class F, class J>
void trapezoid_step(F&& f, J&& jac, double t, Vec& y, double h,
                    double tol = 1e-12, int maxit = 50) {
    (void)f; (void)jac; (void)t; (void)y; (void)h; (void)tol; (void)maxit;
    NP_TODO("ex03::trapezoid_step");
}

// -----------------------------------------------------------------------------
// 3. BDF2:  y_{n+1} = (4/3) y_n - (1/3) y_{n-1} + (2/3) h f(t_{n+1}, y_{n+1})
// `y_prev` is y_{n-1}; `y` holds y_n on entry and y_{n+1} on exit.
// `t` is t_n (so evaluate f at t + h). Newton matrix: I - (2/3) h J.
// -----------------------------------------------------------------------------
template <class F, class J>
void bdf2_step(F&& f, J&& jac, double t, const Vec& y_prev, Vec& y, double h,
               double tol = 1e-12, int maxit = 50) {
    (void)f; (void)jac; (void)t; (void)y_prev; (void)y; (void)h;
    (void)tol; (void)maxit;
    NP_TODO("ex03::bdf2_step");
}

// -----------------------------------------------------------------------------
// 4. ROS2, a 2nd-order L-stable Rosenbrock method. gamma = 1 + 1/sqrt(2).
//        W  = I - gamma h J(t_n, y_n)
//        W k1 = h f(t_n, y_n)
//        W k2 = h f(t_n + h, y_n + k1) - 2 k1
//        y_{n+1} = y_n + (3/2) k1 + (1/2) k2
// Two solves with the SAME matrix W -- in a real code you would factor once.
// -----------------------------------------------------------------------------
template <class F, class J>
void rosenbrock2_step(F&& f, J&& jac, double t, Vec& y, double h) {
    (void)f; (void)jac; (void)t; (void)y; (void)h;
    NP_TODO("ex03::rosenbrock2_step");
}

// -----------------------------------------------------------------------------
// 5. Exponential Euler for the semilinear system  y' = A y + g(t, y)
// with A constant. Solve the linear part EXACTLY by variation of constants,
// freezing g over the step:
//        y_{n+1} = e^{hA} y_n + h phi1(hA) g(t_n, y_n),
//        phi1(z) = (e^z - 1) / z    ==>   h phi1(hA) = A^{-1} (e^{hA} - I).
// You have np::matrix_exp and np::inverse. (Real codes use Krylov / contour
// integrals instead of a dense inverse; the small-A case is the exercise.)
//
// This is EXACT when g is constant, no matter how stiff A is -- which is the
// whole idea, and the seed of the ETDRK4 scheme you will build in ex06.
// `g` has signature g(t, y, out).
// -----------------------------------------------------------------------------
template <class G>
void exp_euler_step(const Mat& A, G&& g, double t, Vec& y, double h) {
    (void)A; (void)g; (void)t; (void)y; (void)h;
    NP_TODO("ex03::exp_euler_step");
}

}  // namespace np::ex03
