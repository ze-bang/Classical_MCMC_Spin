// =============================================================================
// Exercise 10 -- state of the art: colored-noise thermostats and the
//                stochastic Landau-Lifshitz-Gilbert equation
// =============================================================================
// Two things that are genuinely current practice, both built entirely out of
// pieces you already have.
//
// -----------------------------------------------------------------------------
// A. GENERALIZED LANGEVIN EQUATION (Markovian colored-noise thermostat)
// -----------------------------------------------------------------------------
// White-noise Langevin applies the same friction to every frequency, which
// means one gamma cannot simultaneously equilibrate a stiff bond and a slow
// collective mode. The GLE
//        pdot = F(q) - int_0^t K(t-s) p(s) ds + zeta(t)
// gives you a frequency-dependent friction. The trick (Ceriotti, Bussi &
// Parrinello) is that an exponentially-decaying memory kernel is EXACTLY
// reproduced by coupling p to a few auxiliary momenta s with a plain Markovian
// OU process on the extended vector (p, s):
//        d(p,s) = -A_p (p,s) dt + B dW
// Choose A_p and you have designed a memory kernel. The FDT-consistent noise is
// fixed by requiring the stationary covariance to be kT M, and -- exactly as in
// ex08's exact OU step -- the propagator over a finite h is available in closed
// form:
//        (p,s) <- T (p,s) + S xi,   T = exp(-A_p h),   S S^T = C - T C T^T
// with C = m kT I. Drop that in as the "O" of BAOAB and you have a thermostat
// with a tunable spectrum, exact at any h. This machinery is what powers
// quantum (zero-point-energy) thermostats and PIGLET.
//
// -----------------------------------------------------------------------------
// B. STOCHASTIC LANDAU-LIFSHITZ-GILBERT
// -----------------------------------------------------------------------------
// The Langevin equation for a classical spin of FIXED LENGTH:
//        dS/dt = -(1/(1+a^2)) [ S x H + a S x (S x H) ],   H = H_eff + zeta
//        <zeta_i(t) zeta_j(t')> = 2 a kT delta_ij delta(t-t')     (units gamma = mu_s = 1)
// Three features make it different from everything above:
//   * the state lives on a SPHERE, so a good integrator must be a rotation;
//   * the noise is MULTIPLICATIVE (it enters through S x zeta), so the
//     Ito/Stratonovich distinction of ex07 is not academic -- this equation is
//     Stratonovich, and an Ito integrator converges to the wrong stationary
//     distribution;
//   * there is an exact analytic equilibrium to check against: a single spin in
//     a field B has <S_z> = L(B/kT), the Langevin function. That single number
//     validates your noise amplitude, your Stratonovich handling, and your
//     damping term all at once -- and it must NOT depend on alpha.
//
// Rewriting the equation as a precession about an effective axis,
//        dS/dt = A x S,      A = ( H + a S x H ) / (1 + a^2),
// makes the geometry obvious: if A were constant, the exact flow is a rotation.
//
// WHAT TO IMPLEMENT
//   1. Gle::init, Gle::apply
//   2. gle_baoab_step
//   3. llg_axis, llg_rhs
//   4. cayley_rotate
//   5. sllg_heun_step
//   6. sllg_sib_step
//
// GOTCHAS
//   * The thermal FIELD scales as sqrt(2 a kT / dt), not sqrt(2 a kT dt): it is
//     a field, and it multiplies dt inside the step. Get this backwards and
//     <S_z> silently goes to 1.
//   * Heun must use the SAME noise realisation in the predictor and the
//     corrector. Drawing fresh noise for the corrector halves the effective
//     temperature.
//   * Renormalising S every step hides a norm drift but does not make the
//     scheme geometric -- the drift has already polluted the dynamics. SIB
//     needs no renormalisation at all, which is why it tolerates far larger dt.
// =============================================================================
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex10 {

using np::Mat;
using np::Vec;
using np::Vec3;

// ============================ A. GLE thermostat ============================
struct Gle {
    Mat T, S;       // T = exp(-A_p h);  S = chol(C - T C T^T), lower triangular
    int dim = 0;    // 1 + number of auxiliary momenta

    // Build the exact finite-h propagator. C = m kT I.
    // Use np::matrix_exp, np::matmul/transpose/madd/mscale, np::cholesky.
    void init(const Mat& Ap, double h, double m, double kT) {
        (void)Ap; (void)h; (void)m; (void)kT;
        NP_TODO("ex10::Gle::init");
    }

    // ps <- T ps + S xi  with xi ~ N(0, I). S is lower triangular, so
    // (S xi)_i = sum_{j<=i} S(i,j) xi_j.
    template <class Rng>
    void apply(Vec& ps, Rng& g) const {
        (void)ps; (void)g;
        NP_TODO("ex10::Gle::apply");
    }
};

// BAOAB for one degree of freedom with the GLE in the O slot.
// `s` holds the dim-1 auxiliary momenta; pack (p, s) into a length-dim vector,
// call gle.apply, and unpack.
template <class Force, class Rng>
void gle_baoab_step(Force&& force, Vec& q, Vec& p, Vec& f, Vec& s, double m,
                    double h, const Gle& gle, Rng& g) {
    (void)force; (void)q; (void)p; (void)f; (void)s; (void)m; (void)h;
    (void)gle; (void)g;
    NP_TODO("ex10::gle_baoab_step");
}

// ======================= B. stochastic Landau-Lifshitz =====================

// A = ( H + alpha * (S x H) ) / (1 + alpha^2)
inline Vec3 llg_axis(const Vec3& S, const Vec3& H, double alpha) {
    (void)S; (void)H; (void)alpha;
    NP_TODO("ex10::llg_axis");
}

// dS/dt = A x S
inline Vec3 llg_rhs(const Vec3& S, const Vec3& H, double alpha) {
    (void)S; (void)H; (void)alpha;
    NP_TODO("ex10::llg_rhs");
}

// -----------------------------------------------------------------------------
// Exact solution of the implicit midpoint update  S' = S + w x (S + S')/2.
// Closed form (the Cayley transform of the rotation generator), with a = w/2:
//        S' = S + (2 / (1 + |a|^2)) [ a x S + a x (a x S) ]
// Check two things: for small w it reduces to S + w x S, and |S'| = |S| to
// machine precision for ANY w. That exactness is the whole point -- it is what
// lets the SIB scheme run at step sizes where Heun's norm error explodes.
// -----------------------------------------------------------------------------
inline Vec3 cayley_rotate(const Vec3& S, const Vec3& w) {
    (void)S; (void)w;
    NP_TODO("ex10::cayley_rotate");
}

// -----------------------------------------------------------------------------
// Stochastic Heun (Stratonovich), the standard spin-dynamics workhorse:
//        zeta  = sqrt(2 alpha kT / dt) * (N,N,N)          -- ONE draw per step
//        k1    = rhs(S,  heff(S)  + zeta)
//        S'    = S + dt k1
//        k2    = rhs(S', heff(S') + zeta)                 -- same zeta
//        S    += (dt/2)(k1 + k2)
//        renormalise if requested
// -----------------------------------------------------------------------------
template <class Field, class Rng>
void sllg_heun_step(Field&& heff, Vec3& S, double dt, double alpha, double kT,
                    Rng& g, bool renormalize = true) {
    (void)heff; (void)S; (void)dt; (void)alpha; (void)kT; (void)g;
    (void)renormalize;
    NP_TODO("ex10::sllg_heun_step");
}

// -----------------------------------------------------------------------------
// Semi-implicit B (Mentink et al. 2010): the same predictor-corrector shape,
// but each stage is an exact rotation instead of an Euler step, so |S| is
// conserved identically.
//        zeta   = sqrt(2 alpha kT / dt) * (N,N,N)
//        A1     = llg_axis(S,    heff(S)    + zeta)
//        Spred  = cayley_rotate(S, dt A1)
//        Smid   = (S + Spred)/2
//        A2     = llg_axis(Smid, heff(Smid) + zeta)
//        S      = cayley_rotate(S, dt A2)
// -----------------------------------------------------------------------------
template <class Field, class Rng>
void sllg_sib_step(Field&& heff, Vec3& S, double dt, double alpha, double kT,
                   Rng& g) {
    (void)heff; (void)S; (void)dt; (void)alpha; (void)kT; (void)g;
    NP_TODO("ex10::sllg_sib_step");
}

// ---- provided ---------------------------------------------------------------
inline double langevin_function(double x) {
    if (std::fabs(x) < 1e-6) return x / 3.0;
    return 1.0 / std::tanh(x) - 1.0 / x;
}

}  // namespace np::ex10
