#pragma once
/**
 * spin_integrators.h — geometric integrators for classical SO(3) spin
 * dynamics, deterministic and stochastic (Langevin).
 *
 * Equation of motion (Landau-Lifshitz form, γ = ħ = 1, |S_i| = s):
 *
 *     dS_i/dt = S_i × Ω_i,   Ω_i = b_i - (α/s) S_i × b_i,   b_i = B_i + ξ_i,
 *
 * with B_i = -∂E/∂S_i the effective field (including any time-dependent
 * drive) and ξ_i Gaussian white noise, <ξ_ia(t) ξ_jb(t')> = 2D δ_ij δ_ab δ(t-t'),
 * entering both the precession and the damping term (Stratonovich). The
 * Fokker-Planck equation of this SDE has the Gibbs state exp(-E/T) as its
 * stationary solution iff
 *
 *     D = α T / (s (1 + α²))
 *
 * (García-Palacios & Lázaro, PRB 58, 14937 (1998)); a noise variance of
 * 2αT/s, as commonly written, heats the spins to T (1 + α²).
 *
 * Methods — every one preserves each |S_i| to round-off, unlike generic
 * Runge-Kutta steppers whose norm drift grows secularly:
 *
 *   spherical_midpoint  McLachlan, Modin & Verdier, PRE 89, 061301(R) (2014):
 *                       S'_i = S_i + dt Ŝ_i × Ω_i(Ŝ),  Ŝ_i = s (S_i+S'_i)/|S_i+S'_i|,
 *                       solved by fixed-point iteration. Symplectic and
 *                       time-reversible for α = 0 (no secular energy drift),
 *                       second order, any Hamiltonian; with noise it is the
 *                       Stratonovich-consistent midpoint scheme (cf. Mentink
 *                       et al., JPCM 22, 176001 (2010)).
 *   depondt             Depondt & Mertens, JPCM 21, 336005 (2009): Heun
 *                       predictor-corrector in which each update is an exact
 *                       rotation about the (averaged) Ω. Explicit, second
 *                       order, two field evaluations per step; with noise
 *                       held fixed over the step it is the stochastic Heun
 *                       scheme on the sphere.
 *   color_split         Suzuki-Trotter splitting over the sublattice colouring
 *   color_split4        (Krech, Bunker & Landau, CPC 111, 1 (1998); Omelyan,
 *                       Mryglod & Folk, PRL 86, 898 (2001)): spins of one
 *                       colour see a field frozen by the other colours and are
 *                       rotated exactly about it. Symmetric (Strang) ordering
 *                       gives a second-order symplectic map that conserves the
 *                       energy of a time-independent Hamiltonian EXACTLY at
 *                       every step; color_split4 is its Yoshida fourth-order
 *                       composition. Requires a local energy linear in S_i
 *                       (no single-ion anisotropy) and α = T = 0.
 *
 * Model interface (duck-typed):
 *   size_t n_sites() const;
 *   double spin_length() const;
 *   void   field_all(const double* x, double t, double* B) const;      // B_i = -∂E/∂S_i, all i
 *   void   set_time(double t) const;            // caches drive envelopes for field_site
 *   void   field_site(const double* x, double t, size_t i, double* B) const;
 *   size_t n_colors() const;
 *   const size_t* color_sites(size_t c, size_t& count) const;
 *   bool   energy_linear_in_each_spin() const;                         // color_split precondition
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "classical_spin/core/simple_linear_alg.h"  // random_normal_lehman

namespace classical_spin::dynamics {

enum class Method { SphericalMidpoint, Depondt, ColorSplit, ColorSplit4 };

inline bool is_geometric_method(std::string_view name) {
    return name == "spherical_midpoint" || name == "implicit_midpoint" || name == "midpoint_sphere" ||
           name == "depondt" || name == "heun_sphere" ||
           name == "color_split" || name == "suzuki_trotter" || name == "color_split4";
}

inline Method parse_geometric_method(std::string_view name) {
    if (name == "spherical_midpoint" || name == "implicit_midpoint" || name == "midpoint_sphere")
        return Method::SphericalMidpoint;
    if (name == "depondt" || name == "heun_sphere") return Method::Depondt;
    if (name == "color_split" || name == "suzuki_trotter") return Method::ColorSplit;
    if (name == "color_split4") return Method::ColorSplit4;
    throw std::invalid_argument("unknown geometric spin integrator '" + std::string(name) +
                                "' (valid: spherical_midpoint, depondt, color_split, color_split4)");
}

inline const char* method_name(Method m) {
    switch (m) {
        case Method::SphericalMidpoint: return "spherical_midpoint";
        case Method::Depondt:           return "depondt";
        case Method::ColorSplit:        return "color_split";
        case Method::ColorSplit4:       return "color_split4";
    }
    return "?";
}

struct LangevinParams {
    double alpha = 0.0;        // Gilbert/LL damping
    double temperature = 0.0;  // bath temperature (k_B = 1); 0 = deterministic
};

/// Rotate v (3-vector) for a time dt under dv/dt = v × Ω: angle -|Ω| dt about Ω̂.
inline void rotate_by_precession(const double* Om, double dt, double* v) {
    const double w2 = Om[0] * Om[0] + Om[1] * Om[1] + Om[2] * Om[2];
    if (w2 == 0.0) return;
    const double w = std::sqrt(w2);
    const double n[3] = {Om[0] / w, Om[1] / w, Om[2] / w};
    const double th = -w * dt;
    const double c = std::cos(th), s = std::sin(th);
    const double ndv = n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
    const double nxv[3] = {n[1] * v[2] - n[2] * v[1], n[2] * v[0] - n[0] * v[2], n[0] * v[1] - n[1] * v[0]};
    for (int d = 0; d < 3; ++d) v[d] = v[d] * c + nxv[d] * s + n[d] * ndv * (1.0 - c);
}

/// Ω = b - (α/s) S × b for one site.
inline void precession_axis(const double* S, const double* b, double alpha, double s, double* Om) {
    if (alpha == 0.0) { Om[0] = b[0]; Om[1] = b[1]; Om[2] = b[2]; return; }
    const double k = alpha / s;
    Om[0] = b[0] - k * (S[1] * b[2] - S[2] * b[1]);
    Om[1] = b[1] - k * (S[2] * b[0] - S[0] * b[2]);
    Om[2] = b[2] - k * (S[0] * b[1] - S[1] * b[0]);
}

struct StepStats {
    int midpoint_iterations = 0;  // last spherical-midpoint solve
    double midpoint_residual = 0.0;
};

template<class Model>
class SpinIntegrator {
public:
    SpinIntegrator(const Model& model, Method method, LangevinParams lp = {},
                   double midpoint_tol = 1e-13, int midpoint_max_iter = 100)
        : m_(model), method_(method), lp_(lp), tol_(midpoint_tol), max_iter_(midpoint_max_iter) {
        const size_t n3 = 3 * m_.n_sites();
        B_.assign(n3, 0.0);
        B2_.assign(n3, 0.0);
        xi_.assign(n3, 0.0);
        work_.assign(n3, 0.0);
        mid_.assign(n3, 0.0);
        if ((method_ == Method::ColorSplit || method_ == Method::ColorSplit4)) {
            if (lp_.alpha != 0.0 || lp_.temperature != 0.0)
                throw std::invalid_argument("color_split integrators are Hamiltonian: use alpha = T = 0 "
                                            "(spherical_midpoint or depondt support damping and noise)");
            if (!m_.energy_linear_in_each_spin())
                throw std::invalid_argument("color_split needs an energy linear in each spin (no "
                                            "single-ion anisotropy); use spherical_midpoint");
            if (m_.n_colors() == 0)
                throw std::invalid_argument("color_split needs a sublattice colouring");
        }
        if (lp_.temperature < 0.0 || lp_.alpha < 0.0)
            throw std::invalid_argument("Langevin parameters must be non-negative");
    }

    Method method() const { return method_; }
    const StepStats& stats() const { return stats_; }

    /// Advance x (3 N doubles) from time t to t + dt.
    void step(double* x, double t, double dt) {
        draw_noise(dt);
        switch (method_) {
            case Method::SphericalMidpoint: spherical_midpoint(x, t, dt); break;
            case Method::Depondt:           depondt(x, t, dt); break;
            case Method::ColorSplit:        color_split2(x, t, dt); break;
            case Method::ColorSplit4: {
                // Yoshida (1990) triple jump of the symmetric second-order map.
                const double c = std::cbrt(2.0);
                const double w1 = 1.0 / (2.0 - c), w0 = -c / (2.0 - c);
                color_split2(x, t, w1 * dt);
                color_split2(x, t + w1 * dt, w0 * dt);
                color_split2(x, t + (w1 + w0) * dt, w1 * dt);
                break;
            }
        }
    }

private:
    const Model& m_;
    Method method_;
    LangevinParams lp_;
    double tol_;
    int max_iter_;
    std::vector<double> B_, B2_, xi_, work_, mid_;
    StepStats stats_;

    size_t n() const { return m_.n_sites(); }

    bool stochastic() const { return lp_.temperature > 0.0 && lp_.alpha > 0.0; }

    void draw_noise(double dt) {
        if (!stochastic()) return;
        const double s = m_.spin_length();
        const double D = lp_.alpha * lp_.temperature / (s * (1.0 + lp_.alpha * lp_.alpha));
        const double sigma = std::sqrt(2.0 * D / dt);  // piecewise-constant white noise
        const size_t n3 = 3 * n();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n3 >= 3 * 512)
#endif
        for (size_t k = 0; k < n3; ++k) xi_[k] = sigma * random_normal_lehman();
    }

    // b = B (+ ξ) and Ω for every site, from fields already in `B`.
    void axes_from_fields(const double* S, const std::vector<double>& B, double* Om) const {
        const double s = m_.spin_length();
        const bool noisy = stochastic();
        const size_t N = n();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= 512)
#endif
        for (size_t i = 0; i < N; ++i) {
            double b[3] = {B[3 * i], B[3 * i + 1], B[3 * i + 2]};
            if (noisy) for (int d = 0; d < 3; ++d) b[d] += xi_[3 * i + d];
            precession_axis(S + 3 * i, b, lp_.alpha, s, Om + 3 * i);
        }
    }

    void depondt(double* x, double t, double dt) {
        const size_t N = n();
        std::vector<double>& Om1 = B2_;
        m_.field_all(x, t, B_.data());
        axes_from_fields(x, B_, Om1.data());
        // Predictor: rotate a copy about Ω(S_n).
        std::copy(x, x + 3 * N, work_.begin());
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= 512)
#endif
        for (size_t i = 0; i < N; ++i) rotate_by_precession(&Om1[3 * i], dt, &work_[3 * i]);
        // Corrector: rotate S_n about the mean of Ω(S_n) and Ω(S_pred).
        m_.field_all(work_.data(), t + dt, B_.data());
        std::vector<double>& Om2 = mid_;
        axes_from_fields(work_.data(), B_, Om2.data());
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= 512)
#endif
        for (size_t i = 0; i < N; ++i) {
            double Om[3];
            for (int d = 0; d < 3; ++d) Om[d] = 0.5 * (Om1[3 * i + d] + Om2[3 * i + d]);
            rotate_by_precession(Om, dt, &x[3 * i]);
        }
    }

    void spherical_midpoint(double* x, double t, double dt) {
        const size_t N = n();
        const double s = m_.spin_length();
        std::vector<double>& xn = work_;   // S' iterate
        std::vector<double>& mid = mid_;
        std::vector<double>& Om = B2_;
        // Initial guess: one explicit Depondt-like rotation about Ω(S_n).
        m_.field_all(x, t, B_.data());
        axes_from_fields(x, B_, Om.data());
        std::copy(x, x + 3 * N, xn.begin());
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= 512)
#endif
        for (size_t i = 0; i < N; ++i) rotate_by_precession(&Om[3 * i], dt, &xn[3 * i]);

        const double th = t + 0.5 * dt;
        double resid = 0.0;
        int it = 0;
        for (; it < max_iter_; ++it) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(N >= 512)
#endif
            for (size_t i = 0; i < N; ++i) {
                double m3[3], nn = 0.0;
                for (int d = 0; d < 3; ++d) { m3[d] = x[3 * i + d] + xn[3 * i + d]; nn += m3[d] * m3[d]; }
                const double sc = (nn > 0.0) ? s / std::sqrt(nn) : 0.0;
                for (int d = 0; d < 3; ++d) mid[3 * i + d] = sc * m3[d];
            }
            m_.field_all(mid.data(), th, B_.data());
            axes_from_fields(mid.data(), B_, Om.data());
            resid = 0.0;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(max:resid) if(N >= 512)
#endif
            for (size_t i = 0; i < N; ++i) {
                const double* M = &mid[3 * i];
                const double* W = &Om[3 * i];
                const double cr[3] = {M[1] * W[2] - M[2] * W[1], M[2] * W[0] - M[0] * W[2], M[0] * W[1] - M[1] * W[0]};
                for (int d = 0; d < 3; ++d) {
                    const double v = x[3 * i + d] + dt * cr[d];
                    resid = std::max(resid, std::abs(v - xn[3 * i + d]));
                    xn[3 * i + d] = v;
                }
            }
            if (resid <= tol_ * s) { ++it; break; }
        }
        stats_.midpoint_iterations = it;
        stats_.midpoint_residual = resid;
        if (resid > 1e3 * tol_ * s)
            throw std::runtime_error("spherical_midpoint: fixed-point iteration did not converge "
                                     "(residual " + std::to_string(resid) + "); reduce the time step");
        // At the fixed point (S'-S) ⊥ (S+S'), so |S'| = |S|; remove the
        // residual iteration error exactly.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= 512)
#endif
        for (size_t i = 0; i < N; ++i) {
            double nn = 0.0;
            for (int d = 0; d < 3; ++d) nn += xn[3 * i + d] * xn[3 * i + d];
            const double sc = s / std::sqrt(nn);
            for (int d = 0; d < 3; ++d) x[3 * i + d] = sc * xn[3 * i + d];
        }
    }

    // One colour: rotate every site exactly about its frozen field for tau.
    void rotate_color(double* x, double t, size_t c, double tau) {
        m_.set_time(t);
        size_t count = 0;
        const size_t* sites = m_.color_sites(c, count);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(count >= 256)
#endif
        for (size_t k = 0; k < count; ++k) {
            const size_t i = sites[k];
            double B[3];
            m_.field_site(x, t, i, B);
            rotate_by_precession(B, tau, &x[3 * i]);
        }
    }

    void color_split2(double* x, double t, double dt) {
        const size_t C = m_.n_colors();
        const double th = t + 0.5 * dt;  // second order for time-dependent drives
        if (C == 1) { rotate_color(x, th, 0, dt); return; }
        for (size_t c = 0; c + 1 < C; ++c) rotate_color(x, th, c, 0.5 * dt);
        rotate_color(x, th, C - 1, dt);
        for (size_t c = C - 1; c-- > 0;) rotate_color(x, th, c, 0.5 * dt);
    }
};

}  // namespace classical_spin::dynamics
