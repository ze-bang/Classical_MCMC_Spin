// Reference solution -- exercise 10.
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
    Mat T, S;
    int dim = 0;

    void init(const Mat& Ap, double h, double m, double kT) {
        dim = Ap.n;
        T = matrix_exp(mscale(Ap, -h));
        Mat C = mscale(Mat::identity(dim), m * kT);
        Mat rest = madd(C, mscale(matmul(matmul(T, C), transpose(T)), -1.0));
        S = cholesky(rest);
    }

    template <class Rng>
    void apply(Vec& ps, Rng& g) const {
        Vec xi(dim);
        for (int i = 0; i < dim; ++i) xi[i] = g.normal();
        Vec out = matvec(T, ps);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j <= i; ++j) out[i] += S(i, j) * xi[j];
        ps = out;
    }
};

template <class Force, class Rng>
void gle_baoab_step(Force&& force, Vec& q, Vec& p, Vec& f, Vec& s, double m,
                    double h, const Gle& gle, Rng& g) {
    p[0] += 0.5 * h * f[0];
    q[0] += 0.5 * h * p[0] / m;
    {
        Vec ps(gle.dim);
        ps[0] = p[0];
        for (int i = 1; i < gle.dim; ++i) ps[i] = s[i - 1];
        gle.apply(ps, g);
        p[0] = ps[0];
        for (int i = 1; i < gle.dim; ++i) s[i - 1] = ps[i];
    }
    q[0] += 0.5 * h * p[0] / m;
    force(q, f);
    p[0] += 0.5 * h * f[0];
}

// ======================= B. stochastic Landau-Lifshitz =====================
// Units: gyromagnetic ratio = 1, moment mu_s = 1, |S| = 1.
//     dS/dt = A(S,H) x S,      A = ( H + alpha S x H ) / (1 + alpha^2)
// which is the Landau-Lifshitz-Gilbert equation written as a precession about
// an effective axis. Thermal field: <zeta_i zeta_j> = 2 alpha kT delta_ij
// delta(t-t'), i.e. per step add sqrt(2 alpha kT / dt) * N(0,1) to H.

inline Vec3 llg_axis(const Vec3& S, const Vec3& H, double alpha) {
    Vec3 sxh = cross(S, H);
    double c = 1.0 / (1.0 + alpha * alpha);
    return {c * (H[0] + alpha * sxh[0]), c * (H[1] + alpha * sxh[1]),
            c * (H[2] + alpha * sxh[2])};
}

inline Vec3 llg_rhs(const Vec3& S, const Vec3& H, double alpha) {
    return cross(llg_axis(S, H, alpha), S);
}

// Exact solution of  S' = S + w x (S + S')/2  (implicit midpoint = Cayley).
// Preserves |S| to machine precision, for any w.
inline Vec3 cayley_rotate(const Vec3& S, const Vec3& w) {
    Vec3 a = 0.5 * w;
    double a2 = dot3(a, a);
    Vec3 axS = cross(a, S);
    Vec3 aaxS = cross(a, axS);
    double c = 2.0 / (1.0 + a2);
    return {S[0] + c * (axS[0] + aaxS[0]), S[1] + c * (axS[1] + aaxS[1]),
            S[2] + c * (axS[2] + aaxS[2])};
}

template <class Field, class Rng>
void sllg_heun_step(Field&& heff, Vec3& S, double dt, double alpha, double kT,
                    Rng& g, bool renormalize = true) {
    const double sN = std::sqrt(2.0 * alpha * kT / dt);
    Vec3 zeta{sN * g.normal(), sN * g.normal(), sN * g.normal()};
    Vec3 H = heff(S) + zeta;
    Vec3 k1 = llg_rhs(S, H, alpha);
    Vec3 Sp{S[0] + dt * k1[0], S[1] + dt * k1[1], S[2] + dt * k1[2]};
    Vec3 Hp = heff(Sp) + zeta;         // same noise realisation: Stratonovich
    Vec3 k2 = llg_rhs(Sp, Hp, alpha);
    for (int i = 0; i < 3; ++i) S[i] += 0.5 * dt * (k1[i] + k2[i]);
    if (renormalize) S = normalized(S);
}

template <class Field, class Rng>
void sllg_sib_step(Field&& heff, Vec3& S, double dt, double alpha, double kT,
                   Rng& g) {
    const double sN = std::sqrt(2.0 * alpha * kT / dt);
    Vec3 zeta{sN * g.normal(), sN * g.normal(), sN * g.normal()};
    Vec3 A1 = llg_axis(S, heff(S) + zeta, alpha);
    Vec3 Spred = cayley_rotate(S, dt * A1);
    Vec3 Smid = 0.5 * (S + Spred);
    Vec3 A2 = llg_axis(Smid, heff(Smid) + zeta, alpha);
    S = cayley_rotate(S, dt * A2);
}

// ---- provided helpers -------------------------------------------------------
inline double langevin_function(double x) {
    if (std::fabs(x) < 1e-6) return x / 3.0;
    return 1.0 / std::tanh(x) - 1.0 / x;
}

}  // namespace np::ex10
