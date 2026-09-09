// Reference solution -- exercise 06.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex06 {

using np::Cplx;
using np::Vec;

inline void poisson_residual(const Vec& u, const Vec& f, double h, Vec& r) {
    const int n = int(u.size());
    r.assign(n, 0.0);
    const double inv = 1.0 / (h * h);
    for (int j = 1; j + 1 < n; ++j)
        r[j] = f[j] - (-u[j - 1] + 2.0 * u[j] - u[j + 1]) * inv;
}

inline void gauss_seidel_sweep(Vec& u, const Vec& f, double h) {
    const int n = int(u.size());
    for (int j = 1; j + 1 < n; ++j)
        u[j] = 0.5 * (u[j - 1] + u[j + 1] + h * h * f[j]);
}

inline void sor_sweep(Vec& u, const Vec& f, double h, double omega) {
    const int n = int(u.size());
    for (int j = 1; j + 1 < n; ++j) {
        double gs = 0.5 * (u[j - 1] + u[j + 1] + h * h * f[j]);
        u[j] = (1.0 - omega) * u[j] + omega * gs;
    }
}

inline void restrict_full_weighting(const Vec& r, Vec& rc) {
    const int n = int(r.size());
    const int nc = (n + 1) / 2;
    rc.assign(nc, 0.0);
    for (int i = 1; i + 1 < nc; ++i)
        rc[i] = 0.25 * (r[2 * i - 1] + 2.0 * r[2 * i] + r[2 * i + 1]);
}

inline void prolong_linear(const Vec& ec, Vec& e) {
    const int nc = int(ec.size());
    const int n = 2 * nc - 1;
    e.assign(n, 0.0);
    for (int i = 0; i < nc; ++i) e[2 * i] = ec[i];
    for (int i = 0; i + 1 < nc; ++i) e[2 * i + 1] = 0.5 * (ec[i] + ec[i + 1]);
}

inline void mg_vcycle(Vec& u, const Vec& f, double h, int nu1 = 2, int nu2 = 2) {
    const int n = int(u.size());
    if (n <= 3) {
        if (n == 3) u[1] = 0.5 * h * h * f[1];
        return;
    }
    for (int s = 0; s < nu1; ++s) gauss_seidel_sweep(u, f, h);
    Vec r, rc;
    poisson_residual(u, f, h, r);
    restrict_full_weighting(r, rc);
    Vec ec(rc.size(), 0.0);
    mg_vcycle(ec, rc, 2.0 * h, nu1, nu2);
    Vec e;
    prolong_linear(ec, e);
    for (int j = 1; j + 1 < n; ++j) u[j] += e[j];
    for (int s = 0; s < nu2; ++s) gauss_seidel_sweep(u, f, h);
}

inline void fft(std::vector<Cplx>& a, bool inverse) {
    const size_t N = a.size();
    if (N <= 1) return;
    // bit-reversal permutation
    for (size_t i = 1, j = 0; i < N; ++i) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= N; len <<= 1) {
        double ang = 2.0 * pi / double(len) * (inverse ? 1.0 : -1.0);
        Cplx wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < N; i += len) {
            Cplx w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                Cplx x = a[i + k], y = a[i + k + len / 2] * w;
                a[i + k] = x + y;
                a[i + k + len / 2] = x - y;
                w *= wlen;
            }
        }
    }
    if (inverse)
        for (auto& z : a) z /= double(N);
}

// wavenumber of mode m for n samples on [0,L)
inline double wavenumber(int m, int n, double L) {
    int mm = (m < n / 2) ? m : m - n;
    return 2.0 * pi * double(mm) / L;
}

inline void spectral_derivative(const Vec& u, double L, Vec& du) {
    const int n = int(u.size());
    std::vector<Cplx> a(n);
    for (int j = 0; j < n; ++j) a[j] = Cplx(u[j], 0.0);
    fft(a, false);
    for (int m = 0; m < n; ++m) {
        double k = (m == n / 2) ? 0.0 : wavenumber(m, n, L);
        a[m] *= Cplx(0.0, k);
    }
    fft(a, true);
    du.assign(n, 0.0);
    for (int j = 0; j < n; ++j) du[j] = a[j].real();
}

inline void poisson_fft(const Vec& f, double L, Vec& u) {
    const int n = int(f.size());
    std::vector<Cplx> a(n);
    for (int j = 0; j < n; ++j) a[j] = Cplx(f[j], 0.0);
    fft(a, false);
    for (int m = 0; m < n; ++m) {
        double k = wavenumber(m, n, L);
        a[m] = (m == 0) ? Cplx(0.0, 0.0) : a[m] / (k * k);
    }
    fft(a, true);
    u.assign(n, 0.0);
    for (int j = 0; j < n; ++j) u[j] = a[j].real();
}

template <class G>
inline Cplx phi_contour(G&& g, Cplx z, int M = 32, double radius = 1.0) {
    Cplx s = 0.0;
    for (int j = 0; j < M; ++j) {
        double th = 2.0 * pi * (double(j) + 0.5) / double(M);
        s += g(z + radius * Cplx(std::cos(th), std::sin(th)));
    }
    return s / double(M);
}

struct Etdrk4 {
    int n = 0;
    double h = 0;
    std::vector<Cplx> E, E2, Q, f1, f2, f3;

    void init(const std::vector<double>& lam, double dt, int M = 32) {
        n = int(lam.size());
        h = dt;
        E.resize(n); E2.resize(n); Q.resize(n);
        f1.resize(n); f2.resize(n); f3.resize(n);
        for (int m = 0; m < n; ++m) {
            Cplx z = h * lam[m];
            E[m] = std::exp(z);
            E2[m] = std::exp(0.5 * z);
            Q[m] = h * phi_contour([](Cplx w) {
                return (std::exp(0.5 * w) - 1.0) / w;
            }, z, M);
            f1[m] = h * phi_contour([](Cplx w) {
                Cplx ew = std::exp(w);
                return (-4.0 - w + ew * (4.0 - 3.0 * w + w * w)) / (w * w * w);
            }, z, M);
            f2[m] = h * phi_contour([](Cplx w) {
                Cplx ew = std::exp(w);
                return (2.0 + w + ew * (-2.0 + w)) / (w * w * w);
            }, z, M);
            f3[m] = h * phi_contour([](Cplx w) {
                Cplx ew = std::exp(w);
                return (-4.0 - 3.0 * w - w * w + ew * (4.0 - w)) / (w * w * w);
            }, z, M);
        }
    }

    template <class NL>
    void step(std::vector<Cplx>& v, NL&& N) const {
        std::vector<Cplx> Nv(n), a(n), Na(n), b(n), Nb(n), c(n), Nc(n);
        N(v, Nv);
        for (int m = 0; m < n; ++m) a[m] = E2[m] * v[m] + Q[m] * Nv[m];
        N(a, Na);
        for (int m = 0; m < n; ++m) b[m] = E2[m] * v[m] + Q[m] * Na[m];
        N(b, Nb);
        for (int m = 0; m < n; ++m)
            c[m] = E2[m] * a[m] + Q[m] * (2.0 * Nb[m] - Nv[m]);
        N(c, Nc);
        for (int m = 0; m < n; ++m)
            v[m] = E[m] * v[m] + Nv[m] * f1[m] +
                   2.0 * (Na[m] + Nb[m]) * f2[m] + Nc[m] * f3[m];
    }
};

}  // namespace np::ex06
