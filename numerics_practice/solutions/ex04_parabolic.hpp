// Reference solution -- exercise 04.
#pragma once
#include <cmath>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex04 {

using np::Vec;

inline void laplacian_1d(const Vec& u, double dx, Vec& out) {
    const size_t n = u.size();
    out.assign(n, 0.0);
    const double inv = 1.0 / (dx * dx);
    for (size_t j = 1; j + 1 < n; ++j)
        out[j] = (u[j - 1] - 2.0 * u[j] + u[j + 1]) * inv;
}

inline void heat_ftcs_step(Vec& u, double alpha, double dx, double dt) {
    const size_t n = u.size();
    const double r = alpha * dt / (dx * dx);
    static thread_local Vec old;
    old = u;
    for (size_t j = 1; j + 1 < n; ++j)
        u[j] = old[j] + r * (old[j - 1] - 2.0 * old[j] + old[j + 1]);
}

inline void heat_crank_nicolson_step(Vec& u, double alpha, double dx,
                                     double dt) {
    const int n = int(u.size());
    const int m = n - 2;  // interior unknowns
    if (m <= 0) return;
    const double r = alpha * dt / (dx * dx);
    Vec a(m, -0.5 * r), b(m, 1.0 + r), c(m, -0.5 * r), d(m), x;
    a[0] = 0.0;
    c[m - 1] = 0.0;
    for (int i = 0; i < m; ++i) {
        int j = i + 1;
        d[i] = 0.5 * r * u[j - 1] + (1.0 - r) * u[j] + 0.5 * r * u[j + 1];
    }
    thomas(a, b, c, d, x);
    for (int i = 0; i < m; ++i) u[i + 1] = x[i];
}

inline void heat2d_adi_step(Vec& u, int nx, int ny, double alpha, double dx,
                            double dt) {
    const double r = alpha * dt / (dx * dx);   // full-step r; half-steps use r/2
    Vec ustar(u.size(), 0.0);
    const int mx = nx - 2, my = ny - 2;
    if (mx <= 0 || my <= 0) return;

    // ---- half step 1: implicit in x, explicit in y
    {
        Vec a(mx, -0.5 * r), b(mx, 1.0 + r), c(mx, -0.5 * r), d(mx), x;
        a[0] = 0.0;
        c[mx - 1] = 0.0;
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i)
                d[i - 1] = u[size_t(j) * nx + i] +
                           0.5 * r * (u[size_t(j - 1) * nx + i] -
                                      2.0 * u[size_t(j) * nx + i] +
                                      u[size_t(j + 1) * nx + i]);
            thomas(a, b, c, d, x);
            for (int i = 1; i < nx - 1; ++i) ustar[size_t(j) * nx + i] = x[i - 1];
        }
    }
    // ---- half step 2: implicit in y, explicit in x
    {
        Vec a(my, -0.5 * r), b(my, 1.0 + r), c(my, -0.5 * r), d(my), x;
        a[0] = 0.0;
        c[my - 1] = 0.0;
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j)
                d[j - 1] = ustar[size_t(j) * nx + i] +
                           0.5 * r * (ustar[size_t(j) * nx + i - 1] -
                                      2.0 * ustar[size_t(j) * nx + i] +
                                      ustar[size_t(j) * nx + i + 1]);
            thomas(a, b, c, d, x);
            for (int j = 1; j < ny - 1; ++j) u[size_t(j) * nx + i] = x[j - 1];
        }
    }
}

inline double discrete_laplacian_eigenvalue(double k, double dx) {
    double s = std::sin(0.5 * k * pi * dx);
    return -4.0 * s * s / (dx * dx);
}

}  // namespace np::ex04
