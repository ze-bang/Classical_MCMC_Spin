// Reference solution -- exercise 05.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex05 {

using np::Vec;

inline int wrap(int j, int n) { return (j % n + n) % n; }

inline void advect_upwind_step(Vec& u, double a, double dx, double dt) {
    const int n = int(u.size());
    const double C = a * dt / dx;
    static thread_local Vec old;
    old = u;
    for (int j = 0; j < n; ++j)
        u[j] = (a >= 0) ? old[j] - C * (old[j] - old[wrap(j - 1, n)])
                        : old[j] - C * (old[wrap(j + 1, n)] - old[j]);
}

inline void advect_lax_wendroff_step(Vec& u, double a, double dx, double dt) {
    const int n = int(u.size());
    const double C = a * dt / dx;
    static thread_local Vec old;
    old = u;
    for (int j = 0; j < n; ++j) {
        double um = old[wrap(j - 1, n)], up = old[wrap(j + 1, n)];
        u[j] = old[j] - 0.5 * C * (up - um) + 0.5 * C * C * (up - 2 * old[j] + um);
    }
}

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::fabs(a) < std::fabs(b)) ? a : b;
}

inline void advect_muscl_step(Vec& u, double a, double dx, double dt) {
    const int n = int(u.size());
    static thread_local Vec old, sigma, flux;
    old = u;
    sigma.assign(n, 0.0);
    flux.assign(n, 0.0);
    for (int j = 0; j < n; ++j)
        sigma[j] = minmod((old[j] - old[wrap(j - 1, n)]) / dx,
                          (old[wrap(j + 1, n)] - old[j]) / dx);
    const double C = a * dt / dx;
    for (int j = 0; j < n; ++j) {   // flux[j] = F_{j+1/2}
        if (a >= 0) {
            flux[j] = a * (old[j] + 0.5 * dx * (1.0 - C) * sigma[j]);
        } else {
            int jp = wrap(j + 1, n);
            flux[j] = a * (old[jp] - 0.5 * dx * (1.0 + C) * sigma[jp]);
        }
    }
    for (int j = 0; j < n; ++j)
        u[j] = old[j] - (dt / dx) * (flux[j] - flux[wrap(j - 1, n)]);
}

inline void burgers_rusanov_step(Vec& u, double dx, double dt) {
    const int n = int(u.size());
    static thread_local Vec old, flux;
    old = u;
    flux.assign(n, 0.0);
    auto f = [](double v) { return 0.5 * v * v; };
    for (int j = 0; j < n; ++j) {
        double uL = old[j], uR = old[wrap(j + 1, n)];
        double amax = std::max(std::fabs(uL), std::fabs(uR));
        flux[j] = 0.5 * (f(uL) + f(uR)) - 0.5 * amax * (uR - uL);
    }
    for (int j = 0; j < n; ++j)
        u[j] = old[j] - (dt / dx) * (flux[j] - flux[wrap(j - 1, n)]);
}

inline double total_variation(const Vec& u) {
    double tv = 0;
    const int n = int(u.size());
    for (int j = 0; j < n; ++j) tv += std::fabs(u[wrap(j + 1, n)] - u[j]);
    return tv;
}
inline double integral(const Vec& u, double dx) {
    double s = 0;
    for (double v : u) s += v;
    return s * dx;
}

}  // namespace np::ex05
