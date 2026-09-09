// Reference solution -- exercise 01.
#pragma once
#include <algorithm>
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex01 {

using np::Vec;

template <class F>
void euler_step(F&& f, double t, Vec& y, double h) {
    static thread_local Vec k;
    k.assign(y.size(), 0.0);
    f(t, y, k);
    for (size_t i = 0; i < y.size(); ++i) y[i] += h * k[i];
}

template <class F>
void midpoint_step(F&& f, double t, Vec& y, double h) {
    const size_t n = y.size();
    static thread_local Vec k1, k2, ytmp;
    k1.assign(n, 0.0); k2.assign(n, 0.0); ytmp.assign(n, 0.0);
    f(t, y, k1);
    for (size_t i = 0; i < n; ++i) ytmp[i] = y[i] + 0.5 * h * k1[i];
    f(t + 0.5 * h, ytmp, k2);
    for (size_t i = 0; i < n; ++i) y[i] += h * k2[i];
}

template <class F>
void heun_step(F&& f, double t, Vec& y, double h) {
    const size_t n = y.size();
    static thread_local Vec k1, k2, ytmp;
    k1.assign(n, 0.0); k2.assign(n, 0.0); ytmp.assign(n, 0.0);
    f(t, y, k1);
    for (size_t i = 0; i < n; ++i) ytmp[i] = y[i] + h * k1[i];
    f(t + h, ytmp, k2);
    for (size_t i = 0; i < n; ++i) y[i] += 0.5 * h * (k1[i] + k2[i]);
}

template <class F>
void rk4_step(F&& f, double t, Vec& y, double h) {
    const size_t n = y.size();
    static thread_local Vec k1, k2, k3, k4, ytmp;
    k1.assign(n, 0.0); k2.assign(n, 0.0); k3.assign(n, 0.0);
    k4.assign(n, 0.0); ytmp.assign(n, 0.0);
    f(t, y, k1);
    for (size_t i = 0; i < n; ++i) ytmp[i] = y[i] + 0.5 * h * k1[i];
    f(t + 0.5 * h, ytmp, k2);
    for (size_t i = 0; i < n; ++i) ytmp[i] = y[i] + 0.5 * h * k2[i];
    f(t + 0.5 * h, ytmp, k3);
    for (size_t i = 0; i < n; ++i) ytmp[i] = y[i] + h * k3[i];
    f(t + h, ytmp, k4);
    for (size_t i = 0; i < n; ++i)
        y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

template <class F>
void dopri5_step(F&& f, double t, const Vec& y, double h, Vec& y5, Vec& y4) {
    const size_t n = y.size();
    static thread_local Vec k1, k2, k3, k4, k5, k6, k7, ys;
    k1.assign(n, 0.0); k2.assign(n, 0.0); k3.assign(n, 0.0); k4.assign(n, 0.0);
    k5.assign(n, 0.0); k6.assign(n, 0.0); k7.assign(n, 0.0); ys.assign(n, 0.0);
    y5.assign(n, 0.0); y4.assign(n, 0.0);

    f(t, y, k1);
    for (size_t i = 0; i < n; ++i) ys[i] = y[i] + h * (1.0 / 5 * k1[i]);
    f(t + 1.0 / 5 * h, ys, k2);
    for (size_t i = 0; i < n; ++i)
        ys[i] = y[i] + h * (3.0 / 40 * k1[i] + 9.0 / 40 * k2[i]);
    f(t + 3.0 / 10 * h, ys, k3);
    for (size_t i = 0; i < n; ++i)
        ys[i] = y[i] + h * (44.0 / 45 * k1[i] - 56.0 / 15 * k2[i] + 32.0 / 9 * k3[i]);
    f(t + 4.0 / 5 * h, ys, k4);
    for (size_t i = 0; i < n; ++i)
        ys[i] = y[i] + h * (19372.0 / 6561 * k1[i] - 25360.0 / 2187 * k2[i] +
                            64448.0 / 6561 * k3[i] - 212.0 / 729 * k4[i]);
    f(t + 8.0 / 9 * h, ys, k5);
    for (size_t i = 0; i < n; ++i)
        ys[i] = y[i] + h * (9017.0 / 3168 * k1[i] - 355.0 / 33 * k2[i] +
                            46732.0 / 5247 * k3[i] + 49.0 / 176 * k4[i] -
                            5103.0 / 18656 * k5[i]);
    f(t + h, ys, k6);
    for (size_t i = 0; i < n; ++i)
        y5[i] = y[i] + h * (35.0 / 384 * k1[i] + 500.0 / 1113 * k3[i] +
                            125.0 / 192 * k4[i] - 2187.0 / 6784 * k5[i] +
                            11.0 / 84 * k6[i]);
    f(t + h, y5, k7);
    for (size_t i = 0; i < n; ++i)
        y4[i] = y[i] + h * (5179.0 / 57600 * k1[i] + 7571.0 / 16695 * k3[i] +
                            393.0 / 640 * k4[i] - 92097.0 / 339200 * k5[i] +
                            187.0 / 2100 * k6[i] + 1.0 / 40 * k7[i]);
}

struct AdaptiveResult {
    double t = 0;
    long naccepted = 0;
    long nrejected = 0;
    double h_last = 0;
};

template <class F>
AdaptiveResult integrate_adaptive(F&& f, double t0, double t1, Vec& y,
                                  double h0, double atol, double rtol) {
    AdaptiveResult res;
    Vec y5, y4;
    double t = t0, h = h0;
    double err_prev = 1e-4;
    const double alpha = 0.7 / 5.0, beta = 0.4 / 5.0;
    long guard = 0;
    while (t < t1 - 1e-14 * std::fabs(t1)) {
        if (++guard > 10'000'000) break;
        h = std::min(h, t1 - t);
        dopri5_step(f, t, y, h, y5, y4);
        double s = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            double sc = atol + rtol * std::max(std::fabs(y[i]), std::fabs(y5[i]));
            double e = (y5[i] - y4[i]) / sc;
            s += e * e;
        }
        double err = std::sqrt(s / double(y.size()));
        if (err <= 1.0) {
            t += h;
            y = y5;
            ++res.naccepted;
            double fac = 0.9 * std::pow(std::max(err, 1e-10), -alpha) *
                         std::pow(std::max(err_prev, 1e-10), beta);
            h *= std::clamp(fac, 0.2, 5.0);
            err_prev = err;
        } else {
            ++res.nrejected;
            double fac = 0.9 * std::pow(err, -1.0 / 5.0);
            h *= std::clamp(fac, 0.2, 1.0);
        }
    }
    res.t = t;
    res.h_last = h;
    return res;
}

template <class Stepper, class F>
void integrate_fixed(Stepper&& step, F&& f, double t0, double t1, Vec& y,
                     long nsteps) {
    double h = (t1 - t0) / double(nsteps);
    for (long n = 0; n < nsteps; ++n) step(f, t0 + double(n) * h, y, h);
}

}  // namespace np::ex01
