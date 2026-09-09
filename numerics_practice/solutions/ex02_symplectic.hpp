// Reference solution -- exercise 02.
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex02 {

using np::Vec;

template <class Force>
void symplectic_euler_step(Force&& force, Vec& q, Vec& p, double m, double h) {
    static thread_local Vec f;
    f.assign(q.size(), 0.0);
    force(q, f);
    for (size_t i = 0; i < q.size(); ++i) {
        p[i] += h * f[i];
        q[i] += h * p[i] / m;
    }
}

template <class Force>
void velocity_verlet_step(Force&& force, Vec& q, Vec& p, Vec& f, double m,
                          double h) {
    const size_t n = q.size();
    for (size_t i = 0; i < n; ++i) {
        p[i] += 0.5 * h * f[i];
        q[i] += h * p[i] / m;
    }
    force(q, f);
    for (size_t i = 0; i < n; ++i) p[i] += 0.5 * h * f[i];
}

template <class Force>
void leapfrog_step(Force&& force, Vec& q, Vec& p, double m, double h) {
    const size_t n = q.size();
    static thread_local Vec f;
    f.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) q[i] += 0.5 * h * p[i] / m;
    force(q, f);
    for (size_t i = 0; i < n; ++i) {
        p[i] += h * f[i];
        q[i] += 0.5 * h * p[i] / m;
    }
}

template <class Force>
void yoshida4_step(Force&& force, Vec& q, Vec& p, double m, double h) {
    const double cbrt2 = std::cbrt(2.0);
    const double w1 = 1.0 / (2.0 - cbrt2);
    const double w0 = -cbrt2 / (2.0 - cbrt2);
    leapfrog_step(force, q, p, m, w1 * h);
    leapfrog_step(force, q, p, m, w0 * h);
    leapfrog_step(force, q, p, m, w1 * h);
}

}  // namespace np::ex02
