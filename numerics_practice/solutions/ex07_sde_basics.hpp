// Reference solution -- exercise 07.
#pragma once
#include <cmath>
#include <vector>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex07 {

template <class A, class B>
void euler_maruyama_step(A&& a, B&& b, double t, double& x, double dt,
                         double dW) {
    x += a(t, x) * dt + b(t, x) * dW;
}

template <class A, class B, class Bp>
void milstein_step(A&& a, B&& b, Bp&& dbdx, double t, double& x, double dt,
                   double dW) {
    const double bx = b(t, x);
    x += a(t, x) * dt + bx * dW + 0.5 * bx * dbdx(t, x) * (dW * dW - dt);
}

template <class A, class B>
void stratonovich_heun_step(A&& a, B&& b, double t, double& x, double dt,
                            double dW) {
    const double a0 = a(t, x), b0 = b(t, x);
    const double xbar = x + a0 * dt + b0 * dW;
    x += 0.5 * dt * (a0 + a(t + dt, xbar)) + 0.5 * dW * (b0 + b(t + dt, xbar));
}

inline std::vector<double> brownian_path(int n, double dt, Rng& g) {
    std::vector<double> dW(static_cast<size_t>(n));
    const double s = std::sqrt(dt);
    for (int i = 0; i < n; ++i) dW[size_t(i)] = s * g.normal();
    return dW;
}

inline std::vector<double> coarsen(const std::vector<double>& dW, int factor) {
    std::vector<double> out(dW.size() / size_t(factor), 0.0);
    for (size_t i = 0; i < out.size(); ++i)
        for (int k = 0; k < factor; ++k) out[i] += dW[i * size_t(factor) + size_t(k)];
    return out;
}

}  // namespace np::ex07
