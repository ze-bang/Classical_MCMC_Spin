// Reference solution -- exercise 08.
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex08 {

struct Params {
    double gamma = 1.0;
    double kT = 1.0;
};

template <class Force, class Rng>
void bd_euler_step(Force&& F, double& x, double dt, const Params& p, Rng& g) {
    const double s = std::sqrt(2.0 * p.kT * dt / p.gamma);
    x += F(x) / p.gamma * dt + s * g.normal();
}

template <class Rng>
void ou_exact_step(double& x, double dt, double k, const Params& p, Rng& g) {
    const double c = std::exp(-k * dt / p.gamma);
    const double s = std::sqrt((p.kT / k) * (1.0 - c * c));
    x = c * x + s * g.normal();
}

struct LmState {
    double x = 0.0;
    double xi_prev = 0.0;
};

template <class Force, class Rng>
void bd_leimkuhler_matthews_step(Force&& F, LmState& st, double dt,
                                 const Params& p, Rng& g) {
    const double s = std::sqrt(2.0 * p.kT * dt / p.gamma);
    const double xi = g.normal();
    st.x += F(st.x) / p.gamma * dt + s * 0.5 * (st.xi_prev + xi);
    st.xi_prev = xi;
}

template <class U, class Force, class Rng>
bool mala_step(U&& energy, Force&& F, double& x, double dt, const Params& p,
               Rng& g) {
    const double D = p.kT / p.gamma;            // diffusion constant
    const double s2 = 2.0 * D * dt;
    const double mu_fwd = x + (F(x) / p.gamma) * dt;
    const double y = mu_fwd + std::sqrt(s2) * g.normal();
    const double mu_bwd = y + (F(y) / p.gamma) * dt;

    const double logq_fwd = -(y - mu_fwd) * (y - mu_fwd) / (2.0 * s2);
    const double logq_bwd = -(x - mu_bwd) * (x - mu_bwd) / (2.0 * s2);
    const double log_alpha =
        -(energy(y) - energy(x)) / p.kT + logq_bwd - logq_fwd;

    if (log_alpha >= 0.0 || std::log(g.uniform()) < log_alpha) {
        x = y;
        return true;
    }
    return false;
}

// ---- provided helpers -------------------------------------------------------
// <f> under the Boltzmann density exp(-U/kT) on [xmin, xmax] by Simpson's rule.
template <class U, class F>
double boltzmann_average(U&& energy, F&& f, double xmin, double xmax, double kT,
                         int n = 200000) {
    double num = 0, den = 0;
    const double h = (xmax - xmin) / n;
    for (int i = 0; i <= n; ++i) {
        double x = xmin + i * h;
        double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        double p = std::exp(-energy(x) / kT);
        num += w * p * f(x);
        den += w * p;
    }
    return num / den;
}

}  // namespace np::ex08
