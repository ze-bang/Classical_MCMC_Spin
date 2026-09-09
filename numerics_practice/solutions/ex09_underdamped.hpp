// Reference solution -- exercise 09.
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"
#include "np/rng.hpp"

namespace np::ex09 {

using np::Vec;

struct Thermostat {
    double gamma = 1.0;
    double kT = 1.0;
};

template <class Rng>
void ou_step(Vec& p, double m, double h, const Thermostat& th, Rng& g) {
    const double c = std::exp(-th.gamma * h);
    const double s = std::sqrt(m * th.kT * (1.0 - c * c));
    for (double& pi : p) pi = c * pi + s * g.normal();
}

inline void drift(Vec& q, const Vec& p, double m, double h) {
    for (size_t i = 0; i < q.size(); ++i) q[i] += h * p[i] / m;
}
inline void kick(Vec& p, const Vec& f, double h) {
    for (size_t i = 0; i < p.size(); ++i) p[i] += h * f[i];
}

template <class Force, class Rng>
void langevin_euler_step(Force&& force, Vec& q, Vec& p, Vec& f, double m,
                         double h, const Thermostat& th, Rng& g) {
    const double s = std::sqrt(2.0 * th.gamma * m * th.kT * h);
    for (size_t i = 0; i < p.size(); ++i)
        p[i] += h * f[i] - th.gamma * h * p[i] + s * g.normal();
    drift(q, p, m, h);
    force(q, f);
}

template <class Force, class Rng>
void baoab_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
                const Thermostat& th, Rng& g) {
    kick(p, f, 0.5 * h);
    drift(q, p, m, 0.5 * h);
    ou_step(p, m, h, th, g);
    drift(q, p, m, 0.5 * h);
    force(q, f);
    kick(p, f, 0.5 * h);
}

template <class Force, class Rng>
void aboba_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
                const Thermostat& th, Rng& g) {
    drift(q, p, m, 0.5 * h);
    force(q, f);
    kick(p, f, 0.5 * h);
    ou_step(p, m, h, th, g);
    kick(p, f, 0.5 * h);
    drift(q, p, m, 0.5 * h);
    force(q, f);
}

template <class Force, class Rng>
void obabo_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
                const Thermostat& th, Rng& g) {
    ou_step(p, m, 0.5 * h, th, g);
    kick(p, f, 0.5 * h);
    drift(q, p, m, h);
    force(q, f);
    kick(p, f, 0.5 * h);
    ou_step(p, m, 0.5 * h, th, g);
}

template <class Force, class Rng>
void gjf_step(Force&& force, Vec& q, Vec& p, Vec& f, double m, double h,
              const Thermostat& th, Rng& g) {
    const double gh2 = 0.5 * th.gamma * h;
    const double b = 1.0 / (1.0 + gh2);
    const double a = (1.0 - gh2) / (1.0 + gh2);
    const double sbeta = std::sqrt(2.0 * th.gamma * m * th.kT * h);
    static thread_local Vec fold;
    fold = f;
    static thread_local Vec beta;
    beta.assign(q.size(), 0.0);
    for (size_t i = 0; i < q.size(); ++i) beta[i] = sbeta * g.normal();
    for (size_t i = 0; i < q.size(); ++i)
        q[i] += b * h * p[i] / m + b * h * h * fold[i] / (2.0 * m) +
                b * h * beta[i] / (2.0 * m);
    force(q, f);
    for (size_t i = 0; i < p.size(); ++i)
        p[i] = a * p[i] + 0.5 * h * (a * fold[i] + f[i]) + b * beta[i];
}

}  // namespace np::ex09
