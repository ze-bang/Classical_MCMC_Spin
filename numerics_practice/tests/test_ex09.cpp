#include "ex02_symplectic.hpp"
#include "ex08_overdamped.hpp"
#include "ex09_underdamped.hpp"
#include "np/harness.hpp"
#include "np/problems.hpp"
#include "np/rng.hpp"

using namespace np;
using namespace np::problems;

namespace {
constexpr double kM = 1.0, kOmega = 1.0, kKT = 1.0;
auto harmonic_force = [](const Vec& q, Vec& f) { f[0] = -kOmega * kOmega * kM * q[0]; };

// Exact stationary (<q^2>, <p^2>) of a scheme on the harmonic oscillator.
template <class Scheme>
std::pair<double, double> harmonic_moments(Scheme&& scheme, double h,
                                           double gamma) {
    ex09::Thermostat th{gamma, kKT};
    auto step = [&](Vec& z, auto& g) {
        Vec q{z[0]}, p{z[1]}, f(1);
        harmonic_force(q, f);
        scheme(harmonic_force, q, p, f, kM, h, th, g);
        z[0] = q[0];
        z[1] = p[0];
    };
    Mat S = stationary_covariance(step, 2);
    return {S(0, 0), S(1, 1)};
}

#define SCHEME(name)                                                       \
    [](auto&& F, Vec& q, Vec& p, Vec& f, double m, double h,               \
       const ex09::Thermostat& t, auto& g) {                               \
        ex09::name(F, q, p, f, m, h, t, g);                                \
    }

}  // namespace

NP_TEST(ex09_o_step_is_an_exact_ou_propagator) {
    // On its own, O must produce exactly the Maxwell-Boltzmann momentum
    // distribution <p^2> = m kT at any step size, and decay as exp(-gamma h).
    for (double h : {0.01, 0.5, 5.0}) {
        ex09::Thermostat th{2.0, kKT};
        auto step = [&](Vec& z, auto& g) {
            Vec p{z[0]};
            ex09::ou_step(p, kM, h, th, g);
            z[0] = p[0];
        };
        auto pr = probe_linear_sde(step, 1);
        double var = dlyap(pr.M, pr.LL)(0, 0);
        NP_CLOSE(pr.M(0, 0), std::exp(-th.gamma * h), 1e-12);
        NP_CHECK_MSG(std::fabs(var - kM * kKT) < 1e-12,
                     "h = %.2f: <p^2> = %.12f, expected m kT = %.1f", h, var,
                     kM * kKT);
    }
}

NP_TEST(ex09_baoab_reduces_to_velocity_verlet) {
    // gamma = 0 removes the O step entirely, so BAOAB must BE velocity Verlet.
    // If it is not, your A/B half-steps are in the wrong order.
    ex09::Thermostat th{0.0, 0.0};
    Rng g(1);
    Vec q1{0.8}, p1{0.3}, f1(1), q2{0.8}, p2{0.3}, f2(1);
    harmonic_force(q1, f1);
    harmonic_force(q2, f2);
    for (int i = 0; i < 200; ++i) {
        ex09::baoab_step(harmonic_force, q1, p1, f1, kM, 0.1, th, g);
        ex02::velocity_verlet_step(harmonic_force, q2, p2, f2, kM, 0.1);
    }
    NP_CLOSE(q1[0], q2[0], 1e-12);
    NP_CLOSE(p1[0], p2[0], 1e-12);
}

NP_TEST(ex09_exact_sampling_bias_on_a_harmonic_well) {
    // Zero Monte Carlo: the exact stationary moments of each scheme.
    std::printf("      exact answers: <q^2> = kT/(m w^2) = 1, <p^2> = m kT = 1\n");
    for (double h : {0.4, 0.2, 0.1}) {
        auto e = harmonic_moments(SCHEME(langevin_euler_step), h, 1.0);
        auto o = harmonic_moments(SCHEME(obabo_step), h, 1.0);
        auto ab = harmonic_moments(SCHEME(aboba_step), h, 1.0);
        auto ba = harmonic_moments(SCHEME(baoab_step), h, 1.0);
        auto gj = harmonic_moments(SCHEME(gjf_step), h, 1.0);
        std::printf("      h=%.2f  euler(%.6f,%.6f) obabo(%.6f,%.6f) "
                    "aboba(%.6f,%.6f) baoab(%.6f,%.6f) gjf(%.6f,%.6f)\n",
                    h, e.first, e.second, o.first, o.second, ab.first,
                    ab.second, ba.first, ba.second, gj.first, gj.second);

        const double q2_obabo = 1.0 / (1.0 - 0.25 * h * h * kOmega * kOmega);
        const double p2_baoab = 1.0 - 0.25 * h * h * kOmega * kOmega;

        // OBABO: momenta exact, positions inflated by exactly 1/(1 - h^2w^2/4).
        NP_CLOSE(o.second, 1.0, 1e-12);
        NP_CLOSE(o.first, q2_obabo, 1e-12);
        // BAOAB, ABOBA, GJF: positions EXACT at any step size.
        NP_CHECK_MSG(std::fabs(ba.first - 1.0) < 1e-10,
                     "BAOAB <q^2> = %.12f, must be exact for a quadratic "
                     "potential", ba.first);
        NP_CHECK_MSG(std::fabs(ab.first - 1.0) < 1e-10, "ABOBA <q^2> = %.12f",
                     ab.first);
        NP_CHECK_MSG(std::fabs(gj.first - 1.0) < 1e-10, "GJF <q^2> = %.12f",
                     gj.first);
        // ... and their kinetic temperature is low by exactly h^2 w^2 / 4.
        NP_CLOSE(ba.second, p2_baoab, 1e-10);
        NP_CLOSE(gj.second, p2_baoab, 1e-10);
        // Naive Euler is wrong in both.
        NP_CHECK_MSG(std::fabs(e.second - 1.0) > 0.02,
                     "naive Euler should have a visible kinetic-temperature "
                     "error at h = %.2f", h);
    }
}

NP_TEST(ex09_naive_euler_has_a_friction_stability_limit) {
    // The splittings treat friction with an exact exponential and are stable
    // for any gamma*h. Linearised friction is stable only for gamma*h < 2.
    auto e_ok = harmonic_moments(SCHEME(langevin_euler_step), 0.1, 10.0);
    auto e_bad = harmonic_moments(SCHEME(langevin_euler_step), 0.4, 10.0);
    auto b_bad = harmonic_moments(SCHEME(baoab_step), 0.4, 10.0);
    std::printf("      gamma*h = 1: euler <p^2> = %.4f;  gamma*h = 4: euler "
                "<p^2> = %.3e, baoab %.6f\n",
                e_ok.second, e_bad.second, b_bad.second);
    NP_CHECK_MSG(std::isfinite(e_ok.second), "euler should survive gamma*h = 1");
    NP_CHECK_MSG(!std::isfinite(e_bad.second) || e_bad.second > 1e3,
                 "euler should be unstable at gamma*h = 4, got %.4f",
                 e_bad.second);
    NP_CHECK_MSG(std::fabs(b_bad.first - 1.0) < 1e-10,
                 "BAOAB must be unconditionally stable in gamma*h");
}

NP_TEST(ex09_configurational_accuracy_survives_anharmonicity) {
    // The harmonic case flatters BAOAB/ABOBA/GJF equally. On a quartic double
    // well they separate: BAOAB and GJF keep a tiny configurational bias while
    // ABOBA and OBABO do not.
    DoubleWell U{2.0, 1.0};
    auto energy = [&](double x) { return U.energy(x); };
    auto force = [&](const Vec& q, Vec& f) { f[0] = U.force(q[0]); };
    const double ref = ex08::boltzmann_average(
        energy, [](double x) { return x * x; }, -4, 4, kKT);

    const double h = 0.2, gamma = 10.0;
    const long N = 10'000'000;
    ex09::Thermostat th{gamma, kKT};

    auto measure = [&](auto&& scheme) {
        Rng g(5);
        Vec q{1.0}, p{0.0}, f(1);
        force(q, f);
        Stats sq, sp;
        for (long i = 0; i < N; ++i) {
            scheme(force, q, p, f, kM, h, th, g);
            sq.add(q[0] * q[0]);
            sp.add(p[0] * p[0]);
        }
        return std::pair<double, double>{sq.mean, sp.mean};
    };

    auto o = measure(SCHEME(obabo_step));
    auto ab = measure(SCHEME(aboba_step));
    auto ba = measure(SCHEME(baoab_step));
    auto gj = measure(SCHEME(gjf_step));
    std::printf("      double well, h = %.2f, gamma = %.0f (exact <q^2> = %.6f)\n",
                h, gamma, ref);
    std::printf("        OBABO <q^2> bias %+.5f   <p^2> bias %+.5f\n",
                o.first - ref, o.second - kM * kKT);
    std::printf("        ABOBA <q^2> bias %+.5f   <p^2> bias %+.5f\n",
                ab.first - ref, ab.second - kM * kKT);
    std::printf("        BAOAB <q^2> bias %+.5f   <p^2> bias %+.5f\n",
                ba.first - ref, ba.second - kM * kKT);
    std::printf("        GJF   <q^2> bias %+.5f   <p^2> bias %+.5f\n",
                gj.first - ref, gj.second - kM * kKT);

    double b_ba = std::fabs(ba.first - ref), b_gj = std::fabs(gj.first - ref);
    double b_o = std::fabs(o.first - ref), b_ab = std::fabs(ab.first - ref);
    NP_CHECK_MSG(b_ba < 2e-3, "BAOAB configurational bias %.4f too large", b_ba);
    NP_CHECK_MSG(b_gj < 2e-3, "GJF configurational bias %.4f too large", b_gj);
    NP_CHECK_MSG(b_o > 5e-3, "OBABO should show a clear O(h^2) bias here");
    NP_CHECK_MSG(b_ba < b_o / 5.0 && b_ba < b_ab / 5.0,
                 "BAOAB should beat OBABO/ABOBA on positions by >5x");
    // ... and the mirror image: OBABO is the one that gets momenta right.
    NP_CHECK_MSG(std::fabs(o.second - kM * kKT) < 5e-3,
                 "OBABO kinetic temperature bias %+.4f", o.second - kM * kKT);
}

NP_MAIN()
