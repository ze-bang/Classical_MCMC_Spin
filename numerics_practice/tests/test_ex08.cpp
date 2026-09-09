#include "ex08_overdamped.hpp"
#include "np/harness.hpp"
#include "np/problems.hpp"
#include "np/rng.hpp"

using namespace np;
using namespace np::problems;

namespace {
constexpr double kK = 1.0;         // harmonic stiffness
const ex08::Params kP{1.0, 1.0};   // gamma = kT = 1  ->  exact <x^2> = kT/k = 1
}  // namespace

NP_TEST(ex08_harmonic_invariant_measure_bias_is_exact_arithmetic) {
    // No Monte Carlo here: probe each scheme as an affine-Gaussian map and
    // solve the Lyapunov equation for its exact stationary variance.
    auto force = [](double x) { return -kK * x; };
    std::vector<double> dts, e_em;

    for (double dt : {0.2, 0.1, 0.05, 0.025}) {
        auto em = [&](Vec& z, auto& g) {
            ex08::bd_euler_step(force, z[0], dt, kP, g);
        };
        double var = stationary_covariance(em, 1)(0, 0);
        // Analytically: var = (kT/k) / (1 - k dt / (2 gamma)).
        dts.push_back(dt);
        e_em.push_back(std::fabs(var - 1.0));
        std::printf("      dt = %.3f : Euler-Maruyama <x^2> = %.6f "
                    "(exact 1, closed form %.6f)\n",
                    dt, var, 1.0 / (1.0 - kK * dt / (2.0 * kP.gamma)));
    }
    NP_CHECK_ORDER("Euler-Maruyama invariant measure", dts, e_em, 1.0, 0.1);

    // The exact OU propagator is exact at ANY step size -- including dt = 10,
    // where it is simply drawing an independent Boltzmann sample.
    for (double dt : {0.1, 1.0, 10.0}) {
        auto ou = [&](Vec& z, auto& g) {
            ex08::ou_exact_step(z[0], dt, kK, kP, g);
        };
        double var = stationary_covariance(ou, 1)(0, 0);
        NP_CHECK_MSG(std::fabs(var - 1.0) < 1e-12,
                     "exact OU at dt=%.1f gave <x^2> = %.12f", dt, var);
    }

    // Leimkuhler-Matthews: the chain is 2-D (x, xi_prev) because consecutive
    // steps share a random number. For a QUADRATIC potential its bias is not
    // just O(dt^2) -- it vanishes identically.
    for (double dt : {0.2, 0.05}) {
        auto lm = [&](Vec& z, auto& g) {
            ex08::LmState st{z[0], z[1]};
            ex08::bd_leimkuhler_matthews_step(force, st, dt, kP, g);
            z[0] = st.x;
            z[1] = st.xi_prev;
        };
        double var = stationary_covariance(lm, 2)(0, 0);
        std::printf("      dt = %.3f : Leimkuhler-Matthews <x^2> = %.12f\n", dt,
                    var);
        NP_CHECK_MSG(std::fabs(var - 1.0) < 1e-10,
                     "LM should be exact for a harmonic well, got %.12f", var);
    }
}

NP_TEST(ex08_euler_is_biased_and_can_be_transient) {
    // Quartic double well: U = h (x^2/a^2 - 1)^2, barrier 2 kT.
    DoubleWell U{2.0, 1.0};
    auto energy = [&](double x) { return U.energy(x); };
    auto force = [&](double x) { return U.force(x); };
    const double ref =
        ex08::boltzmann_average(energy, [](double x) { return x * x; }, -4, 4,
                                kP.kT);
    std::printf("      reference <x^2> = %.6f\n", ref);

    const long N = 20'000'000;
    std::vector<double> dts, errs;
    for (double dt : {0.025, 0.0125, 0.00625}) {
        Rng g(1);
        double x = 1.0;
        Stats s;
        for (long i = 0; i < N; ++i) {
            ex08::bd_euler_step(force, x, dt, kP, g);
            s.add(x * x);
        }
        std::printf("      dt = %.5f : <x^2> = %.6f  bias %+.5f\n", dt, s.mean,
                    s.mean - ref);
        dts.push_back(dt);
        errs.push_back(std::fabs(s.mean - ref));
    }
    NP_CHECK_ORDER("Euler-Maruyama bias (double well)", dts, errs, 1.0, 0.25);

    // dt = 0.05 with a cubic force: the chain is transient and escapes.
    Rng g(1);
    double x = 1.0;
    bool blew_up = false;
    for (long i = 0; i < 20'000'000 && !blew_up; ++i) {
        ex08::bd_euler_step(force, x, 0.05, kP, g);
        blew_up = !std::isfinite(x) || std::fabs(x) > 1e6;
    }
    NP_CHECK_MSG(blew_up,
                 "unadjusted Euler-Maruyama with a superlinear drift should be "
                 "transient at dt = 0.05 (Roberts & Tweedie); it survived");
}

NP_TEST(ex08_leimkuhler_matthews_kills_most_of_the_bias) {
    DoubleWell U{2.0, 1.0};
    auto energy = [&](double x) { return U.energy(x); };
    auto force = [&](double x) { return U.force(x); };
    const double ref = ex08::boltzmann_average(
        energy, [](double x) { return x * x; }, -4, 4, kP.kT);

    const long N = 20'000'000;
    const double dt = 0.025;

    Rng g1(1);
    double x = 1.0;
    Stats se;
    for (long i = 0; i < N; ++i) {
        ex08::bd_euler_step(force, x, dt, kP, g1);
        se.add(x * x);
    }
    Rng g2(1);
    ex08::LmState st{1.0, g2.normal()};
    Stats sl;
    for (long i = 0; i < N; ++i) {
        ex08::bd_leimkuhler_matthews_step(force, st, dt, kP, g2);
        sl.add(st.x * st.x);
    }
    double be = std::fabs(se.mean - ref), bl = std::fabs(sl.mean - ref);
    std::printf("      dt = %.4f : Euler bias %+.5f, Leimkuhler-Matthews %+.5f "
                "(%.0fx smaller)\n", dt, se.mean - ref, sl.mean - ref, be / bl);
    NP_CHECK_MSG(bl < be / 20.0,
                 "LM bias %.3e is not much better than Euler's %.3e", bl, be);
    NP_CHECK_MSG(bl < 2e-3, "LM bias %.3e too large", bl);
}

NP_TEST(ex08_mala_is_unbiased_where_euler_diverges) {
    DoubleWell U{2.0, 1.0};
    auto energy = [&](double x) { return U.energy(x); };
    auto force = [&](double x) { return U.force(x); };
    const double ref = ex08::boltzmann_average(
        energy, [](double x) { return x * x; }, -4, 4, kP.kT);
    const double ref4 = ex08::boltzmann_average(
        energy, [](double x) { return x * x * x * x; }, -4, 4, kP.kT);

    Rng g(1);
    double x = 1.0;
    long acc = 0;
    const long N = 4'000'000;
    Stats s2, s4;
    for (long i = 0; i < N; ++i) {
        acc += ex08::mala_step(energy, force, x, 0.05, kP, g) ? 1 : 0;
        s2.add(x * x);
        s4.add(x * x * x * x);
    }
    std::printf("      MALA at dt = 0.05: acceptance %.3f, <x^2> = %.6f "
                "(exact %.6f), <x^4> = %.6f (exact %.6f)\n",
                double(acc) / N, s2.mean, ref, s4.mean, ref4);
    NP_CHECK_MSG(double(acc) / N > 0.5 && double(acc) / N < 0.999,
                 "acceptance %.3f is implausible -- check the q ratio",
                 double(acc) / N);
    NP_CHECK_MSG(std::fabs(s2.mean - ref) < 3e-3, "MALA <x^2> bias %+.4f",
                 s2.mean - ref);
    NP_CHECK_MSG(std::fabs(s4.mean - ref4) < 2e-2, "MALA <x^4> bias %+.4f",
                 s4.mean - ref4);
}

NP_MAIN()
