#include "ex01_ode_explicit.hpp"
#include "ex02_symplectic.hpp"
#include "np/harness.hpp"
#include "np/problems.hpp"
#include "np/rng.hpp"

using namespace np;
using namespace np::problems;

namespace {

// Harmonic well with omega = 1, m = 1.
auto harmonic_force = [](const Vec& q, Vec& f) { f[0] = -q[0]; };
inline double harmonic_energy(const Vec& q, const Vec& p) {
    return 0.5 * p[0] * p[0] + 0.5 * q[0] * q[0];
}

// Kepler force in 2-D, GM = 1.
auto kepler_force = [](const Vec& q, Vec& f) {
    double r2 = q[0] * q[0] + q[1] * q[1];
    double r3 = r2 * std::sqrt(r2);
    f[0] = -q[0] / r3;
    f[1] = -q[1] / r3;
};

const std::vector<double> kSteps = {1.0 / 10, 1.0 / 20, 1.0 / 40, 1.0 / 80};

// Global error of a (q,p) stepper on the harmonic oscillator at T.
template <class Step>
std::vector<double> harmonic_orders(Step&& step, double T) {
    std::vector<double> errs;
    for (double h : kSteps) {
        Vec q{1.0}, p{0.0};
        long n = long(std::lround(T / h));
        for (long i = 0; i < n; ++i) step(q, p, h);
        double eq = std::fabs(q[0] - std::cos(T)), ep = std::fabs(p[0] + std::sin(T));
        errs.push_back(std::max(eq, ep));
    }
    return errs;
}

}  // namespace

NP_TEST(ex02_orders) {
    auto e1 = harmonic_orders([](Vec& q, Vec& p, double h) {
        ex02::symplectic_euler_step(harmonic_force, q, p, 1.0, h);
    }, 2.0);
    NP_CHECK_ORDER("symplectic Euler", kSteps, e1, 1.0, 0.15);

    auto e2 = harmonic_orders([](Vec& q, Vec& p, double h) {
        Vec f(1);
        harmonic_force(q, f);
        ex02::velocity_verlet_step(harmonic_force, q, p, f, 1.0, h);
    }, 2.0);
    NP_CHECK_ORDER("velocity Verlet", kSteps, e2, 2.0, 0.15);

    auto e3 = harmonic_orders([](Vec& q, Vec& p, double h) {
        ex02::leapfrog_step(harmonic_force, q, p, 1.0, h);
    }, 2.0);
    NP_CHECK_ORDER("leapfrog", kSteps, e3, 2.0, 0.15);

    auto e4 = harmonic_orders([](Vec& q, Vec& p, double h) {
        ex02::yoshida4_step(harmonic_force, q, p, 1.0, h);
    }, 2.0);
    NP_CHECK_ORDER("Yoshida 4", kSteps, e4, 4.0, 0.2);
}

NP_TEST(ex02_one_force_call_per_step) {
    // Velocity Verlet must evaluate the force exactly once per step.
    long calls = 0;
    auto counting = [&](const Vec& q, Vec& f) { ++calls; harmonic_force(q, f); };
    Vec q{1.0}, p{0.0}, f(1);
    counting(q, f);
    calls = 0;
    for (int i = 0; i < 100; ++i)
        ex02::velocity_verlet_step(counting, q, p, f, 1.0, 0.05);
    NP_CHECK_MSG(calls == 100,
                 "velocity Verlet used %ld force evaluations for 100 steps "
                 "(should be 100 -- reuse the cached force)", calls);
}

NP_TEST(ex02_map_is_area_preserving) {
    // A symplectic map on one degree of freedom has |det J| = 1 exactly.
    // Explicit Euler does not: det = 1 + h^2 w^2. This is the definition test.
    const double h = 0.1, eps = 1e-6;
    auto jac_det = [&](auto&& step) {
        double base[2] = {0.7, -0.3};
        double J[2][2];
        for (int j = 0; j < 2; ++j) {
            Vec qp{base[0]}, pp{base[1]}, qm{base[0]}, pm{base[1]};
            (j == 0 ? qp[0] : pp[0]) += eps;
            (j == 0 ? qm[0] : pm[0]) -= eps;
            step(qp, pp, h);
            step(qm, pm, h);
            J[0][j] = (qp[0] - qm[0]) / (2 * eps);
            J[1][j] = (pp[0] - pm[0]) / (2 * eps);
        }
        return J[0][0] * J[1][1] - J[0][1] * J[1][0];
    };

    double d_se = jac_det([](Vec& q, Vec& p, double s) {
        ex02::symplectic_euler_step(harmonic_force, q, p, 1.0, s);
    });
    double d_vv = jac_det([](Vec& q, Vec& p, double s) {
        Vec f(1);
        harmonic_force(q, f);
        ex02::velocity_verlet_step(harmonic_force, q, p, f, 1.0, s);
    });
    double d_y4 = jac_det([](Vec& q, Vec& p, double s) {
        ex02::yoshida4_step(harmonic_force, q, p, 1.0, s);
    });
    NP_CLOSE(d_se, 1.0, 1e-8);
    NP_CLOSE(d_vv, 1.0, 1e-8);
    NP_CLOSE(d_y4, 1.0, 1e-8);
}

NP_TEST(ex02_verlet_is_time_reversible) {
    // Step forward N times, then backward with -h: exact return to the start.
    Vec q{0.9}, p{0.4}, f(1);
    const Vec q0 = q, p0 = p;
    harmonic_force(q, f);
    for (int i = 0; i < 500; ++i)
        ex02::velocity_verlet_step(harmonic_force, q, p, f, 1.0, 0.07);
    harmonic_force(q, f);
    for (int i = 0; i < 500; ++i)
        ex02::velocity_verlet_step(harmonic_force, q, p, f, 1.0, -0.07);
    NP_CLOSE(q[0], q0[0], 1e-11);
    NP_CLOSE(p[0], p0[0], 1e-11);
}

NP_TEST(ex02_energy_error_is_bounded_not_secular) {
    // The headline property. Run 200k steps; compare the *mean* energy over the
    // first and second halves. Verlet: no drift. RK4: measurable secular decay,
    // even though its per-step error is far smaller.
    const double h = 0.1;
    const long N = 200000;

    Stats first, second;
    {
        Vec q{1.0}, p{0.0}, f(1);
        harmonic_force(q, f);
        for (long i = 0; i < N; ++i) {
            ex02::velocity_verlet_step(harmonic_force, q, p, f, 1.0, h);
            (i < N / 2 ? first : second).add(harmonic_energy(q, p));
        }
    }
    double verlet_drift = std::fabs(second.mean - first.mean);

    Stats rfirst, rsecond;
    {
        Harmonic sys{1.0};
        Vec y{1.0, 0.0};
        for (long i = 0; i < N; ++i) {
            ex01::rk4_step(sys, double(i) * h, y, h);
            (i < N / 2 ? rfirst : rsecond).add(sys.energy(y));
        }
    }
    double rk4_drift = std::fabs(rsecond.mean - rfirst.mean);

    std::printf("      Verlet mean-energy drift = %.3e   RK4 = %.3e\n",
                verlet_drift, rk4_drift);
    // The 1e-6 floor is the window-boundary artefact of averaging an
    // oscillatory E(t) over a non-integer number of periods, not real drift.
    NP_CHECK_MSG(verlet_drift < 1e-6,
                 "Verlet energy drifted by %.3e -- symplecticity is broken",
                 verlet_drift);
    NP_CHECK_MSG(rk4_drift > 1e2 * verlet_drift,
                 "sanity: RK4 should drift much more (%.3e vs %.3e)",
                 rk4_drift, verlet_drift);
}

NP_TEST(ex02_verlet_conserves_angular_momentum_exactly) {
    // For a central force, leapfrog/Verlet conserves L = q x p to roundoff --
    // not to O(h^2). Energy is only O(h^2)-conserved; L is exact.
    Vec q{Kepler::init(0.5)[0], Kepler::init(0.5)[1]};
    Vec p{Kepler::init(0.5)[2], Kepler::init(0.5)[3]};
    Vec f(2);
    kepler_force(q, f);
    auto L = [&] { return q[0] * p[1] - q[1] * p[0]; };
    const double L0 = L();
    double maxdL = 0, maxdE = 0;
    auto E = [&] {
        return 0.5 * (p[0] * p[0] + p[1] * p[1]) - 1.0 / std::hypot(q[0], q[1]);
    };
    const double E0 = E();
    for (long i = 0; i < 200000; ++i) {
        ex02::velocity_verlet_step(kepler_force, q, p, f, 1.0, 1e-3);
        maxdL = std::max(maxdL, std::fabs(L() - L0));
        maxdE = std::max(maxdE, std::fabs(E() - E0));
    }
    std::printf("      max |dL| = %.3e   max |dE| = %.3e\n", maxdL, maxdE);
    NP_CHECK_MSG(maxdL < 1e-12, "angular momentum drifted by %.3e", maxdL);
    NP_CHECK_MSG(maxdE < 1e-4, "energy error %.3e too large for h=1e-3", maxdE);
}

NP_MAIN()
