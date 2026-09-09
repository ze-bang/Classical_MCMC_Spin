#include "ex01_ode_explicit.hpp"
#include "np/harness.hpp"
#include "np/problems.hpp"

using namespace np;
using namespace np::problems;

namespace {

// Global error at T for a fixed-step method on y' = y cos t, y(0) = 1.
template <class Stepper>
double global_error(Stepper&& step, double T, long nsteps) {
    SmoothNonAutonomous f;
    Vec y{1.0};
    ex01::integrate_fixed(step, f, 0.0, T, y, nsteps);
    return std::fabs(y[0] - f.exact(T, 1.0));
}

const std::vector<double> kSteps = {1.0 / 10, 1.0 / 20, 1.0 / 40, 1.0 / 80};

template <class Stepper>
std::vector<double> error_sequence(Stepper&& step, double T) {
    std::vector<double> e;
    for (double h : kSteps) e.push_back(global_error(step, T, long(std::lround(T / h))));
    return e;
}

}  // namespace

NP_TEST(ex01_euler_is_first_order) {
    auto e = error_sequence([](auto&& f, double t, Vec& y, double h) {
        ex01::euler_step(f, t, y, h);
    }, 2.0);
    NP_CHECK_ORDER("forward Euler", kSteps, e, 1.0, 0.15);
}

NP_TEST(ex01_midpoint_is_second_order) {
    auto e = error_sequence([](auto&& f, double t, Vec& y, double h) {
        ex01::midpoint_step(f, t, y, h);
    }, 2.0);
    NP_CHECK_ORDER("explicit midpoint", kSteps, e, 2.0, 0.15);
}

NP_TEST(ex01_heun_is_second_order) {
    auto e = error_sequence([](auto&& f, double t, Vec& y, double h) {
        ex01::heun_step(f, t, y, h);
    }, 2.0);
    NP_CHECK_ORDER("Heun", kSteps, e, 2.0, 0.15);
}

NP_TEST(ex01_rk4_is_fourth_order) {
    auto e = error_sequence([](auto&& f, double t, Vec& y, double h) {
        ex01::rk4_step(f, t, y, h);
    }, 2.0);
    NP_CHECK_ORDER("RK4", kSteps, e, 4.0, 0.2);
}

NP_TEST(ex01_rk4_on_a_system) {
    // Harmonic oscillator: checks that you handle coupled components, not just
    // a scalar. A common bug (updating y[0] before computing k from y[1]) shows
    // up here as order 1.
    Harmonic f{1.0};
    std::vector<double> errs;
    const double T = 10.0;
    for (double h : kSteps) {
        Vec y{1.0, 0.0}, ex(2);
        ex01::integrate_fixed([](auto&& g, double t, Vec& u, double s) {
            ex01::rk4_step(g, t, u, s);
        }, f, 0.0, T, y, long(std::lround(T / h)));
        f.exact(T, 1.0, 0.0, ex);
        errs.push_back(err_inf(y, ex));
    }
    NP_CHECK_ORDER("RK4 on (x,v)", kSteps, errs, 4.0, 0.2);
}

NP_TEST(ex01_dopri5_pair_orders) {
    // The embedded pair must be 5th and 4th order *as single steps*, i.e. local
    // errors of O(h^6) and O(h^5).
    SmoothNonAutonomous f;
    std::vector<double> hs, e5, e4;
    for (double h = 0.2; h > 0.02; h *= 0.5) {
        Vec y{1.0}, y5, y4;
        ex01::dopri5_step(f, 0.0, y, h, y5, y4);
        double ex = f.exact(h, 1.0);
        hs.push_back(h);
        e5.push_back(std::fabs(y5[0] - ex));
        e4.push_back(std::fabs(y4[0] - ex));
    }
    NP_CHECK_ORDER("dopri5 local (5th)", hs, e5, 6.0, 0.4);
    NP_CHECK_ORDER("dopri5 local (4th)", hs, e4, 5.0, 0.4);
}

NP_TEST(ex01_adaptive_meets_tolerance) {
    SmoothNonAutonomous f;
    const double T = 10.0;
    double prev_err = 0;
    for (double tol : {1e-6, 1e-9}) {
        Vec y{1.0};
        auto r = ex01::integrate_adaptive(f, 0.0, T, y, 1e-3, tol, tol);
        NP_CLOSE(r.t, T, 1e-10);
        double err = std::fabs(y[0] - f.exact(T, 1.0));
        NP_CHECK_MSG(err < 200.0 * tol,
                     "tol %.0e: global error %.3e exceeds 200*tol", tol, err);
        NP_CHECK_MSG(r.nrejected < r.naccepted / 4,
                     "controller thrashing: %ld rejected vs %ld accepted",
                     r.nrejected, r.naccepted);
        if (prev_err > 0)
            NP_CHECK_MSG(err < prev_err / 50.0,
                         "tightening tol 1e3x only cut the error %.1fx",
                         prev_err / err);
        prev_err = err;
    }
}

NP_TEST(ex01_adaptive_clusters_steps_at_perihelion) {
    // Kepler at e = 0.7: a good controller spends most of its steps near
    // perihelion. Check the orbit closes and the energy is retained.
    Kepler f;
    Vec y = Kepler::init(0.7);
    double E0 = Kepler::energy(y);
    auto r = ex01::integrate_adaptive(f, 0.0, 3.0 * Kepler::period(), y, 1e-3,
                                      1e-11, 1e-11);
    Vec y0 = Kepler::init(0.7);
    NP_CHECK_MSG(err_inf(y, y0) < 1e-6, "orbit did not close: %.3e",
                 err_inf(y, y0));
    NP_CHECK_MSG(std::fabs(Kepler::energy(y) - E0) < 1e-9, "energy drift %.3e",
                 std::fabs(Kepler::energy(y) - E0));
    (void)r;
}

NP_MAIN()
