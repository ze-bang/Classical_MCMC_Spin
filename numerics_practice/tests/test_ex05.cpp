#include "ex05_hyperbolic.hpp"
#include "np/harness.hpp"

using namespace np;

namespace {

constexpr double kCFL = 0.4;

Vec smooth_ic(int n) {
    Vec u(n);
    for (int j = 0; j < n; ++j) u[j] = std::sin(2.0 * pi * (j + 0.5) / n);
    return u;
}
Vec square_ic(int n) {
    Vec u(n, 0.0);
    for (int j = 0; j < n; ++j) {
        double x = (j + 0.5) / n;
        u[j] = (x > 0.3 && x < 0.6) ? 1.0 : 0.0;
    }
    return u;
}
double l1(const Vec& a, const Vec& b, double dx) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += std::fabs(a[i] - b[i]);
    return s * dx;
}

// Advect a smooth profile exactly one period; return L1 error.
template <class Step>
double advect_one_period(Step&& step, int n) {
    double dx = 1.0 / n, a = 1.0, dt = kCFL * dx / a;
    long ns = long(std::lround(1.0 / dt));
    dt = 1.0 / double(ns);            // land exactly on t = 1
    Vec u = smooth_ic(n);
    for (long s = 0; s < ns; ++s) step(u, a, dx, dt);
    return l1(u, smooth_ic(n), dx);
}

}  // namespace

NP_TEST(ex05_minmod_semantics) {
    NP_CLOSE(ex05::minmod(2.0, 3.0), 2.0, 0);
    NP_CLOSE(ex05::minmod(3.0, 2.0), 2.0, 0);
    NP_CLOSE(ex05::minmod(-2.0, -3.0), -2.0, 0);
    NP_CLOSE(ex05::minmod(2.0, -3.0), 0.0, 0);
    NP_CLOSE(ex05::minmod(0.0, 5.0), 0.0, 0);
}

NP_TEST(ex05_advection_orders) {
    const std::vector<double> hs = {1.0 / 40, 1.0 / 80, 1.0 / 160, 1.0 / 320};
    std::vector<double> e_up, e_lw, e_mu;
    for (double h : hs) {
        int n = int(std::lround(1.0 / h));
        e_up.push_back(advect_one_period(
            [](Vec& u, double a, double dx, double dt) {
                ex05::advect_upwind_step(u, a, dx, dt);
            }, n));
        e_lw.push_back(advect_one_period(
            [](Vec& u, double a, double dx, double dt) {
                ex05::advect_lax_wendroff_step(u, a, dx, dt);
            }, n));
        e_mu.push_back(advect_one_period(
            [](Vec& u, double a, double dx, double dt) {
                ex05::advect_muscl_step(u, a, dx, dt);
            }, n));
    }
    NP_CHECK_ORDER("upwind (L1)", hs, e_up, 1.0, 0.15);
    NP_CHECK_ORDER("Lax-Wendroff (L1)", hs, e_lw, 2.0, 0.15);
    // minmod clips the two smooth extrema of a sine, so MUSCL loses a little
    // of its formal 2nd order in L1. Anything >= 1.6 means the limiter and the
    // half-step term are both right.
    double p = fit_order(hs, e_mu);
    std::printf("      MUSCL (L1) fitted order = %.3f\n", p);
    NP_CHECK_MSG(p > 1.6, "MUSCL order %.2f -- did you include the "
                          "(1 - a dt/dx) half-step term?", p);
    NP_CHECK_MSG(e_mu.back() < e_up.back() / 10.0,
                 "MUSCL should be far more accurate than upwind");
}

NP_TEST(ex05_square_wave_monotonicity) {
    const int n = 200;
    const double dx = 1.0 / n, a = 1.0, dt = kCFL * dx;
    long ns = long(std::lround(0.5 / dt));

    auto run = [&](auto&& step) {
        Vec u = square_ic(n);
        double tv0 = ex05::total_variation(u), tvmax = tv0;
        for (long s = 0; s < ns; ++s) {
            step(u, a, dx, dt);
            tvmax = std::max(tvmax, ex05::total_variation(u));
        }
        double lo = *std::min_element(u.begin(), u.end());
        double hi = *std::max_element(u.begin(), u.end());
        return std::array<double, 4>{lo, hi, tvmax, tv0};
    };

    auto up = run([](Vec& u, double a, double dx, double dt) {
        ex05::advect_upwind_step(u, a, dx, dt);
    });
    auto lw = run([](Vec& u, double a, double dx, double dt) {
        ex05::advect_lax_wendroff_step(u, a, dx, dt);
    });
    auto mu = run([](Vec& u, double a, double dx, double dt) {
        ex05::advect_muscl_step(u, a, dx, dt);
    });

    std::printf("      upwind: min %.4f max %.4f  TVmax/TV0 %.4f\n", up[0], up[1], up[2] / up[3]);
    std::printf("      Lax-W : min %.4f max %.4f  TVmax/TV0 %.4f\n", lw[0], lw[1], lw[2] / lw[3]);
    std::printf("      MUSCL : min %.4f max %.4f  TVmax/TV0 %.4f\n", mu[0], mu[1], mu[2] / mu[3]);

    NP_CHECK_MSG(up[2] <= up[3] * (1 + 1e-10), "upwind must be TVD");
    NP_CHECK_MSG(mu[2] <= mu[3] * (1 + 1e-10), "MUSCL must be TVD");
    NP_CHECK_MSG(mu[0] > -1e-10 && mu[1] < 1.0 + 1e-10,
                 "MUSCL overshoot: min %.3e max %.6f", mu[0], mu[1]);
    NP_CHECK_MSG(lw[0] < -0.05,
                 "Lax-Wendroff should undershoot at a jump (got min %.4f) -- "
                 "if it does not, you have added viscosity somewhere", lw[0]);
}

NP_TEST(ex05_burgers_conservation_and_shock_speed) {
    // Riemann data uL = 1, uR = 0 with the jump at x = 0.25. Exact shock speed
    // s = (uL + uR)/2 = 0.5, so at T = 0.4 the front sits at x = 0.45.
    const int n = 800;
    const double dx = 1.0 / n;
    Vec u(n);
    for (int j = 0; j < n; ++j) u[j] = ((j + 0.5) / n < 0.25) ? 1.0 : 0.0;

    const double mass0 = ex05::integral(u, dx);
    const double tv0 = ex05::total_variation(u);
    double t = 0;
    const double T = 0.4;
    double tvmax = tv0;
    while (t < T - 1e-14) {
        double umax = 0;
        for (double v : u) umax = std::max(umax, std::fabs(v));
        double dt = std::min(kCFL * dx / std::max(umax, 1e-12), T - t);
        ex05::burgers_rusanov_step(u, dx, dt);
        tvmax = std::max(tvmax, ex05::total_variation(u));
        t += dt;
    }
    // Find where the profile crosses 0.5 (the shock).
    double xs = -1;
    for (int j = 0; j + 1 < n; ++j)
        if (u[j] >= 0.5 && u[j + 1] < 0.5) { xs = (j + 0.5) / n; break; }

    std::printf("      shock at x = %.4f (exact 0.45), mass drift %.3e, "
                "TVmax/TV0 %.6f\n",
                xs, std::fabs(ex05::integral(u, dx) - mass0), tvmax / tv0);
    NP_CHECK_MSG(std::fabs(ex05::integral(u, dx) - mass0) < 1e-12,
                 "conservative form violated: mass drifted %.3e",
                 std::fabs(ex05::integral(u, dx) - mass0));
    NP_CHECK_MSG(std::fabs(xs - 0.45) < 3 * dx,
                 "shock at %.4f, expected 0.45 -- wrong shock speed means a "
                 "non-conservative update", xs);
    NP_CHECK_MSG(tvmax <= tv0 * (1 + 1e-10), "Rusanov must be TVD");
}

NP_TEST(ex05_burgers_forms_a_shock_from_smooth_data) {
    // u0 = 1 + 0.5 sin(2 pi x): characteristics cross at t = 1/(2 pi * 0.5).
    // Past that time the solution must stay bounded and monotone-total-
    // variation-decreasing, and mass must be conserved exactly.
    const int n = 400;
    const double dx = 1.0 / n;
    Vec u(n);
    for (int j = 0; j < n; ++j)
        u[j] = 1.0 + 0.5 * std::sin(2.0 * pi * (j + 0.5) / n);
    const double mass0 = ex05::integral(u, dx);
    double t = 0;
    while (t < 1.0) {
        double umax = 0;
        for (double v : u) umax = std::max(umax, std::fabs(v));
        double dt = std::min(kCFL * dx / umax, 1.0 - t);
        ex05::burgers_rusanov_step(u, dx, dt);
        t += dt;
    }
    NP_CHECK_MSG(std::fabs(ex05::integral(u, dx) - mass0) < 1e-12,
                 "mass not conserved past the shock");
    NP_CHECK_MSG(norm_inf(u) < 1.6, "solution grew past the shock: %.4f",
                 norm_inf(u));
}

NP_MAIN()
