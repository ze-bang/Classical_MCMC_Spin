#include "ex04_parabolic.hpp"
#include "np/harness.hpp"

using namespace np;

namespace {
constexpr double kAlpha = 0.7;

Vec sine_ic(int n) {
    Vec u(n, 0.0);
    double dx = 1.0 / (n - 1);
    for (int j = 0; j < n; ++j) u[j] = std::sin(pi * j * dx);
    return u;
}
// Exact solution of the SEMI-discrete system (space error removed).
Vec semi_discrete(int n, double t) {
    Vec u = sine_ic(n);
    double dx = 1.0 / (n - 1);
    double lam = kAlpha * ex04::discrete_laplacian_eigenvalue(1.0, dx);
    for (auto& v : u) v *= std::exp(lam * t);
    return u;
}
}  // namespace

NP_TEST(ex04_laplacian_is_second_order_accurate) {
    std::vector<double> hs, errs;
    for (int n : {21, 41, 81, 161}) {
        double dx = 1.0 / (n - 1);
        Vec u = sine_ic(n), lap;
        ex04::laplacian_1d(u, dx, lap);
        double e = 0;
        for (int j = 1; j + 1 < n; ++j)
            e = std::max(e, std::fabs(lap[j] + pi * pi * u[j]));
        hs.push_back(dx);
        errs.push_back(e);
    }
    NP_CHECK_ORDER("d2/dx2 stencil", hs, errs, 2.0, 0.1);
}

NP_TEST(ex04_ftcs_stability_threshold) {
    // r = alpha dt / dx^2. Stable iff r <= 1/2. Seed with a sawtooth so the
    // unstable Nyquist mode is actually excited.
    const int n = 41;
    const double dx = 1.0 / (n - 1);
    auto run = [&](double r) {
        Vec u(n, 0.0);
        for (int j = 1; j + 1 < n; ++j) u[j] = (j % 2 ? 1.0 : -1.0) * 1e-3;
        double dt = r * dx * dx / kAlpha;
        for (int s = 0; s < 400; ++s) ex04::heat_ftcs_step(u, kAlpha, dx, dt);
        return norm_inf(u);
    };
    double stable = run(0.49), unstable = run(0.51);
    std::printf("      |u| after 400 steps: r=0.49 -> %.3e,  r=0.51 -> %.3e\n",
                stable, unstable);
    NP_CHECK_MSG(stable < 1e-6, "r = 0.49 must decay, got %.3e", stable);
    NP_CHECK_MSG(!std::isfinite(unstable) || unstable > 1e3,
                 "r = 0.51 must blow up, got %.3e", unstable);
}

NP_TEST(ex04_time_orders_on_a_fixed_grid) {
    const int n = 65;
    const double dx = 1.0 / (n - 1), T = 0.02;
    std::vector<double> dts, e_ftcs, e_cn;
    // Keep every dt inside the FTCS stability limit so we measure accuracy,
    // not blow-up: r_max = alpha*dt/dx^2 <= 0.5  ->  dt <= 1.7e-4 here.
    for (double dt : {1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5}) {
        long ns = long(std::lround(T / dt));
        Vec uf = sine_ic(n), uc = sine_ic(n);
        for (long s = 0; s < ns; ++s) ex04::heat_ftcs_step(uf, kAlpha, dx, dt);
        for (long s = 0; s < ns; ++s)
            ex04::heat_crank_nicolson_step(uc, kAlpha, dx, dt);
        Vec ref = semi_discrete(n, T);
        dts.push_back(dt);
        e_ftcs.push_back(err_inf(uf, ref));
        e_cn.push_back(err_inf(uc, ref));
    }
    NP_CHECK_ORDER("FTCS in time", dts, e_ftcs, 1.0, 0.15);
    NP_CHECK_ORDER("Crank-Nicolson in time", dts, e_cn, 2.0, 0.2);
}

NP_TEST(ex04_space_order_against_the_pde) {
    // Now refine dx with dt tiny, and compare to the true PDE solution.
    const double T = 0.02;
    std::vector<double> hs, errs;
    for (int n : {21, 41, 81, 161}) {
        double dx = 1.0 / (n - 1), dt = 1e-6;
        long ns = long(std::lround(T / dt));
        Vec u = sine_ic(n);
        for (long s = 0; s < ns; ++s)
            ex04::heat_crank_nicolson_step(u, kAlpha, dx, dt);
        double e = 0;
        for (int j = 0; j < n; ++j)
            e = std::max(e, std::fabs(u[j] - std::sin(pi * j * dx) *
                                                  std::exp(-kAlpha * pi * pi * T)));
        hs.push_back(dx);
        errs.push_back(e);
    }
    NP_CHECK_ORDER("CN in space", hs, errs, 2.0, 0.15);
}

NP_TEST(ex04_crank_nicolson_is_unconditionally_stable_but_rings) {
    // dt 1000x past the FTCS limit: CN stays bounded (A-stable) ...
    const int n = 65;
    const double dx = 1.0 / (n - 1);
    Vec u(n, 0.0);
    for (int j = 1; j + 1 < n; ++j) u[j] = (j > n / 3 && j < 2 * n / 3) ? 1.0 : 0.0;
    const double dt = 0.05;   // r = 143
    Vec u0 = u;
    for (int s = 0; s < 20; ++s) ex04::heat_crank_nicolson_step(u, kAlpha, dx, dt);
    NP_CHECK_MSG(norm_inf(u) < norm_inf(u0), "CN must not amplify");

    // ... but it is not L-stable: after ONE big step off a discontinuity the
    // high modes are only multiplied by ~-1, so the profile oscillates.
    Vec u1 = u0;
    ex04::heat_crank_nicolson_step(u1, kAlpha, dx, dt);
    double umin = 0;
    for (double v : u1) umin = std::min(umin, v);
    std::printf("      CN undershoot after one big step: %.4f\n", umin);
    NP_CHECK_MSG(umin < -1e-3,
                 "expected the classic CN undershoot (got min = %.3e); if your "
                 "solution does not ring, check the right-hand side assembly",
                 umin);
}

NP_TEST(ex04_adi_2d) {
    const int n = 41;
    const double dx = 1.0 / (n - 1);
    auto ic = [&](int nn) {
        Vec u(size_t(nn) * nn, 0.0);
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i)
                u[size_t(j) * nn + i] =
                    std::sin(pi * i * dx) * std::sin(pi * j * dx);
        return u;
    };
    const double lam =
        2.0 * kAlpha * ex04::discrete_laplacian_eigenvalue(1.0, dx);

    std::vector<double> dts, errs;
    const double T = 0.02;
    for (double dt : {2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4}) {
        long ns = long(std::lround(T / dt));
        Vec u = ic(n);
        for (long s = 0; s < ns; ++s)
            ex04::heat2d_adi_step(u, n, n, kAlpha, dx, dt);
        Vec ref = ic(n);
        for (auto& v : ref) v *= std::exp(lam * T);
        dts.push_back(dt);
        errs.push_back(err_inf(u, ref));
    }
    NP_CHECK_ORDER("ADI in time", dts, errs, 2.0, 0.2);

    // Unconditional stability: r = alpha dt/dx^2 = 1120 and it still decays.
    Vec u = ic(n);
    for (int s = 0; s < 50; ++s) ex04::heat2d_adi_step(u, n, n, kAlpha, dx, 0.7);
    NP_CHECK_MSG(norm_inf(u) < 1e-6, "ADI should be unconditionally stable "
                                     "(got |u| = %.3e)", norm_inf(u));
}

NP_MAIN()
