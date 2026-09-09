#include "ex06_elliptic_spectral.hpp"
#include "np/harness.hpp"
#include "np/rng.hpp"

using namespace np;

namespace {
double resnorm(const Vec& u, const Vec& f, double h) {
    Vec r;
    ex06::poisson_residual(u, f, h, r);
    return norm_inf(r);
}
}  // namespace

NP_TEST(ex06_relaxation_is_a_smoother_not_a_solver) {
    // Solve -u'' = 0 (exact answer: u = 0) starting from an error that is one
    // low mode plus one high mode. Gauss-Seidel annihilates the high mode in a
    // handful of sweeps and barely touches the low one. THAT is why multigrid
    // works: after smoothing, what is left is representable on a coarse grid.
    const int n = 129;
    const double h = 1.0 / (n - 1);
    Vec f(n, 0.0);
    auto amplitude = [&](int k, int sweeps) {
        Vec u(n, 0.0);
        for (int j = 0; j < n; ++j) u[j] = std::sin(k * pi * j * h);
        u[0] = u[n - 1] = 0.0;
        for (int s = 0; s < sweeps; ++s) ex06::gauss_seidel_sweep(u, f, h);
        return norm_inf(u);
    };
    double lo = amplitude(1, 10), hi = amplitude(60, 10);
    std::printf("      after 10 GS sweeps: k=1 amplitude %.4f, k=60 %.3e "
                "(%.0fx more damped)\n", lo, hi, lo / hi);
    NP_CHECK_MSG(lo > 0.9, "low mode should barely decay, got %.4f", lo);
    NP_CHECK_MSG(hi < 0.05, "high mode should be crushed, got %.3e", hi);
    NP_CHECK_MSG(lo / hi > 20.0, "smoothing property not visible");
}

NP_TEST(ex06_sor_beats_gauss_seidel) {
    const int n = 129;
    const double h = 1.0 / (n - 1);
    Vec f(n);
    for (int j = 0; j < n; ++j) f[j] = std::sin(pi * j * h);
    const double omega_opt = 2.0 / (1.0 + std::sin(pi / (n - 1)));

    auto iters = [&](double omega) {
        Vec u(n, 0.0);
        double r0 = resnorm(u, f, h);
        for (int it = 1; it <= 200000; ++it) {
            if (omega == 1.0) ex06::gauss_seidel_sweep(u, f, h);
            else ex06::sor_sweep(u, f, h, omega);
            if (resnorm(u, f, h) < 1e-8 * r0) return it;
        }
        return -1;
    };
    int gs = iters(1.0), sor = iters(omega_opt);
    std::printf("      GS %d sweeps, SOR(omega=%.4f) %d sweeps\n", gs,
                omega_opt, sor);
    NP_CHECK_MSG(gs > 0 && sor > 0, "did not converge");
    NP_CHECK_MSG(sor * 5 < gs, "optimal SOR should be >5x faster than GS");
}

NP_TEST(ex06_multigrid_is_grid_independent) {
    // The headline property: V-cycles to convergence does NOT grow with n.
    int prev = 0;
    for (int n : {65, 129, 257, 513, 1025}) {
        double h = 1.0 / (n - 1);
        Vec f(n), u(n, 0.0);
        for (int j = 0; j < n; ++j)
            f[j] = std::sin(pi * j * h) + 0.5 * std::sin(37.0 * pi * j * h);
        double r0 = resnorm(u, f, h);
        int cycles = 0;
        while (resnorm(u, f, h) > 1e-10 * r0 && cycles < 200) {
            ex06::mg_vcycle(u, f, h, 2, 2);
            ++cycles;
        }
        std::printf("      n = %4d : %d V-cycles\n", n, cycles);
        NP_CHECK_MSG(cycles > 0 && cycles <= 15,
                     "n=%d needed %d V-cycles (expected O(1), <= 15)", n, cycles);
        if (prev) NP_CHECK_MSG(cycles <= prev + 2,
                               "cycle count is growing with n -- the coarse-grid "
                               "correction is probably not being ADDED to u");
        prev = cycles;

        // ... and the answer is right.
        Vec exact(n);
        for (int j = 0; j < n; ++j)
            exact[j] = std::sin(pi * j * h) / (pi * pi) +
                       0.5 * std::sin(37.0 * pi * j * h) / (37.0 * 37.0 * pi * pi);
        NP_CHECK_MSG(err_inf(u, exact) < 5e-5,
                     "n=%d: solution error %.3e", n, err_inf(u, exact));
    }
}

NP_TEST(ex06_fft_round_trip_and_dft_agreement) {
    const int n = 64;
    Rng g(7);
    std::vector<Cplx> a(n), a0(n);
    for (int j = 0; j < n; ++j) a[j] = Cplx(g.normal(), g.normal());
    a0 = a;
    ex06::fft(a, false);

    // brute-force DFT of the first few bins
    for (int k : {0, 1, 5, 31, 63}) {
        Cplx s = 0.0;
        for (int j = 0; j < n; ++j) {
            double th = -2.0 * pi * j * k / n;
            s += a0[j] * Cplx(std::cos(th), std::sin(th));
        }
        NP_CLOSE(a[k].real(), s.real(), 1e-10);
        NP_CLOSE(a[k].imag(), s.imag(), 1e-10);
    }
    ex06::fft(a, true);
    double e = 0;
    for (int j = 0; j < n; ++j) e = std::max(e, std::abs(a[j] - a0[j]));
    NP_CHECK_MSG(e < 1e-12, "round trip error %.3e", e);
}

NP_TEST(ex06_spectral_accuracy) {
    // d/dx exp(sin x) = cos(x) exp(sin x) on [0, 2pi).
    const double L = 2.0 * pi;
    std::vector<double> es;
    for (int n : {16, 32, 64}) {
        Vec u(n), du, ex(n);
        for (int j = 0; j < n; ++j) {
            double x = L * j / n;
            u[j] = std::exp(std::sin(x));
            ex[j] = std::cos(x) * u[j];
        }
        ex06::spectral_derivative(u, L, du);
        double e = err_inf(du, ex);
        std::printf("      n = %3d : spectral error %.3e\n", n, e);
        es.push_back(e);
    }
    // Doubling from 16 to 32 points buys ~8 orders of magnitude and lands on
    // roundoff; there is nothing left for n = 64 to improve.
    NP_CHECK_MSG(es[1] < 1e-5 * es[0], "not spectrally accurate: %.3e -> %.3e",
                 es[0], es[1]);
    NP_CHECK_MSG(es[1] < 1e-12 && es[2] < 1e-12,
                 "n>=32 should be at roundoff (%.3e, %.3e)", es[1], es[2]);
    const double prev = std::max(es[2], 1e-15);

    // For contrast: 2nd-order central differences on the same 64 points.
    const int n = 64;
    double dx = L / n, efd = 0;
    for (int j = 0; j < n; ++j) {
        double xm = L * ((j - 1 + n) % n) / n, xp = L * ((j + 1) % n) / n;
        double x = L * j / n;
        double fd = (std::exp(std::sin(xp)) - std::exp(std::sin(xm))) / (2 * dx);
        efd = std::max(efd, std::fabs(fd - std::cos(x) * std::exp(std::sin(x))));
    }
    std::printf("      n =  64 : 2nd-order FD error %.3e  (%.0e x worse)\n",
                efd, efd / prev);
    NP_CHECK_MSG(efd > 1e6 * prev, "spectral should crush FD here");
}

NP_TEST(ex06_poisson_fft) {
    const int n = 64;
    const double L = 2.0 * pi;
    Vec f(n), u, ex(n);
    for (int j = 0; j < n; ++j) {
        double x = L * j / n;
        f[j] = 9.0 * std::sin(3.0 * x) + 4.0 * std::cos(2.0 * x);
        ex[j] = std::sin(3.0 * x) + std::cos(2.0 * x);
    }
    ex06::poisson_fft(f, L, u);
    NP_CHECK_MSG(err_inf(u, ex) < 1e-12, "error %.3e", err_inf(u, ex));
}

NP_TEST(ex06_etdrk4_is_fourth_order_and_unconditionally_stable) {
    // Allen-Cahn:  u_t = eps u_xx + u - u^3  on [0, 2pi), periodic.
    // Linear symbol lam_m = -eps k_m^2 + 1 ; nonlinear term N(u) = -u^3.
    const int n = 256;
    const double L = 2.0 * pi, eps = 0.05, T = 0.5;

    std::vector<double> lam(n);
    for (int m = 0; m < n; ++m) {
        int mm = (m < n / 2) ? m : m - n;
        double k = 2.0 * pi * mm / L;
        lam[m] = -eps * k * k + 1.0;
    }
    std::printf("      stiffness: min lam = %.1f  -> explicit RK4 needs "
                "dt < %.2e\n", *std::min_element(lam.begin(), lam.end()),
                2.78 / std::fabs(*std::min_element(lam.begin(), lam.end())));

    auto nonlinear = [&](const std::vector<Cplx>& v, std::vector<Cplx>& out) {
        std::vector<Cplx> w = v;
        ex06::fft(w, true);
        for (int j = 0; j < n; ++j) {
            double uj = w[j].real();
            w[j] = Cplx(-uj * uj * uj, 0.0);
        }
        ex06::fft(w, false);
        out = w;
    };
    auto initial = [&] {
        std::vector<Cplx> v(n);
        for (int j = 0; j < n; ++j) {
            double x = L * j / n;
            v[j] = Cplx(0.3 * std::sin(x) + 0.1 * std::cos(2 * x), 0.0);
        }
        ex06::fft(v, false);
        return v;
    };
    auto run = [&](double dt) {
        ex06::Etdrk4 s;
        s.init(lam, dt);
        auto v = initial();
        long ns = long(std::lround(T / dt));
        for (long i = 0; i < ns; ++i) s.step(v, nonlinear);
        ex06::fft(v, true);
        Vec u(n);
        for (int j = 0; j < n; ++j) u[j] = v[j].real();
        return u;
    };

    Vec ref = run(T / 4096.0);
    std::vector<double> dts, errs;
    for (double dt : {T / 32, T / 64, T / 128, T / 256}) {
        dts.push_back(dt);
        errs.push_back(err_inf(run(dt), ref));
    }
    NP_CHECK_ORDER("ETDRK4", dts, errs, 4.0, 0.4);

    // Stability: dt = T/32 is ~250x past the explicit-RK4 limit, and the answer
    // is still accurate to ~1e-8.
    NP_CHECK_MSG(errs[0] < 1e-6,
                 "dt = %.4f should already be accurate (got %.3e)", dts[0],
                 errs[0]);
}

NP_MAIN()
