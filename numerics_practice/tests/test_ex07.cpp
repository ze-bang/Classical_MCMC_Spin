#include "ex07_sde_basics.hpp"
#include "np/harness.hpp"
#include "np/rng.hpp"

using namespace np;

namespace {

// Geometric Brownian motion: dX = mu X dt + sigma X dW (Ito).
// Exact:  X(T) = X0 exp( (mu - sigma^2/2) T + sigma W_T ).
constexpr double kMu = 0.5, kSigma = 0.8, kX0 = 1.0, kT = 1.0;

auto gbm_a = [](double, double x) { return kMu * x; };
auto gbm_b = [](double, double x) { return kSigma * x; };
auto gbm_db = [](double, double) { return kSigma; };

double gbm_exact(double WT) {
    return kX0 * std::exp((kMu - 0.5 * kSigma * kSigma) * kT + kSigma * WT);
}

}  // namespace

NP_TEST(ex07_brownian_increments_have_the_right_variance) {
    Rng g(11);
    auto dW = ex07::brownian_path(200000, 0.01, g);
    Stats s;
    for (double v : dW) s.add(v);
    std::printf("      mean %.5f (0), var %.5f (0.01)\n", s.mean, s.var());
    NP_CHECK_MSG(std::fabs(s.mean) < 5e-4, "mean %.3e", s.mean);
    NP_CHECK_MSG(std::fabs(s.var() - 0.01) < 1e-4,
                 "var %.5f -- dW must be sqrt(dt)*N(0,1), not dt*N(0,1)",
                 s.var());
}

NP_TEST(ex07_strong_orders) {
    // Pathwise error against the exact solution driven by the SAME path.
    const int nfine = 1 << 14;
    const double dtf = kT / nfine;
    const int npaths = 3000;
    const std::vector<int> factors = {1 << 8, 1 << 6, 1 << 4, 1 << 2};

    std::vector<double> hs, e_em, e_mil, e_heun;
    std::vector<Stats> sem(factors.size()), smil(factors.size()),
        sheun(factors.size());

    Rng g(2024);
    for (int p = 0; p < npaths; ++p) {
        auto fine = ex07::brownian_path(nfine, dtf, g);
        double WT = 0;
        for (double v : fine) WT += v;
        const double exact = gbm_exact(WT);

        for (size_t li = 0; li < factors.size(); ++li) {
            auto dW = ex07::coarsen(fine, factors[li]);
            const double dt = dtf * factors[li];
            double xe = kX0, xm = kX0, xh = kX0;
            // The Stratonovich form of the same process has drift
            // a_S = mu x - (1/2) sigma^2 x, so Heun reproduces the same X(T).
            auto strat_a = [](double, double x) {
                return (kMu - 0.5 * kSigma * kSigma) * x;
            };
            for (size_t i = 0; i < dW.size(); ++i) {
                double t = double(i) * dt;
                ex07::euler_maruyama_step(gbm_a, gbm_b, t, xe, dt, dW[i]);
                ex07::milstein_step(gbm_a, gbm_b, gbm_db, t, xm, dt, dW[i]);
                ex07::stratonovich_heun_step(strat_a, gbm_b, t, xh, dt, dW[i]);
            }
            sem[li].add(std::fabs(xe - exact));
            smil[li].add(std::fabs(xm - exact));
            sheun[li].add(std::fabs(xh - exact));
        }
    }
    for (size_t li = 0; li < factors.size(); ++li) {
        hs.push_back(dtf * factors[li]);
        e_em.push_back(sem[li].mean);
        e_mil.push_back(smil[li].mean);
        e_heun.push_back(sheun[li].mean);
    }
    print_convergence("Euler-Maruyama (strong)", hs, e_em);
    print_convergence("Milstein (strong)", hs, e_mil);
    print_convergence("Stratonovich Heun (strong)", hs, e_heun);
    NP_CHECK_ORDER("Euler-Maruyama (strong)", hs, e_em, 0.5, 0.15);
    NP_CHECK_ORDER("Milstein (strong)", hs, e_mil, 1.0, 0.15);
    NP_CHECK_ORDER("Stratonovich Heun (strong)", hs, e_heun, 1.0, 0.2);
}

NP_TEST(ex07_weak_order_of_euler_maruyama) {
    // E[X_T] = X0 exp(mu T) exactly. Euler-Maruyama's mean obeys
    // E[X_N] = X0 (1 + mu dt)^N, so the bias is O(dt): weak order 1, even
    // though the strong order is only 1/2.
    const double exact = kX0 * std::exp(kMu * kT);
    std::vector<double> hs, errs;
    for (int n : {4, 8, 16, 32}) {
        const double dt = kT / n;
        Rng g(4242 + n);
        Stats s;
        // Antithetic pairs: using (+xi, -xi) cancels the leading Monte Carlo
        // error in the mean, which is what makes 200k samples enough here.
        for (int p = 0; p < 200000; ++p) {
            double xp = kX0, xm = kX0;
            for (int i = 0; i < n; ++i) {
                double dW = std::sqrt(dt) * g.normal();
                ex07::euler_maruyama_step(gbm_a, gbm_b, double(i) * dt, xp, dt, dW);
                ex07::euler_maruyama_step(gbm_a, gbm_b, double(i) * dt, xm, dt, -dW);
            }
            s.add(0.5 * (xp + xm));
        }
        hs.push_back(dt);
        errs.push_back(std::fabs(s.mean - exact));
        std::printf("      dt = %.4f : E[X] = %.5f (exact %.5f, sem %.1e)\n",
                    dt, s.mean, exact, s.sem());
    }
    NP_CHECK_ORDER("Euler-Maruyama (weak)", hs, errs, 1.0, 0.2);
}

NP_TEST(ex07_ito_versus_stratonovich_are_different_processes) {
    // The SAME coefficients a = 0, b = sigma x, driven by the SAME path:
    //   Ito         dX = sigma X dW        -> X = X0 exp(sigma W - sigma^2 T/2)
    //   Stratonovich dX = sigma X o dW     -> X = X0 exp(sigma W)
    // The two answers differ by exp(sigma^2 T / 2) = 4.6% here. This is not a
    // discretisation error -- it does not go away as dt -> 0.
    const int n = 1 << 16;
    const double dt = kT / n;
    Rng g(99);
    auto dW = ex07::brownian_path(n, dt, g);
    double WT = 0;
    for (double v : dW) WT += v;

    auto zero = [](double, double) { return 0.0; };
    double x_ito = kX0, x_str = kX0;
    for (int i = 0; i < n; ++i) {
        ex07::euler_maruyama_step(zero, gbm_b, double(i) * dt, x_ito, dt, dW[i]);
        ex07::stratonovich_heun_step(zero, gbm_b, double(i) * dt, x_str, dt, dW[i]);
    }
    const double ito_exact = kX0 * std::exp(kSigma * WT - 0.5 * kSigma * kSigma * kT);
    const double str_exact = kX0 * std::exp(kSigma * WT);
    std::printf("      Ito  %.6f (exact %.6f)\n      Str  %.6f (exact %.6f)\n",
                x_ito, ito_exact, x_str, str_exact);
    // Tolerances are relative: at this dt the residual Euler-Maruyama strong
    // error is a few 0.1%, which is orders of magnitude smaller than the 38%
    // gap between the two interpretations.
    NP_CLOSE(x_ito, ito_exact, 5e-3 * ito_exact);
    NP_CLOSE(x_str, str_exact, 5e-3 * str_exact);
    const double ratio_exact = std::exp(0.5 * kSigma * kSigma * kT);
    NP_CHECK_MSG(std::fabs(x_str / x_ito - ratio_exact) < 0.01 * ratio_exact,
                 "the Ito/Stratonovich ratio is %.5f, should be "
                 "exp(sigma^2 T/2) = %.5f", x_str / x_ito, ratio_exact);
}

NP_MAIN()
