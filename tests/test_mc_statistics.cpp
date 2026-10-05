// test_mc_statistics.cpp — error-analysis helpers vs. a process with known
// autocorrelation.
//
// The AR(1) process x_t = rho x_{t-1} + sqrt(1 - rho^2) xi_t has unit variance,
// autocorrelation rho^|t| and integrated autocorrelation time
//     tau_int = 1/2 + sum_{t>=1} rho^t = (1 + rho) / (2 (1 - rho)),
// so the true standard error of the mean of n samples is sqrt(2 tau_int / n).
#include "physics_test_util.h"

#include <random>

using namespace phys_test;

namespace {

std::vector<double> ar1(size_t n, double rho, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<double> x(n);
    double v = N01(gen);
    const double c = std::sqrt(1.0 - rho * rho);
    for (size_t i = 0; i < n; ++i) {
        v = rho * v + c * N01(gen);
        x[i] = v;
    }
    return x;
}

void test_ar1(double rho) {
    const size_t n = 1 << 20;
    const double tau = (1 + rho) / (2 * (1 - rho));
    const double true_err = std::sqrt(2 * tau / double(n));
    const std::string tag = " (rho=" + std::to_string(rho) + ", tau=" + std::to_string(tau) + ")";

    // Average estimates over independent realisations: the estimators are
    // themselves noisy, and we want to test their bias, not their luck.
    double err_bin = 0, tau_bin = 0, tau_sokal = 0;
    const int reps = 8;
    for (int r = 0; r < reps; ++r) {
        auto x = ar1(n, rho, 1000 + r);
        auto b = mc::binning_analysis(x);
        err_bin += b.error / reps;
        tau_bin += b.tau_int / reps;
        auto a = mc::compute_autocorrelation(x, 1);
        tau_sokal += a.tau_int / reps;
    }
    check(std::abs(err_bin / true_err - 1.0) < 0.15,
          "binning error / true error = " + std::to_string(err_bin / true_err) + tag);
    check(std::abs(tau_bin / tau - 1.0) < 0.3,
          "binning tau_int / exact = " + std::to_string(tau_bin / tau) + tag);
    check(std::abs(tau_sokal / tau - 1.0) < 0.15,
          "Sokal tau_int / exact = " + std::to_string(tau_sokal / tau) + tag);
}

void test_large_offset_variance() {
    // Variance of data with a large common offset (E ~ -1e6 for a big lattice)
    // must not suffer catastrophic cancellation.
    auto x = ar1(1 << 16, 0.0, 77);
    for (double& v : x) v = 1e7 + 1e-3 * v;
    auto b = mc::binning_analysis(x);
    const double expected = 1e-3 / std::sqrt(double(x.size()));
    check(std::abs(b.errors_by_level.at(0) / expected - 1.0) < 0.05,
          "binning level-0 error with 1e7 offset: ratio " +
              std::to_string(b.errors_by_level.at(0) / expected));
}

// ------------------------------------------------------------ Gamma method
void test_gamma_ar1(double rho) {
    // Wolff's automatic windowing must recover the exact tau_int of AR(1)
    // (no lag cap, no stop at the first negative rho) and the true error.
    const size_t n = 1 << 18;
    const double tau = (1 + rho) / (2 * (1 - rho));
    const double true_err = std::sqrt(2 * tau / double(n));
    double t = 0, e = 0, dt = 0;
    const int reps = 8;
    for (int r = 0; r < reps; ++r) {
        auto g = mc::gamma_method(ar1(n, rho, 500 + r));
        t += g.tau_int / reps;
        e += g.error / reps;
        dt += g.tau_int_error / reps;
    }
    const std::string tag = " (rho=" + std::to_string(rho) + ", exact tau=" + std::to_string(tau) + ")";
    check(std::abs(t / tau - 1.0) < 0.08, "Gamma tau_int / exact = " + std::to_string(t / tau) + tag);
    check(std::abs(e / true_err - 1.0) < 0.06, "Gamma error / true error = " + std::to_string(e / true_err) + tag);
    // The quoted d tau must be a sensible fraction of tau (Madras-Sokal scale).
    check(dt > 0 && dt < 0.2 * tau + 0.05, "Gamma d tau_int = " + std::to_string(dt) + tag);
}

void test_gamma_coverage() {
    // Error bars must cover: |mean| < 1.96 err in ~95% of independent runs.
    const int reps = 400;
    int inside = 0;
    for (int r = 0; r < reps; ++r) {
        auto g = mc::gamma_method(ar1(4000, 0.9, 9000 + r));
        if (std::abs(g.mean) < 1.96 * g.error) ++inside;
    }
    const double cov = double(inside) / reps;
    check(cov > 0.90 && cov < 0.99, "Gamma 95% interval coverage on AR(1) rho=0.9: " + std::to_string(cov));
}

void test_autocovariance_fft() {
    auto x = ar1(3001, 0.7, 4242);
    for (double& v : x) v = 5.0 + v;
    const size_t L = 200;
    auto g = mc::autocovariance(x, L);
    double m = 0;
    for (double v : x) m += v;
    m /= double(x.size());
    double worst = 0;
    for (size_t t = 0; t <= L; ++t) {
        double s = 0;
        for (size_t i = 0; i + t < x.size(); ++i) s += (x[i] - m) * (x[i + t] - m);
        worst = std::max(worst, std::abs(g[t] - s / double(x.size() - t)));
    }
    check(worst < 1e-12, "FFT autocovariance equals the direct sum (max diff " + std::to_string(worst) + ")");
}

void test_degenerate_series() {
    auto c = mc::gamma_method(std::vector<double>(100, 3.25));
    check(c.mean == 3.25 && c.error == 0.0 && std::isfinite(c.tau_int), "constant series: exact mean, zero error");
    auto one = mc::gamma_method({1.5});
    check(one.mean == 1.5 && one.error == 0.0, "single sample: mean, zero error, no NaN");
    auto b = mc::binning_analysis({2.0});
    check(std::isfinite(b.error) && b.mean == 2.0, "binning of one sample is finite");
    mc::ThermodynamicObservables th;
    mc::energy_statistics({-3.0}, 1.0, 4, th);
    check(std::isfinite(th.specific_heat.value) && std::isfinite(th.specific_heat.error),
          "energy statistics of one sample are finite");
    auto big = mc::gamma_method([] { auto v = ar1(1 << 14, 0.0, 31); for (double& q : v) q = 1e8 + 1e-4 * q; return v; }());
    const double expected = 1e-4 / std::sqrt(double(1 << 14));
    check(std::abs(big.error / expected - 1.0) < 0.1,
          "Gamma error with 1e8 offset: ratio " + std::to_string(big.error / expected));
}

// ---------------------------------------------------- blocked jackknife
void test_jackknife_variance() {
    // Var(x) of AR(1) is exactly 1. The jackknife over tau-sized blocks must
    // be unbiased and its error must match the run-to-run spread.
    const int reps = 200;
    const size_t n = 20000;
    std::vector<double> est, err;
    for (int r = 0; r < reps; ++r) {
        auto o = mc::scaled_variance(ar1(n, 0.9, 7000 + r), 1.0);
        est.push_back(o.value);
        err.push_back(o.error);
    }
    double m = 0, s2 = 0, e = 0;
    for (int r = 0; r < reps; ++r) { m += est[r] / reps; e += err[r] / reps; }
    for (int r = 0; r < reps; ++r) s2 += (est[r] - m) * (est[r] - m) / (reps - 1);
    const double spread = std::sqrt(s2);
    check(std::abs(m - 1.0) < 4 * spread / std::sqrt(double(reps)),
          "jackknife Var(x) unbiased: mean " + std::to_string(m) + " (exact 1)");
    check(e / spread > 0.8 && e / spread < 1.25,
          "jackknife error / empirical spread = " + std::to_string(e / spread));
}

void test_jackknife_uses_all_samples() {
    // Covariance of identical series = variance; remainder samples must count.
    std::vector<double> x = ar1(199, 0.0, 99);
    auto c = mc::covariance(x, x);
    double m = 0, v = 0;
    for (double q : x) m += q / 199.0;
    for (double q : x) v += (q - m) * (q - m) / 199.0;
    check(std::abs(c.value - v) < 0.02 * v && c.error > 0, "covariance(x, x) = Var(x) incl. remainder samples");
}

// ---------------------------------------------------- order parameters
void test_order_parameter_gaussian() {
    // m ~ N(0, 1_3): <m^2> = 3, Var|m| = 3 - 8/pi (chi distribution, 3 dof),
    // U4 = 1 - <m^4>/(3<m^2>^2) = 1 - 15/27 = 4/9.
    std::mt19937_64 gen(2024);
    std::normal_distribution<double> N01;
    mc::OrderParameterSeries op;
    op.name = "m";
    op.dim = 3;
    op.n_sites = 1;
    const size_t n = 200000;
    for (size_t i = 0; i < 3 * n; ++i) op.data.push_back(N01(gen));
    auto st = mc::order_parameter_statistics(op, 1.0);
    check_stat(st.m2.value, st.m2.error, 3.0, "<m^2> of a Gaussian 3-vector", 5.0);
    check_stat(st.susceptibility.value, st.susceptibility.error, 3.0 - 8.0 / M_PI,
               "chi = beta N Var|m| of a Gaussian 3-vector", 5.0);
    check_stat(st.binder.value, st.binder.error, 4.0 / 9.0, "Binder U4 of a Gaussian 3-vector", 5.0);
    check_stat(st.abs.value, st.abs.error, 2.0 * std::sqrt(2.0 / M_PI), "<|m|> of a Gaussian 3-vector", 5.0);
    check_stat(st.mean.values[1], st.mean.errors[1], 0.0, "<m_y> of a Gaussian 3-vector", 5.0);
}

}  // namespace

int main() {
    std::printf("\n== Error analysis on AR(1) processes ==\n");
    for (double rho : {0.0, 0.5, 0.9, 0.98}) test_ar1(rho);
    test_large_offset_variance();
    std::printf("\n== Gamma method (Wolff 2004) ==\n");
    for (double rho : {0.0, 0.5, 0.9, 0.98}) test_gamma_ar1(rho);
    test_gamma_coverage();
    test_autocovariance_fft();
    test_degenerate_series();
    std::printf("\n== Blocked jackknife and order-parameter estimators ==\n");
    test_jackknife_variance();
    test_jackknife_uses_all_samples();
    test_order_parameter_gaussian();
    return finish("test_mc_statistics");
}
