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

}  // namespace

int main() {
    std::printf("\n== Error analysis on AR(1) processes ==\n");
    for (double rho : {0.0, 0.5, 0.9, 0.98}) test_ar1(rho);
    test_large_offset_variance();
    return finish("test_mc_statistics");
}
