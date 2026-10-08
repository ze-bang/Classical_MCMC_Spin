// test_gpu_tableaux.cpp — the Runge-Kutta coefficients used by the GPU
// steppers (gpu/ode/rk_tableaux.h, the same constants the .cu code reads),
// verified on the CPU without a GPU:
//
//   - consistency: row sums Σ_j a_ij = c_i, Σ b = Σ b_hat = 1;
//   - measured convergence order on y' = y (exact e^t) and on the resonantly
//     forced oscillator x'' = -x + F cos t (exact x = cos t + (F/2) t sin t,
//     non-autonomous, so the nodes c_i matter): 8 for Fehlberg's propagated
//     solution and 7 for its embedded one, 5(4) for Dormand-Prince and
//     Cash-Karp, 4 for RK4;
//   - the tableau the GPU used before (a53 and a54 swapped) fails the order
//     test, so the check discriminates;
//   - GPU method names: unsupported names (bulirsch_stoer, adams, geometric,
//     typos) throw instead of silently running another method.
#include "classical_spin/gpu/ode/rk_tableaux.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu::ode;
using Vec = std::vector<double>;
using Rhs = std::function<Vec(double, const Vec&)>;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// One explicit RK step with weights b (or b_hat): the CPU twin of the GPU stepper.
Vec rk_step(const ExplicitTableau& T, bool embedded, const Rhs& f, double t, const Vec& y, double h) {
    std::vector<Vec> k(T.stages);
    for (int i = 0; i < T.stages; ++i) {
        Vec stage = y;
        for (int j = 0; j < i; ++j)
            for (size_t n = 0; n < y.size(); ++n) stage[n] += h * T.a[i][j] * k[j][n];
        k[i] = f(t + T.c[i] * h, stage);
    }
    Vec out = y;
    for (int j = 0; j < T.stages; ++j)
        for (size_t n = 0; n < y.size(); ++n) out[n] += h * (embedded ? T.b_hat[j] : T.b[j]) * k[j][n];
    return out;
}

constexpr double kF = 0.3;  // forcing amplitude

// Global error at t_end after n steps.
double global_error(const ExplicitTableau& T, bool embedded, bool oscillator, int n) {
    const double t_end = oscillator ? 10.0 : 4.0;
    Rhs f;
    Vec y;
    if (oscillator) {
        f = [](double t, const Vec& s) { return Vec{s[1], -s[0] + kF * std::cos(t)}; };
        y = {1.0, 0.0};
    } else {
        f = [](double, const Vec& s) { return Vec{s[0]}; };
        y = {1.0};
    }
    const double h = t_end / n;
    for (int i = 0; i < n; ++i) y = rk_step(T, embedded, f, i * h, y, h);
    if (!oscillator) return std::abs(y[0] - std::exp(t_end));
    const double x = std::cos(t_end) + 0.5 * kF * t_end * std::sin(t_end);
    const double v = -std::sin(t_end) + 0.5 * kF * (std::sin(t_end) + t_end * std::cos(t_end));
    return std::hypot(y[0] - x, y[1] - v);
}

double observed_order(const ExplicitTableau& T, bool embedded, bool oscillator, int n) {
    return std::log2(global_error(T, embedded, oscillator, n) / global_error(T, embedded, oscillator, 2 * n));
}

void check_order(const char* name, const ExplicitTableau& T, bool embedded, bool oscillator, int n, int expected) {
    const double p = observed_order(T, embedded, oscillator, n);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s %s solution on %s: observed order %.2f (expected %d)", name,
                  embedded ? "embedded" : "propagated", oscillator ? "forced oscillator" : "y' = y", p, expected);
    check(std::abs(p - expected) < 0.35, buf);
}

void check_consistency(const char* name, const ExplicitTableau& T) {
    double worst = 0.0, sb = 0.0, sbh = 0.0;
    for (int i = 0; i < T.stages; ++i) {
        double row = 0.0, scale = 1.0;   // round-off relative to Σ_j |a_ij| (up to ~45 for rkf78)
        for (int j = 0; j < i; ++j) {
            row += T.a[i][j];
            scale += std::abs(T.a[i][j]);
        }
        worst = std::max(worst, std::abs(row - T.c[i]) / scale);
        for (int j = i; j < ExplicitTableau::kMaxStages; ++j) worst = std::max(worst, std::abs(T.a[i][j]));  // explicit
        sb += T.b[i];
        sbh += T.b_hat[i];
    }
    check(worst < 4e-16, std::string(name) + ": explicit, row sums equal the nodes c_i");
    check(std::abs(sb - 1.0) < 1e-15, std::string(name) + ": weights b sum to 1");
    if (T.embedded_order > 0) check(std::abs(sbh - 1.0) < 1e-15, std::string(name) + ": embedded weights sum to 1");
}

bool throws(const char* name) {
    try {
        (void)parse_method(name);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    check_consistency("rkf78", kFehlberg78);
    check_consistency("dopri5", kDormandPrince54);
    check_consistency("cash_karp54", kCashKarp54);
    check_consistency("rk4", kRK4);

    check_order("rkf78", kFehlberg78, false, false, 16, 8);
    check_order("rkf78", kFehlberg78, false, true, 32, 8);
    check_order("rkf78", kFehlberg78, true, false, 32, 7);
    check_order("rkf78", kFehlberg78, true, true, 32, 7);
    check_order("dopri5", kDormandPrince54, false, true, 64, 5);
    check_order("dopri5", kDormandPrince54, true, true, 64, 4);
    check_order("dopri5", kDormandPrince54, false, false, 128, 5);
    check_order("cash_karp54", kCashKarp54, false, true, 64, 5);
    check_order("cash_karp54", kCashKarp54, true, true, 64, 4);
    check_order("rk4", kRK4, false, true, 64, 4);

    // The tableau of the old GPU stepper: a53 and a54 swapped.
    ExplicitTableau swapped = kFehlberg78;
    std::swap(swapped.a[4][2], swapped.a[4][3]);
    const double p_bad = observed_order(swapped, false, true, 32);
    check(p_bad < 6.5, "the old GPU tableau (a53 <-> a54 swapped) fails: observed order " + std::to_string(p_bad));

    check(parse_method("rk78") == Method::Fehlberg78 && parse_method("rkf78") == Method::Fehlberg78 &&
              parse_method("dopri5") == Method::Dopri5 && parse_method("rk54") == Method::CashKarp54 &&
              parse_method("rk4") == Method::RK4 && parse_method("ssprk53") == Method::SSPRK53,
          "GPU method names parse");
    check(is_adaptive(Method::Dopri5) && is_adaptive(Method::Fehlberg78) && is_adaptive(Method::CashKarp54) &&
              !is_adaptive(Method::RK4) && !is_adaptive(Method::SSPRK53),
          "error-controlled vs fixed-step GPU methods");
    check(adaptive_tableau(Method::Fehlberg78) == &kFehlberg78 && adaptive_tableau(Method::RK4) == nullptr,
          "adaptive methods carry their embedded tableau");
    check(throws("bulirsch_stoer") && throws("bs") && throws("adams_bashforth") && throws("spherical_midpoint") &&
              throws("rk45"),
          "unsupported or unknown GPU method names throw std::invalid_argument");

    if (failures == 0) {
        std::printf("\ntest_gpu_tableaux: all checks passed\n");
        return 0;
    }
    std::printf("\ntest_gpu_tableaux: %d check(s) FAILED\n", failures);
    return 1;
}
