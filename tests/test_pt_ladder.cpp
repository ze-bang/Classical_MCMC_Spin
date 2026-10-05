// test_pt_ladder.cpp — temperature-ladder updates on synthetic data.
//
//  1. NRPT schedule update (Syed et al. 2022): with the per-edge rejection
//     generated from a known barrier Lambda(beta) (r_e = Lambda(b_{e+1}) -
//     Lambda(b_e)), iterating the update must converge to the equal-barrier
//     ladder Lambda^{-1}(i Lambda / (R-1)) and equalise the rejections; a
//     ladder with equal rejections is a fixed point.
//  2. Katzgraber-Trebst-Huse-Troyer flow feedback: linear f(T) in the replica
//     index is a fixed point; the update places equal shares of
//     sum sqrt(-df) between temperatures; non-monotone f is made monotone.
//  3. DEO round-trip-rate formula and the shared exchange uniform.
#include "physics_test_util.h"

#include "classical_spin/mc/parallel_tempering.h"

using namespace phys_test;

namespace {

// Barrier density with a sharp peak at beta = 1.5 (a "transition").
double Lambda(double b) {
    const double a = 0.3, h = 1.2, b0 = 1.5, w = 0.15;
    return a * b + h * 0.5 * (1.0 + std::erf((b - b0) / (std::sqrt(2.0) * w)));
}

std::vector<double> rejections(const std::vector<double>& T) {
    std::vector<double> r(T.size() - 1);
    for (size_t k = 0; k + 1 < T.size(); ++k) r[k] = Lambda(1.0 / T[k]) - Lambda(1.0 / T[k + 1]);
    return r;
}

bool strictly_increasing(const std::vector<double>& T) {
    for (size_t k = 1; k < T.size(); ++k)
        if (!(T[k] > T[k - 1])) return false;
    return true;
}

void test_nrpt_update() {
    std::printf("\n== NRPT schedule update (equal rejection) ==\n");
    const size_t R = 12;
    const double Tmin = 0.2, Tmax = 5.0;
    std::vector<double> T = mc::generate_geometric_temperature_ladder(Tmin, Tmax, R);
    auto spread = [](const std::vector<double>& r) {
        const auto [lo, hi] = std::minmax_element(r.begin(), r.end());
        double m = 0;
        for (double x : r) m += x / double(r.size());
        return (*hi - *lo) / m;
    };
    const double s0 = spread(rejections(T));
    auto T1 = mc::nrpt_schedule_update(T, rejections(T));
    const double s1 = spread(rejections(T1));
    check(s1 < 0.5 * s0, "one update halves the rejection spread (" + std::to_string(s0) + " -> " +
                             std::to_string(s1) + ")");
    check(T1.front() == Tmin && T1.back() == Tmax && strictly_increasing(T1), "end points fixed, ladder monotone");
    for (int it = 0; it < 8; ++it) T1 = mc::nrpt_schedule_update(T1, rejections(T1));
    const double s8 = spread(rejections(T1));
    check(s8 < 0.01, "iterated updates equalise the rejections (relative spread " + std::to_string(s8) + ")");

    // Compare with the exact equal-barrier ladder from bisection on Lambda.
    const double b_lo = 1.0 / Tmax, b_hi = 1.0 / Tmin, L_tot = Lambda(b_hi) - Lambda(b_lo);
    double worst = 0.0;
    for (size_t i = 1; i + 1 < R; ++i) {
        const double target = Lambda(b_lo) + L_tot * double(i) / double(R - 1);
        double a = b_lo, b = b_hi;
        for (int k = 0; k < 200; ++k) { const double m = 0.5 * (a + b); (Lambda(m) < target ? a : b) = m; }
        const double T_exact = 1.0 / (0.5 * (a + b));
        worst = std::max(worst, std::abs(T1[R - 1 - i] - T_exact) / T_exact);
    }
    check(worst < 2e-3, "converged ladder = exact equal-barrier ladder (max rel. dev " + std::to_string(worst) + ")");

    // Fixed point: equal rejections leave any ladder unchanged.
    const auto T2 = mc::nrpt_schedule_update(T, std::vector<double>(R - 1, 0.37));
    check(mc::ladder_change(T, T2) < 1e-10, "equal rejections are a fixed point");
    // Zero measured rejection on some edges stays well defined.
    std::vector<double> rz(R - 1, 0.0);
    rz[3] = 0.9;
    const auto T3 = mc::nrpt_schedule_update(T, rz);
    check(strictly_increasing(T3) && T3.front() == Tmin && T3.back() == Tmax, "zero-rejection edges handled");
    check(T3[4] - T3[3] < T[4] - T[3], "the bottleneck edge shrinks");
}

void test_katzgraber_update() {
    std::printf("\n== Katzgraber-Trebst-Huse-Troyer flow feedback ==\n");
    const size_t R = 8;
    const std::vector<double> T = mc::generate_geometric_temperature_ladder(0.5, 4.0, R);
    std::vector<double> f(R);
    for (size_t k = 0; k < R; ++k) f[k] = 1.0 - double(k) / double(R - 1);
    check(mc::ladder_change(T, mc::katzgraber_schedule_update(T, f)) < 1e-12,
          "f linear in the replica index is a fixed point");

    const std::vector<double> fs = {1.0, 0.97, 0.94, 0.35, 0.30, 0.2, 0.1, 0.0};
    const auto T1 = mc::katzgraber_schedule_update(T, fs);
    check(strictly_increasing(T1) && T1.front() == T.front() && T1.back() == T.back(), "end points fixed, monotone");
    // Equal shares of G = sum sqrt(f_k - f_{k+1}) (piecewise linear in T).
    std::vector<double> G(R, 0.0);
    for (size_t k = 0; k + 1 < R; ++k) G[k + 1] = G[k] + std::sqrt(fs[k] - fs[k + 1]);
    double worst = 0.0;
    for (size_t i = 1; i + 1 < R; ++i) {
        size_t k = 0;
        while (T[k + 1] < T1[i]) ++k;
        const double g = G[k] + (T1[i] - T[k]) / (T[k + 1] - T[k]) * (G[k + 1] - G[k]);
        worst = std::max(worst, std::abs(g / G.back() - double(i) / double(R - 1)));
    }
    check(worst < 1e-12, "new ladder splits sum sqrt(-df) equally (dev " + std::to_string(worst) + ")");
    size_t inside = 0;
    for (double t : T1) inside += (t > T[2] && t < T[3]) ? 1 : 0;
    check(inside >= 2, "two temperatures move into the interval where f drops steeply");

    // Non-monotone noisy f is pooled (isotonic regression), unobserved points interpolated.
    const std::vector<double> fn = {1.0, 0.6, 0.7, 0.3, std::nan(""), 0.1, 0.05, 0.0};
    const auto T2 = mc::katzgraber_schedule_update(T, fn, {1, 10, 10, 10, 0, 10, 10, 1});
    check(strictly_increasing(T2), "non-monotone / unobserved f handled");
    const auto iso = mc::detail::isotonic_nonincreasing({1.0, 0.6, 0.7, 0.3}, {1, 1, 1, 1});
    check(std::abs(iso[1] - 0.65) < 1e-14 && std::abs(iso[2] - 0.65) < 1e-14, "PAVA pools 0.6, 0.7 -> 0.65");
}

void test_round_trip_formula_and_uniform() {
    std::printf("\n== DEO round-trip rate and shared exchange uniform ==\n");
    check_close(mc::nrpt_round_trip_rate({0.0, 0.0, 0.0}), 0.5, 1e-15, "no rejection: 1/2 round trip per round");
    check_close(mc::nrpt_round_trip_rate({0.5, 0.5}), 1.0 / 6.0, 1e-15, "r = 1/2 on two edges: 1/6");
    check(mc::nrpt_round_trip_rate({0.3, 1.0}) == 0.0, "a blocked edge gives no round trips");
    // Exchange uniforms: identical for identical keys, uniform otherwise.
    check(mc::detail::exchange_uniform(7, 11, 3) == mc::detail::exchange_uniform(7, 11, 3),
          "shared uniform is a pure function of (seed, round, edge)");
    double m = 0, m2 = 0;
    const int n = 200000;
    for (int i = 0; i < n; ++i) {
        const double u = mc::detail::exchange_uniform(12345, uint64_t(i / 7), uint64_t(i % 7));
        m += u / n;
        m2 += u * u / n;
    }
    check(std::abs(m - 0.5) < 5 * std::sqrt(1.0 / 12.0 / n) && std::abs(m2 - 1.0 / 3.0) < 0.003,
          "shared uniforms have the U(0,1) mean and second moment");
}

}  // namespace

int main() {
    test_nrpt_update();
    test_katzgraber_update();
    test_round_trip_formula_and_uniform();
    return finish("test_pt_ladder");
}
