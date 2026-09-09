#include "ex09_underdamped.hpp"
#include "ex10_gle_sllg.hpp"
#include "np/harness.hpp"
#include "np/rng.hpp"

using namespace np;

namespace {

// A 3-dimensional drift matrix (p plus two auxiliary momenta). Its symmetric
// part is diag(1,2,3) -- positive definite, so the extended OU process is
// stable; its antisymmetric part is what makes the memory kernel oscillatory.
Mat drift_matrix() {
    Mat A(3, 3);
    A(0, 0) = 1.0;  A(0, 1) = 1.0;  A(0, 2) = 0.5;
    A(1, 0) = -1.0; A(1, 1) = 2.0;  A(1, 2) = 0.3;
    A(2, 0) = -0.5; A(2, 1) = -0.3; A(2, 2) = 3.0;
    return A;
}

constexpr double kM = 1.0, kKT = 1.0;

}  // namespace

// ================================ part A ===================================

NP_TEST(ex10_gle_reduces_to_white_noise_langevin) {
    // With no auxiliary momenta, A_p = [gamma] and the GLE propagator must BE
    // the exact OU step from ex09.
    const double gamma = 1.7, h = 0.3;
    Mat A(1, 1);
    A(0, 0) = gamma;
    ex10::Gle gle;
    gle.init(A, h, kM, kKT);
    NP_CLOSE(gle.T(0, 0), std::exp(-gamma * h), 1e-13);
    NP_CLOSE(gle.S(0, 0),
             std::sqrt(kM * kKT * (1.0 - std::exp(-2.0 * gamma * h))), 1e-13);
}

NP_TEST(ex10_gle_satisfies_the_fluctuation_dissipation_theorem) {
    // The stationary covariance of the extended vector must be exactly m kT I:
    // every auxiliary momentum thermalised, no cross-correlation.
    for (double h : {0.5, 0.05}) {
        ex10::Gle gle;
        gle.init(drift_matrix(), h, kM, kKT);
        auto step = [&](Vec& z, auto& g) { gle.apply(z, g); };
        Mat S = stationary_covariance(step, 3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                NP_CHECK_MSG(std::fabs(S(i, j) - (i == j ? kM * kKT : 0.0)) < 1e-11,
                             "h=%.2f Sigma(%d,%d) = %.12f", h, i, j, S(i, j));
    }
}

NP_TEST(ex10_gle_memory_kernel_is_actually_coloured) {
    // C(t) = <p(0) p(t)> = m kT (T^n)_00. A white-noise thermostat gives a pure
    // decaying exponential, which can never go negative. This one does.
    ex10::Gle gle;
    gle.init(drift_matrix(), 0.05, kM, kKT);
    Mat P = Mat::identity(3);
    double cmin = 1e9;
    for (int n = 0; n < 200; ++n) {
        cmin = std::min(cmin, kM * kKT * P(0, 0));
        P = matmul(P, gle.T);
    }
    std::printf("      min <p(0)p(t)> = %+.5f (a white-noise bath cannot go "
                "below 0)\n", cmin);
    NP_CHECK_MSG(cmin < -1e-4, "kernel is not coloured (min C = %.3e)", cmin);
}

NP_TEST(ex10_gle_baoab_samples_configurations_exactly) {
    // Colored noise, arbitrary step size, three different oscillator
    // frequencies -- and <q^2> is still exactly kT/(m w^2). Deterministic.
    for (double h : {0.4, 0.1}) {
        ex10::Gle gle;
        gle.init(drift_matrix(), h, kM, kKT);
        for (double w : {0.5, 1.0, 4.0}) {
            auto force = [&](const Vec& q, Vec& f) { f[0] = -w * w * kM * q[0]; };
            auto step = [&](Vec& z, auto& g) {
                Vec q{z[0]}, p{z[1]}, f(1), s{z[2], z[3]};
                force(q, f);
                ex10::gle_baoab_step(force, q, p, f, s, kM, h, gle, g);
                z[0] = q[0]; z[1] = p[0]; z[2] = s[0]; z[3] = s[1];
            };
            Mat C = stationary_covariance(step, 4);
            std::printf("      h=%.2f w=%.1f : <q^2> = %.10f (exact %.10f)\n", h,
                        w, C(0, 0), kKT / (kM * w * w));
            NP_CHECK_MSG(std::fabs(C(0, 0) - kKT / (kM * w * w)) <
                             1e-9 * kKT / (kM * w * w),
                         "GLE-BAOAB <q^2> = %.10f, exact %.10f", C(0, 0),
                         kKT / (kM * w * w));
        }
    }
}

// ================================ part B ===================================

NP_TEST(ex10_cayley_rotate_is_an_exact_rotation) {
    Rng g(3);
    for (int trial = 0; trial < 200; ++trial) {
        Vec3 S = normalized(Vec3{g.normal(), g.normal(), g.normal()});
        double scale = std::exp(4.0 * (g.uniform() - 0.5));   // |w| over 3 decades
        Vec3 w{scale * g.normal(), scale * g.normal(), scale * g.normal()};
        Vec3 R = ex10::cayley_rotate(S, w);
        NP_CHECK_MSG(std::fabs(norm3(R) - 1.0) < 1e-14,
                     "|S| drifted by %.3e at |w| = %.3f", norm3(R) - 1.0,
                     norm3(w));
        // It must satisfy the defining implicit-midpoint relation exactly.
        Vec3 mid = 0.5 * (S + R);
        Vec3 lhs = R - S, rhs = cross(w, mid);
        NP_CHECK_MSG(norm3(lhs - rhs) < 1e-12,
                     "cayley_rotate does not solve S' = S + w x (S+S')/2");
    }
    // small-w limit
    Vec3 S{0, 0, 1}, w{1e-6, 0, 0};
    Vec3 R = ex10::cayley_rotate(S, w);
    NP_CLOSE(R[1], -1e-6, 1e-15);
}

NP_TEST(ex10_damped_precession_matches_the_analytic_solution) {
    // Zero temperature, uniform field B along z. Exact solution:
    //   tan(theta/2) = tan(theta0/2) exp(-lambda t),  phi = omega t
    //   omega = B/(1+a^2),  lambda = a B/(1+a^2)
    // This pins BOTH the precession frequency (the 1/(1+a^2) prefactor) and the
    // damping rate, which the equilibrium test in the next case cannot see.
    const double a = 0.5, B = 1.0, T = 2.0, th0 = 1.0;
    const double w = B / (1 + a * a), lam = a * B / (1 + a * a);
    auto field = [&](const Vec3&) -> Vec3 { return {0.0, 0.0, B}; };
    const double th = 2.0 * std::atan(std::tan(0.5 * th0) * std::exp(-lam * T));
    const Vec3 exact{std::sin(th) * std::cos(w * T),
                     std::sin(th) * std::sin(w * T), std::cos(th)};

    const std::vector<double> dts = {0.02, 0.01, 0.005, 0.0025};
    std::vector<double> e_sib, e_heun;
    for (double dt : dts) {
        Rng g1(1), g2(1);
        Vec3 A{std::sin(th0), 0, std::cos(th0)}, Bs = A;
        long n = long(std::lround(T / dt));
        for (long i = 0; i < n; ++i) ex10::sllg_sib_step(field, A, dt, a, 0.0, g1);
        for (long i = 0; i < n; ++i) ex10::sllg_heun_step(field, Bs, dt, a, 0.0, g2);
        e_sib.push_back(norm3(A - exact));
        e_heun.push_back(norm3(Bs - exact));
    }
    NP_CHECK_ORDER("SIB (deterministic)", dts, e_sib, 2.0, 0.15);
    NP_CHECK_ORDER("Heun (deterministic)", dts, e_heun, 2.0, 0.15);
    std::printf("      dt=0.02 errors: SIB %.3e, Heun %.3e\n", e_sib[0],
                e_heun[0]);
}

NP_TEST(ex10_sib_conserves_the_spin_length_identically) {
    auto field = [](const Vec3& S) -> Vec3 {
        return {0.3, 0.0, 1.0 + 0.5 * S[0]};   // includes an anisotropy-like term
    };
    const long N = 10000;
    Rng g1(1);
    Vec3 Sh{0, 0, 1};
    for (long i = 0; i < N; ++i)
        ex10::sllg_heun_step(field, Sh, 0.01, 0.2, 0.5, g1, /*renormalize=*/false);
    Rng g2(1);
    Vec3 Ss{0, 0, 1};
    for (long i = 0; i < N; ++i) ex10::sllg_sib_step(field, Ss, 0.01, 0.2, 0.5, g2);
    std::printf("      after %ld steps: Heun (no renorm) |S|-1 = %+.3e, "
                "SIB |S|-1 = %+.3e\n", N, norm3(Sh) - 1.0, norm3(Ss) - 1.0);
    NP_CHECK_MSG(std::fabs(norm3(Ss) - 1.0) < 1e-12,
                 "SIB must be exactly norm preserving, got %+.3e",
                 norm3(Ss) - 1.0);
    NP_CHECK_MSG(std::fabs(norm3(Sh) - 1.0) > 1e-3,
                 "unrenormalised Heun should visibly drift (got %+.3e) -- if it "
                 "does not, are you renormalising unconditionally?",
                 norm3(Sh) - 1.0);
}

NP_TEST(ex10_equilibrium_is_boltzmann_and_independent_of_damping) {
    // THE test. A single spin in a field B at temperature kT must satisfy
    //     <S_z>   = L(x) = coth x - 1/x,      x = B/kT
    //     <S_z^2> = 1 - 2 L(x)/x
    // and neither may depend on alpha. That last clause is the fluctuation-
    // dissipation theorem: alpha appears in both the damping and the noise
    // amplitude, and only the correct pairing cancels out of the equilibrium
    // distribution. A wrong factor of (1+alpha^2), a missing 2, or an Ito
    // reading of the multiplicative noise all break it.
    const double B = 1.0, kT = 0.5, x = B / kT;
    const double sz_exact = ex10::langevin_function(x);
    const double sz2_exact = 1.0 - 2.0 * sz_exact / x;
    auto field = [&](const Vec3&) -> Vec3 { return {0.0, 0.0, B}; };

    std::printf("      exact: <S_z> = %.5f, <S_z^2> = %.5f\n", sz_exact,
                sz2_exact);
    std::vector<double> means;
    for (double alpha : {0.2, 0.5}) {
        Rng g(3);
        Vec3 S{0, 0, 1};
        Stats sz, sz2;
        const long N = 10'000'000;
        for (long i = 0; i < N; ++i) {
            ex10::sllg_sib_step(field, S, 0.005, alpha, kT, g);
            sz.add(S[2]);
            sz2.add(S[2] * S[2]);
        }
        std::printf("      alpha = %.2f : <S_z> = %.5f, <S_z^2> = %.5f, "
                    "|S|-1 = %.1e\n", alpha, sz.mean, sz2.mean, norm3(S) - 1.0);
        NP_CHECK_MSG(std::fabs(sz.mean - sz_exact) < 0.01,
                     "alpha=%.2f: <S_z> = %.5f vs Langevin function %.5f",
                     alpha, sz.mean, sz_exact);
        NP_CHECK_MSG(std::fabs(sz2.mean - sz2_exact) < 0.01,
                     "alpha=%.2f: <S_z^2> = %.5f vs exact %.5f", alpha,
                     sz2.mean, sz2_exact);
        means.push_back(sz.mean);
    }
    NP_CHECK_MSG(std::fabs(means[0] - means[1]) < 0.01,
                 "equilibrium depends on the damping (%.5f vs %.5f) -- the "
                 "fluctuation-dissipation balance is wrong", means[0], means[1]);
}

NP_MAIN()
