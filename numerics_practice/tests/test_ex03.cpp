#include "ex01_ode_explicit.hpp"
#include "ex03_implicit_stiff.hpp"
#include "np/harness.hpp"
#include "np/problems.hpp"

using namespace np;
using namespace np::problems;

namespace {

// Nonstiff scalar problem with an exact solution, plus its Jacobian.
struct Scalar {
    void operator()(double t, const Vec& y, Vec& dy) const {
        dy[0] = y[0] * std::cos(t);
    }
    void jac(double t, const Vec&, Mat& J) const { J(0, 0) = std::cos(t); }
    double exact(double t) const { return std::exp(std::sin(t)); }
};

const std::vector<double> kSteps = {1.0 / 10, 1.0 / 20, 1.0 / 40, 1.0 / 80};

}  // namespace

NP_TEST(ex03_orders) {
    Scalar s;
    auto jac = [&](double t, const Vec& y, Mat& J) { s.jac(t, y, J); };
    const double T = 2.0;

    auto run = [&](auto&& stepper) {
        std::vector<double> errs;
        for (double h : kSteps) {
            Vec y{1.0};
            long n = long(std::lround(T / h));
            for (long i = 0; i < n; ++i) stepper(y, double(i) * h, h);
            errs.push_back(std::fabs(y[0] - s.exact(T)));
        }
        return errs;
    };

    auto e_be = run([&](Vec& y, double t, double h) {
        ex03::backward_euler_step(s, jac, t, y, h);
    });
    NP_CHECK_ORDER("backward Euler", kSteps, e_be, 1.0, 0.15);

    auto e_tr = run([&](Vec& y, double t, double h) {
        ex03::trapezoid_step(s, jac, t, y, h);
    });
    NP_CHECK_ORDER("trapezoid", kSteps, e_tr, 2.0, 0.15);

    auto e_ros = run([&](Vec& y, double t, double h) {
        ex03::rosenbrock2_step(s, jac, t, y, h);
    });
    NP_CHECK_ORDER("ROS2", kSteps, e_ros, 2.0, 0.2);

    // BDF2 needs a start-up value; use the exact one so the test isolates BDF2.
    std::vector<double> e_bdf;
    for (double h : kSteps) {
        Vec yprev{1.0}, y{s.exact(h)};
        long n = long(std::lround(T / h));
        for (long i = 1; i < n; ++i) {
            Vec keep = y;
            ex03::bdf2_step(s, jac, double(i) * h, yprev, y, h);
            yprev = keep;
        }
        e_bdf.push_back(std::fabs(y[0] - s.exact(T)));
    }
    NP_CHECK_ORDER("BDF2", kSteps, e_bdf, 2.0, 0.2);
}

NP_TEST(ex03_A_stability_and_L_stability) {
    // One step of y' = lam y with h*lam = -100. The amplification factor R(z)
    // tells you everything:
    //   forward Euler   R = 1 + z          = -99      (explodes)
    //   backward Euler  R = 1/(1-z)        = 0.00990  (L-stable: kills it)
    //   trapezoid       R = (1+z/2)/(1-z/2)= -0.9608  (A- but not L-stable:
    //                                                  rings with alternating sign)
    ExpDecay f{-1000.0};
    auto jac = [&](double t, const Vec& y, Mat& J) { f.jac(t, y, J); };
    const double h = 0.1;

    Vec y{1.0};
    ex01::euler_step(f, 0.0, y, h);
    NP_CLOSE(y[0], -99.0, 1e-9);

    y = {1.0};
    ex03::backward_euler_step(f, jac, 0.0, y, h);
    NP_CLOSE(y[0], 1.0 / 101.0, 1e-10);

    y = {1.0};
    ex03::trapezoid_step(f, jac, 0.0, y, h);
    NP_CLOSE(y[0], -49.0 / 51.0, 1e-10);

    y = {1.0};
    ex03::rosenbrock2_step(f, jac, 0.0, y, h);
    NP_CHECK_MSG(std::fabs(y[0]) < 0.05, "ROS2 is not L-stable here: R = %.4f",
                 y[0]);

    // The consequence: over 200 steps the trapezoidal transient is still ~2%
    // of its initial size while backward Euler annihilated it immediately.
    Vec ytr{1.0}, ybe{1.0};
    for (int i = 0; i < 200; ++i) {
        ex03::trapezoid_step(f, jac, double(i) * h, ytr, h);
        ex03::backward_euler_step(f, jac, double(i) * h, ybe, h);
    }
    std::printf("      after 200 steps: trapezoid %.3e, backward Euler %.3e\n",
                ytr[0], ybe[0]);
    NP_CHECK_MSG(std::fabs(ytr[0]) > 1e-4,
                 "trapezoid should still be ringing (got %.3e)", ytr[0]);
    // (backward Euler bottoms out at the Newton residual tolerance, not zero)
    NP_CHECK_MSG(std::fabs(ybe[0]) < 1e-10,
                 "backward Euler should have annihilated the mode (got %.3e)",
                 ybe[0]);
}

NP_TEST(ex03_stiff_system_accuracy) {
    // Two time scales, 1 and 1e4. Step at h = 1e-2 (h*lam_fast = -100):
    // an explicit method is unusable, an L-stable one is accurate on the slow
    // mode.  Initial data c1 = c2 = 1.
    Stiff2 sys{1e4};
    auto jac = [&](double t, const Vec& y, Mat& J) { sys.jac(t, y, J); };
    const double T = 1.0, h = 1e-2;
    const long n = long(std::lround(T / h));
    Vec ex(2);
    sys.exact(T, 1.0, 1.0, ex);

    Vec y0{2.0, 1.0};
    Vec ye = y0;
    for (long i = 0; i < n; ++i) ex01::euler_step(sys, double(i) * h, ye, h);
    NP_CHECK_MSG(!std::isfinite(ye[0]) || std::fabs(ye[0]) > 1e10,
                 "forward Euler should blow up at h*lam = -100, got %.3e",
                 ye[0]);

    Vec ybe = y0;
    for (long i = 0; i < n; ++i)
        ex03::backward_euler_step(sys, jac, double(i) * h, ybe, h);
    double e_be = err_inf(ybe, ex);

    Vec yros = y0;
    for (long i = 0; i < n; ++i)
        ex03::rosenbrock2_step(sys, jac, double(i) * h, yros, h);
    double e_ros = err_inf(yros, ex);

    std::printf("      stiff errors: BE %.3e   ROS2 %.3e   (exact y1 = %.6f)\n",
                e_be, e_ros, ex[0]);
    NP_CHECK_MSG(e_be < 5e-3, "backward Euler stiff error %.3e", e_be);
    NP_CHECK_MSG(e_ros < e_be, "ROS2 (2nd order) should beat BE (1st order)");
}

NP_TEST(ex03_van_der_pol_survives) {
    // mu = 100: the classic stiff relaxation oscillator, limit-cycle amplitude
    // exactly 2. An explicit method needs h < 2/mu just to stay alive; backward
    // Euler is stable at any h and converges (slowly, being 1st order and very
    // dissipative) to the right amplitude.
    VanDerPol vdp{100.0};
    auto jac = [&](double t, const Vec& y, Mat& J) { vdp.jac(t, y, J); };

    auto amplitude = [&](auto&& stepper, double h) {
        Vec y{2.0, 0.0};
        double xmax = 0;
        long n = long(std::lround(1000.0 / h));
        for (long i = 0; i < n; ++i) {
            stepper(y, double(i) * h, h);
            if (!std::isfinite(y[0])) return std::nan("");
            if (double(i) * h > 500.0) xmax = std::max(xmax, std::fabs(y[0]));
        }
        return xmax;
    };

    double a_expl = amplitude([&](Vec& y, double t, double h) {
        ex01::euler_step(vdp, t, y, h);
    }, 1e-2);
    NP_CHECK_MSG(!std::isfinite(a_expl),
                 "forward Euler should be unstable at h*mu = 1 (got %.3f)",
                 a_expl);

    double a1 = amplitude([&](Vec& y, double t, double h) {
        ex03::backward_euler_step(vdp, jac, t, y, h, 1e-10, 100);
    }, 1e-2);
    double a2 = amplitude([&](Vec& y, double t, double h) {
        ex03::backward_euler_step(vdp, jac, t, y, h, 1e-10, 100);
    }, 1e-3);
    std::printf("      limit-cycle amplitude: h=1e-2 -> %.4f, h=1e-3 -> %.4f "
                "(exact 2)\n", a1, a2);
    NP_CHECK_MSG(std::isfinite(a1) && a1 > 1.5 && a1 < 2.2,
                 "backward Euler unstable/wrong at h=1e-2: %.4f", a1);
    NP_CHECK_MSG(a2 > 1.90 && a2 < 2.02,
                 "backward Euler at h=1e-3 gave amplitude %.4f, expected ~2", a2);
    NP_CHECK_MSG(a2 > a1, "refining h should approach the true amplitude");
}

NP_TEST(ex03_exponential_euler_is_exact_for_constant_forcing) {
    // y' = A y + g with constant g: the exponential integrator has NO time
    // discretisation error at all, at any h, however stiff A is.
    Mat A(2, 2);
    A(0, 0) = -1.0;  A(0, 1) = 9999.0;
    A(1, 0) = 0.0;   A(1, 1) = -1e4;
    Vec g{3.0, -2.0};
    auto gfun = [&](double, const Vec&, Vec& out) { out = g; };

    // Reference: one big step vs many small ones must agree to roundoff.
    Vec y_one{1.0, 1.0};
    ex03::exp_euler_step(A, gfun, 0.0, y_one, 1.0);

    Vec y_many{1.0, 1.0};
    for (int i = 0; i < 1000; ++i)
        ex03::exp_euler_step(A, gfun, double(i) * 1e-3, y_many, 1e-3);

    std::printf("      one step %.12f  vs  1000 steps %.12f\n", y_one[0],
                y_many[0]);
    NP_CLOSE(y_one[0], y_many[0], 1e-9);
    NP_CLOSE(y_one[1], y_many[1], 1e-9);

    // And it agrees with the analytic steady state as h -> inf.
    Vec y_inf{1.0, 1.0};
    ex03::exp_euler_step(A, gfun, 0.0, y_inf, 50.0);
    Vec ss = g;                      // solve A y_ss = -g
    for (auto& v : ss) v = -v;
    lu_solve(A, ss);
    NP_CLOSE(y_inf[0], ss[0], 1e-6);
    NP_CLOSE(y_inf[1], ss[1], 1e-6);
}

NP_MAIN()
