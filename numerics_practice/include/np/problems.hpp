// np/problems.hpp -- shared reference problems with known exact solutions or
// exact invariants.  Provided infrastructure: you never need to edit this.
#pragma once
#include <cmath>

#include "np/linalg.hpp"

namespace np::problems {

// ------------------------------------------------------------------ y' = L y
struct ExpDecay {
    double lam = -1.5;
    void operator()(double, const Vec& y, Vec& dy) const { dy[0] = lam * y[0]; }
    void jac(double, const Vec&, Mat& J) const { J(0, 0) = lam; }
    double exact(double t, double y0) const { return y0 * std::exp(lam * t); }
};

// -------------------------------------------- y' = y cos t,  y = exp(sin t)
struct SmoothNonAutonomous {
    void operator()(double t, const Vec& y, Vec& dy) const {
        dy[0] = y[0] * std::cos(t);
    }
    double exact(double t, double y0) const {
        return y0 * std::exp(std::sin(t));
    }
};

// ------------------------------------------------- harmonic oscillator (x,v)
struct Harmonic {
    double w = 1.0;
    void operator()(double, const Vec& y, Vec& dy) const {
        dy[0] = y[1];
        dy[1] = -w * w * y[0];
    }
    void exact(double t, double x0, double v0, Vec& y) const {
        y[0] = x0 * std::cos(w * t) + (v0 / w) * std::sin(w * t);
        y[1] = -x0 * w * std::sin(w * t) + v0 * std::cos(w * t);
    }
    double energy(const Vec& y) const {
        return 0.5 * y[1] * y[1] + 0.5 * w * w * y[0] * y[0];
    }
};

// ------------------------------- Kepler two-body, y = (x, y, vx, vy), GM = 1
struct Kepler {
    void operator()(double, const Vec& y, Vec& dy) const {
        double r2 = y[0] * y[0] + y[1] * y[1];
        double r3 = r2 * std::sqrt(r2);
        dy[0] = y[2];
        dy[1] = y[3];
        dy[2] = -y[0] / r3;
        dy[3] = -y[1] / r3;
    }
    static double energy(const Vec& y) {
        return 0.5 * (y[2] * y[2] + y[3] * y[3]) -
               1.0 / std::hypot(y[0], y[1]);
    }
    static double angular_momentum(const Vec& y) {
        return y[0] * y[3] - y[1] * y[2];
    }
    // Elliptic orbit of eccentricity e, semi-major axis 1, started at perihelion.
    static Vec init(double e) {
        return {1.0 - e, 0.0, 0.0, std::sqrt((1.0 + e) / (1.0 - e))};
    }
    static double period() { return 2.0 * pi; }
};

// ------------------------- stiff linear 2x2, eigenvalues -1 and -lam (lam>>1)
// A = V diag(-1, -lam) V^{-1} with V = [[1,1],[0,1]] = [[-1, 1-lam],[0, -lam]].
struct Stiff2 {
    double lam = 1000.0;
    void operator()(double, const Vec& y, Vec& dy) const {
        dy[0] = -1.0 * y[0] + (1.0 - lam) * y[1];
        dy[1] = -lam * y[1];
    }
    void jac(double, const Vec&, Mat& J) const {
        J(0, 0) = -1.0; J(0, 1) = 1.0 - lam;
        J(1, 0) = 0.0;  J(1, 1) = -lam;
    }
    // y(0) = (c1 + c2, c2)
    void exact(double t, double c1, double c2, Vec& y) const {
        y[0] = c1 * std::exp(-t) + c2 * std::exp(-lam * t);
        y[1] = c2 * std::exp(-lam * t);
    }
};

// ----------------------------------------------------------- Van der Pol
struct VanDerPol {
    double mu = 100.0;
    void operator()(double, const Vec& y, Vec& dy) const {
        dy[0] = y[1];
        dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
    void jac(double, const Vec& y, Mat& J) const {
        J(0, 0) = 0.0; J(0, 1) = 1.0;
        J(1, 0) = -2.0 * mu * y[0] * y[1] - 1.0;
        J(1, 1) = mu * (1.0 - y[0] * y[0]);
    }
};

// ------------------------------------------------ 1-D potentials for Langevin
struct HarmonicWell {
    double k = 1.0;
    double energy(double x) const { return 0.5 * k * x * x; }
    double force(double x) const { return -k * x; }
};

struct DoubleWell {
    // U(x) = h * (x^2/a^2 - 1)^2 ; minima at +-a, barrier h at x = 0.
    double h = 5.0, a = 1.0;
    double energy(double x) const {
        double u = x * x / (a * a) - 1.0;
        return h * u * u;
    }
    double force(double x) const {
        double u = x * x / (a * a) - 1.0;
        return -4.0 * h * u * x / (a * a);
    }
};

}  // namespace np::problems
