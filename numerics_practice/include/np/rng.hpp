// np/rng.hpp -- reproducible RNG, running statistics, and the deterministic
// "probe" machinery used to test stochastic integrators without Monte Carlo
// noise.  Provided infrastructure: you never need to edit this file.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "np/linalg.hpp"

namespace np {

// PCG32.  Deterministic across platforms/compilers, unlike <random>'s
// distributions -- which is what makes the statistical tests reproducible.
struct Rng {
    std::uint64_t state = 0, inc = 1;
    bool have_cached = false;
    double cached = 0.0;

    explicit Rng(std::uint64_t seed = 42u, std::uint64_t seq = 54u) {
        inc = (seq << 1u) | 1u;
        next_u32();
        state += seed;
        next_u32();
    }
    std::uint32_t next_u32() {
        std::uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        auto xs = std::uint32_t(((old >> 18u) ^ old) >> 27u);
        auto rot = std::uint32_t(old >> 59u);
        return (xs >> rot) | (xs << ((~rot + 1u) & 31u));
    }
    // Uniform on (0,1).
    double uniform() { return (next_u32() + 0.5) * (1.0 / 4294967296.0); }
    // Standard normal (Box-Muller, cached pair).
    double normal() {
        if (have_cached) { have_cached = false; return cached; }
        double u1 = uniform(), u2 = uniform();
        double r = std::sqrt(-2.0 * std::log(u1)), th = 2.0 * pi * u2;
        cached = r * std::sin(th);
        have_cached = true;
        return r * std::cos(th);
    }
};

// An RNG that hands back a prescribed sequence (0 once exhausted).  Used to
// turn a stochastic step into a deterministic affine map for exact testing.
struct ScriptedRng {
    std::vector<double> vals;
    std::size_t i = 0;
    double normal() { return i < vals.size() ? vals[i++] : 0.0; }
    double uniform() { return 0.5; }
};

// Counts how many normal() draws one step consumes.
struct CountingRng {
    Rng g;
    long count = 0;
    double normal() { ++count; return g.normal(); }
    double uniform() { return g.uniform(); }
};

// Welford running mean/variance.
struct Stats {
    long n = 0;
    double mean = 0, m2 = 0;
    void add(double x) {
        ++n;
        double d = x - mean;
        mean += d / double(n);
        m2 += d * (x - mean);
    }
    double var() const { return n > 1 ? m2 / double(n - 1) : 0.0; }
    double sd() const { return std::sqrt(var()); }
    double sem() const { return n > 0 ? sd() / std::sqrt(double(n)) : 0.0; }
};

// ---------------------------------------------------------------------------
// Exact stationary distribution of a *linear* stochastic integrator.
//
// Any of the Langevin schemes here, applied to a harmonic oscillator, is an
// affine-Gaussian map   z_{n+1} = M z_n + b + L xi,   xi ~ N(0, I).
// We recover M, b, L by calling the step with scripted noise, then solve the
// discrete Lyapunov equation for the exact stationary covariance
//     Sigma = M Sigma M^T + L L^T.
// No Monte Carlo, no statistical error: the sampling bias of a scheme becomes
// a deterministic number you can fit a convergence order to.
//
// `step` has signature  void(Vec& z, Rng& g)  for state dimension `dim`.
// ---------------------------------------------------------------------------
template <class Step>
struct LinearSdeMap {
    Mat M, LL;   // LL = L L^T (the noise covariance per step)
    Vec b;
    int nnoise = 0;
};

template <class Step>
LinearSdeMap<Step> probe_linear_sde(Step&& step, int dim) {
    LinearSdeMap<Step> out;
    // How many normals does one step consume?
    {
        CountingRng c;
        Vec z(dim, 0.0);
        step(z, c);
        out.nnoise = int(c.count);
    }
    auto apply = [&](Vec z, std::vector<double> xi) {
        ScriptedRng g{std::move(xi), 0};
        step(z, g);
        return z;
    };
    const std::vector<double> zero(out.nnoise, 0.0);
    out.b = apply(Vec(dim, 0.0), zero);
    out.M = Mat(dim, dim);
    for (int j = 0; j < dim; ++j) {
        Vec e(dim, 0.0);
        e[j] = 1.0;
        Vec r = apply(e, zero);
        for (int i = 0; i < dim; ++i) out.M(i, j) = r[i] - out.b[i];
    }
    Mat L(dim, out.nnoise ? out.nnoise : 1);
    for (int k = 0; k < out.nnoise; ++k) {
        std::vector<double> xi(out.nnoise, 0.0);
        xi[k] = 1.0;
        Vec r = apply(Vec(dim, 0.0), xi);
        for (int i = 0; i < dim; ++i) L(i, k) = r[i] - out.b[i];
    }
    out.LL = out.nnoise ? matmul(L, transpose(L)) : Mat(dim, dim);
    return out;
}

// Exact stationary covariance of the affine-Gaussian map produced by `step`.
template <class Step>
Mat stationary_covariance(Step&& step, int dim) {
    auto p = probe_linear_sde(step, dim);
    return dlyap(p.M, p.LL);
}

}  // namespace np
