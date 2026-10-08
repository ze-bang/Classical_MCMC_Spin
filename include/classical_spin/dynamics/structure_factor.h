#pragma once
/**
 * structure_factor.h — classical dynamical structure factor estimator.
 *
 *     S^{ab}(q, ω) = (1/2π) ∫ dt e^{iωt} < A^a_q(t) A^b_q(0)* >,
 *     A_q(t)       = N^{-1/2} Σ_i e^{-i q·r_i} S_i(t)      (global-frame components).
 *
 * Estimator for one trajectory sampled at t_k = k Δ, k < n, with window w_k
 * (Hann or none), zero-padded to M >= 2n points so that the transform is the
 * one of the APERIODIC correlation (no wrap-around):
 *
 *     Ã^a(q, ω_m) = Δ Σ_k w_k A^a_q(t_k) e^{iω_m t_k},   ω_m = 2π m / (M Δ),
 *     Ŝ^{ab}(q, ω_m) = Ã^a Ã^b* / (2π T_w),               T_w = Δ Σ_k w_k².
 *
 * By Parseval, Σ_m Δω Ŝ^{ab}(q, ω_m) (Δω = 2π/(MΔ)) equals the window-weighted
 * equal-time correlation Σ_k w_k² A^a A^b* / Σ_k w_k² of the same samples
 * exactly: the frequency sum rule ∫ dω S(q, ω) = S(q) holds by construction
 * and is what the tests check. Averaging Ŝ over independent thermal samples
 * gives S(q, ω) convolved with the window's resolution function (Zhang,
 * Batista et al., PRL 122, 167203 (2019); Sunny.jl SampledCorrelations).
 *
 * Frequencies are returned in increasing order, ω = 2π m / (MΔ) for
 * m = -M/2 .. M/2 - 1.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace classical_spin::dynamics {

inline std::size_t next_power_of_two(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/**
 * In-place radix-2 FFT, x_m <- Σ_k x_k exp(sign · 2πi k m / M), M a power of
 * two (no normalisation).
 */
inline void fft_radix2(std::vector<std::complex<double>>& x, int sign) {
    const std::size_t M = x.size();
    if (M == 0 || (M & (M - 1)) != 0) throw std::invalid_argument("fft_radix2: size must be a power of two");
    for (std::size_t i = 1, j = 0; i < M; ++i) {      // bit reversal
        std::size_t bit = M >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (std::size_t len = 2; len <= M; len <<= 1) {
        const double ang = sign * 2.0 * M_PI / double(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < M; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = x[i + k], v = x[i + k + len / 2] * w;
                x[i + k] = u + v;
                x[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

class DSSFAccumulator {
public:
    using cplx = std::complex<double>;

    /**
     * @param n_q   number of wave vectors
     * @param n_t   samples per trajectory (>= 2)
     * @param dt    sample spacing Δ
     * @param hann  Hann window (reduces spectral leakage) instead of none
     */
    DSSFAccumulator(std::size_t n_q, std::size_t n_t, double dt, bool hann)
        : n_q_(n_q), n_t_(n_t), dt_(dt), M_(next_power_of_two(2 * n_t)), w_(n_t, 1.0) {
        if (n_q == 0 || n_t < 2 || !(dt > 0.0))
            throw std::invalid_argument("DSSFAccumulator: need n_q >= 1, n_t >= 2 and dt > 0");
        if (hann)
            for (std::size_t k = 0; k < n_t; ++k) w_[k] = 0.5 * (1.0 - std::cos(2.0 * M_PI * k / double(n_t - 1)));
        double w2 = 0.0;
        for (double w : w_) w2 += w * w;
        w2_sum_ = w2;
        sum_.assign(n_q_ * M_ * 9, cplx(0.0));
        sum2_.assign(n_q_ * M_ * 9, 0.0);
        static_.assign(n_q_ * 9, cplx(0.0));
    }

    std::size_t n_q() const { return n_q_; }
    std::size_t n_t() const { return n_t_; }
    std::size_t n_omega() const { return M_; }
    std::size_t samples() const { return samples_; }
    double dt() const { return dt_; }
    double d_omega() const { return 2.0 * M_PI / (double(M_) * dt_); }
    /// ω of output index j (increasing): 2π (j - M/2) / (M Δ).
    double omega(std::size_t j) const { return (double(j) - double(M_ / 2)) * d_omega(); }

    /**
     * Add one trajectory: A[(q * n_t + k) * 3 + a] = A^a_q(t_k).
     */
    void add_sample(const std::vector<cplx>& A) {
        if (A.size() != n_q_ * n_t_ * 3) throw std::invalid_argument("DSSFAccumulator::add_sample: wrong size");
        const double norm = 1.0 / (2.0 * M_PI * dt_ * w2_sum_);
        std::vector<cplx> F[3];
        for (std::size_t q = 0; q < n_q_; ++q) {
            for (int a = 0; a < 3; ++a) {
                F[a].assign(M_, cplx(0.0));
                for (std::size_t k = 0; k < n_t_; ++k) F[a][k] = dt_ * w_[k] * A[(q * n_t_ + k) * 3 + a];
                fft_radix2(F[a], +1);   // e^{+iωt}
            }
            for (std::size_t m = 0; m < M_; ++m) {
                const std::size_t j = (m + M_ / 2) % M_;   // shift to increasing ω
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        const cplx s = F[a][m] * std::conj(F[b][m]) * norm;
                        const std::size_t idx = ((q * M_ + j) * 3 + a) * 3 + b;
                        sum_[idx] += s;
                        sum2_[idx] += s.real() * s.real();
                    }
            }
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    cplx c(0.0);
                    for (std::size_t k = 0; k < n_t_; ++k)
                        c += w_[k] * w_[k] * A[(q * n_t_ + k) * 3 + a] * std::conj(A[(q * n_t_ + k) * 3 + b]);
                    static_[(q * 3 + a) * 3 + b] += c / w2_sum_;
                }
        }
        ++samples_;
    }

    /// Sample mean of S^{ab}(q, ω_j).
    cplx S(std::size_t q, std::size_t j, int a, int b) const {
        return sum_[((q * M_ + j) * 3 + a) * 3 + b] / double(samples_);
    }
    /// Standard error of Re S^{ab}(q, ω_j) over the samples (0 for one sample).
    double S_err(std::size_t q, std::size_t j, int a, int b) const {
        if (samples_ < 2) return 0.0;
        const std::size_t idx = ((q * M_ + j) * 3 + a) * 3 + b;
        const double n = double(samples_), mean = sum_[idx].real() / n;
        const double var = std::max(0.0, (sum2_[idx] / n - mean * mean) * n / (n - 1.0));
        return std::sqrt(var / n);
    }
    /// Sample mean of the window-weighted equal-time correlation <A^a_q A^b_q*>.
    cplx S_static(std::size_t q, int a, int b) const {
        return static_[(q * 3 + a) * 3 + b] / double(samples_);
    }

private:
    std::size_t n_q_, n_t_;
    double dt_;
    std::size_t M_;
    std::vector<double> w_;
    double w2_sum_ = 0.0;
    std::size_t samples_ = 0;
    std::vector<cplx> sum_;
    std::vector<double> sum2_;
    std::vector<cplx> static_;
};

}  // namespace classical_spin::dynamics
