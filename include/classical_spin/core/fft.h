#pragma once
/**
 * fft.h — complex discrete Fourier transforms of any length, in 1 and 3 dimensions.
 *
 *     X_m = Σ_{j=0}^{n-1} x_j exp(sign · 2πi j m / n),    sign = ±1, no normalisation.
 *
 * Algorithms (all O(n log n) asymptotically):
 *   - n a power of two: iterative radix-2 Cooley–Tukey with a precomputed
 *     twiddle table;
 *   - other n <= 16: direct DFT from an exact table of the n-th roots of unity
 *     (cheaper and more accurate than any factorisation at these sizes);
 *   - every other n: Bluestein's chirp-z algorithm (L. I. Bluestein, IEEE Trans.
 *     Audio Electroacoust. 18, 451 (1970)), which writes the DFT as a cyclic
 *     convolution of length m >= 2n - 1 evaluated with radix-2 FFTs, so prime
 *     lengths cost O(n log n) too. The chirp phase π j²/n is evaluated with
 *     j² reduced modulo 2n so it stays exact for large j.
 *
 * Plans keep mutable scratch buffers: one plan must not execute concurrently
 * from several threads (give each thread its own plan).
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace classical_spin::fft {

using cplx = std::complex<double>;

namespace detail {

inline bool is_power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

inline std::size_t next_power_of_two(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/// exp(2πi k / n) for k = 0 .. n-1, evaluated directly (no recurrence, so no
/// accumulated round-off).
inline std::vector<cplx> roots_of_unity(std::size_t n) {
    std::vector<cplx> w(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double phi = 2.0 * M_PI * double(k) / double(n);
        w[k] = cplx(std::cos(phi), std::sin(phi));
    }
    return w;
}

/// In-place radix-2 transform of a contiguous power-of-two array. `w` holds
/// exp(2πi k / n), k < n/2; sign = -1 uses its conjugates.
inline void radix2(cplx* x, std::size_t n, const std::vector<cplx>& w, int sign) {
    for (std::size_t i = 1, j = 0; i < n; ++i) {  // bit-reversal permutation
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len / 2, stride = n / len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const cplx tw = (sign > 0) ? w[k * stride] : std::conj(w[k * stride]);
                const cplx u = x[i + k], v = x[i + k + half] * tw;
                x[i + k] = u + v;
                x[i + k + half] = u - v;
            }
        }
    }
}

}  // namespace detail

/** One-dimensional transform of a fixed length n >= 1. */
class Plan1D {
public:
    explicit Plan1D(std::size_t n = 1) : n_(n) {
        if (n == 0) throw std::invalid_argument("fft::Plan1D: length must be >= 1");
        if (n == 1) {
            kind_ = Kind::Trivial;
        } else if (detail::is_power_of_two(n)) {
            kind_ = Kind::Radix2;
            w_ = detail::roots_of_unity(n);
            w_.resize(n / 2);
        } else if (n <= 16) {
            kind_ = Kind::Direct;
            w_ = detail::roots_of_unity(n);
        } else {
            kind_ = Kind::Bluestein;
            m_ = detail::next_power_of_two(2 * n - 1);
            w_ = detail::roots_of_unity(m_);
            w_.resize(m_ / 2);
            // Chirp c_j = exp(-iπ j²/n); j² mod 2n keeps the phase exact.
            chirp_.resize(n);
            for (std::size_t j = 0; j < n; ++j) {
                const std::uint64_t j2 = (std::uint64_t(j) * std::uint64_t(j)) % (2 * std::uint64_t(n));
                const double phi = M_PI * double(j2) / double(n);
                chirp_[j] = cplx(std::cos(phi), -std::sin(phi));
            }
            // Convolution kernel b_j = conj(c_j) on the cyclic index range
            // (-n, n), transformed once.
            kernel_.assign(m_, cplx(0.0));
            kernel_[0] = std::conj(chirp_[0]);
            for (std::size_t j = 1; j < n; ++j) kernel_[j] = kernel_[m_ - j] = std::conj(chirp_[j]);
            detail::radix2(kernel_.data(), m_, w_, -1);
        }
    }

    std::size_t size() const { return n_; }

    /** In place on x[0], x[stride], ..., x[(n-1) stride]. */
    void execute(cplx* x, std::size_t stride, int sign) const {
        if (sign != 1 && sign != -1) throw std::invalid_argument("fft::Plan1D: sign must be +1 or -1");
        switch (kind_) {
            case Kind::Trivial:
                return;
            case Kind::Radix2:
                if (stride == 1) {
                    detail::radix2(x, n_, w_, sign);
                } else {
                    gather(x, stride);
                    detail::radix2(line_.data(), n_, w_, sign);
                    scatter(x, stride);
                }
                return;
            case Kind::Direct:
                direct(x, stride, sign);
                return;
            case Kind::Bluestein:
                bluestein(x, stride, sign);
                return;
        }
    }

private:
    enum class Kind { Trivial, Radix2, Direct, Bluestein };

    void gather(const cplx* x, std::size_t stride) const {
        line_.resize(n_);
        for (std::size_t j = 0; j < n_; ++j) line_[j] = x[j * stride];
    }
    void scatter(cplx* x, std::size_t stride) const {
        for (std::size_t j = 0; j < n_; ++j) x[j * stride] = line_[j];
    }

    void direct(cplx* x, std::size_t stride, int sign) const {
        gather(x, stride);
        for (std::size_t m = 0; m < n_; ++m) {
            cplx acc(0.0);
            std::size_t jm = 0;  // (j m) mod n, updated incrementally
            for (std::size_t j = 0; j < n_; ++j) {
                acc += line_[j] * ((sign > 0) ? w_[jm] : std::conj(w_[jm]));
                jm += m;
                if (jm >= n_) jm -= n_;
            }
            x[m * stride] = acc;
        }
    }

    // X_k = c_k Σ_j (x_j c_j) conj(c_{k-j}) with c_j = exp(-iπ j²/n): a cyclic
    // convolution of length m. sign = +1 is the conjugate of the sign = -1
    // transform of conj(x).
    void bluestein(cplx* x, std::size_t stride, int sign) const {
        work_.assign(m_, cplx(0.0));
        for (std::size_t j = 0; j < n_; ++j) {
            const cplx xj = (sign > 0) ? std::conj(x[j * stride]) : x[j * stride];
            work_[j] = xj * chirp_[j];
        }
        detail::radix2(work_.data(), m_, w_, -1);
        for (std::size_t j = 0; j < m_; ++j) work_[j] *= kernel_[j];
        detail::radix2(work_.data(), m_, w_, +1);
        const double inv_m = 1.0 / double(m_);
        for (std::size_t k = 0; k < n_; ++k) {
            const cplx Xk = work_[k] * chirp_[k] * inv_m;
            x[k * stride] = (sign > 0) ? std::conj(Xk) : Xk;
        }
    }

    std::size_t n_ = 1, m_ = 0;
    Kind kind_ = Kind::Trivial;
    std::vector<cplx> w_, chirp_, kernel_;
    mutable std::vector<cplx> line_, work_;
};

/**
 * Three-dimensional transform of a row-major n1 x n2 x n3 array
 * (index (i1 n2 + i2) n3 + i3), applied axis by axis.
 */
class Plan3D {
public:
    Plan3D() = default;
    Plan3D(std::size_t n1, std::size_t n2, std::size_t n3) : n_{n1, n2, n3}, p1_(n1), p2_(n2), p3_(n3) {}

    std::size_t size() const { return n_[0] * n_[1] * n_[2]; }

    void execute(cplx* a, int sign) const {
        const std::size_t n1 = n_[0], n2 = n_[1], n3 = n_[2];
        if (n3 > 1)
            for (std::size_t i = 0; i < n1 * n2; ++i) p3_.execute(a + i * n3, 1, sign);
        if (n2 > 1)
            for (std::size_t i1 = 0; i1 < n1; ++i1)
                for (std::size_t i3 = 0; i3 < n3; ++i3) p2_.execute(a + i1 * n2 * n3 + i3, n3, sign);
        if (n1 > 1)
            for (std::size_t i = 0; i < n2 * n3; ++i) p1_.execute(a + i, n2 * n3, sign);
    }

private:
    std::size_t n_[3] = {1, 1, 1};
    Plan1D p1_, p2_, p3_;
};

}  // namespace classical_spin::fft
