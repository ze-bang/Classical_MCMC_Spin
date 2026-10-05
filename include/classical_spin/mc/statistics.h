#pragma once
/**
 * statistics.h — Monte Carlo error analysis (one implementation for every
 * sampler in the code base: SA final measurements, parallel tempering, the
 * MixedLattice / PhononLattice drivers and the tests).
 *
 *  - binning_analysis      : Flyvbjerg-Petersen blocking with the Wolff / Lee
 *                            plateau criterion (kept for its level table).
 *  - gamma_method          : U. Wolff's Gamma method with automatic windowing
 *                            ("UWerr", Comput. Phys. Commun. 156, 143 (2004))
 *                            for the mean, its error, tau_int and d tau_int of
 *                            a primary observable. The autocovariance is
 *                            computed by FFT in O(N log N); there is no fixed
 *                            lag cap and no early stop at the first negative
 *                            rho (both biased tau_int low before).
 *  - blocked_jackknife     : delete-one-block jackknife for nonlinear
 *                            functions of several primary means (specific
 *                            heat, susceptibility, Binder cumulant, E-M cross
 *                            terms) in O(N) via block sums, with the block
 *                            length chosen from the Gamma-method tau_int.
 *  - compute_thermodynamics: energy / specific heat per site and, for every
 *                            vector order parameter m (per-site normalised),
 *                            <m_a>, <|m|>, <m^2>, chi = beta N (<m^2> - <|m|>^2)
 *                            and U4 = 1 - <m^4> / (3 <m^2>^2), all with errors.
 *
 * Conventions: series are in measurement order (time-ordered); tau_int is in
 * units of the measurement spacing, tau_int = 1/2 for uncorrelated data, and
 * the error of the mean is sqrt(2 tau_int Var / N).
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mc {

// ============================================================
// RESULT TYPES
// ============================================================

struct Observable {
    double value = 0.0;
    double error = 0.0;
    Observable(double v = 0.0, double e = 0.0) : value(v), error(e) {}
};

struct VectorObservable {
    std::vector<double> values;
    std::vector<double> errors;
    VectorObservable() = default;
    explicit VectorObservable(size_t dim) : values(dim, 0.0), errors(dim, 0.0) {}
};

struct BinningResult {
    double mean = 0.0;
    double error = 0.0;
    double tau_int = 1.0;
    size_t optimal_bin_level = 0;
    std::vector<double> errors_by_level;
};

/// Output of the Gamma method for one primary observable.
struct GammaResult {
    size_t n = 0;                 ///< number of samples
    double mean = 0.0;
    double error = 0.0;           ///< error of the mean including autocorrelations
    double error_of_error = 0.0;  ///< statistical uncertainty of `error`
    double variance = 0.0;        ///< Gamma(0) (bias corrected)
    double tau_int = 0.5;         ///< integrated autocorrelation time (samples)
    double tau_int_error = 0.0;
    size_t window = 0;            ///< summation window W chosen automatically
    bool reliable = false;        ///< window found and N >= 50 tau_int
};

struct JackknifeResult {
    double value = 0.0;           ///< estimator on the full sample
    double bias_corrected = 0.0;  ///< B value - (B - 1) mean of the leave-one-out values
    double error = 0.0;
    size_t n_blocks = 0;
    size_t block_length = 0;
};

/// Time series of one vector order parameter (per-site normalised, e.g.
/// m = sum_i S_i / N), stored row-major: data[s * dim + a].
struct OrderParameterSeries {
    std::string name;
    size_t dim = 0;
    size_t n_sites = 1;  ///< N entering chi = beta N Var(|m|)
    std::vector<double> data;
    size_t n_samples() const { return dim ? data.size() / dim : 0; }
};

/// Statistics of one vector order parameter m at temperature T.
struct OrderParameterStats {
    std::string name;
    size_t dim = 0;
    size_t n_sites = 1;
    VectorObservable mean;       ///< <m_a>
    Observable abs;              ///< <|m|>
    Observable m2;               ///< <m^2>
    Observable susceptibility;   ///< beta N (<m^2> - <|m|>^2)
    Observable binder;           ///< 1 - <m^4> / (3 <m^2>^2)
    double tau_int_abs = 0.5;    ///< tau_int of |m| (samples)
};

struct ThermodynamicObservables {
    double temperature = 0.0;
    size_t n_samples = 0;
    Observable energy;                 ///< <E>/N
    double energy_tau_int = 0.5;       ///< tau_int of E (samples)
    Observable specific_heat;          ///< Var(E) / (N T^2)
    VectorObservable magnetization;
    std::vector<VectorObservable> sublattice_magnetization;
    std::vector<VectorObservable> energy_sublattice_cross;
    std::vector<OrderParameterStats> order_parameters;
};

// ============================================================
// HELPERS
// ============================================================

namespace detail {

/// Two-pass (compensated) mean.
inline double mean(const double* x, size_t n) {
    if (n == 0) return 0.0;
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m += x[i];
    m /= double(n);
    double r = 0.0;
    for (size_t i = 0; i < n; ++i) r += x[i] - m;
    return m + r / double(n);
}

inline size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/// exp(-2 pi i k / n) for k < n/2 (evaluated directly, no recurrence, so the
/// round-off of the transforms stays at O(eps log n)).
inline void twiddles(size_t n, std::vector<double>& c, std::vector<double>& s) {
    c.resize(n / 2);
    s.resize(n / 2);
    for (size_t k = 0; k < n / 2; ++k) {
        const double a = 2.0 * M_PI * double(k) / double(n);
        c[k] = std::cos(a);
        s[k] = -std::sin(a);
    }
}

/// Forward radix-2 decimation-in-frequency FFT (size a power of two):
/// natural-order input, bit-reversed output. Split real/imaginary arrays and
/// written-out complex products (std::complex multiplication is IEEE-checked
/// and slow without -ffast-math).
inline void fft_dif(std::vector<double>& re, std::vector<double>& im,
                    const std::vector<double>& c, const std::vector<double>& s) {
    const size_t n = re.size();
    for (size_t len = n; len >= 2; len >>= 1) {
        const size_t half = len / 2, step = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half; ++j) {
                const size_t p = i + j, q = p + half;
                const double ur = re[p], ui = im[p], vr = re[q], vi = im[q];
                re[p] = ur + vr;
                im[p] = ui + vi;
                const double dr = ur - vr, di = ui - vi, wr = c[j * step], wi = s[j * step];
                re[q] = dr * wr - di * wi;
                im[q] = dr * wi + di * wr;
            }
        }
    }
}

/// Inverse radix-2 decimation-in-time FFT: bit-reversed input (as produced by
/// fft_dif), natural-order output, scaled by 1/n. Pairing the two avoids the
/// bit-reversal permutation entirely when only pointwise products are taken
/// in between (as for the autocovariance).
inline void ifft_dit(std::vector<double>& re, std::vector<double>& im,
                     const std::vector<double>& c, const std::vector<double>& s) {
    const size_t n = re.size();
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t half = len / 2, step = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half; ++j) {
                const size_t p = i + j, q = p + half;
                const double wr = c[j * step], wi = -s[j * step];  // conjugate twiddle
                const double tr = re[q] * wr - im[q] * wi, ti = re[q] * wi + im[q] * wr;
                re[q] = re[p] - tr;
                im[q] = im[p] - ti;
                re[p] += tr;
                im[p] += ti;
            }
        }
    }
    const double inv = 1.0 / double(n);
    for (size_t k = 0; k < n; ++k) { re[k] *= inv; im[k] *= inv; }
}

}  // namespace detail

/**
 * Autocovariance Gamma(t) = 1/(N-t) sum_i (x_i - xbar)(x_{i+t} - xbar) for
 * t = 0..max_lag (max_lag < N), by zero-padded FFT for long series and by
 * direct summation for short ones.
 */
inline std::vector<double> autocovariance(const std::vector<double>& x, size_t max_lag) {
    const size_t n = x.size();
    if (n == 0) return {};
    max_lag = std::min(max_lag, n - 1);
    const double m = detail::mean(x.data(), n);
    std::vector<double> gamma(max_lag + 1, 0.0);
    if (n < 128 || max_lag <= 64) {  // direct sum is cheaper for few lags
        for (size_t t = 0; t <= max_lag; ++t) {
            double s = 0.0;
            for (size_t i = 0; i + t < n; ++i) s += (x[i] - m) * (x[i + t] - m);
            gamma[t] = s / double(n - t);
        }
        return gamma;
    }
    // Wiener-Khinchin with zero padding to >= 2N (no circular wrap-around).
    const size_t nfft = detail::next_pow2(2 * n);
    std::vector<double> re(nfft, 0.0), im(nfft, 0.0), c, sn;
    for (size_t i = 0; i < n; ++i) re[i] = x[i] - m;
    detail::twiddles(nfft, c, sn);
    detail::fft_dif(re, im, c, sn);
    for (size_t k = 0; k < nfft; ++k) { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0.0; }
    detail::ifft_dit(re, im, c, sn);
    for (size_t t = 0; t <= max_lag; ++t) gamma[t] = re[t] / double(n - t);
    return gamma;
}

/**
 * Gamma method for a primary observable (Wolff 2004, Sec. 3.3, S_tau = S).
 * If `rho` is given it receives the normalised autocorrelation function
 * rho(t) = Gamma(t)/Gamma(0) for t up to ~4 W.
 *
 * The window W is the first W with g(W) = exp(-W/tau_W) - tau_W/sqrt(W N) < 0,
 * tau_W = S / ln((2 tau_int(W) + 1) / (2 tau_int(W) - 1)), balancing the
 * truncation bias exp(-W/tau) against the statistical error sqrt(W/N) of the
 * summed autocorrelations. The O(tau/N) bias of Gamma(t) from using the
 * sample mean is then removed (Gamma(t) += C_F/N) and
 *   sigma^2 = C_F / N,  C_F = Gamma(0) + 2 sum_{t=1}^{W} Gamma(t),
 *   tau_int = C_F / (2 Gamma(0)),  d tau_int = 2 tau_int sqrt((W - tau_int + 1/2)/N).
 */
inline GammaResult gamma_method(const std::vector<double>& x, double S = 1.5,
                                std::vector<double>* rho = nullptr) {
    GammaResult r;
    r.n = x.size();
    if (r.n == 0) return r;
    r.mean = detail::mean(x.data(), r.n);
    if (r.n < 2) return r;
    const size_t n = r.n;
    const size_t w_max = n / 2;
    std::vector<double> gamma = autocovariance(x, std::max<size_t>(w_max, 1));
    if (!(gamma[0] > 0.0) || !std::isfinite(gamma[0])) {
        // Constant series: no fluctuations, exactly known mean.
        r.variance = 0.0;
        r.reliable = std::isfinite(gamma[0]);
        return r;
    }
    const double nd = double(n);
    size_t W = 0;
    bool found = false;
    double tau_w_sum = 0.5;
    for (size_t w = 1; w <= w_max; ++w) {
        tau_w_sum += gamma[w] / gamma[0];
        double tau;
        if (tau_w_sum <= 0.5) {
            tau = std::numeric_limits<double>::min();
        } else {
            tau = S / std::log((2.0 * tau_w_sum + 1.0) / (2.0 * tau_w_sum - 1.0));
        }
        const double g = std::exp(-double(w) / tau) - tau / std::sqrt(double(w) * nd);
        if (g < 0.0) { W = w; found = true; break; }
    }
    if (!found) W = std::max<size_t>(w_max, 1);
    if (W >= gamma.size()) W = gamma.size() - 1;
    if (rho) {  // normalised autocorrelation up to a few windows (before bias correction)
        const size_t L = std::min(gamma.size() - 1, std::max<size_t>(4 * W, 10));
        rho->assign(gamma.begin(), gamma.begin() + long(L) + 1);
        for (double& v : *rho) v /= gamma[0];
    }

    double cf = gamma[0];
    for (size_t t = 1; t <= W; ++t) cf += 2.0 * gamma[t];
    const double bias = cf / nd;  // Wolff eq. (49): <Gamma(t)> is shifted by -C_F/N
    for (size_t t = 0; t <= W; ++t) gamma[t] += bias;
    cf = gamma[0];
    for (size_t t = 1; t <= W; ++t) cf += 2.0 * gamma[t];
    cf = std::max(cf, 0.0);

    r.variance = gamma[0];
    r.window = W;
    r.error = std::sqrt(cf / nd);
    r.tau_int = std::max(0.5 * cf / gamma[0], 0.0);
    r.tau_int_error = 2.0 * r.tau_int * std::sqrt(std::max(double(W) - r.tau_int + 0.5, 0.0) / nd);
    r.error_of_error = r.error * std::sqrt((double(W) + 0.5) / nd);
    r.reliable = found && nd >= 50.0 * std::max(r.tau_int, 0.5);
    return r;
}

/// Error of the mean of a (correlated) series via the Gamma method.
inline Observable mean_with_error(const std::vector<double>& x) {
    const GammaResult g = gamma_method(x);
    return Observable(g.mean, g.error);
}

/**
 * Jackknife block length for data with integrated autocorrelation time tau:
 * blocks of ~8 tau make the block means nearly independent (the residual
 * downward bias of the variance is ~tau_exp / b, about 10%), while keeping
 * at least `min_blocks` blocks when the series is long enough.
 */
inline size_t jackknife_block_length(size_t n, double tau_int, size_t min_blocks = 16) {
    if (n < 2) return 1;
    size_t b = size_t(std::ceil(8.0 * std::max(tau_int, 0.5)));
    const size_t b_cap = std::max<size_t>(1, n / std::max<size_t>(min_blocks, 2));
    return std::max<size_t>(1, std::min(b, b_cap));
}

/**
 * Delete-one-block jackknife for F(<a_1>, ..., <a_k>), a nonlinear function
 * of the means of k primary series of equal length n. The n samples are cut
 * into B = n / block_length contiguous blocks whose sizes differ by at most
 * one (every sample is used). Leave-one-out means come from block sums, so
 * the cost is O(k n + k B). Each series is shifted by its first element
 * before summing, so large common offsets do not cancel catastrophically;
 * the estimator receives the true means.
 *
 *   error^2 = (B-1)/B sum_b (F_b - F_bar)^2,  bias corrected = B F - (B-1) F_bar.
 */
template <class Estimator>
JackknifeResult blocked_jackknife(const std::vector<const std::vector<double>*>& series,
                                  Estimator&& estimator, size_t block_length) {
    JackknifeResult r;
    const size_t k = series.size();
    if (k == 0) throw std::invalid_argument("blocked_jackknife: no series");
    const size_t n = series[0]->size();
    for (const auto* s : series)
        if (s->size() != n) throw std::invalid_argument("blocked_jackknife: series lengths differ");
    std::vector<double> means(k, 0.0);
    if (n == 0) return r;
    std::vector<double> shift(k), total(k, 0.0);
    for (size_t a = 0; a < k; ++a) shift[a] = (*series[a])[0];

    const size_t B = std::max<size_t>(1, std::min(n, n / std::max<size_t>(block_length, 1)));
    std::vector<double> block_sum(B * k, 0.0);
    std::vector<size_t> block_n(B, 0);
    for (size_t b = 0; b < B; ++b) {
        const size_t lo = b * n / B, hi = (b + 1) * n / B;
        block_n[b] = hi - lo;
        for (size_t a = 0; a < k; ++a) {
            const std::vector<double>& x = *series[a];
            double s = 0.0;
            for (size_t i = lo; i < hi; ++i) s += x[i] - shift[a];
            block_sum[b * k + a] = s;
            total[a] += s;
        }
    }
    for (size_t a = 0; a < k; ++a) means[a] = shift[a] + total[a] / double(n);
    r.value = estimator(static_cast<const double*>(means.data()));
    r.n_blocks = B;
    r.block_length = n / B;
    if (B < 2) {
        r.bias_corrected = r.value;
        return r;
    }
    std::vector<double> loo(B);
    std::vector<double> m(k);
    for (size_t b = 0; b < B; ++b) {
        const double nb = double(n - block_n[b]);
        for (size_t a = 0; a < k; ++a) m[a] = shift[a] + (total[a] - block_sum[b * k + a]) / nb;
        loo[b] = estimator(static_cast<const double*>(m.data()));
    }
    const double fbar = detail::mean(loo.data(), B);
    double var = 0.0;
    for (double f : loo) var += (f - fbar) * (f - fbar);
    var *= double(B - 1) / double(B);
    r.error = std::sqrt(std::max(var, 0.0));
    r.bias_corrected = double(B) * r.value - double(B - 1) * fbar;
    return r;
}

// ============================================================
// BINNING (Flyvbjerg-Petersen)
// ============================================================

/**
 * Binning analysis (Flyvbjerg-Petersen blocking) for the standard error of
 * the mean of a correlated time series, with the integrated autocorrelation
 * time tau_int = (err_l / err_0)^2 / 2 read off the chosen level.
 *
 * Level choice: the smallest block length B = 2^l satisfying the
 * Wolff / Lee et al. criterion  B^3 >= 2 n (err_l / err_0)^4 , i.e. blocks
 * long compared with the autocorrelation time (B^3 > 8 n tau^2) while
 * keeping as many blocks as possible. (Taking the maximum error over levels,
 * as was done previously, is biased upward by the noise of the few-block
 * levels.) If no level qualifies, the deepest level with at least 16 blocks
 * is used and the error is a lower bound.
 *
 * Variances are accumulated on mean-shifted data so that a large common
 * offset (total energies of big lattices) does not cancel catastrophically.
 */
inline BinningResult binning_analysis(const std::vector<double>& data) {
    BinningResult result;
    if (data.empty()) return result;

    const size_t n = data.size();
    result.mean = detail::mean(data.data(), n);

    std::vector<double> binned(n);
    for (size_t i = 0; i < n; ++i) binned[i] = data[i] - result.mean;

    if (n < 4) {
        double var = 0.0;
        for (double x : binned) var += x * x;
        result.error = (n > 1) ? std::sqrt(var / (double(n) * double(n - 1))) : 0.0;
        return result;
    }

    while (binned.size() >= 4) {
        const size_t m = binned.size();
        double s = 0.0;
        for (double x : binned) s += x;
        const double mean_l = s / double(m);
        double var_l = 0.0;
        for (double x : binned) var_l += (x - mean_l) * (x - mean_l);
        var_l /= double(m);
        result.errors_by_level.push_back(std::sqrt(var_l / double(m - 1)));

        std::vector<double> next(m / 2);
        for (size_t i = 0; i < m / 2; ++i)
            next[i] = 0.5 * (binned[2 * i] + binned[2 * i + 1]);
        binned = std::move(next);
    }

    const size_t n_levels = result.errors_by_level.size();
    const double err0 = result.errors_by_level[0];
    size_t chosen = n_levels;
    if (err0 > 0.0) {
        for (size_t l = 0; l < n_levels; ++l) {
            const double B = std::ldexp(1.0, int(l));
            const double r2 = (result.errors_by_level[l] / err0) * (result.errors_by_level[l] / err0);
            if (B * B * B >= 2.0 * double(n) * r2 * r2) { chosen = l; break; }
        }
    }
    if (chosen == n_levels) {
        chosen = 0;
        for (size_t l = 0; l < n_levels; ++l)
            if ((n >> l) >= 16) chosen = l;
    }
    result.optimal_bin_level = chosen;
    result.error = result.errors_by_level[chosen];
    if (err0 > 0.0) {
        const double ratio = result.error / err0;
        result.tau_int = std::max(0.5, 0.5 * ratio * ratio);
    } else {
        result.tau_int = 0.5;
    }
    return result;
}

/// Component-wise binning analysis for vector observables.
template <typename SpinVec>
inline std::vector<BinningResult> binning_analysis_vector(const std::vector<SpinVec>& data) {
    if (data.empty()) return {};
    const size_t dim = data[0].size();
    std::vector<BinningResult> results(dim);
    std::vector<double> comp(data.size());
    for (size_t d = 0; d < dim; ++d) {
        for (size_t i = 0; i < data.size(); ++i) comp[i] = data[i](d);
        results[d] = binning_analysis(comp);
    }
    return results;
}

// ============================================================
// THERMODYNAMIC ESTIMATORS
// ============================================================

/**
 * Jackknife estimate of N Var(x) / T^2-type quantities: returns
 * scale * Var(x) with Var the (biased, 1/n) sample variance, computed from
 * the centred primaries x - xbar and (x - xbar)^2 so that large offsets do
 * not cancel. The block length follows the larger tau_int of the two.
 */
inline Observable scaled_variance(const std::vector<double>& x, double scale) {
    const size_t n = x.size();
    if (n < 2) return Observable(0.0, 0.0);
    const double c = detail::mean(x.data(), n);
    std::vector<double> d(n), d2(n);
    for (size_t i = 0; i < n; ++i) { d[i] = x[i] - c; d2[i] = d[i] * d[i]; }
    const double tau = std::max(gamma_method(d).tau_int, gamma_method(d2).tau_int);
    auto var = [scale](const double* m) { return scale * std::max(m[1] - m[0] * m[0], 0.0); };
    const JackknifeResult jk = blocked_jackknife({&d, &d2}, var, jackknife_block_length(n, tau));
    return Observable(jk.bias_corrected, jk.error);
}

/**
 * Energy per site and specific heat per site c = Var(E) / (N T^2) from a
 * series of TOTAL energies (energy fluctuations are extensive, Var(E) ~ N).
 */
inline void energy_statistics(const std::vector<double>& E_total, double T, size_t N,
                              ThermodynamicObservables& out) {
    out.temperature = T;
    out.n_samples = E_total.size();
    if (E_total.empty() || N == 0) return;
    std::vector<double> e(E_total.size());
    for (size_t i = 0; i < e.size(); ++i) e[i] = E_total[i] / double(N);
    const GammaResult g = gamma_method(e);
    out.energy = Observable(g.mean, g.error);
    out.energy_tau_int = g.tau_int;
    out.specific_heat = (T > 0.0) ? scaled_variance(E_total, 1.0 / (double(N) * T * T))
                                  : Observable(0.0, 0.0);
}

/**
 * Statistics of a vector order parameter m (per-site normalised):
 * component means, <|m|>, <m^2>, the connected susceptibility
 * chi = beta N (<m^2> - <|m|>^2) = beta N Var(|m|) (the finite-size estimator
 * that stays finite in the ordered phase, cf. Binder & Landau), and the Binder
 * cumulant U4 = 1 - <m^4> / (3 <m^2>^2). For an n-component order parameter
 * U4 -> 1 - (n+2)/(3n) in the disordered (Gaussian) phase and -> 2/3 deep in
 * the ordered phase.
 */
inline OrderParameterStats order_parameter_statistics(const OrderParameterSeries& op, double T) {
    OrderParameterStats st;
    st.name = op.name;
    st.dim = op.dim;
    st.n_sites = op.n_sites;
    st.mean = VectorObservable(op.dim);
    const size_t n = op.n_samples();
    if (n == 0 || op.dim == 0) return st;
    std::vector<double> comp(n), absm(n), m2(n), m4(n);
    for (size_t s = 0; s < n; ++s) {
        double q = 0.0;
        for (size_t a = 0; a < op.dim; ++a) q += op.data[s * op.dim + a] * op.data[s * op.dim + a];
        m2[s] = q;
        m4[s] = q * q;
        absm[s] = std::sqrt(q);
    }
    for (size_t a = 0; a < op.dim; ++a) {
        for (size_t s = 0; s < n; ++s) comp[s] = op.data[s * op.dim + a];
        const GammaResult g = gamma_method(comp);
        st.mean.values[a] = g.mean;
        st.mean.errors[a] = g.error;
    }
    const GammaResult ga = gamma_method(absm);
    st.abs = Observable(ga.mean, ga.error);
    st.tau_int_abs = ga.tau_int;
    st.m2 = mean_with_error(m2);
    if (T > 0.0) st.susceptibility = scaled_variance(absm, double(op.n_sites) / T);
    if (n >= 2) {
        const double tau = std::max(gamma_method(m2).tau_int, gamma_method(m4).tau_int);
        auto binder = [](const double* m) {
            return m[0] > 0.0 ? 1.0 - m[1] / (3.0 * m[0] * m[0]) : 0.0;
        };
        const JackknifeResult jk = blocked_jackknife({&m2, &m4}, binder, jackknife_block_length(n, tau));
        st.binder = Observable(jk.bias_corrected, jk.error);
    } else {
        st.binder = Observable(m2[0] > 0.0 ? 1.0 - m4[0] / (3.0 * m2[0] * m2[0]) : 0.0, 0.0);
    }
    return st;
}

/**
 * Covariance <x y> - <x><y> with a blocked-jackknife error (centred
 * primaries, block length from the largest tau_int).
 */
inline Observable covariance(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    if (n < 2 || y.size() != n) return Observable(0.0, 0.0);
    const double cx = detail::mean(x.data(), n), cy = detail::mean(y.data(), n);
    std::vector<double> dx(n), dy(n), dxy(n);
    for (size_t i = 0; i < n; ++i) { dx[i] = x[i] - cx; dy[i] = y[i] - cy; dxy[i] = dx[i] * dy[i]; }
    const double tau = std::max({gamma_method(dx).tau_int, gamma_method(dy).tau_int,
                                 gamma_method(dxy).tau_int});
    auto cov = [](const double* m) { return m[2] - m[0] * m[1]; };
    const JackknifeResult jk = blocked_jackknife({&dx, &dy, &dxy}, cov, jackknife_block_length(n, tau));
    return Observable(jk.bias_corrected, jk.error);
}

/**
 * Energy, specific heat and order-parameter statistics at temperature T from
 * total energies and per-site order-parameter series (all of equal length).
 */
inline ThermodynamicObservables compute_thermodynamics(const std::vector<double>& E_total,
                                                       const std::vector<OrderParameterSeries>& ops,
                                                       double T, size_t N) {
    ThermodynamicObservables obs;
    energy_statistics(E_total, T, N, obs);
    obs.order_parameters.reserve(ops.size());
    for (const auto& op : ops) obs.order_parameters.push_back(order_parameter_statistics(op, T));
    return obs;
}

}  // namespace mc
