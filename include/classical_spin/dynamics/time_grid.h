#pragma once
/**
 * time_grid.h — exact, integer-indexed sampling grids for spin dynamics.
 *
 * Every trajectory produced by the dynamics drivers is sampled at
 *
 *     t_k = t0 + k dt,   k = 0, 1, ..., n - 1,
 *
 * with t_k computed from the integer k (never by accumulating t += dt), so
 * two trajectories on the same grid carry bitwise-identical time stamps and
 * the same number of samples whatever happens inside the integrator. This is
 * what makes difference signals such as the 2DCS non-linear response
 * M_NL = M01 - M0 - M1 well defined sample by sample.
 *
 * Background: odeint's integrate_const decides whether one more interval
 * fits by comparing times with an ABSOLUTE epsilon (2.2e-16), so an interval
 * whose end differs from t0 + k dt by ~1e-14 is silently dropped. Integrating
 * pulse-window segments one after the other that way lost one step at every
 * seam, with a delay-dependent lag (audit: chunk-seam-drops-last-interval).
 */

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace classical_spin::dynamics {

/**
 * Number of whole steps of size `step` that fit into `span` (both > 0 or
 * span == 0), tolerant to round-off: 0.3 / 0.1 = 2.9999999999999996 counts
 * as 3, while a genuinely fractional ratio such as 2.5 is truncated to 2 so
 * that the last grid point never overshoots the requested end.
 */
inline std::size_t whole_steps(double span, double step) {
    const double r = span / step;
    const double tol = 1e-9 + 8.0 * std::numeric_limits<double>::epsilon() * std::abs(r);
    return static_cast<std::size_t>(std::floor(r + tol));
}

struct TimeGrid {
    double t0 = 0.0;
    double dt = 1.0;
    std::size_t n = 1;  // number of samples (>= 1)

    double operator[](std::size_t k) const { return t0 + static_cast<double>(k) * dt; }
    double t_end() const { return (*this)[n - 1]; }

    std::vector<double> times() const {
        std::vector<double> t(n);
        for (std::size_t k = 0; k < n; ++k) t[k] = (*this)[k];
        return t;
    }

    /**
     * Grid starting at t_start with spacing dt and as many samples as fit
     * into [t_start, t_end] (t_end is included when it lies on the grid up
     * to round-off). Throws std::invalid_argument for non-finite input,
     * dt <= 0 or t_end < t_start.
     */
    static TimeGrid covering(double t_start, double t_end, double dt,
                             const std::string& what = "time grid") {
        if (!std::isfinite(t_start) || !std::isfinite(t_end) || !std::isfinite(dt))
            throw std::invalid_argument(what + ": non-finite start, end or step");
        if (!(dt > 0.0))
            throw std::invalid_argument(what + ": step must be positive (got " + std::to_string(dt) + ")");
        if (t_end < t_start)
            throw std::invalid_argument(what + ": end time " + std::to_string(t_end) +
                                        " precedes start time " + std::to_string(t_start));
        return TimeGrid{t_start, dt, whole_steps(t_end - t_start, dt) + 1};
    }
};

/**
 * Delay values start + i * step, i = 0 .. n-1, covering [start, end] (or
 * [end, start] for a negative step). The count uses whole_steps, so
 * 0 -> 0.3 in steps of 0.1 gives the four points 0, 0.1, 0.2, 0.3 (the old
 * truncating formula gave three). Throws std::invalid_argument for a zero
 * or non-finite step, or a step whose sign points away from `end`.
 */
inline std::vector<double> delay_grid(double start, double end, double step,
                                      const std::string& what = "delay grid") {
    if (!std::isfinite(start) || !std::isfinite(end) || !std::isfinite(step))
        throw std::invalid_argument(what + ": non-finite start, end or step");
    if (step == 0.0) throw std::invalid_argument(what + ": step must be non-zero");
    const double span = end - start;
    if (span != 0.0 && (span > 0.0) != (step > 0.0))
        throw std::invalid_argument(what + ": step " + std::to_string(step) +
                                    " points away from end " + std::to_string(end));
    const std::size_t n = (span == 0.0) ? 1 : whole_steps(span, step) + 1;
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = start + static_cast<double>(i) * step;
    return v;
}

}  // namespace classical_spin::dynamics
