#pragma once
/**
 * grid_integrate.h — integrate a generic ODE  dx/dt = f(x, t)  and observe it
 * exactly on an integer-indexed TimeGrid (dynamics/time_grid.h).
 *
 * Boost's integrate_const on [T_start, T_end] decides whether one more step
 * fits with an absolute epsilon, so a floating-point end time one ulp below
 * t0 + n dt silently drops the last step and the state lags its time label;
 * chained pulse-window segments hit this at every seam (see time_grid.h).
 * Here every observation is made at t_k = t0 + k dt, k = 0..n-1, computed
 * from the integer k, and the observer sees every k exactly once, in order:
 *
 *   * dopri5: dense output (Hairer, Norsett & Wanner, "Solving ODEs I",
 *     Sec. II.6). Steps are chosen by the error controller alone, samples
 *     are interpolated at t_k, so free precession is not forced down to the
 *     output cadence (the measured cost of integrate_const capped at the
 *     cadence is 1.6-3.8x more RHS calls). Inside caller-supplied windows
 *     the step is capped (max_dt) so a narrow pulse cannot be stepped over.
 *   * Cash-Karp 5(4), Fehlberg 7(8), Bulirsch-Stoer: integrate_times, which
 *     steps exactly onto every t_k.
 *   * fixed-step methods: exactly `substeps` steps of dt/substeps between
 *     samples, each step start time computed from integers.
 *
 * The state type only needs to be an odeint-compatible container
 * (std::vector<double> in practice).
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include "classical_spin/dynamics/ode_method.h"
#include "classical_spin/dynamics/time_grid.h"

namespace classical_spin::dynamics {

/// A range of sample indices [k_begin, k_end] with its own adaptive step cap.
struct GridSegment {
    std::size_t k_begin = 0;
    std::size_t k_end = 0;
    double max_dt = 0.0;  // <= 0: no cap
};

/**
 * Split the grid into segments: within each window [lo, hi] (absolute times,
 * widened outward to grid points and merged when overlapping) the adaptive
 * step is capped at `window_max_dt`; elsewhere it is free. Segment ends are
 * grid indices, so consecutive segments join exactly at a sample.
 */
inline std::vector<GridSegment> segments_with_windows(const TimeGrid& grid,
                                                      const std::vector<std::pair<double, double>>& windows,
                                                      double window_max_dt) {
    const std::size_t last = grid.n - 1;
    std::vector<std::pair<std::size_t, std::size_t>> idx;
    for (const auto& w : windows) {
        const double lo = std::floor((w.first - grid.t0) / grid.dt);
        const double hi = std::ceil((w.second - grid.t0) / grid.dt);
        if (hi < 0.0 || lo > static_cast<double>(last)) continue;
        idx.emplace_back(static_cast<std::size_t>(std::max(0.0, lo)),
                         static_cast<std::size_t>(std::min(static_cast<double>(last), hi)));
    }
    std::sort(idx.begin(), idx.end());
    std::vector<std::pair<std::size_t, std::size_t>> merged;
    for (const auto& w : idx) {
        if (!merged.empty() && w.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, w.second);
        } else {
            merged.push_back(w);
        }
    }
    std::vector<GridSegment> segs;
    std::size_t cursor = 0;
    for (const auto& w : merged) {
        if (w.first > cursor) segs.push_back({cursor, w.first, 0.0});
        if (w.second > std::max(w.first, cursor)) segs.push_back({std::max(w.first, cursor), w.second, window_max_dt});
        cursor = std::max(cursor, w.second);
    }
    if (cursor < last || segs.empty()) segs.push_back({cursor, last, 0.0});
    return segs;
}

namespace detail {
// integrate_times observer adaptor: the first call of every segment is the
// segment's starting sample, which the previous segment (or the k = 0 call)
// has already reported; later calls map to consecutive grid indices.
template <class Observer>
struct SegmentObserver {
    Observer* obs;
    std::size_t k;
    bool first = true;
    template <class S>
    void operator()(const S& x, double /*t*/) {
        if (first) { first = false; return; }
        (*obs)(x, ++k);
    }
};
}  // namespace detail

/**
 * Integrate from grid[k_start] (state x) to grid.t_end(), calling
 * obs(const State&, size_t k) at every sample k = k_start..n-1 (the first
 * call reports the initial state). Fixed-step methods take `substeps` steps
 * of grid.dt / substeps per sample interval; for adaptive methods
 * grid.dt / substeps is the initial step. `segments` may cap the dopri5 step
 * in sub-ranges (empty: no cap anywhere). Non-positive tolerances select the
 * method defaults (1e-6; 1e-8 for Bulirsch-Stoer). On return `x` is the state
 * at grid.t_end().
 */
template <class State, class System, class Observer>
void integrate_on_time_grid(System&& sys, State& x, const TimeGrid& grid, std::size_t substeps,
                            OdeMethod method, double abs_tol, double rel_tol, Observer&& obs,
                            const std::vector<GridSegment>& segments = {}, std::size_t k_start = 0) {
    namespace odeint = boost::numeric::odeint;
    if (grid.n == 0) throw std::invalid_argument("integrate_on_time_grid: empty time grid");
    if (substeps == 0) throw std::invalid_argument("integrate_on_time_grid: substeps must be >= 1");
    if (is_geometric(method)) {
        throw std::invalid_argument(std::string("integrate_on_time_grid: '") + ode_method_name(method) +
                                    "' is a geometric SO(3) integrator, not available for this model");
    }
    if (abs_tol <= 0.0) abs_tol = (method == OdeMethod::BulirschStoer) ? 1e-8 : 1e-6;
    if (rel_tol <= 0.0) rel_tol = (method == OdeMethod::BulirschStoer) ? 1e-8 : 1e-6;

    if (k_start >= grid.n) throw std::invalid_argument("integrate_on_time_grid: k_start beyond the grid");

    using Obs = std::remove_reference_t<Observer>;
    obs(static_cast<const State&>(x), k_start);
    const std::size_t last = grid.n - 1;
    if (k_start == last) return;
    const double h = grid.dt / static_cast<double>(substeps);

    auto fixed = [&](auto&& stepper) {
        for (std::size_t k = k_start; k < last; ++k) {
            for (std::size_t j = 0; j < substeps; ++j) {
                const double t = grid.t0 + (static_cast<double>(k) * static_cast<double>(substeps) +
                                            static_cast<double>(j)) * h;
                stepper.do_step(sys, x, t, h);
            }
            obs(static_cast<const State&>(x), k + 1);
        }
    };
    auto times_of = [&](std::size_t k0, std::size_t k1) {
        std::vector<double> ts(k1 - k0 + 1);
        for (std::size_t i = 0; i < ts.size(); ++i) ts[i] = grid[k0 + i];
        return ts;
    };

    switch (method) {
        case OdeMethod::Euler:           fixed(odeint::euler<State>()); return;
        case OdeMethod::RK2:             fixed(odeint::modified_midpoint<State>()); return;
        case OdeMethod::RK4:             fixed(odeint::runge_kutta4<State>()); return;
        case OdeMethod::AdamsBashforth5: fixed(odeint::adams_bashforth<5, State>()); return;
        case OdeMethod::AdamsMoulton5:   fixed(odeint::adams_bashforth_moulton<5, State>()); return;
        case OdeMethod::Dopri5: {
            std::vector<GridSegment> segs = segments;
            if (segs.empty()) segs.push_back({0, last, 0.0});
            for (auto s : segs) {
                if (s.k_end <= k_start) continue;
                s.k_begin = std::max(s.k_begin, k_start);
                if (s.k_end <= s.k_begin) continue;
                const auto ts = times_of(s.k_begin, s.k_end);
                detail::SegmentObserver<Obs> so{&obs, s.k_begin};
                if (s.max_dt > 0.0) {
                    odeint::integrate_times(
                        odeint::make_dense_output(abs_tol, rel_tol, s.max_dt,
                                                  odeint::runge_kutta_dopri5<State>()),
                        sys, x, ts.begin(), ts.end(), std::min(h, s.max_dt), std::ref(so));
                } else {
                    odeint::integrate_times(
                        odeint::make_dense_output(abs_tol, rel_tol, odeint::runge_kutta_dopri5<State>()),
                        sys, x, ts.begin(), ts.end(), h, std::ref(so));
                }
            }
            return;
        }
        case OdeMethod::CashKarp54:
        case OdeMethod::Fehlberg78:
        case OdeMethod::BulirschStoer: {
            const auto ts = times_of(k_start, last);
            detail::SegmentObserver<Obs> so{&obs, k_start};
            if (method == OdeMethod::CashKarp54) {
                odeint::integrate_times(
                    odeint::make_controlled<odeint::runge_kutta_cash_karp54<State>>(abs_tol, rel_tol),
                    sys, x, ts.begin(), ts.end(), h, std::ref(so));
            } else if (method == OdeMethod::Fehlberg78) {
                odeint::integrate_times(
                    odeint::make_controlled<odeint::runge_kutta_fehlberg78<State>>(abs_tol, rel_tol),
                    sys, x, ts.begin(), ts.end(), h, std::ref(so));
            } else {
                odeint::integrate_times(odeint::bulirsch_stoer<State>(abs_tol, rel_tol),
                                        sys, x, ts.begin(), ts.end(), h, std::ref(so));
            }
            return;
        }
        default:
            break;
    }
    throw std::invalid_argument(std::string("integrate_on_time_grid: unsupported method '") +
                                ode_method_name(method) + "'");
}

}  // namespace classical_spin::dynamics
