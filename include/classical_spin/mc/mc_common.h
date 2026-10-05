#pragma once
/**
 * mc_common.h — Common Monte Carlo types and helpers shared by Lattice,
 * MixedLattice and PhononLattice.
 *
 *  - statistics (binning, Gamma method, jackknife, thermodynamic
 *    estimators) live in mc/statistics.h and are re-exported here;
 *  - the replica-exchange engine and temperature-ladder tuning live in
 *    mc/parallel_tempering.h;
 *  - this header keeps the annealing schedule, the proposal-width
 *    controller, the geometric ladder, greedy quench and the legacy
 *    (energy, sublattice magnetisation) observable interface.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "classical_spin/mc/statistics.h"

namespace mc {

using std::vector;
using std::string;
using std::cout;
using std::endl;

struct AutocorrelationResult {
    double tau_int = 1.0;
    size_t sampling_interval = 100;
    vector<double> correlation_function;
};

// ============================================================
// AUTOCORRELATION (legacy interface over the Gamma method)
// ============================================================

/**
 * Integrated autocorrelation time of a series by the Gamma method
 * (statistics.h) plus the normalised autocorrelation function rho(t) up to a
 * few windows, and a suggested measurement spacing of 2 tau_int samples
 * (in sweeps: times base_interval; at least 100 sweeps, as before).
 */
inline AutocorrelationResult compute_autocorrelation(const vector<double>& series,
                                                     size_t base_interval = 10) {
    AutocorrelationResult result;
    result.sampling_interval = base_interval;
    if (series.size() < 10) return result;
    const GammaResult g = gamma_method(series, 1.5, &result.correlation_function);
    if (g.variance <= 0.0) return result;
    result.tau_int = g.tau_int;
    const size_t tau_sweeps = static_cast<size_t>(std::ceil(g.tau_int)) * base_interval;
    result.sampling_interval = std::max(size_t(2) * tau_sweeps, size_t(100));
    return result;
}

/// Legacy out-parameter form of compute_autocorrelation.
inline void estimate_autocorrelation_time(const vector<double>& series, size_t base_interval,
                                          double& tau_int_out, size_t& sampling_interval_out) {
    const AutocorrelationResult r = compute_autocorrelation(series, base_interval);
    tau_int_out = r.tau_int;
    sampling_interval_out = r.sampling_interval;
}

// ============================================================
// SCHEDULES AND PROPOSAL CONTROL
// ============================================================

/**
 * Geometric annealing schedule T_k = T_start r^k, truncated at T_end, which
 * is always the last entry (the previous `while (T > T_end)` loops never
 * simulated T_end itself). Throws on invalid input instead of looping
 * forever (r >= 1) or dividing by zero (T <= 0).
 */
inline vector<double> annealing_schedule(double T_start, double T_end, double cooling_rate,
                                         size_t max_steps = 1000000) {
    if (!(T_start > 0.0) || !(T_end > 0.0) || !std::isfinite(T_start) || !std::isfinite(T_end))
        throw std::invalid_argument("annealing: temperatures must be positive and finite");
    if (T_end > T_start)
        throw std::invalid_argument("annealing: T_end must not exceed T_start");
    vector<double> schedule;
    if (T_start == T_end) return {T_end};
    if (!(cooling_rate > 0.0 && cooling_rate < 1.0))
        throw std::invalid_argument("annealing: cooling_rate must lie in (0, 1)");
    const double n_est = std::log(T_end / T_start) / std::log(cooling_rate);
    if (n_est > double(max_steps))
        throw std::invalid_argument("annealing: schedule would need more than " +
                                    std::to_string(max_steps) + " temperatures");
    for (double T = T_start; T > T_end * (1.0 + 1e-12); T *= cooling_rate) schedule.push_back(T);
    schedule.push_back(T_end);
    return schedule;
}

/**
 * Robbins-Monro controller for the width σ of local Gaussian proposals:
 *     log σ ← log σ + γ_k (A_k - A*),   γ_k = (k + 5)^-0.6,
 * steering the acceptance A toward A* (≈ 0.4-0.5 for continuous spins;
 * Alzate-Cardona et al., JPCM 31, 095802 (2019)). Adapt only during
 * equilibration and freeze σ while measuring, so the sampling chain stays
 * time-homogeneous (detailed balance holds for any fixed σ).
 */
class StepSizeController {
public:
    explicit StepSizeController(double sigma0 = 2.0, double target = 0.45,
                                double sigma_min = 1e-4, double sigma_max = 10.0)
        : log_sigma_(std::log(sigma0)), target_(target),
          log_min_(std::log(sigma_min)), log_max_(std::log(sigma_max)) {}

    double sigma() const { return std::exp(log_sigma_); }
    double target() const { return target_; }
    /// Restart the gain sequence (e.g. at a new temperature); σ is kept.
    void restart() { k_ = 0; }
    void update(double acceptance) {
        const double gain = std::pow(double(k_) + 5.0, -0.6);
        log_sigma_ = std::clamp(log_sigma_ + gain * (acceptance - target_), log_min_, log_max_);
        ++k_;
    }

private:
    double log_sigma_, target_, log_min_, log_max_;
    size_t k_ = 0;
};

/**
 * Geometrically spaced temperature ladder T_i = T_min (T_max/T_min)^(i/(R-1)),
 * ascending (cold -> hot). Throws on T_min <= 0, T_max < T_min or R = 0.
 */
inline vector<double> generate_geometric_temperature_ladder(double Tmin, double Tmax, size_t R) {
    if (R == 0) throw std::invalid_argument("temperature ladder: R must be >= 1");
    if (!(Tmin > 0.0) || !std::isfinite(Tmin) || !std::isfinite(Tmax) || !(Tmax >= Tmin))
        throw std::invalid_argument("temperature ladder: need 0 < T_min <= T_max (got T_min=" +
                                    std::to_string(Tmin) + ", T_max=" + std::to_string(Tmax) + ")");
    vector<double> temps(R);
    if (R == 1) { temps[0] = Tmin; return temps; }
    for (size_t i = 0; i < R; ++i)
        temps[i] = Tmin * std::pow(Tmax / Tmin, double(i) / double(R - 1));
    temps.back() = Tmax;
    return temps;
}

// ============================================================
// LEGACY OBSERVABLE INTERFACE
// ============================================================

/**
 * Thermodynamic observables from total energies and per-sample sublattice
 * magnetisations (local frames, as returned by magnetization_sublattice()).
 * Energy and specific heat use statistics.h (Gamma-method error; jackknife
 * over autocorrelation-sized blocks for c); `magnetization` is the average
 * of the sublattice vectors; the cross terms <E S_a> - <E><S_a> carry
 * jackknife errors whose blocks cover all samples.
 */
template<typename SpinVec>
inline ThermodynamicObservables compute_thermodynamic_observables(
    const vector<double>& energies,
    const vector<vector<SpinVec>>& sublattice_mags,
    double temperature, size_t lattice_size) {

    ThermodynamicObservables obs;
    energy_statistics(energies, temperature, lattice_size, obs);
    const size_t n = energies.size();
    if (n == 0 || sublattice_mags.size() != n || sublattice_mags[0].empty()) return obs;

    const size_t n_sub = sublattice_mags[0].size();
    const size_t sdim = sublattice_mags[0][0].size();
    obs.sublattice_magnetization.assign(n_sub, VectorObservable(sdim));
    obs.energy_sublattice_cross.assign(n_sub, VectorObservable(sdim));
    obs.magnetization = VectorObservable(sdim);
    vector<double> comp(n), total(n);
    for (size_t d = 0; d < sdim; ++d) {
        std::fill(total.begin(), total.end(), 0.0);
        for (size_t a = 0; a < n_sub; ++a) {
            for (size_t i = 0; i < n; ++i) {
                comp[i] = sublattice_mags[i][a](d);
                total[i] += comp[i] / double(n_sub);
            }
            const Observable m = mean_with_error(comp);
            obs.sublattice_magnetization[a].values[d] = m.value;
            obs.sublattice_magnetization[a].errors[d] = m.error;
            const Observable c = covariance(energies, comp);
            obs.energy_sublattice_cross[a].values[d] = c.value;
            obs.energy_sublattice_cross[a].errors[d] = c.error;
        }
        const Observable m = mean_with_error(total);
        obs.magnetization.values[d] = m.value;
        obs.magnetization.errors[d] = m.error;
    }
    return obs;
}

// ============================================================
// TEMPLATE: MC ALGORITHMS
// ============================================================

/**
 * Greedy quench: iterate deterministic sweeps until energy converges.
 */
template<typename L>
inline void greedy_quench(L& lat, double rel_tol = 1e-12,
                          size_t max_sweeps = 10000) {
    double E_prev = lat.total_energy();
    for (size_t s = 0; s < max_sweeps; ++s) {
        lat.deterministic_sweep(1);
        double E = lat.total_energy();
        double dE = std::abs(E - E_prev);
        if (E_prev != 0.0 && dE / (std::abs(E_prev) + 1e-18) < rel_tol) {
            cout << "Greedy quench converged at sweep " << s + 1
                 << ", E/N = " << E / lat.lattice_size << endl;
            return;
        }
        E_prev = E;
    }
    cout << "Greedy quench reached max sweeps (" << max_sweeps
         << "), E/N = " << E_prev / lat.lattice_size << endl;
}

// ============================================================
// SFINAE trait: detect if lattice L has extra DOF to exchange
// (e.g., strain fields in StrainPhononLattice)
// Required methods: extra_dof_size(), pack_extra_dof(double*), unpack_extra_dof(const double*)
// ============================================================
template<typename T, typename = void>
struct has_extra_dof : std::false_type {};

template<typename T>
struct has_extra_dof<T, std::void_t<
    decltype(std::declval<const T&>().extra_dof_size()),
    decltype(std::declval<const T&>().pack_extra_dof(std::declval<double*>())),
    decltype(std::declval<T&>().unpack_extra_dof(std::declval<const double*>()))
>> : std::true_type {};

} // namespace mc
