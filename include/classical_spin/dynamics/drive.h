#pragma once
/**
 * drive.h — time-dependent external fields for spin dynamics.
 *
 * A drive is a sum of Gaussian-enveloped carrier pulses, each with its own
 * centre, amplitude, width, frequency and per-sublattice polarisation:
 *
 *     B_drive(t, site of sublattice a) = Σ_p f_p(t) e_{p,a},
 *     f_p(t) = A_p exp(-((t - t_p) / (2 w_p))²) cos(ω_p (t - t_p)).
 *
 * (w is σ_eff/√2 of the textbook Gaussian; this is the convention used by
 * every pulse driver in the code base.) The polarisations e_{p,a} are stored
 * in the frame of the spin variables, i.e. already rotated out of the global
 * frame by the model (Lattice::make_pulse applies F_a^T, see there).
 *
 * The schedule is an immutable value passed explicitly to the right-hand
 * side, so concurrent trajectories with different pulses (the 2DCS delay
 * scan) can share one const lattice instead of each mutating -- or deep
 * copying -- the model.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace classical_spin::dynamics {

/// Half-width of the region where a pulse acts, in units of its width w:
/// beyond |t - t_p| = 9 w the envelope is below exp(-81/4) ≈ 1.6e-9.
inline constexpr double kPulseSupportWidths = 9.0;

struct Pulse {
    double t_center = 0.0;
    double amplitude = 0.0;
    double width = 1.0;
    double frequency = 0.0;

    /// Amplitude-weighted envelope f(t).
    double envelope(double t) const {
        const double u = (t - t_center) / (2.0 * width);
        return amplitude * std::exp(-u * u) * std::cos(frequency * (t - t_center));
    }
    /// |t - t_center| beyond which the pulse is negligible (1.6e-9 relative).
    double half_support() const { return kPulseSupportWidths * width; }
};

class DriveSchedule {
public:
    static constexpr std::size_t kMaxPulses = 8;

    DriveSchedule() = default;
    DriveSchedule(std::size_t n_sublattices, std::size_t spin_dim)
        : n_sub_(n_sublattices), dim_(spin_dim) {}

    /**
     * Add a pulse with polarisation `pol` (n_sublattices * spin_dim values,
     * frame of the spin variables). Pulses with zero amplitude or zero
     * polarisation are dropped: they cannot act, and skipping them keeps the
     * right-hand side free of dead work.
     */
    void add(const Pulse& p, const std::vector<double>& pol) {
        if (pol.size() != n_sub_ * dim_)
            throw std::invalid_argument("DriveSchedule::add: polarisation has " + std::to_string(pol.size()) +
                                        " entries, expected " + std::to_string(n_sub_ * dim_));
        if (!std::isfinite(p.amplitude) || !std::isfinite(p.t_center) || !std::isfinite(p.frequency))
            throw std::invalid_argument("DriveSchedule::add: non-finite pulse parameter");
        if (p.amplitude == 0.0) return;
        if (!(p.width > 0.0) || !std::isfinite(p.width))
            throw std::invalid_argument("DriveSchedule::add: pulse width must be positive and finite");
        bool any = false;
        for (double v : pol) {
            if (!std::isfinite(v)) throw std::invalid_argument("DriveSchedule::add: non-finite polarisation");
            any = any || (v != 0.0);
        }
        if (!any) return;
        if (pulses_.size() == kMaxPulses)
            throw std::invalid_argument("DriveSchedule::add: at most " + std::to_string(kMaxPulses) + " pulses");
        pulses_.push_back(p);
        pol_.insert(pol_.end(), pol.begin(), pol.end());
    }

    bool empty() const { return pulses_.empty(); }
    std::size_t size() const { return pulses_.size(); }
    const Pulse& pulse(std::size_t p) const { return pulses_[p]; }
    const double* polarisation(std::size_t p, std::size_t sublattice) const {
        return pol_.data() + (p * n_sub_ + sublattice) * dim_;
    }

    /// f[p] = f_p(t) for every pulse (size() values).
    void envelopes(double t, double* f) const {
        for (std::size_t p = 0; p < pulses_.size(); ++p) f[p] = pulses_[p].envelope(t);
    }

    /// B[0..dim) += Σ_p f[p] e_{p,a}.
    void accumulate(std::size_t sublattice, const double* f, double* B) const {
        for (std::size_t p = 0; p < pulses_.size(); ++p) {
            const double* e = polarisation(p, sublattice);
            for (std::size_t d = 0; d < dim_; ++d) B[d] += f[p] * e[d];
        }
    }

    /**
     * Upper bound for an adaptive step: a quarter of the shortest drive time
     * scale (pulse width or carrier period). Without it an error-controlled
     * stepper starting from a stationary state sees zero local error, grows
     * its step without bound and can jump over a pulse entirely. Returns 0
     * (no bound) for an empty schedule.
     */
    double max_step() const {
        double h = std::numeric_limits<double>::infinity();
        for (const auto& p : pulses_) {
            h = std::min(h, p.width);
            if (p.frequency != 0.0) h = std::min(h, 2.0 * M_PI / std::abs(p.frequency));
        }
        return pulses_.empty() ? 0.0 : 0.25 * h;
    }

    /// True when every pulse is negligible on [t_lo, t_hi].
    bool negligible_on(double t_lo, double t_hi) const {
        for (const auto& p : pulses_) {
            if (t_hi > p.t_center - p.half_support() && t_lo < p.t_center + p.half_support()) return false;
        }
        return true;
    }

private:
    std::size_t n_sub_ = 0, dim_ = 0;
    std::vector<Pulse> pulses_;
    std::vector<double> pol_;
};

}  // namespace classical_spin::dynamics
