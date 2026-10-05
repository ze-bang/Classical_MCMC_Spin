/**
 * @file mixed_lattice_md.cpp
 * @brief MixedLattice (SU(2) + SU(3)) dynamics: equations of motion, pulse
 *        drives, molecular dynamics and pump-probe / 2DCS spectroscopy.
 *
 * Conventions (see core/su3_coherent_state.h for the derivation):
 *   E = <psi|H|psi> for every term, local field H = dE/dS (dE/dn),
 *   SU(2):  dS/dt = H x S                         (+ LL-Gilbert damping)
 *   SU(3):  dn^a/dt = c f_abc H^b n^c,  c = su3_bracket = 2   (+ Bloch relaxation)
 * All trajectories are sampled on exact integer-indexed grids
 * (dynamics/time_grid.h, dynamics/grid_integrate.h).
 */

#include "classical_spin/lattice/mixed_lattice.h"

#include "classical_spin/dynamics/grid_integrate.h"
#include "classical_spin/dynamics/time_grid.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

namespace dyn = classical_spin::dynamics;

// =============================================================
// Inner kernels of the MD local field, on the flat row-major packed buffers
// of MixedLattice::build_packed_interaction_buffers():
//   bilinear_kernel<Da,Db>(J, s, H):      H[a] += sum_b J[a*Db + b] s[b]
//   trilinear_kernel<Da,Db,Dc>(T, s1, s2, H):
//                                         H[a] += sum_{b,c} T[(a*Db+b)*Dc + c] s1[b] s2[c]
// Compile-time sizes (3x3, 3x8, 8x8, 3x3x3, 3x3x8, 8x3x3, 8x8x8) let the
// compiler unroll and vectorise; MixedLattice only admits (3, 8) spins.
// =============================================================
template <size_t Da, size_t Db>
inline void bilinear_kernel(const double* __restrict J,
                            const double* __restrict s,
                            double* __restrict H) {
    for (size_t a = 0; a < Da; ++a) {
        double acc = 0.0;
        const double* __restrict Ja = J + a * Db;
        for (size_t b = 0; b < Db; ++b) acc += Ja[b] * s[b];
        H[a] += acc;
    }
}

template <size_t Da, size_t Db>
inline void scaled_bilinear_kernel(const double* __restrict J,
                                   const double* __restrict s,
                                   double scale, double* __restrict H) {
    for (size_t a = 0; a < Da; ++a) {
        double acc = 0.0;
        const double* __restrict Ja = J + a * Db;
        for (size_t b = 0; b < Db; ++b) acc += Ja[b] * s[b];
        H[a] += scale * acc;
    }
}

template <size_t Da, size_t Db, size_t Dc>
inline void trilinear_kernel(const double* __restrict T,
                             const double* __restrict s1,
                             const double* __restrict s2,
                             double* __restrict H) {
    // a outer, b middle, c inner: unit stride in c, s1[b] hoisted.
    for (size_t a = 0; a < Da; ++a) {
        double acc = 0.0;
        for (size_t b = 0; b < Db; ++b) {
            const double s1b = s1[b];
            const double* __restrict Tab = T + (a * Db + b) * Dc;
            for (size_t c = 0; c < Dc; ++c) acc += Tab[c] * s1b * s2[c];
        }
        H[a] += acc;
    }
}

// Lie-Poisson SU(3) torque out^a = c f_abc H^b n^c (sparse Gell-Mann f).
inline void su3_torque(const double* H, const double* n, double c, double* out) {
    double Hc[8];
    for (int a = 0; a < 8; ++a) Hc[a] = c * H[a];
    cross_prod_SU3_flat(Hc, n, out, /*accumulate=*/false);
}

inline void require_finite(const std::vector<double>& x, double t, const char* who) {
    for (double v : x) {
        if (!std::isfinite(v)) {
            std::ostringstream os;
            os << who << ": non-finite state at t = " << t
               << " (integration diverged; reduce the time step or tolerances)";
            throw std::runtime_error(os.str());
        }
    }
}

// The Gaussian pulse shape used by every driver: exp(-(dt / 2w)^2).
inline double gaussian_envelope(double dt, double width) {
    const double u = dt / (2.0 * width);
    return std::exp(-u * u);
}

void validate_pulse_directions(const std::vector<SpinVector>& field, size_t n_atoms, size_t dim,
                               const char* what) {
    if (field.empty()) return;  // empty = zero direction
    if (field.size() < n_atoms) {
        throw std::invalid_argument(std::string(what) + ": need at least " + std::to_string(n_atoms) +
                                    " direction vectors (one per atom of the unit cell), got " +
                                    std::to_string(field.size()));
    }
    for (size_t a = 0; a < n_atoms; ++a) {
        if (field[a].size() != static_cast<Eigen::Index>(dim)) {
            throw std::invalid_argument(std::string(what) + ": direction vectors must have " +
                                        std::to_string(dim) + " components");
        }
        if (!field[a].allFinite()) {
            throw std::invalid_argument(std::string(what) + ": non-finite direction vector");
        }
    }
}

void validate_pulse_shape(double amp, double width, double freq, const char* what) {
    if (!std::isfinite(amp) || !std::isfinite(width) || !std::isfinite(freq)) {
        throw std::invalid_argument(std::string(what) + ": non-finite pulse amplitude, width or frequency");
    }
    if (amp != 0.0 && !(width > 0.0)) {
        throw std::invalid_argument(std::string(what) + ": pulse width must be > 0 (got " +
                                    std::to_string(width) + ")");
    }
}

}  // namespace

// =============================================================
// Pulse configuration
// =============================================================

// ---- MixedLattice::set_pulse_SU2 ----
    void MixedLattice::set_pulse_SU2(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq) {
        validate_pulse_directions(field_in1, N_atoms_SU2, spin_dim_SU2, "set_pulse_SU2 (pulse 1)");
        validate_pulse_directions(field_in2, N_atoms_SU2, spin_dim_SU2, "set_pulse_SU2 (pulse 2)");
        validate_pulse_shape(amp, width, freq, "set_pulse_SU2");
        if (!std::isfinite(t_B1) || !std::isfinite(t_B2)) {
            throw std::invalid_argument("set_pulse_SU2: non-finite pulse time");
        }
        const vector<SpinVector>* in[2] = {&field_in1, &field_in2};
        for (int k = 0; k < 2; ++k) {
            field_drive_SU2[k] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
            if (in[k]->empty()) {
                field_drive_global_SU2[k] = SpinVector::Zero(spin_dim_SU2);
                continue;
            }
            // B_local = F^T B_global (the field transforms covariantly).
            for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                field_drive_SU2[k].segment(atom * spin_dim_SU2, spin_dim_SU2) =
                    sublattice_frames_SU2[atom].transpose() * (*in[k])[atom];
            }
            // Lab-frame representative for the polarisation-resolved gates
            // B_x/B_y/B_z(t): the THz pump is a uniform plane wave.
            field_drive_global_SU2[k] = (*in[k])[0];
        }
        t_pulse_SU2 = {t_B1, t_B2};
        field_drive_amp_SU2 = amp;
        field_drive_width_SU2 = width;
        field_drive_freq_SU2 = freq;
        n_active_pulses = 2;
    }

// ---- MixedLattice::set_pulse_SU3 ----
    void MixedLattice::set_pulse_SU3(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq) {
        validate_pulse_directions(field_in1, N_atoms_SU3, spin_dim_SU3, "set_pulse_SU3 (pulse 1)");
        validate_pulse_directions(field_in2, N_atoms_SU3, spin_dim_SU3, "set_pulse_SU3 (pulse 2)");
        validate_pulse_shape(amp, width, freq, "set_pulse_SU3");
        if (!std::isfinite(t_B1) || !std::isfinite(t_B2)) {
            throw std::invalid_argument("set_pulse_SU3: non-finite pulse time");
        }
        const vector<SpinVector>* in[2] = {&field_in1, &field_in2};
        for (int k = 0; k < 2; ++k) {
            field_drive_SU3[k] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
            if (in[k]->empty()) continue;
            // B^(i) = chi^T D_i h = F_i^T B^(0)
            for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
                field_drive_SU3[k].segment(atom * spin_dim_SU3, spin_dim_SU3) =
                    sublattice_frames_SU3[atom].transpose() * (*in[k])[atom];
            }
        }
        t_pulse_SU3 = {t_B1, t_B2};
        field_drive_amp_SU3 = amp;
        field_drive_width_SU3 = width;
        field_drive_freq_SU3 = freq;
        n_active_pulses = 2;
    }

// ---- MixedLattice::configure_pulse_train ----
    void MixedLattice::configure_pulse_train(size_t n_pulses,
                               const vector<SpinVector>& field1_SU2, const vector<SpinVector>& field1_SU3,
                               double t1,
                               const vector<SpinVector>& field2_SU2, const vector<SpinVector>& field2_SU3,
                               double t2,
                               double amp_SU2, double width_SU2, double freq_SU2,
                               double amp_SU3, double width_SU3, double freq_SU3) {
        if (n_pulses < 1 || n_pulses > 2) {
            throw std::invalid_argument("configure_pulse_train: 1 or 2 pulses supported");
        }
        static const vector<SpinVector> none;
        set_pulse_SU2(field1_SU2, t1, n_pulses == 2 ? field2_SU2 : none, n_pulses == 2 ? t2 : t1,
                      amp_SU2, width_SU2, freq_SU2);
        set_pulse_SU3(field1_SU3, t1, n_pulses == 2 ? field2_SU3 : none, n_pulses == 2 ? t2 : t1,
                      amp_SU3, width_SU3, freq_SU3);
        n_active_pulses = n_pulses;
    }

// ---- MixedLattice::drive_envelopes_SU2 ----
    // The two pulse factors amp * envelope * carrier at time t; pulse slots
    // k >= n_active_pulses are zero (no phantom second pulse).
    void MixedLattice::drive_envelopes_SU2(double t, double& factor1, double& factor2) const {
        factor1 = factor2 = 0.0;
        if (n_active_pulses == 0 || field_drive_amp_SU2 == 0.0) return;
        double f[2] = {0.0, 0.0};
        for (size_t k = 0; k < std::min<size_t>(n_active_pulses, 2); ++k) {
            const double dt = t - t_pulse_SU2[k];
            f[k] = tabulated_pulse_times.empty()
                 ? field_drive_amp_SU2 * gaussian_envelope(dt, field_drive_width_SU2)
                       * std::cos(field_drive_freq_SU2 * dt)
                 : field_drive_amp_SU2 * interp_tabulated_pulse(dt);
        }
        factor1 = f[0];
        factor2 = f[1];
    }

// ---- MixedLattice::drive_envelopes_SU3 ----
    void MixedLattice::drive_envelopes_SU3(double t, double& factor1, double& factor2) const {
        factor1 = factor2 = 0.0;
        if (n_active_pulses == 0 || field_drive_amp_SU3 == 0.0) return;
        double f[2] = {0.0, 0.0};
        for (size_t k = 0; k < std::min<size_t>(n_active_pulses, 2); ++k) {
            const double dt = t - t_pulse_SU3[k];
            if (!tabulated_pulse_times.empty()) {
                // Same physical pulse as the SU(2) drive: shared table.
                f[k] = field_drive_amp_SU3 * interp_tabulated_pulse(dt);
                continue;
            }
            // Two-colour carrier when field_drive_freq_SU3_2 != 0: one Gaussian
            // drives two CEF lines, so the f_257 Raman product (E13 x E23)
            // lands on E12 = E13 - E23 resonantly.
            double carrier = std::cos(field_drive_freq_SU3 * dt);
            if (field_drive_freq_SU3_2 != 0.0) carrier += std::cos(field_drive_freq_SU3_2 * dt);
            f[k] = field_drive_amp_SU3 * gaussian_envelope(dt, field_drive_width_SU3) * carrier;
        }
        factor1 = f[0];
        factor2 = f[1];
    }

// ---- MixedLattice::interp_tabulated_pulse (private helper) ----
    // Linear interpolation of the normalised tabulated pulse at offset dt from
    // the pulse centre; 0 outside the data range.
    double MixedLattice::interp_tabulated_pulse(double dt) const {
        if (tabulated_pulse_times.empty()) return 0.0;
        if (dt <= tabulated_pulse_times.front() || dt >= tabulated_pulse_times.back()) return 0.0;
        auto it = std::lower_bound(tabulated_pulse_times.begin(), tabulated_pulse_times.end(), dt);
        const size_t idx = static_cast<size_t>(it - tabulated_pulse_times.begin());
        const double t0 = tabulated_pulse_times[idx - 1];
        const double t1 = tabulated_pulse_times[idx];
        const double v0 = tabulated_pulse_values[idx - 1];
        const double v1 = tabulated_pulse_values[idx];
        return v0 + (v1 - v0) * (dt - t0) / (t1 - t0);  // t1 > t0 (validated at load)
    }

// ---- MixedLattice::load_tabulated_pulse ----
    void MixedLattice::load_tabulated_pulse(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open()) {
            throw std::runtime_error("load_tabulated_pulse: cannot open \"" + filename + "\"");
        }
        std::vector<double> times, values;
        std::string line;
        while (std::getline(f, line)) {
            const auto comment_pos = line.find('#');
            if (comment_pos != std::string::npos) line.erase(comment_pos);
            if (line.empty()) continue;
            for (char& c : line) { if (c == ',') c = ' '; }   // comma- or space-separated
            std::istringstream ss(line);
            double t_val, e_val;
            if (ss >> t_val >> e_val) {
                times.push_back(t_val);
                values.push_back(e_val);
            }
        }
        if (times.size() < 2) {
            throw std::runtime_error("load_tabulated_pulse: fewer than 2 data points in \"" + filename + "\"");
        }
        double e_max = 0.0;
        size_t idx_peak = 0;
        for (size_t i = 0; i < times.size(); ++i) {
            if (!std::isfinite(times[i]) || !std::isfinite(values[i])) {
                throw std::runtime_error("load_tabulated_pulse: non-finite entry in \"" + filename + "\"");
            }
            if (i > 0 && !(times[i] > times[i - 1])) {
                throw std::runtime_error("load_tabulated_pulse: times must be strictly increasing in \"" +
                                         filename + "\"");
            }
            if (std::abs(values[i]) > e_max) {
                e_max = std::abs(values[i]);
                idx_peak = i;
            }
        }
        if (!(e_max > 0.0)) {
            throw std::runtime_error("load_tabulated_pulse: the pulse is identically zero in \"" + filename + "\"");
        }
        // Centre the time axis at the peak and normalise to max|E| = 1
        // (amplitudes are applied by field_drive_amp_SU2/SU3).
        const double t_peak = times[idx_peak];
        for (double& tv : times) tv -= t_peak;
        for (double& ev : values) ev /= e_max;
        tabulated_pulse_times = std::move(times);
        tabulated_pulse_values = std::move(values);
        // Gaussian-equivalent width such that kPulseWindowSigmas * width
        // covers the whole table (used for the pulse window of the drivers).
        const double max_extent = std::max(std::abs(tabulated_pulse_times.front()),
                                           std::abs(tabulated_pulse_times.back()));
        tabulated_pulse_sigma = max_extent / classical_spin_pulse_chunking::kPulseWindowSigmas;
    }

// ---- MixedLattice::drive_field_SU2_at_time ----
    SpinVector MixedLattice::drive_field_SU2_at_time(double t, size_t site_index) const {
        const size_t atom = site_index % N_atoms_SU2;
        double factor1, factor2;
        drive_envelopes_SU2(t, factor1, factor2);
        return factor1 * field_drive_SU2[0].segment(atom * spin_dim_SU2, spin_dim_SU2) +
               factor2 * field_drive_SU2[1].segment(atom * spin_dim_SU2, spin_dim_SU2);
    }

// ---- MixedLattice::drive_field_SU3_at_time ----
    SpinVector MixedLattice::drive_field_SU3_at_time(double t, size_t site_index) const {
        const size_t atom = site_index % N_atoms_SU3;
        double factor1, factor2;
        drive_envelopes_SU3(t, factor1, factor2);
        return factor1 * field_drive_SU3[0].segment(atom * spin_dim_SU3, spin_dim_SU3) +
               factor2 * field_drive_SU3[1].segment(atom * spin_dim_SU3, spin_dim_SU3);
    }

// =============================================================
// Equations of motion
// =============================================================

// ---- MixedLattice::drive_factors ----
    MixedLattice::DriveFactors MixedLattice::drive_factors(double t) const {
        DriveFactors d;
        drive_envelopes_SU2(t, d.su2[0], d.su2[1]);
        drive_envelopes_SU3(t, d.su3[0], d.su3[1]);
        if (has_mixed_bilinear_drive) {
            // Field-assisted Fe-Tm exchange: H_{E chi} follows the SU(3)
            // (electric) envelope, H_{B chi} the SU(2) (magnetic) one; only
            // active pulses contribute (the factors of inactive slots are 0).
            d.env_E = d.su3[0] + d.su3[1];
            d.env_B = d.su2[0] + d.su2[1];
            // Polarisation-resolved gates: lab-frame B_eta(t) = sum_k dir_k,eta f_k.
            // A B_z-gated vertex is identically zero for an H||a pump.
            const auto& g0 = field_drive_global_SU2[0];
            const auto& g1 = field_drive_global_SU2[1];
            if (g0.size() >= 3 && g1.size() >= 3) {
                d.env_Bx = g0(0) * d.su2[0] + g1(0) * d.su2[1];
                d.env_By = g0(1) * d.su2[0] + g1(1) * d.su2[1];
                d.env_Bz = g0(2) * d.su2[0] + g1(2) * d.su2[1];
            }
        }
        return d;
    }

// ---- MixedLattice::ode_system ----
    void MixedLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) const {
        evaluate_rhs(x, dxdt, drive_factors(t));
    }

// ---- MixedLattice::landau_lifshitz ----
    void MixedLattice::landau_lifshitz(const ODEState& state, ODEState& dsdt, double t) const {
        evaluate_rhs(state, dsdt, drive_factors(t));
    }

// ---- MixedLattice::evaluate_rhs ----
    void MixedLattice::evaluate_rhs(const ODEState& state, ODEState& dsdt, const DriveFactors& d) const {
        const size_t n_spin = spin_state_size();
        const bool reservoir = (state.size() == n_spin + 1);
        if (!reservoir && state.size() != n_spin) {
            throw std::invalid_argument("MixedLattice::landau_lifshitz: state has " + std::to_string(state.size()) +
                                        " entries, expected " + std::to_string(n_spin) + " (spins) or " +
                                        std::to_string(n_spin + 1) + " (spins + thermal reservoir)");
        }
        if (dsdt.size() != state.size()) dsdt.resize(state.size());
        const size_t offset_SU3 = lattice_size_SU2 * 3;

        // Thermal reservoir: E_dep is the last state entry (0 when absent).
        const double E_dep = reservoir ? state[n_spin] : 0.0;
        double eq3_shift = thermal_heat * E_dep;
        if (eq3_shift > thermal_cap) eq3_shift = thermal_cap;
        const bool accumulate_power = reservoir && thermal_heat != 0.0;

        // Per-channel Bloch relaxation rates, hoisted out of the site loop.
        double gamma[8];
        bool any_gamma = false;
        for (int a = 0; a < 8; ++a) {
            gamma[a] = damping_rates_SU3.size() == 8 ? damping_rates_SU3(a) : 0.0;
            any_gamma = any_gamma || gamma[a] != 0.0;
        }
        const bool linear_torque = linear_drive_torque_SU2 && (d.su2[0] != 0.0 || d.su2[1] != 0.0);
        const double c3 = su3_bracket;

        double power = 0.0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        // One OpenMP team per RHS for both species (disjoint dsdt slices,
        // read-only state); `nowait` lets threads move on to SU(3) sites.
#ifdef _OPENMP
        #pragma omp parallel if(total_sites >= 64) reduction(+:power)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (size_t site = 0; site < lattice_size_SU2; ++site) {
                const size_t idx = site * 3;
                double H[3];
                get_local_field_SU2_flat_into(site, state, offset_SU3, d.su2[0], d.su2[1], H,
                                              d.env_E, d.env_B, d.env_Bx, d.env_By, d.env_Bz);
                const double Sx = state[idx + 0], Sy = state[idx + 1], Sz = state[idx + 2];
                double* out = &dsdt[idx];
                out[0] = H[1] * Sz - H[2] * Sy;
                out[1] = H[2] * Sx - H[0] * Sz;
                out[2] = H[0] * Sy - H[1] * Sx;

                // Field-mediated-conversion ablation: H carries -h(t), so the
                // torque contains -h x S; adding h x (S - S0) leaves -h x S0,
                // dropping the term bilinear in (pulse field, magnon amplitude).
                // See set_linear_drive_torque_SU2().
                if (linear_torque) {
                    const size_t atom = site % N_atoms_SU2;
                    const double* __restrict fd0 = field_drive_SU2[0].data() + atom * 3;
                    const double* __restrict fd1 = field_drive_SU2[1].data() + atom * 3;
                    const double hx = fd0[0] * d.su2[0] + fd1[0] * d.su2[1];
                    const double hy = fd0[1] * d.su2[0] + fd1[1] * d.su2[1];
                    const double hz = fd0[2] * d.su2[0] + fd1[2] * d.su2[1];
                    const double dx = Sx - drive_ref_SU2[idx + 0];
                    const double dy = Sy - drive_ref_SU2[idx + 1];
                    const double dz = Sz - drive_ref_SU2[idx + 2];
                    out[0] += hy * dz - hz * dy;
                    out[1] += hz * dx - hx * dz;
                    out[2] += hx * dy - hy * dx;
                }

                // Gilbert damping (LL form): dS/dt += (alpha/|S|) S x (S x H),
                // S x (S x H) = S (S.H) - H |S|^2.
                if (alpha_gilbert != 0.0) {
                    const double S2 = Sx * Sx + Sy * Sy + Sz * Sz;
                    const double SdotH = Sx * H[0] + Sy * H[1] + Sz * H[2];
                    const double a_S = (S2 > 0.0) ? alpha_gilbert / std::sqrt(S2) : 0.0;
                    out[0] += a_S * (Sx * SdotH - H[0] * S2);
                    out[1] += a_S * (Sy * SdotH - H[1] * S2);
                    out[2] += a_S * (Sz * SdotH - H[2] * S2);
                }
            }

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (size_t site = 0; site < lattice_size_SU3; ++site) {
                const size_t idx = offset_SU3 + site * 8;
                double H[8];
                get_local_field_SU3_flat_into(site, state, offset_SU3, d.su3[0], d.su3[1], H,
                                              d.env_E, d.env_B, d.env_Bx, d.env_By, d.env_Bz);
                su3_torque(H, &state[idx], c3, &dsdt[idx]);

                // Casimir-preserving LL damping -(alpha/|n|) c f(n, P), P = torque.
                if (alpha_SU3 != 0.0) {
                    const double* n = &state[idx];
                    double C[8];
                    su3_torque(n, &dsdt[idx], c3, C);
                    double n2 = 0.0;
                    for (int a = 0; a < 8; ++a) n2 += n[a] * n[a];
                    const double k = (n2 > 0.0) ? alpha_SU3 / std::sqrt(n2) : 0.0;
                    for (int a = 0; a < 8; ++a) dsdt[idx + a] -= k * C[a];
                }

                // Bloch relaxation -Gamma_a (n^a - n^a_eq); the lambda3 target
                // follows the thermal reservoir. The dissipation proxy P omits
                // the population channels lambda3, lambda8 (see the header).
                if (any_gamma) {
                    const SpinVector& eq = equilibrium_SU3[site];
                    for (int a = 0; a < 8; ++a) {
                        if (gamma[a] == 0.0) continue;
                        const double target = (a == 2) ? eq(a) - eq3_shift : eq(a);
                        dsdt[idx + a] -= gamma[a] * (state[idx + a] - target);
                        if (accumulate_power && a != 2 && a != 7) {
                            const double dev = state[idx + a] - eq(a);
                            power += gamma[a] * dev * dev;
                        }
                    }
                }
            }
        }  // end omp parallel

        if (reservoir) dsdt[n_spin] = power - thermal_cool * E_dep;
    }

// ---- MixedLattice::get_local_field_SU2_flat_into ----
    // Full local field dE/dS at an SU(2) site (gradient convention), written
    // into H[0..2]: field, on-site, SU(2)-SU(2), SU(2)-SU(3) and field-assisted
    // bilinears, both trilinears, and the pulse drive with pre-computed factors.
    void MixedLattice::get_local_field_SU2_flat_into(
        size_t site, const ODEState& state, size_t offset_SU3,
        double drive_factor1, double drive_factor2,
        double* __restrict H, double env_E, double env_B,
        double env_Bx, double env_By, double env_Bz) const {
        // Envelope by field-assisted bond tag: 0=E 1=B 2=B_x 3=B_y 4=B_z.
        const double env_lut[5] = { env_E, env_B, env_Bx, env_By, env_Bz };
        const size_t idx = site * 3;
        const double* __restrict field0 = field_SU2[site].data();
        for (size_t a = 0; a < 3; ++a) H[a] = -field0[a];

        // On-site: 2 A S.
        const auto& A = onsite_interaction_SU2[site];
        for (size_t a = 0; a < 3; ++a) {
            double acc = 0.0;
            for (size_t b = 0; b < 3; ++b) acc += A(a, b) * state[idx + b];
            H[a] += 2.0 * acc;
        }

        const auto& bp = bilinear_partners_SU2[site];
        if (!bp.empty()) {
            const double* __restrict J = bilinear_packed_SU2[site].data();
            for (size_t n = 0; n < bp.size(); ++n)
                bilinear_kernel<3, 3>(J + n * 9, &state[bp[n] * 3], H);
        }

        const auto& mbp = mixed_bilinear_partners_SU2[site];
        if (!mbp.empty()) {
            const double* __restrict J = mixed_bilinear_packed_SU2[site].data();
            for (size_t n = 0; n < mbp.size(); ++n)
                bilinear_kernel<3, 8>(J + n * 24, &state[offset_SU3 + mbp[n] * 8], H);
        }

        // Field-assisted (pulse-gated) Fe-Tm exchange H_{E chi} / H_{B chi}.
        const auto& mbpd = mixed_bilinear_drive_partners_SU2[site];
        if (!mbpd.empty() && (env_E != 0.0 || env_B != 0.0 || env_Bx != 0.0 ||
                              env_By != 0.0 || env_Bz != 0.0)) {
            const double* __restrict J = mixed_bilinear_drive_packed_SU2[site].data();
            const auto& tags = mixed_bilinear_drive_envelope_SU2[site];
            for (size_t n = 0; n < mbpd.size(); ++n) {
                const int tag = tags[n];
                const double env = env_lut[(tag >= 0 && tag < 5) ? tag : 1];
                if (env == 0.0) continue;
                scaled_bilinear_kernel<3, 8>(J + n * 24, &state[offset_SU3 + mbpd[n] * 8], env, H);
            }
        }

        const auto& tp = trilinear_partners_SU2[site];
        if (!tp.empty()) {
            const double* __restrict T = trilinear_packed_SU2[site].data();
            for (size_t n = 0; n < tp.size(); ++n)
                trilinear_kernel<3, 3, 3>(T + n * 27, &state[tp[n][0] * 3], &state[tp[n][1] * 3], H);
        }

        // SU(2)-SU(2)-SU(3): the same tensor as the energy and the MC field
        // (no hidden reference subtraction; see set_mixed_trilinear_reference_SU3).
        const auto& mtp = mixed_trilinear_partners_SU2[site];
        if (!mtp.empty()) {
            const double* __restrict T = mixed_trilinear_packed_SU2[site].data();
            for (size_t n = 0; n < mtp.size(); ++n)
                trilinear_kernel<3, 3, 8>(T + n * 72, &state[mtp[n][0] * 3],
                                          &state[offset_SU3 + mtp[n][1] * 8], H);
        }

        if (drive_factor1 != 0.0 || drive_factor2 != 0.0) {
            const size_t atom = site % N_atoms_SU2;
            const double* fd0 = field_drive_SU2[0].data() + atom * 3;
            const double* fd1 = field_drive_SU2[1].data() + atom * 3;
            for (size_t a = 0; a < 3; ++a) H[a] -= fd0[a] * drive_factor1 + fd1[a] * drive_factor2;
        }
    }

// ---- MixedLattice::get_local_field_SU3_flat_into ----
    // Full local field dE/dn at an SU(3) site, written into H[0..7].
    void MixedLattice::get_local_field_SU3_flat_into(
        size_t site, const ODEState& state, size_t offset_SU3,
        double drive_factor1, double drive_factor2,
        double* __restrict H, double env_E, double env_B,
        double env_Bx, double env_By, double env_Bz) const {
        const double env_lut[5] = { env_E, env_B, env_Bx, env_By, env_Bz };
        const size_t idx = offset_SU3 + site * 8;
        const double* __restrict field0 = field_SU3[site].data();
        for (size_t a = 0; a < 8; ++a) H[a] = -field0[a];

        const auto& A = onsite_interaction_SU3[site];
        for (size_t a = 0; a < 8; ++a) {
            double acc = 0.0;
            for (size_t b = 0; b < 8; ++b) acc += A(a, b) * state[idx + b];
            H[a] += 2.0 * acc;
        }

        const auto& bp = bilinear_partners_SU3[site];
        if (!bp.empty()) {
            const double* __restrict J = bilinear_packed_SU3[site].data();
            for (size_t n = 0; n < bp.size(); ++n)
                bilinear_kernel<8, 8>(J + n * 64, &state[offset_SU3 + bp[n] * 8], H);
        }

        const auto& mbp = mixed_bilinear_partners_SU3[site];
        if (!mbp.empty()) {
            const double* __restrict J = mixed_bilinear_packed_SU3[site].data();
            for (size_t n = 0; n < mbp.size(); ++n)
                bilinear_kernel<8, 3>(J + n * 24, &state[mbp[n] * 3], H);
        }

        const auto& mbpd = mixed_bilinear_drive_partners_SU3[site];
        if (!mbpd.empty() && (env_E != 0.0 || env_B != 0.0 || env_Bx != 0.0 ||
                              env_By != 0.0 || env_Bz != 0.0)) {
            const double* __restrict J = mixed_bilinear_drive_packed_SU3[site].data();
            const auto& tags = mixed_bilinear_drive_envelope_SU3[site];
            for (size_t n = 0; n < mbpd.size(); ++n) {
                const int tag = tags[n];
                const double env = env_lut[(tag >= 0 && tag < 5) ? tag : 1];
                if (env == 0.0) continue;
                scaled_bilinear_kernel<8, 3>(J + n * 24, &state[mbpd[n] * 3], env, H);
            }
        }

        const auto& tp = trilinear_partners_SU3[site];
        if (!tp.empty()) {
            const double* __restrict T = trilinear_packed_SU3[site].data();
            for (size_t n = 0; n < tp.size(); ++n)
                trilinear_kernel<8, 8, 8>(T + n * 512, &state[offset_SU3 + tp[n][0] * 8],
                                          &state[offset_SU3 + tp[n][1] * 8], H);
        }

        const auto& mtp = mixed_trilinear_partners_SU3[site];
        if (!mtp.empty()) {
            const double* __restrict T = mixed_trilinear_packed_SU3[site].data();
            for (size_t n = 0; n < mtp.size(); ++n)
                trilinear_kernel<8, 3, 3>(T + n * 72, &state[mtp[n][0] * 3], &state[mtp[n][1] * 3], H);
        }

        if (drive_factor1 != 0.0 || drive_factor2 != 0.0) {
            const size_t atom = site % N_atoms_SU3;
            const double* fd0 = field_drive_SU3[0].data() + atom * 8;
            const double* fd1 = field_drive_SU3[1].data() + atom * 8;
            for (size_t a = 0; a < 8; ++a) H[a] -= fd0[a] * drive_factor1 + fd1[a] * drive_factor2;
        }
    }

// ---- MixedLattice::relative_stationarity_residual ----
    double MixedLattice::relative_stationarity_residual(const ODEState& state) const {
        if (state.size() < spin_state_size()) {
            throw std::invalid_argument("relative_stationarity_residual: state too short");
        }
        const size_t offset_SU3 = lattice_size_SU2 * 3;
        double worst = 0.0;
        for (size_t site = 0; site < lattice_size_SU2; ++site) {
            double H[3], dS[3];
            get_local_field_SU2_flat_into(site, state, offset_SU3, 0.0, 0.0, H);
            const double* S = &state[site * 3];
            dS[0] = H[1] * S[2] - H[2] * S[1];
            dS[1] = H[2] * S[0] - H[0] * S[2];
            dS[2] = H[0] * S[1] - H[1] * S[0];
            const double scale = std::sqrt((H[0] * H[0] + H[1] * H[1] + H[2] * H[2]) *
                                           (S[0] * S[0] + S[1] * S[1] + S[2] * S[2]));
            const double rate = std::sqrt(dS[0] * dS[0] + dS[1] * dS[1] + dS[2] * dS[2]);
            if (scale > 0.0) worst = std::max(worst, rate / scale);
        }
        const bool damped = damping_rates_SU3.size() == 8 && damping_rates_SU3.cwiseAbs().maxCoeff() > 0.0;
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            double H[8], dn[8];
            get_local_field_SU3_flat_into(site, state, offset_SU3, 0.0, 0.0, H);
            const double* n = &state[offset_SU3 + site * 8];
            su3_torque(H, n, su3_bracket, dn);
            // Bloch relaxation towards n_eq also moves a torque-free state.
            if (damped) {
                for (int a = 0; a < 8; ++a)
                    dn[a] -= damping_rates_SU3(a) * (n[a] - equilibrium_SU3[site](a));
            }
            double h2 = 0.0, n2 = 0.0, r2 = 0.0;
            for (int a = 0; a < 8; ++a) { h2 += H[a] * H[a]; n2 += n[a] * n[a]; r2 += dn[a] * dn[a]; }
            const double scale = su3_bracket * std::sqrt(h2 * n2);
            if (scale > 0.0) worst = std::max(worst, std::sqrt(r2) / scale);
        }
        return worst;
    }

// ---- MixedLattice::max_dSdt_norm_no_drive ----
    double MixedLattice::max_dSdt_norm_no_drive() const {
        const ODEState state = spins_to_state();
        ODEState dsdt(state.size(), 0.0);
        evaluate_rhs(state, dsdt, DriveFactors{});
        double max_norm = 0.0;
        for (size_t i = 0; i < spin_state_size(); ++i) max_norm = std::max(max_norm, std::abs(dsdt[i]));
        return max_norm;
    }

// ---- MixedLattice::gpu_unsupported_reason ----
    string MixedLattice::gpu_unsupported_reason(bool want_spin_states) const {
        string why;
        auto add = [&why](const char* what) {
            if (!why.empty()) why += ", ";
            why += what;
        };
        auto any_entries = [](const auto& table) {
            for (const auto& per_site : table) if (!per_site.empty()) return true;
            return false;
        };
        if (any_entries(trilinear_partners_SU2) || any_entries(trilinear_partners_SU3))
            add("SU(2)-SU(2)-SU(2) / SU(3)-SU(3)-SU(3) trilinear couplings");
        if (any_entries(mixed_trilinear_partners_SU2) || any_entries(mixed_trilinear_partners_SU3))
            add("mixed SU(2)-SU(2)-SU(3) trilinear couplings");
        if (has_mixed_bilinear_drive) add("field-assisted Fe-Tm exchange");
        if (alpha_gilbert != 0.0) add("SU(2) Gilbert damping");
        if (alpha_SU3 != 0.0) add("SU(3) Landau-Lifshitz damping");
        if (damping_rates_SU3.size() > 0 && damping_rates_SU3.cwiseAbs().maxCoeff() > 0.0)
            add("SU(3) Bloch damping");
        if (thermal_heat != 0.0) add("thermal reservoir");
        if (field_drive_freq_SU3_2 != 0.0) add("two-colour SU(3) pulse");
        if (!tabulated_pulse_times.empty()) add("tabulated pulse waveform");
        if (linear_drive_torque_SU2) add("linearised SU(2) drive torque");
        if (su3_bracket != classical_spin::su3::kGellMannBracket) add("legacy SU(3) bracket convention");
        if (want_spin_states) add("spin-state trajectory output");
        return why;
    }

// ---- MixedLattice::set_mixed_trilinear_reference_SU3 ----
    void MixedLattice::set_mixed_trilinear_reference_SU3(const SpinConfigSU3& reference) {
        if (!reference.empty()) {
            if (reference.size() != lattice_size_SU3) {
                throw std::invalid_argument("set_mixed_trilinear_reference_SU3: expected " +
                                            std::to_string(lattice_size_SU3) + " SU(3) vectors (got " +
                                            std::to_string(reference.size()) + ")");
            }
            for (const auto& r : reference) {
                if (r.size() != 8 || !r.allFinite()) {
                    throw std::invalid_argument("set_mixed_trilinear_reference_SU3: every reference "
                                                "vector must have 8 finite components");
                }
            }
        }
        // Undo a previous reference: its Fe-Fe bonds were appended at the end
        // of each site's bilinear list, its on-site part is restored from the
        // saved matrices.
        if (!reference_bond_slots_.empty()) {
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                const size_t k = reference_bond_slots_[i].size();
                bilinear_partners_SU2[i].resize(bilinear_partners_SU2[i].size() - k);
                bilinear_interaction_SU2[i].resize(bilinear_interaction_SU2[i].size() - k);
            }
            onsite_interaction_SU2 = reference_saved_onsite_SU2_;
        }
        reference_bond_slots_.clear();
        reference_saved_onsite_SU2_.clear();
        trilinear_reference_SU3_.clear();

        if (!reference.empty()) {
            // T(S_i, S_j, n_k - r_k) = T(S_i, S_j, n_k) - S_i^a J^{ab} S_j^b,
            // J^{ab} = sum_c T_i^a(b, c) r_k^c, for every SU(2)-side entry
            // (i; j, k) of a triple. Distinct j: each triple has an entry at i
            // and one at j, giving the bonds i->j (J) and j->i (J^T), counted
            // 1/2 + 1/2 like any bond. j == i (a TmFeO3 W vertex is
            // T(S_i, S_i, n_k)): the triple has two entries at i, with J and
            // J^T, and the subtraction -S_i^T J S_i is on-site; each entry adds
            // A -= (J + J^T)/4 (E = S^T A S, field 2 A S, A symmetric).
            reference_saved_onsite_SU2_ = onsite_interaction_SU2;
            reference_bond_slots_.assign(lattice_size_SU2, {});
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t n = 0; n < mixed_trilinear_partners_SU2[i].size(); ++n) {
                    const size_t j = mixed_trilinear_partners_SU2[i][n][0];
                    const size_t k = mixed_trilinear_partners_SU2[i][n][1];
                    const auto& T = mixed_trilinear_interaction_SU2[i][n];
                    SpinMatrix J = SpinMatrix::Zero(spin_dim_SU2, spin_dim_SU2);
                    for (size_t a = 0; a < spin_dim_SU2; ++a)
                        for (size_t b = 0; b < spin_dim_SU2; ++b)
                            for (size_t c = 0; c < spin_dim_SU3; ++c)
                                J(a, b) += T[a](b, c) * reference[k](c);
                    if (j == i) {
                        onsite_interaction_SU2[i] -= 0.25 * (J + J.transpose());
                    } else {
                        bilinear_partners_SU2[i].push_back(j);
                        bilinear_interaction_SU2[i].push_back(-J);
                        reference_bond_slots_[i].push_back({bilinear_partners_SU2[i].size() - 1, n});
                    }
                }
            }
            trilinear_reference_SU3_ = reference;
        }

        num_bi_SU2 = 0;
        for (const auto& bp : bilinear_partners_SU2) num_bi_SU2 = std::max(num_bi_SU2, bp.size());
        build_packed_interaction_buffers();
        build_color_partition();
        invalidate_all_fields();
    }

// =============================================================
// Pulse drives
// =============================================================

// ---- MixedLattice::integrate_pulse_train ----
    MixedLattice::PumpProbeTrajectory MixedLattice::integrate_pulse_train(
        const dyn::TimeGrid& grid, size_t k_start, ODEState x, const string& method,
        vector<vector<double>>* spin_state_out, double abs_tol, double rel_tol,
        const vector<size_t>& capture_at, vector<ODEState>* captured) {
        // The installed pulse train belongs to this trajectory only: remove it
        // on every exit path, including exceptions.
        struct PulseGuard {
            MixedLattice* lat;
            ~PulseGuard() { lat->reset_pulse(); }
        } guard{this};

        const auto method_id = dyn::parse_ode_method(method);
        if (k_start >= grid.n) throw std::invalid_argument("pulse drive: start index beyond the time grid");
        if (x.size() != ode_state_size()) {
            throw std::invalid_argument("pulse drive: initial state has " + std::to_string(x.size()) +
                                        " entries, expected " + std::to_string(ode_state_size()));
        }

        // Windows where a pulse acts (its envelope exceeds ~1.6e-9 of the
        // peak): the adaptive step is capped there at min(T_step, width/4,
        // carrier period/4) so a pulse can never be stepped over; elsewhere the
        // error controller alone sets the step.
        const bool tabulated = !tabulated_pulse_times.empty();
        std::vector<std::pair<double, double>> windows;
        double cap = grid.dt;
        auto add_species = [&](double amp, double width, double freq, double freq2,
                               const array<double, 2>& centres) {
            if (amp == 0.0) return;
            for (size_t k = 0; k < std::min<size_t>(n_active_pulses, 2); ++k) {
                if (tabulated) {
                    windows.emplace_back(centres[k] + tabulated_pulse_times.front(),
                                         centres[k] + tabulated_pulse_times.back());
                } else {
                    const double W = classical_spin_pulse_chunking::kPulseWindowSigmas * width;
                    windows.emplace_back(centres[k] - W, centres[k] + W);
                }
            }
            if (!tabulated) {
                cap = std::min(cap, 0.25 * width);
                for (double f : {freq, freq2}) {
                    if (f != 0.0) cap = std::min(cap, 0.25 * 2.0 * M_PI / std::abs(f));
                }
            }
        };
        add_species(field_drive_amp_SU2, field_drive_width_SU2, field_drive_freq_SU2, 0.0, t_pulse_SU2);
        add_species(field_drive_amp_SU3, field_drive_width_SU3, field_drive_freq_SU3,
                    field_drive_freq_SU3_2, t_pulse_SU3);
        const auto segments = dyn::segments_with_windows(grid, windows, cap);

        PumpProbeTrajectory trajectory;
        trajectory.reserve(grid.n - k_start);
        if (spin_state_out) {
            spin_state_out->clear();
            spin_state_out->reserve(grid.n - k_start);
        }
        if (captured) captured->assign(capture_at.size(), ODEState());
        const size_t n_spin = spin_state_size();
        auto observer = [&](const ODEState& s, size_t k) {
            require_finite(s, grid[k], "pulse drive");
            trajectory.emplace_back(grid[k], observe(s.data()));
            if (spin_state_out) spin_state_out->emplace_back(s.begin(), s.begin() + n_spin);
            if (captured) {
                for (size_t c = 0; c < capture_at.size(); ++c) {
                    if (capture_at[c] == k) (*captured)[c] = s;
                }
            }
        };
        auto system = [this](const ODEState& s, ODEState& ds, double t) { landau_lifshitz(s, ds, t); };
        dyn::integrate_on_time_grid(system, x, grid, 1, method_id, abs_tol, rel_tol, observer,
                                    segments, k_start);
        return trajectory;
    }

// ---- MixedLattice::single_pulse_drive ----
    MixedLattice::PumpProbeTrajectory MixedLattice::single_pulse_drive(
               const vector<SpinVector>& field_in_SU2, const vector<SpinVector>& field_in_SU3,
               double t_B,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method, bool use_gpu,
               vector<vector<double>>* spin_state_out,
               bool pulse_window_chunking,
               double abs_tol, double rel_tol) {
        (void) pulse_window_chunking;  // superseded by exact-grid integration
        if (use_gpu) {
#ifdef CUDA_ENABLED
            check_gpu_supported(spin_state_out != nullptr);
            return single_pulse_drive_gpu(field_in_SU2, field_in_SU3, t_B,
                            pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                            pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                            T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED); "
                         "running single_pulse_drive on the CPU." << endl;
#endif
        }
        const auto grid = dyn::TimeGrid::covering(T_start, T_end, step_size,
                                                  "single_pulse_drive (T_start, T_end, T_step)");
        configure_pulse_train(1, field_in_SU2, field_in_SU3, t_B, {}, {}, t_B,
                              pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                              pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        return integrate_pulse_train(grid, 0, spins_to_state(), method, spin_state_out, abs_tol, rel_tol);
    }

// ---- MixedLattice::double_pulse_drive ----
    MixedLattice::PumpProbeTrajectory MixedLattice::double_pulse_drive(
               const vector<SpinVector>& field_in_1_SU2, const vector<SpinVector>& field_in_1_SU3,
               double t_B_1,
               const vector<SpinVector>& field_in_2_SU2, const vector<SpinVector>& field_in_2_SU3,
               double t_B_2,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method, bool use_gpu,
               vector<vector<double>>* spin_state_out,
               bool pulse_window_chunking,
               double abs_tol, double rel_tol) {
        (void) pulse_window_chunking;
        if (use_gpu) {
#ifdef CUDA_ENABLED
            check_gpu_supported(spin_state_out != nullptr);
            return double_pulse_drive_gpu(field_in_1_SU2, field_in_1_SU3, t_B_1,
                                field_in_2_SU2, field_in_2_SU3, t_B_2,
                                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED); "
                         "running double_pulse_drive on the CPU." << endl;
#endif
        }
        const auto grid = dyn::TimeGrid::covering(T_start, T_end, step_size,
                                                  "double_pulse_drive (T_start, T_end, T_step)");
        configure_pulse_train(2, field_in_1_SU2, field_in_1_SU3, t_B_1, field_in_2_SU2, field_in_2_SU3, t_B_2,
                              pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                              pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        return integrate_pulse_train(grid, 0, spins_to_state(), method, spin_state_out, abs_tol, rel_tol);
    }

// =============================================================
// Molecular dynamics
// =============================================================

namespace {

// Report how far the initial state is from a stationary, physical one. Free
// dynamics from a thermal state is legitimate, so these are warnings.
void report_initial_state(const MixedLattice& lat, const MixedLattice::ODEState& x, const char* who) {
    const double residual = lat.relative_stationarity_residual(x);
    const double rho_min = lat.min_SU3_density_eigenvalue();
    cout << who << ": initial state stationarity residual max|dS/dt|/(|H||S|) = " << residual
         << ", min SU(3) density-matrix eigenvalue = " << rho_min << endl;
    if (rho_min < -1e-8) {
        std::cerr << "Warning (" << who << "): an SU(3) state is not a physical density matrix "
                  << "(eigenvalue " << rho_min << " < 0); consider physicalize_SU3_state()." << endl;
    }
}

#ifdef HDF5_ENABLED
// Append 1-D double datasets to a group of an existing file (SEC2 driver, so
// it also works against a parallel HDF5 build).
void append_hdf5_columns(const std::string& filename, const std::string& group_name,
                         const std::vector<std::pair<std::string, const std::vector<double>*>>& columns) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_sec2(fapl);
    hid_t fid = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
    H5Pclose(fapl);
    if (fid < 0) throw std::runtime_error("cannot reopen " + filename + " to write " + group_name);
    // H5File(hid_t) takes an additional reference; drop ours so close()
    // really closes the file.
    H5::H5File file(fid);
    H5Idec_ref(fid);
    H5::Group group = file.createGroup(group_name);
    for (const auto& [name, data] : columns) {
        hsize_t dims[1] = {data->size()};
        H5::DataSpace space(1, dims);
        H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
        ds.write(data->data(), H5::PredType::NATIVE_DOUBLE);
    }
    group.close();
    file.close();
}
#endif

}  // namespace

// ---- MixedLattice::molecular_dynamics ----
    void MixedLattice::molecular_dynamics(double T_start, double T_end, double dt_initial,
                           const string& out_dir, size_t save_interval,
                           const string& method, bool use_gpu,
                           double abs_tol, double rel_tol) {
        if (use_gpu) {
#ifdef CUDA_ENABLED
            check_gpu_supported();
            (void) abs_tol; (void) rel_tol;  // the GPU integrators are fixed-step
            molecular_dynamics_gpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
            return;
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED); "
                         "running molecular dynamics on the CPU." << endl;
#endif
        }
        molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method, abs_tol, rel_tol);
    }

// ---- MixedLattice::molecular_dynamics_cpu ----
    void MixedLattice::molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           const string& out_dir, size_t save_interval,
                           const string& method,
                           double abs_tol, double rel_tol) {
#ifndef HDF5_ENABLED
        (void) T_start; (void) T_end; (void) dt_initial; (void) out_dir;
        (void) save_interval; (void) method; (void) abs_tol; (void) rel_tol;
        throw std::runtime_error("molecular_dynamics requires HDF5 output; rebuild with HDF5_ENABLED");
#else
        if (save_interval == 0) throw std::invalid_argument("molecular_dynamics: save_interval must be >= 1");
        if (!(dt_initial > 0.0)) throw std::invalid_argument("molecular_dynamics: time step must be > 0");
        const auto method_id = dyn::parse_ode_method(method);
        const auto grid = dyn::TimeGrid::covering(T_start, T_end, dt_initial * static_cast<double>(save_interval),
                                                  "molecular_dynamics (T_start, T_end, dt * save_interval)");
        ensure_directory_exists(out_dir);

        cout << "Mixed-lattice molecular dynamics: t = " << T_start << " -> " << grid.t_end()
             << ", method " << dyn::ode_method_name(method_id) << ", dt " << dt_initial
             << (dyn::is_adaptive(method_id) ? " (initial step)" : "")
             << ", " << grid.n << " samples every " << grid.dt << endl;

        ODEState state = spins_to_state();
        report_initial_state(*this, state, "molecular_dynamics");

        std::unique_ptr<HDF5MixedMDWriter> writer;
        const string hdf5_file = out_dir.empty() ? string() : out_dir + "/trajectory.h5";
        if (!hdf5_file.empty()) {
            writer = std::make_unique<HDF5MixedMDWriter>(
                hdf5_file,
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3, dyn::ode_method_name(method_id),
                dt_initial, T_start, grid.t_end(), save_interval,
                spin_length_SU2, spin_length_SU3,
                &site_positions_SU2, &site_positions_SU3, grid.n);
        }

        // Conserved quantities of the undamped, undriven flow, per site:
        // |S_i| and the SU(3) Casimirs |n_i|^2, d_abc n^a n^b n^c.
        const size_t offset_SU3 = lattice_size_SU2 * 3;
        vector<double> S0(lattice_size_SU2), C2_0(lattice_size_SU3), C3_0(lattice_size_SU3);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const double* S = &state[i * 3];
            S0[i] = std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            C2_0[i] = classical_spin::su3::casimir2(&state[offset_SU3 + i * 8]);
            C3_0[i] = classical_spin::su3::casimir3(&state[offset_SU3 + i * 8]);
        }
        vector<double> d_time, d_energy, d_spin, d_c2, d_c3;
        for (auto* v : {&d_time, &d_energy, &d_spin, &d_c2, &d_c3}) v->reserve(grid.n);
        const double n_sites = static_cast<double>(lattice_size_SU2 + lattice_size_SU3);
        const size_t progress_every = std::max<size_t>(1, grid.n / 10);

        auto observer = [&](const ODEState& x, size_t k) {
            const double t = grid[k];
            require_finite(x, t, "molecular_dynamics");
            const Observables o = observe(x.data());
            if (writer) {
                writer->write_flat_step(t, o.first[0], o.first[1], o.first[2],
                                        o.second[0], o.second[1], o.second[2], x.data());
            }
            double ds = 0.0, dc2 = 0.0, dc3 = 0.0;
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                const double* S = &x[i * 3];
                ds = std::max(ds, std::abs(std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]) - S0[i]));
            }
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                const double* n = &x[offset_SU3 + i * 8];
                dc2 = std::max(dc2, std::abs(classical_spin::su3::casimir2(n) - C2_0[i]));
                dc3 = std::max(dc3, std::abs(classical_spin::su3::casimir3(n) - C3_0[i]));
            }
            const double e = total_energy_flat(x.data()) / n_sites;
            d_time.push_back(t);
            d_energy.push_back(e);
            d_spin.push_back(ds);
            d_c2.push_back(dc2);
            d_c3.push_back(dc3);
            if (k % progress_every == 0 || k + 1 == grid.n) {
                cout << "t=" << t << ", E/N=" << std::setprecision(12) << e << std::setprecision(6)
                     << ", |M_SU2|=" << o.first[1].norm() << ", |M_SU3|=" << o.second[1].norm()
                     << ", max||S|-S0|=" << ds << ", max|dC2|=" << dc2 << endl;
            }
        };
        auto system = [this](const ODEState& x, ODEState& dxdt, double t) { landau_lifshitz(x, dxdt, t); };
        dyn::integrate_on_time_grid(system, state, grid, save_interval, method_id, abs_tol, rel_tol, observer);

        if (writer) {
            writer->close();
            append_hdf5_columns(hdf5_file, "/diagnostics",
                                {{"times", &d_time},
                                 {"energy_per_site", &d_energy},
                                 {"spin_length_drift_SU2", &d_spin},
                                 {"casimir2_drift_SU3", &d_c2},
                                 {"casimir3_drift_SU3", &d_c3}});
            cout << "Trajectory (" << grid.n << " samples) and diagnostics written to " << hdf5_file << endl;
        }
        cout << "Energy drift |E(t_end) - E(t0)|/N = " << std::abs(d_energy.back() - d_energy.front()) << endl;
#endif // HDF5_ENABLED
    }

// =============================================================
// Pump-probe / 2DCS spectroscopy
// =============================================================

// ---- MixedLattice::synthesize_M1_from_M0 ----
    MixedLattice::PumpProbeTrajectory MixedLattice::synthesize_M1_from_M0(
        const PumpProbeTrajectory& M_pulse_trajectory,
        const Observables& M_ground,
        double tau,
        double T_start, double T_end, double T_step) const {
        (void) T_start;
        (void) T_end;
        if (!(T_step > 0.0)) throw std::invalid_argument("synthesize_M1_from_M0: T_step must be > 0");
        const double r = tau / T_step;
        const double m = std::round(r);
        if (std::abs(r - m) > 1e-9 * std::max(1.0, std::abs(m))) {
            throw std::invalid_argument("synthesize_M1_from_M0: tau = " + std::to_string(tau) +
                                        " is not a multiple of T_step = " + std::to_string(T_step));
        }
        if (m < 0.0) throw std::invalid_argument("synthesize_M1_from_M0: requires tau >= 0");
        const size_t shift = static_cast<size_t>(m);
        PumpProbeTrajectory M1;
        M1.reserve(M_pulse_trajectory.size());
        for (size_t k = 0; k < M_pulse_trajectory.size(); ++k) {
            M1.emplace_back(M_pulse_trajectory[k].first,
                            k < shift ? M_ground : M_pulse_trajectory[k - shift].second);
        }
        return M1;
    }

// Validated description of one pump-probe scan, shared by the serial and the
// MPI driver so both run exactly the same physics.
struct MixedLattice::SpectroscopyPlan {
    const vector<SpinVector>* pump_SU2 = nullptr;
    const vector<SpinVector>* pump_SU3 = nullptr;
    const vector<SpinVector>* probe_SU2 = nullptr;
    const vector<SpinVector>* probe_SU3 = nullptr;
    double amp_SU2 = 0.0, width_SU2 = 1.0, freq_SU2 = 0.0;
    double amp_SU3 = 0.0, width_SU3 = 1.0, freq_SU3 = 0.0;
    string method;
    double abs_tol = 0.0, rel_tol = 0.0;
    bool gpu = false;
    bool save_spins = false;
    dyn::TimeGrid grid;
    vector<double> taus;
    bool distinct_probe = false;
    double residual = 0.0;          // relative stationarity of the ground state
    bool w1 = false;                // synthesise M1 from M0
    string w1_note;                 // why W1 is off (empty when on or not requested)
    bool m01_from_m0 = false;       // continue M01 from stored M0 states
    double probe_lead = 0.0;        // a pulse is negligible before its centre - probe_lead
    bool pump_truncated = false;    // the pump (t = 0) already acts at T_start
    vector<size_t> m01_start;       // per delay: grid index where M01 is integrated from
    vector<size_t> checkpoint_indices;  // distinct m01_start > 0 (states kept from the M0 run)
};

struct MixedLattice::DelayResult {
    PumpProbeTrajectory M1;          // empty when synthesised (W1)
    PumpProbeTrajectory M01;         // samples k0 .. n-1
    vector<double> M1_spins;         // flat, same samples as M1 (save_spins only)
    vector<double> M01_spins;        // flat, samples k0 .. n-1 (save_spins only)
    size_t k0 = 0;
};

namespace {

vector<double> flatten_states(const vector<vector<double>>& states) {
    vector<double> flat;
    if (states.empty()) return flat;
    flat.reserve(states.size() * states.front().size());
    for (const auto& s : states) flat.insert(flat.end(), s.begin(), s.end());
    return flat;
}

// Observables are 1 + 3*3 + 3*8 doubles per sample (time first).
constexpr size_t kObsDoubles = 1 + 3 * 3 + 3 * 8;

void flatten_trajectory(const MixedLattice::PumpProbeTrajectory& tr, vector<double>& buf) {
    buf.resize(tr.size() * kObsDoubles);
    for (size_t k = 0; k < tr.size(); ++k) {
        double* p = &buf[k * kObsDoubles];
        *p++ = tr[k].first;
        for (const auto& v : tr[k].second.first) for (Eigen::Index d = 0; d < 3; ++d) *p++ = v(d);
        for (const auto& v : tr[k].second.second) for (Eigen::Index d = 0; d < 8; ++d) *p++ = v(d);
    }
}

MixedLattice::PumpProbeTrajectory unflatten_trajectory(const vector<double>& buf) {
    MixedLattice::PumpProbeTrajectory tr(buf.size() / kObsDoubles);
    for (size_t k = 0; k < tr.size(); ++k) {
        const double* p = &buf[k * kObsDoubles];
        tr[k].first = *p++;
        for (auto& v : tr[k].second.first) { v = Eigen::Map<const Eigen::VectorXd>(p, 3); p += 3; }
        for (auto& v : tr[k].second.second) { v = Eigen::Map<const Eigen::VectorXd>(p, 8); p += 8; }
    }
    return tr;
}

// M01 over the whole grid: samples before k0 are those of M0 (identical drive
// up to the probe lead), samples from k0 on come from the continued run.
MixedLattice::PumpProbeTrajectory assemble_m01(const MixedLattice::PumpProbeTrajectory& M0,
                                               MixedLattice::PumpProbeTrajectory&& partial, size_t k0) {
    if (k0 == 0) return std::move(partial);
    MixedLattice::PumpProbeTrajectory full;
    full.reserve(k0 + partial.size());
    full.insert(full.end(), M0.begin(), M0.begin() + static_cast<std::ptrdiff_t>(k0));
    full.insert(full.end(), std::make_move_iterator(partial.begin()), std::make_move_iterator(partial.end()));
    return full;
}

vector<double> assemble_m01_spins(const vector<double>& M0_spins, vector<double>&& partial, size_t k0,
                                  size_t state_dim) {
    if (k0 == 0 || partial.empty()) return std::move(partial);
    vector<double> full;
    full.reserve(k0 * state_dim + partial.size());
    full.insert(full.end(), M0_spins.begin(), M0_spins.begin() + static_cast<std::ptrdiff_t>(k0 * state_dim));
    full.insert(full.end(), partial.begin(), partial.end());
    return full;
}

void require_grid_length(const MixedLattice::PumpProbeTrajectory& tr, size_t n, const char* what) {
    if (tr.size() != n) {
        throw std::runtime_error(std::string("pump-probe: ") + what + " has " + std::to_string(tr.size()) +
                                 " samples, the time grid has " + std::to_string(n));
    }
}

}  // namespace

// ---- MixedLattice::plan_spectroscopy ----
    MixedLattice::SpectroscopyPlan MixedLattice::plan_spectroscopy(
        const vector<SpinVector>& field_in_SU2, const vector<SpinVector>& field_in_SU3,
        const vector<SpinVector>& field_in_SU2_B, const vector<SpinVector>& field_in_SU3_B,
        double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
        double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
        double tau_start, double tau_end, double tau_step,
        double T_start, double T_end, double T_step,
        const string& method, bool use_gpu, bool save_spin_trajectories,
        bool reuse_m0_for_m1, double stationarity_tol,
        bool reuse_m0_for_m01, double abs_tol, double rel_tol) const {
        SpectroscopyPlan p;
        validate_pulse_directions(field_in_SU2, N_atoms_SU2, spin_dim_SU2, "pump direction (SU2)");
        validate_pulse_directions(field_in_SU3, N_atoms_SU3, spin_dim_SU3, "pump direction (SU3)");
        validate_pulse_directions(field_in_SU2_B, N_atoms_SU2, spin_dim_SU2, "probe direction (SU2)");
        validate_pulse_directions(field_in_SU3_B, N_atoms_SU3, spin_dim_SU3, "probe direction (SU3)");
        validate_pulse_shape(pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2, "pump-probe (SU2 pulse)");
        validate_pulse_shape(pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3, "pump-probe (SU3 pulse)");
        (void) dyn::parse_ode_method(method);

        p.pump_SU2 = &field_in_SU2;
        p.pump_SU3 = &field_in_SU3;
        p.distinct_probe = !field_in_SU2_B.empty() || !field_in_SU3_B.empty();
        p.probe_SU2 = field_in_SU2_B.empty() ? &field_in_SU2 : &field_in_SU2_B;
        p.probe_SU3 = field_in_SU3_B.empty() ? &field_in_SU3 : &field_in_SU3_B;
        p.amp_SU2 = pulse_amp_SU2; p.width_SU2 = pulse_width_SU2; p.freq_SU2 = pulse_freq_SU2;
        p.amp_SU3 = pulse_amp_SU3; p.width_SU3 = pulse_width_SU3; p.freq_SU3 = pulse_freq_SU3;
        p.method = method;
        p.abs_tol = abs_tol;
        p.rel_tol = rel_tol;
#ifdef CUDA_ENABLED
        p.gpu = use_gpu;
        if (p.gpu) check_gpu_supported(save_spin_trajectories);
#else
        (void) use_gpu;
#endif
        p.save_spins = save_spin_trajectories;
        p.grid = dyn::TimeGrid::covering(T_start, T_end, T_step, "pump-probe time grid (T_start, T_end, T_step)");
        p.taus = dyn::delay_grid(tau_start, tau_end, tau_step, "pump-probe delay grid (tau_start, tau_end, tau_step)");

        // Lead of a pulse before its centre: 9 widths for the Gaussian (envelope
        // < 1.6e-9 of the peak), the start of the table for a tabulated pulse.
        double lead = 0.0;
        if (!tabulated_pulse_times.empty()) {
            lead = -tabulated_pulse_times.front();
        } else {
            if (pulse_amp_SU2 != 0.0) lead = std::max(lead, classical_spin_pulse_chunking::kPulseWindowSigmas * pulse_width_SU2);
            if (pulse_amp_SU3 != 0.0) lead = std::max(lead, classical_spin_pulse_chunking::kPulseWindowSigmas * pulse_width_SU3);
        }
        p.probe_lead = lead;
        p.pump_truncated = (T_start > -lead);

        p.residual = relative_stationarity_residual(spins_to_state());
        if (reuse_m0_for_m1) {
            bool on_grid = true, non_negative = true;
            for (double tau : p.taus) {
                const double r = tau / T_step;
                on_grid = on_grid && std::abs(r - std::round(r)) <= 1e-9 * std::max(1.0, std::abs(r));
                non_negative = non_negative && tau >= 0.0;
            }
            if (p.gpu) p.w1_note = "GPU backend";
            else if (p.distinct_probe) p.w1_note = "probe directions differ from the pump";
            else if (p.save_spins) p.w1_note = "spin-state output requested";
            else if (!non_negative) p.w1_note = "negative delays";
            else if (!on_grid) p.w1_note = "delays are not multiples of T_step";
            else if (p.pump_truncated) p.w1_note = "the pump is already on at T_start";
            else if (!(p.residual <= stationarity_tol)) {
                std::ostringstream os;
                os << "initial state not stationary (residual " << p.residual << " > tol " << stationarity_tol << ")";
                p.w1_note = os.str();
            } else {
                p.w1 = true;
            }
        }

        p.m01_from_m0 = reuse_m0_for_m01 && !p.gpu;
        p.m01_start.assign(p.taus.size(), 0);
        if (p.m01_from_m0) {
            for (size_t j = 0; j < p.taus.size(); ++j) {
                const double lo = (p.taus[j] - lead - T_start) / T_step;
                if (lo >= 1.0) {
                    p.m01_start[j] = std::min(static_cast<size_t>(std::floor(lo)), p.grid.n - 1);
                }
            }
            for (size_t k : p.m01_start) if (k > 0) p.checkpoint_indices.push_back(k);
            std::sort(p.checkpoint_indices.begin(), p.checkpoint_indices.end());
            p.checkpoint_indices.erase(std::unique(p.checkpoint_indices.begin(), p.checkpoint_indices.end()),
                                       p.checkpoint_indices.end());
        }
        return p;
    }

// ---- MixedLattice::run_reference_trajectory ----
    MixedLattice::PumpProbeTrajectory MixedLattice::run_reference_trajectory(
        const SpectroscopyPlan& p, const ODEState& x_ground,
        vector<vector<double>>* spin_states, vector<ODEState>* checkpoints) {
        if (p.gpu) {
            return single_pulse_drive(*p.pump_SU2, *p.pump_SU3, 0.0,
                                      p.amp_SU2, p.width_SU2, p.freq_SU2, p.amp_SU3, p.width_SU3, p.freq_SU3,
                                      p.grid.t0, p.grid.t_end(), p.grid.dt, p.method, true, nullptr, true,
                                      p.abs_tol, p.rel_tol);
        }
        configure_pulse_train(1, *p.pump_SU2, *p.pump_SU3, 0.0, {}, {}, 0.0,
                              p.amp_SU2, p.width_SU2, p.freq_SU2, p.amp_SU3, p.width_SU3, p.freq_SU3);
        return integrate_pulse_train(p.grid, 0, x_ground, p.method, spin_states, p.abs_tol, p.rel_tol,
                                     p.checkpoint_indices, checkpoints);
    }

// ---- MixedLattice::compute_delay ----
    void MixedLattice::compute_delay(const SpectroscopyPlan& p, size_t j, const ODEState& x_ground,
                                     const ODEState* checkpoint, DelayResult& out) {
        const double tau = p.taus[j];
        out = DelayResult{};
        if (!p.w1) {
            if (p.gpu) {
                out.M1 = single_pulse_drive(*p.probe_SU2, *p.probe_SU3, tau,
                                            p.amp_SU2, p.width_SU2, p.freq_SU2, p.amp_SU3, p.width_SU3, p.freq_SU3,
                                            p.grid.t0, p.grid.t_end(), p.grid.dt, p.method, true, nullptr, true,
                                            p.abs_tol, p.rel_tol);
            } else {
                vector<vector<double>> states;
                configure_pulse_train(1, *p.probe_SU2, *p.probe_SU3, tau, {}, {}, tau,
                                      p.amp_SU2, p.width_SU2, p.freq_SU2, p.amp_SU3, p.width_SU3, p.freq_SU3);
                out.M1 = integrate_pulse_train(p.grid, 0, x_ground, p.method, p.save_spins ? &states : nullptr,
                                               p.abs_tol, p.rel_tol);
                out.M1_spins = flatten_states(states);
            }
        }
        if (p.gpu) {
            out.M01 = double_pulse_drive(*p.pump_SU2, *p.pump_SU3, 0.0, *p.probe_SU2, *p.probe_SU3, tau,
                                         p.amp_SU2, p.width_SU2, p.freq_SU2, p.amp_SU3, p.width_SU3, p.freq_SU3,
                                         p.grid.t0, p.grid.t_end(), p.grid.dt, p.method, true, nullptr, true,
                                         p.abs_tol, p.rel_tol);
            return;
        }
        out.k0 = p.m01_from_m0 ? p.m01_start[j] : 0;
        if (out.k0 > 0 && (checkpoint == nullptr || checkpoint->size() != ode_state_size())) {
            throw std::logic_error("pump-probe: missing M0 state to continue M01 from");
        }
        vector<vector<double>> states;
        configure_pulse_train(2, *p.pump_SU2, *p.pump_SU3, 0.0, *p.probe_SU2, *p.probe_SU3, tau,
                              p.amp_SU2, p.width_SU2, p.freq_SU2, p.amp_SU3, p.width_SU3, p.freq_SU3);
        out.M01 = integrate_pulse_train(p.grid, out.k0, out.k0 > 0 ? *checkpoint : x_ground, p.method,
                                        p.save_spins ? &states : nullptr, p.abs_tol, p.rel_tol);
        out.M01_spins = flatten_states(states);
    }

namespace {

void print_plan(std::ostream& os, size_t n_tau, size_t n_t, double residual, bool w1,
                const std::string& w1_note, bool m01, size_t n_checkpoints, bool pump_truncated) {
    os << "  Time grid: " << n_t << " samples; delays: " << n_tau << endl;
    if (pump_truncated) {
        os << "  WARNING: T_start is inside the pump window (start at least 9 pulse widths before "
              "t = 0); every trajectory misses the leading part of the pump." << endl;
    }
    os << "  Ground-state stationarity residual max|dS/dt|/(|H||S|) = " << residual << endl;
    os << "  [W1] M1 from time-shifted M0: " << (w1 ? "on" : "off");
    if (!w1 && !w1_note.empty()) os << " (" << w1_note << ")";
    os << endl;
    os << "  [M01] continued from stored M0 states: " << (m01 ? "on" : "off");
    if (m01) os << " (" << n_checkpoints << " stored states)";
    os << endl;
}

}  // namespace

// ---- MixedLattice::pump_probe_spectroscopy ----
    void MixedLattice::pump_probe_spectroscopy(const vector<SpinVector>& field_in_SU2,
                                 const vector<SpinVector>& field_in_SU3,
                                 double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                 double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start, double Temp_end,
                                 size_t n_anneal,
                                 bool T_zero_quench, size_t quench_sweeps,
                                 string dir_name,
                                 string method, bool use_gpu,
                                 bool save_spin_trajectories,
                                 bool reuse_m0_for_m1,
                                 double stationarity_tol,
                                 int outer_omp_threads,
                                 bool pulse_window_chunking,
                                 double abs_tol, double rel_tol,
                                 const vector<SpinVector>& field_in_SU2_B,
                                 const vector<SpinVector>& field_in_SU3_B,
                                 bool reuse_m0_for_m01) {
        (void) pulse_window_chunking;
        const SpectroscopyPlan plan = plan_spectroscopy(
            field_in_SU2, field_in_SU3, field_in_SU2_B, field_in_SU3_B,
            pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2, pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
            tau_start, tau_end, tau_step, T_start, T_end, T_step, method, use_gpu, save_spin_trajectories,
            reuse_m0_for_m1, stationarity_tol, reuse_m0_for_m01, abs_tol, rel_tol);
        const size_t n_tau = plan.taus.size();
        const size_t n_t = plan.grid.n;
        const size_t state_dim = spin_state_size();

        std::filesystem::create_directories(dir_name);
        cout << "\n==========================================\n"
             << "Mixed Lattice Pump-Probe Spectroscopy\n"
             << "==========================================" << endl;
        cout << "SU(2) pulse: amp=" << pulse_amp_SU2 << ", width=" << pulse_width_SU2
             << ", freq=" << pulse_freq_SU2 << "; SU(3) pulse: amp=" << pulse_amp_SU3
             << ", width=" << pulse_width_SU3 << ", freq=" << pulse_freq_SU3 << endl;
        cout << "Delays: " << tau_start << " -> " << plan.taus.back() << " (step " << tau_step << "); time: "
             << T_start << " -> " << plan.grid.t_end() << " (step " << T_step << ")" << endl;
        print_plan(cout, n_tau, n_t, plan.residual, plan.w1, plan.w1_note, plan.m01_from_m0,
                   plan.checkpoint_indices.size(), plan.pump_truncated);

        // The ground state is never modified by the drivers; restore it anyway
        // on every exit path.
        const SpinConfigSU2 ground_SU2 = spins_SU2;
        const SpinConfigSU3 ground_SU3 = spins_SU3;
        struct SpinRestore {
            MixedLattice* lat; const SpinConfigSU2* s2; const SpinConfigSU3* s3;
            ~SpinRestore() { lat->spins_SU2 = *s2; lat->spins_SU3 = *s3; }
        } restore{this, &ground_SU2, &ground_SU3};

        const ODEState x_ground = spins_to_state();
        const Observables M_ground = observe(x_ground.data());
        const double E_ground = energy_density();
        cout << "Ground state: E/N = " << E_ground << ", |M_SU2| = " << magnetization_SU2().norm()
             << ", |M_SU3| = " << magnetization_SU3().norm() << endl;
        save_positions_to_dir(dir_name);
        save_spin_config_to_dir(dir_name, "initial_spins");

        cout << "Computing M0 (pump only)..." << endl;
        vector<vector<double>> M0_states;
        vector<ODEState> checkpoints;
        const PumpProbeTrajectory M0 = run_reference_trajectory(
            plan, x_ground, plan.save_spins ? &M0_states : nullptr, &checkpoints);
        require_grid_length(M0, n_t, "M0");
        const vector<double> M0_spins = flatten_states(M0_states);
        M0_states.clear();
        auto checkpoint_for = [&](size_t j) -> const ODEState* {
            const size_t k0 = plan.m01_start[j];
            if (k0 == 0) return nullptr;
            const auto it = std::lower_bound(plan.checkpoint_indices.begin(), plan.checkpoint_indices.end(), k0);
            return &checkpoints[static_cast<size_t>(it - plan.checkpoint_indices.begin())];
        };

#ifdef HDF5_ENABLED
        const string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        HDF5MixedPumpProbeWriter writer(
            hdf5_file,
            lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
            lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
            dim1, dim2, dim3, spin_length_SU2, spin_length_SU3,
            pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
            pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
            T_start, T_end, T_step, method,
            tau_start, tau_end, tau_step,
            E_ground, magnetization_SU2(), magnetization_SU3(),
            Temp_start, Temp_end, n_anneal, T_zero_quench, quench_sweeps,
            save_spin_trajectories,
            &field_in_SU2, &field_in_SU3,
            &site_positions_SU2, &site_positions_SU3);
        writer.write_reference_trajectory(M0, plan.save_spins ? &M0_spins : nullptr);
#else
        (void) Temp_start; (void) Temp_end; (void) n_anneal; (void) T_zero_quench; (void) quench_sweeps;
        cout << "Note: HDF5 support not enabled; trajectories are computed but not written." << endl;
#endif

        // One delay point: compute, assemble on the full grid, write.
        auto finish_delay = [&](size_t j, DelayResult& r) {
            PumpProbeTrajectory M1 = plan.w1
                ? synthesize_M1_from_M0(M0, M_ground, plan.taus[j], T_start, T_end, T_step)
                : std::move(r.M1);
            PumpProbeTrajectory M01 = assemble_m01(M0, std::move(r.M01), r.k0);
            vector<double> M01_spins = assemble_m01_spins(M0_spins, std::move(r.M01_spins), r.k0, state_dim);
            require_grid_length(M1, n_t, "M1");
            require_grid_length(M01, n_t, "M01");
#ifdef HDF5_ENABLED
            writer.write_tau_trajectory(static_cast<int>(j), plan.taus[j], M1, M01,
                                        plan.save_spins ? &r.M1_spins : nullptr,
                                        plan.save_spins ? &M01_spins : nullptr);
#else
            (void) M01_spins;
#endif
        };

        int n_outer = 1;
#ifdef _OPENMP
        n_outer = (outer_omp_threads <= 0) ? std::max(1, omp_get_max_threads()) : outer_omp_threads;
        n_outer = std::min<int>(n_outer, static_cast<int>(n_tau));
        if (plan.gpu) n_outer = 1;  // one backend and one device stream for the whole scan
#else
        (void) outer_omp_threads;
#endif

        if (n_outer <= 1) {
            for (size_t j = 0; j < n_tau; ++j) {
                cout << "--- Delay " << (j + 1) << "/" << n_tau << ": tau = " << plan.taus[j] << " ---" << endl;
                DelayResult r;
                compute_delay(plan, j, x_ground, checkpoint_for(j), r);
                finish_delay(j, r);
            }
        } else {
#ifdef _OPENMP
            // Outer parallelism over delays: every thread drives its own copy
            // of the lattice (the pulse train is per-object state) with nested
            // parallelism off; exceptions are carried out of the region.
            cout << "  Distributing " << n_tau << " delays over " << n_outer << " OpenMP threads" << endl;
            const int saved_levels = omp_get_max_active_levels();
            omp_set_max_active_levels(1);
            std::exception_ptr failure;
            std::atomic<bool> failed{false};
            #pragma omp parallel num_threads(n_outer)
            {
                std::unique_ptr<MixedLattice> local;
                try {
                    local = std::make_unique<MixedLattice>(*this);
                } catch (...) {
                    #pragma omp critical(mixed_pump_probe_failure)
                    if (!failure) failure = std::current_exception();
                    failed = true;
                }
                #pragma omp for schedule(dynamic, 1)
                for (long jj = 0; jj < static_cast<long>(n_tau); ++jj) {
                    if (!local || failed) continue;
                    const size_t j = static_cast<size_t>(jj);
                    std::exception_ptr error;
                    try {
                        DelayResult r;
                        local->compute_delay(plan, j, x_ground, checkpoint_for(j), r);
                        // No exception may leave a critical construct.
                        #pragma omp critical(mixed_pump_probe_hdf5_write)
                        {
                            try { finish_delay(j, r); } catch (...) { error = std::current_exception(); }
                        }
                    } catch (...) {
                        error = std::current_exception();
                    }
                    if (error) {
                        #pragma omp critical(mixed_pump_probe_failure)
                        if (!failure) failure = error;
                        failed = true;
                    }
                }
            }
            omp_set_max_active_levels(saved_levels);
            if (failure) std::rethrow_exception(failure);
#endif
        }

#ifdef HDF5_ENABLED
        writer.close();
        cout << "\nAll trajectories written to " << hdf5_file << endl;
#endif
        cout << "Pump-probe spectroscopy complete (" << n_tau << " delays)." << endl;
    }

namespace {

// Message tags of the MPI scan (fixed: the delay index travels in the payload,
// so tags never approach MPI_TAG_UB however many delays there are).
constexpr int kTagResult     = 7101;  // worker -> 0: int64 header {delay, status, k0, msg_len}
constexpr int kTagM1         = 7102;
constexpr int kTagM01        = 7103;
constexpr int kTagM1Spins    = 7104;
constexpr int kTagM01Spins   = 7105;
constexpr int kTagError      = 7106;
constexpr int kTagAssign     = 7110;  // 0 -> worker: int64 delay index (-1 = stop)
constexpr int kTagCheckpoint = 7111;  // 0 -> worker: M0 state to continue M01 from

// Large buffers go out in chunks whose element count fits an int.
constexpr size_t kMaxMessageDoubles = size_t(1) << 27;

void send_doubles(const double* p, size_t n, int dest, int tag, MPI_Comm comm) {
    size_t off = 0;
    do {
        const size_t c = std::min(kMaxMessageDoubles, n - off);
        MPI_Send(p + off, static_cast<int>(c), MPI_DOUBLE, dest, tag, comm);
        off += c;
    } while (off < n);
}

void recv_doubles(double* p, size_t n, int src, int tag, MPI_Comm comm) {
    size_t off = 0;
    do {
        const size_t c = std::min(kMaxMessageDoubles, n - off);
        MPI_Recv(p + off, static_cast<int>(c), MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
        off += c;
    } while (off < n);
}

// Collective error agreement: every rank passes its local error message
// (empty = success). If any rank failed, every rank throws the message of the
// lowest failing rank, so no rank is left waiting in a later collective.
void agree_on_errors(MPI_Comm comm, const std::string& local_error, const char* phase) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int mine = local_error.empty() ? INT_MAX : rank;
    int first = INT_MAX;
    MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MIN, comm);
    if (first == INT_MAX) return;
    int len = (rank == first) ? static_cast<int>(local_error.size()) : 0;
    MPI_Bcast(&len, 1, MPI_INT, first, comm);
    std::string msg = (rank == first) ? local_error : std::string(static_cast<size_t>(len), ' ');
    MPI_Bcast(msg.data(), len, MPI_CHAR, first, comm);
    throw std::runtime_error(std::string(phase) + " failed on rank " + std::to_string(first) + ": " + msg);
}

}  // namespace

// ---- MixedLattice::pump_probe_spectroscopy_mpi ----
    void MixedLattice::pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in_SU2,
                                     const vector<SpinVector>& field_in_SU3,
                                     double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                     double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start, double Temp_end,
                                     size_t n_anneal,
                                     bool T_zero_quench, size_t quench_sweeps,
                                     string dir_name,
                                     string method, bool use_gpu,
                                     bool save_spin_trajectories,
                                     bool reuse_m0_for_m1,
                                     double stationarity_tol,
                                     bool pulse_window_chunking,
                                     double abs_tol, double rel_tol,
                                     const vector<SpinVector>& field_in_SU2_B,
                                     const vector<SpinVector>& field_in_SU3_B,
                                     bool reuse_m0_for_m01) {
        int mpi_on = 0;
        MPI_Initialized(&mpi_on);
        int rank = 0, size = 1;
        if (mpi_on) {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
        }
        if (size <= 1) {
            pump_probe_spectroscopy(field_in_SU2, field_in_SU3,
                                    pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                    pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                    tau_start, tau_end, tau_step, T_start, T_end, T_step,
                                    Temp_start, Temp_end, n_anneal, T_zero_quench, quench_sweeps,
                                    dir_name, method, use_gpu, save_spin_trajectories,
                                    reuse_m0_for_m1, stationarity_tol, /*outer_omp_threads=*/0,
                                    pulse_window_chunking, abs_tol, rel_tol,
                                    field_in_SU2_B, field_in_SU3_B, reuse_m0_for_m01);
            return;
        }
        (void) pulse_window_chunking;
        const MPI_Comm comm = MPI_COMM_WORLD;

        // ---- Phase 1: validate on every rank (identical inputs give identical
        // verdicts; agree_on_errors makes that robust) and open the output file.
        std::unique_ptr<SpectroscopyPlan> plan;
        std::string error;
        try {
            plan = std::make_unique<SpectroscopyPlan>(plan_spectroscopy(
                field_in_SU2, field_in_SU3, field_in_SU2_B, field_in_SU3_B,
                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2, pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                tau_start, tau_end, tau_step, T_start, T_end, T_step, method, use_gpu, save_spin_trajectories,
                reuse_m0_for_m1, stationarity_tol, reuse_m0_for_m01, abs_tol, rel_tol));
        } catch (const std::exception& e) {
            error = e.what();
        }
        agree_on_errors(comm, error, "pump_probe_spectroscopy_mpi: input validation");
        // Rank 0's W1 verdict is authoritative (the residual is a floating-point
        // reduction over a state the runner broadcast).
        int w1 = plan->w1 ? 1 : 0;
        MPI_Bcast(&w1, 1, MPI_INT, 0, comm);
        plan->w1 = (w1 != 0);

        const size_t n_tau = plan->taus.size();
        const size_t n_t = plan->grid.n;
        const size_t state_dim = spin_state_size();
        const size_t ode_dim = ode_state_size();
        const SpinConfigSU2 ground_SU2 = spins_SU2;
        const SpinConfigSU3 ground_SU3 = spins_SU3;
        struct SpinRestore {
            MixedLattice* lat; const SpinConfigSU2* s2; const SpinConfigSU3* s3;
            ~SpinRestore() { lat->spins_SU2 = *s2; lat->spins_SU3 = *s3; }
        } restore{this, &ground_SU2, &ground_SU3};
        const ODEState x_ground = spins_to_state();
        const Observables M_ground = observe(x_ground.data());

#ifdef HDF5_ENABLED
        std::unique_ptr<HDF5MixedPumpProbeWriter> writer;
#endif
        const string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        if (rank == 0) {
            try {
                std::filesystem::create_directories(dir_name);
                cout << "\n==========================================\n"
                     << "Mixed Lattice Pump-Probe Spectroscopy (MPI, " << size << " ranks: 1 writer + "
                     << (size - 1) << " workers, dynamic scheduling)\n"
                     << "==========================================" << endl;
                print_plan(cout, n_tau, n_t, plan->residual, plan->w1, plan->w1_note, plan->m01_from_m0,
                           plan->checkpoint_indices.size(), plan->pump_truncated);
                if (plan->distinct_probe) {
                    cout << "  Probe directions differ from the pump directions." << endl;
                }
                save_positions_to_dir(dir_name);
                save_spin_config_to_dir(dir_name, "initial_spins");
                save_energy_to_dir(dir_name, "energy_initial");
#ifdef HDF5_ENABLED
                writer = std::make_unique<HDF5MixedPumpProbeWriter>(
                    hdf5_file,
                    lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                    lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                    dim1, dim2, dim3, spin_length_SU2, spin_length_SU3,
                    pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                    pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                    T_start, T_end, T_step, method,
                    tau_start, tau_end, tau_step,
                    energy_density(), magnetization_SU2(), magnetization_SU3(),
                    Temp_start, Temp_end, n_anneal, T_zero_quench, quench_sweeps,
                    save_spin_trajectories,
                    &field_in_SU2, &field_in_SU3,
                    &site_positions_SU2, &site_positions_SU3);
#else
                (void) Temp_start; (void) Temp_end; (void) n_anneal; (void) T_zero_quench; (void) quench_sweeps;
                cout << "Note: HDF5 support not enabled; trajectories are computed but not written." << endl;
#endif
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        agree_on_errors(comm, error, "pump_probe_spectroscopy_mpi: output setup");

        // ---- Phase 2: reference M0 on rank 0 (also the source of the M1
        // synthesis and of the states M01 is continued from).
        PumpProbeTrajectory M0;
        vector<double> M0_spins;
        vector<ODEState> checkpoints;
        if (rank == 0) {
            try {
                cout << "Computing M0 (pump only) on rank 0..." << endl;
                vector<vector<double>> states;
                M0 = run_reference_trajectory(*plan, x_ground, plan->save_spins ? &states : nullptr, &checkpoints);
                require_grid_length(M0, n_t, "M0");
                M0_spins = flatten_states(states);
#ifdef HDF5_ENABLED
                writer->write_reference_trajectory(M0, plan->save_spins ? &M0_spins : nullptr);
#endif
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        agree_on_errors(comm, error, "pump_probe_spectroscopy_mpi: reference trajectory M0");

        // ---- Phase 3: dynamic master-worker scan. A worker's result message
        // doubles as its request for the next delay; rank 0 writes each result
        // as soon as it arrives. After the first error rank 0 hands out no more
        // work but keeps serving requests until every worker has stopped.
        // Expected failures (a worker's integration, rank 0's HDF5 writes) are
        // reported in-band above; anything escaping (e.g. std::bad_alloc while
        // a peer is blocked in a matching send/receive) cannot be recovered
        // without a deadlock, so it aborts the job with a message.
        auto abort_job = [&](const char* where, const std::exception& e) {
            std::cerr << "pump_probe_spectroscopy_mpi: fatal error on rank " << rank << " (" << where
                      << "): " << e.what() << endl;
            MPI_Abort(comm, 1);
        };
        if (rank == 0) {
            try {
                size_t next = 0;
                int active = size - 1;
                size_t done = 0;
                const size_t progress_every = std::max<size_t>(1, n_tau / 20);
                vector<double> buf_M1, buf_M01, buf_s1, buf_s01;
                while (active > 0) {
                    int64_t hdr[4];
                    MPI_Status st;
                    MPI_Recv(hdr, 4, MPI_INT64_T, MPI_ANY_SOURCE, kTagResult, comm, &st);
                    const int src = st.MPI_SOURCE;
                    if (hdr[0] >= 0) {
                        const size_t j = static_cast<size_t>(hdr[0]);
                        if (hdr[1] == 0) {
                            const size_t k0 = static_cast<size_t>(hdr[2]);
                            if (!plan->w1) {
                                buf_M1.resize(n_t * kObsDoubles);
                                recv_doubles(buf_M1.data(), buf_M1.size(), src, kTagM1, comm);
                            }
                            buf_M01.resize((n_t - k0) * kObsDoubles);
                            recv_doubles(buf_M01.data(), buf_M01.size(), src, kTagM01, comm);
                            if (plan->save_spins) {
                                buf_s1.resize(n_t * state_dim);
                                recv_doubles(buf_s1.data(), buf_s1.size(), src, kTagM1Spins, comm);
                                buf_s01.resize((n_t - k0) * state_dim);
                                recv_doubles(buf_s01.data(), buf_s01.size(), src, kTagM01Spins, comm);
                            }
                            if (error.empty()) {
                                try {
                                    PumpProbeTrajectory M1 = plan->w1
                                        ? synthesize_M1_from_M0(M0, M_ground, plan->taus[j], T_start, T_end, T_step)
                                        : unflatten_trajectory(buf_M1);
                                    PumpProbeTrajectory M01 = assemble_m01(M0, unflatten_trajectory(buf_M01), k0);
                                    vector<double> M01_spins = plan->save_spins
                                        ? assemble_m01_spins(M0_spins, std::move(buf_s01), k0, state_dim)
                                        : vector<double>();
                                    require_grid_length(M1, n_t, "M1");
                                    require_grid_length(M01, n_t, "M01");
#ifdef HDF5_ENABLED
                                    writer->write_tau_trajectory(static_cast<int>(j), plan->taus[j], M1, M01,
                                                                 plan->save_spins ? &buf_s1 : nullptr,
                                                                 plan->save_spins ? &M01_spins : nullptr);
#endif
                                    if (++done % progress_every == 0 || done == n_tau) {
                                        cout << "  " << done << "/" << n_tau << " delays written" << endl;
                                    }
                                } catch (const std::exception& e) {
                                    error = std::string("writing delay ") + std::to_string(j) + ": " + e.what();
                                }
                            }
                        } else {
                            std::string msg(static_cast<size_t>(hdr[3]), '\0');
                            MPI_Recv(msg.data(), static_cast<int>(hdr[3]), MPI_CHAR, src, kTagError, comm,
                                     MPI_STATUS_IGNORE);
                            if (error.empty()) {
                                error = "rank " + std::to_string(src) + ", delay " + std::to_string(j) + ": " + msg;
                            }
                        }
                    }
                    const int64_t assign = (error.empty() && next < n_tau) ? static_cast<int64_t>(next++) : -1;
                    MPI_Send(&assign, 1, MPI_INT64_T, src, kTagAssign, comm);
                    if (assign < 0) {
                        --active;
                    } else if (plan->m01_from_m0 && plan->m01_start[static_cast<size_t>(assign)] > 0) {
                        const size_t k0 = plan->m01_start[static_cast<size_t>(assign)];
                        const auto it = std::lower_bound(plan->checkpoint_indices.begin(),
                                                         plan->checkpoint_indices.end(), k0);
                        const ODEState& cp = checkpoints[static_cast<size_t>(it - plan->checkpoint_indices.begin())];
                        send_doubles(cp.data(), cp.size(), src, kTagCheckpoint, comm);
                    }
                }
#ifdef HDF5_ENABLED
                try {
                    writer->close();
                } catch (const std::exception& e) {
                    if (error.empty()) error = std::string("closing ") + hdf5_file + ": " + e.what();
                }
#endif
            } catch (const std::exception& e) {
                abort_job("scheduler", e);
            }
        } else {
            try {
                int64_t hdr[4] = {-1, 0, 0, 0};
                MPI_Send(hdr, 4, MPI_INT64_T, 0, kTagResult, comm);
                ODEState checkpoint;
                vector<double> buf;
                while (true) {
                    int64_t assign = -1;
                    MPI_Recv(&assign, 1, MPI_INT64_T, 0, kTagAssign, comm, MPI_STATUS_IGNORE);
                    if (assign < 0) break;
                    const size_t j = static_cast<size_t>(assign);
                    const size_t k0 = plan->m01_from_m0 ? plan->m01_start[j] : 0;
                    if (k0 > 0) {
                        checkpoint.resize(ode_dim);
                        recv_doubles(checkpoint.data(), ode_dim, 0, kTagCheckpoint, comm);
                    }
                    DelayResult r;
                    std::string msg;
                    try {
                        compute_delay(*plan, j, x_ground, k0 > 0 ? &checkpoint : nullptr, r);
                        // Rank 0 sizes its receives from (n_t, k0): check before sending.
                        if (!plan->w1) require_grid_length(r.M1, n_t, "M1");
                        if (r.M01.size() != n_t - k0) throw std::runtime_error("M01 has an unexpected number of samples");
                        if (plan->save_spins &&
                            (r.M1_spins.size() != n_t * state_dim || r.M01_spins.size() != (n_t - k0) * state_dim)) {
                            throw std::runtime_error("spin-state trajectory has an unexpected size");
                        }
                    } catch (const std::exception& e) {
                        msg = e.what();
                        if (msg.empty()) msg = "unknown error";
                    }
                    if (msg.empty()) {
                        hdr[0] = assign; hdr[1] = 0; hdr[2] = static_cast<int64_t>(k0); hdr[3] = 0;
                        MPI_Send(hdr, 4, MPI_INT64_T, 0, kTagResult, comm);
                        if (!plan->w1) {
                            flatten_trajectory(r.M1, buf);
                            send_doubles(buf.data(), buf.size(), 0, kTagM1, comm);
                        }
                        flatten_trajectory(r.M01, buf);
                        send_doubles(buf.data(), buf.size(), 0, kTagM01, comm);
                        if (plan->save_spins) {
                            send_doubles(r.M1_spins.data(), r.M1_spins.size(), 0, kTagM1Spins, comm);
                            send_doubles(r.M01_spins.data(), r.M01_spins.size(), 0, kTagM01Spins, comm);
                        }
                    } else {
                        hdr[0] = assign; hdr[1] = 1; hdr[2] = 0; hdr[3] = static_cast<int64_t>(msg.size());
                        MPI_Send(hdr, 4, MPI_INT64_T, 0, kTagResult, comm);
                        MPI_Send(msg.data(), static_cast<int>(msg.size()), MPI_CHAR, 0, kTagError, comm);
                    }
                }
            } catch (const std::exception& e) {
                abort_job("worker", e);
            }
        }
        agree_on_errors(comm, error, "pump_probe_spectroscopy_mpi: delay scan");
        if (rank == 0) {
            cout << "Pump-probe spectroscopy (MPI) complete: " << n_tau << " delays in " << hdf5_file << endl;
        }
    }
