/**
 * @file lattice_md.cpp
 * @brief Lattice spin dynamics: equation of motion, the exact-grid
 *        integration engine, molecular dynamics and the pump-probe / 2DCS
 *        drivers (OpenMP and MPI).
 *
 * Design (see the SPIN DYNAMICS block of lattice.h):
 *   - one right-hand side, landau_lifshitz_rhs, with the drive passed
 *     explicitly as an immutable DriveSchedule;
 *   - one integration engine, integrate_on_grid, that samples every
 *     trajectory on the exact grid t_k = t0 + k dt whatever the method;
 *   - one definition of the magnetisation channels, measure_magnetizations;
 *   - one delay-scan implementation, run_pump_probe_scan, behind both the
 *     OpenMP and the MPI entry points, writing through HDF5PumpProbeWriter.
 */

#include "classical_spin/lattice/lattice.h"
#include "classical_spin/dynamics/ode_method.h"
#include "classical_spin/dynamics/structure_factor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/odeint.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using classical_spin::dynamics::DriveSchedule;
using classical_spin::dynamics::OdeMethod;
using classical_spin::dynamics::Pulse;
using classical_spin::dynamics::TimeGrid;

// Sites per RHS call above which the site loop is shared among threads; below
// it the fork/join costs more than the work.
constexpr size_t kParallelRhsMinSites = 512;

void require_dynamics_dim(size_t spin_dim, const char* where) {
    if (spin_dim != 3 && spin_dim != 8) {
        throw std::invalid_argument(std::string(where) + ": spin dynamics needs spin_dim 3 (SU(2)) or "
                                    "8 (SU(3)); got " + std::to_string(spin_dim) +
                                    " (no Lie-algebra cross product exists for other dimensions)");
    }
}

void warn_no_gpu() {
    static bool warned = false;
    if (warned) return;
    warned = true;
    std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED); "
                 "running on the CPU." << std::endl;
}

#ifdef CUDA_ENABLED
// use_gpu honoured only when the GPU kernels implement the model.
bool gpu_usable(const Lattice& lat) {
    std::string reason;
    if (lat.gpu_supports_model(reason)) return true;
    std::cerr << "Warning: the GPU right-hand side does not support " << reason
              << "; running on the CPU." << std::endl;
    return false;
}
#endif

}  // namespace

// ============================================================
// Equation of motion
// ============================================================

void Lattice::landau_lifshitz_rhs(const double* x, double* dxdt, double t, const DriveSchedule& drive) const {
    // The pulse envelopes depend only on t: evaluate them once per call
    // (two transcendentals per pulse) instead of once per site.
    double f[DriveSchedule::kMaxPulses];
    drive.envelopes(t, f);
    const bool driven = !drive.empty();
    const double alpha = alpha_gilbert;
    // Overall prefactor: 1 in Landau-Lifshitz form, 1/(1+α²) in Gilbert form.
    const double g = classical_spin::dynamics::LangevinParams{alpha, 0.0, damping_form}.prefactor();
    const size_t N = lattice_size;

    if (spin_dim == 3) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= kParallelRhsMinSites)
#endif
        for (size_t i = 0; i < N; ++i) {
            const double* S = x + 3 * i;
            double H[3];                      // H = ∂E/∂S_i = -B_i
            get_local_field_flat(x, i, H);
            if (driven) {
                double B[3] = {0.0, 0.0, 0.0};
                drive.accumulate(i % N_atoms, f, B);
                H[0] -= B[0]; H[1] -= B[1]; H[2] -= B[2];
            }
            // Precession dS/dt = H × S = S × B.
            double d0 = H[1] * S[2] - H[2] * S[1];
            double d1 = H[2] * S[0] - H[0] * S[2];
            double d2 = H[0] * S[1] - H[1] * S[0];
            if (alpha > 0.0) {
                // Damping -(α/|S|) S × (S × B) = -(α/|S|) S × (H × S).
                const double c0 = S[1] * d2 - S[2] * d1;
                const double c1 = S[2] * d0 - S[0] * d2;
                const double c2 = S[0] * d1 - S[1] * d0;
                const double k = alpha / std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
                d0 -= k * c0; d1 -= k * c1; d2 -= k * c2;
            }
            dxdt[3 * i + 0] = g * d0;
            dxdt[3 * i + 1] = g * d1;
            dxdt[3 * i + 2] = g * d2;
        }
    } else if (spin_dim == 8) {
        // SU(3): (a × b)_i = f_ijk a_j b_k over the 54 non-zero structure
        // constants (cross_prod_SU3_flat). The damping term has the same
        // double-bracket form as for SU(2): with P = H × S,
        //   dE/dt = H·(P - (α/|S|) S × P) = -(α/|S|) |P|²  (H·(S × P) = P·(H × S)),
        // and S·(S × P) = 0 conserves |S|.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= kParallelRhsMinSites)
#endif
        for (size_t i = 0; i < N; ++i) {
            const double* S = x + 8 * i;
            double H[8];
            get_local_field_flat(x, i, H);
            if (driven) {
                double B[8] = {0.0};
                drive.accumulate(i % N_atoms, f, B);
                for (int d = 0; d < 8; ++d) H[d] -= B[d];
            }
            double* out = dxdt + 8 * i;
            cross_prod_SU3_flat(H, S, out, /*accumulate=*/false);
            if (alpha > 0.0) {
                double C[8];
                cross_prod_SU3_flat(S, out, C, /*accumulate=*/false);
                double n2 = 0.0;
                for (int d = 0; d < 8; ++d) n2 += S[d] * S[d];
                const double k = alpha / std::sqrt(n2);
                for (int d = 0; d < 8; ++d) out[d] = g * (out[d] - k * C[d]);
            }
        }
    } else {
        require_dynamics_dim(spin_dim, "landau_lifshitz_rhs");
    }
}

SpinVector Lattice::drive_field_at_time(double t, size_t site_index) const {
    double f[DriveSchedule::kMaxPulses];
    active_drive.envelopes(t, f);
    SpinVector B = SpinVector::Zero(spin_dim);
    active_drive.accumulate(site_index % N_atoms, f, B.data());
    return B;
}

vector<double> Lattice::local_polarisation(const vector<SpinVector>& field_global) const {
    if (field_global.size() != N_atoms) {
        throw std::invalid_argument("pulse polarisation: need one direction per sublattice (" +
                                    std::to_string(N_atoms) + "), got " + std::to_string(field_global.size()));
    }
    vector<double> e(N_atoms * spin_dim);
    for (size_t a = 0; a < N_atoms; ++a) {
        if (static_cast<size_t>(field_global[a].size()) != spin_dim) {
            throw std::invalid_argument("pulse polarisation: direction of sublattice " + std::to_string(a) +
                                        " has dimension " + std::to_string(field_global[a].size()) +
                                        ", expected spin_dim = " + std::to_string(spin_dim));
        }
        // S_global = F S  =>  B_spin = F^T B_global.
        Eigen::Map<Eigen::VectorXd>(e.data() + a * spin_dim, spin_dim) =
            sublattice_frames[a].transpose() * field_global[a];
    }
    return e;
}

void Lattice::set_pulse(const vector<SpinVector>& field_in1, double t_B1,
                        const vector<SpinVector>& field_in2, double t_B2,
                        double pulse_amp, double pulse_width, double pulse_freq) {
    const vector<double> e1 = local_polarisation(field_in1);
    const vector<double> e2 = local_polarisation(field_in2);
    DriveSchedule drive = make_drive();
    drive.add(Pulse{t_B1, pulse_amp, pulse_width, pulse_freq}, e1);
    drive.add(Pulse{t_B2, pulse_amp, pulse_width, pulse_freq}, e2);
    active_drive = std::move(drive);
    // Mirror in the two-pulse layout uploaded by the GPU paths.
    field_drive[0] = Eigen::Map<const Eigen::VectorXd>(e1.data(), Eigen::Index(e1.size()));
    field_drive[1] = Eigen::Map<const Eigen::VectorXd>(e2.data(), Eigen::Index(e2.size()));
    t_pulse[0] = t_B1;
    t_pulse[1] = t_B2;
    field_drive_amp = pulse_amp;
    field_drive_width = pulse_width;
    field_drive_freq = pulse_freq;
}

void Lattice::clear_pulse() {
    active_drive = make_drive();
    field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
    field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
    field_drive_amp = 0.0;
}

void Lattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
    landau_lifshitz_rhs(x.data(), dxdt.data(), t, active_drive);
}

// ============================================================
// Observables
// ============================================================

void Lattice::measure_magnetizations(const double* x, double* out, double* scratch) const {
    const size_t D = spin_dim;
    std::fill(scratch, scratch + N_atoms * D, 0.0);
    for (size_t i = 0; i < lattice_size; ++i) {
        double* P = scratch + (i % N_atoms) * D;
        const double* S = x + i * D;
        for (size_t d = 0; d < D; ++d) P[d] += S[d];
    }
    std::fill(out, out + 3 * D, 0.0);
    for (size_t a = 0; a < N_atoms; ++a) {
        const double* P = scratch + a * D;
        const SpinMatrix& F = sublattice_frames[a];
        const double sgn = afm_sublattice_signs[a];
        for (size_t mu = 0; mu < D; ++mu) {
            double g = 0.0;
            for (size_t nu = 0; nu < D; ++nu) g += F(mu, nu) * P[nu];
            out[mu] += sgn * g;
            out[D + mu] += P[mu];
            out[2 * D + mu] += g;
        }
    }
    const double inv_n = 1.0 / double(lattice_size);
    for (size_t k = 0; k < 3 * D; ++k) out[k] *= inv_n;
}

array<SpinVector, 3> Lattice::measure_magnetizations(const double* x) const {
    vector<double> out(3 * spin_dim), scratch(N_atoms * spin_dim);
    measure_magnetizations(x, out.data(), scratch.data());
    const Eigen::Index D = Eigen::Index(spin_dim);
    return {SpinVector(Eigen::Map<const Eigen::VectorXd>(out.data(), D)),
            SpinVector(Eigen::Map<const Eigen::VectorXd>(out.data() + D, D)),
            SpinVector(Eigen::Map<const Eigen::VectorXd>(out.data() + 2 * D, D))};
}

double Lattice::stationarity_residual() const {
    require_dynamics_dim(spin_dim, "stationarity_residual");
    const ODEState x = spins_to_state(spins);
    const size_t D = spin_dim;
    double max_torque = 0.0, max_scale = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:max_torque,max_scale) if(lattice_size >= kParallelRhsMinSites)
#endif
    for (size_t i = 0; i < lattice_size; ++i) {
        const double* S = x.data() + i * D;
        double H[MAX_SPIN_DIM], T[MAX_SPIN_DIM];
        get_local_field_flat(x.data(), i, H);
        if (D == 3) {
            T[0] = H[1] * S[2] - H[2] * S[1];
            T[1] = H[2] * S[0] - H[0] * S[2];
            T[2] = H[0] * S[1] - H[1] * S[0];
        } else {
            cross_prod_SU3_flat(H, S, T, false);
        }
        double t2 = 0.0, h2 = 0.0, s2 = 0.0;
        for (size_t d = 0; d < D; ++d) { t2 += T[d] * T[d]; h2 += H[d] * H[d]; s2 += S[d] * S[d]; }
        max_torque = std::max(max_torque, std::sqrt(t2));
        max_scale = std::max(max_scale, std::sqrt(h2 * s2));
    }
    return (max_scale > 0.0) ? max_torque / max_scale : 0.0;
}

double Lattice::max_dSdt_norm_no_drive() const {
    require_dynamics_dim(spin_dim, "max_dSdt_norm_no_drive");
    const ODEState x = spins_to_state(spins);
    ODEState f(x.size());
    landau_lifshitz_rhs(x.data(), f.data(), 0.0, make_drive());
    double m = 0.0;
    for (double v : f) m = std::max(m, std::abs(v));
    return m;
}

// ============================================================
// Integration engine
// ============================================================

void Lattice::integrate_on_grid(ODEState& state, const TimeGrid& grid, const DriveSchedule& drive,
                                const DynamicsSettings& s, const GridObserver& observer) const {
    namespace odeint = boost::numeric::odeint;
    namespace dyn = classical_spin::dynamics;

    require_dynamics_dim(spin_dim, "integrate_on_grid");
    const OdeMethod method = dyn::parse_ode_method(s.method);
    if (state.size() != lattice_size * spin_dim)
        throw std::invalid_argument("integrate_on_grid: state has " + std::to_string(state.size()) +
                                    " entries, expected " + std::to_string(lattice_size * spin_dim));
    if (grid.n == 0 || !(grid.dt > 0.0) || !std::isfinite(grid.t0))
        throw std::invalid_argument("integrate_on_grid: invalid time grid");
    if (!(s.dt > 0.0) || !std::isfinite(s.dt))
        throw std::invalid_argument("integrate_on_grid: step dt must be positive (got " + std::to_string(s.dt) + ")");
    if (!(alpha_gilbert >= 0.0) || !(langevin_temperature >= 0.0))
        throw std::invalid_argument("integrate_on_grid: alpha_gilbert and langevin_temperature must be >= 0");
    if (langevin_temperature > 0.0) {
        if (!dyn::is_geometric(method))
            throw std::invalid_argument("Langevin dynamics (langevin_temperature > 0) needs a geometric "
                                        "integrator: spherical_midpoint or depondt (got '" + s.method + "')");
        if (alpha_gilbert == 0.0)
            throw std::invalid_argument("langevin_temperature > 0 needs alpha_gilbert > 0: the bath couples "
                                        "to the spins through the damping (fluctuation-dissipation)");
    }

    auto rhs = [this, &drive](const ODEState& x, ODEState& dxdt, double t) {
        landau_lifshitz_rhs(x.data(), dxdt.data(), t, drive);
    };

    if (!dyn::is_adaptive(method)) {
        // Fixed step: m integer sub-steps per sample interval, every time
        // computed from integer indices (no accumulated t += h).
        const double ratio = grid.dt / s.dt;
        const size_t m = std::max<size_t>(1, static_cast<size_t>(std::llround(ratio)));
        const double h = grid.dt / double(m);
        auto run = [&](auto&& advance) {
            observer(state.data(), 0, grid[0]);
            for (size_t k = 0; k + 1 < grid.n; ++k) {
                const double tk = grid[k];
                for (size_t j = 0; j < m; ++j) advance(tk + double(j) * h, h);
                observer(state.data(), k + 1, grid[k + 1]);
            }
        };
        switch (method) {
            case OdeMethod::SphericalMidpoint:
            case OdeMethod::Depondt:
            case OdeMethod::ColorSplit:
            case OdeMethod::ColorSplit4: {
                if (spin_dim != 3)
                    throw std::invalid_argument("geometric spin integrators need spin_dim == 3 (got " +
                                                std::to_string(spin_dim) + "); use rk4 or dopri5 for SU(3)");
                DynamicsModel model{*this, &drive};
                dyn::SpinIntegrator<DynamicsModel> integrator(model, dyn::to_geometric(method),
                                                              {alpha_gilbert, langevin_temperature, damping_form});
                run([&](double t, double dt) { integrator.step(state.data(), t, dt); });
                break;
            }
            case OdeMethod::Euler: {
                odeint::euler<ODEState> st;
                run([&](double t, double dt) { st.do_step(rhs, state, t, dt); });
                break;
            }
            case OdeMethod::RK2: {
                odeint::modified_midpoint<ODEState> st;
                run([&](double t, double dt) { st.do_step(rhs, state, t, dt); });
                break;
            }
            case OdeMethod::RK4: {
                odeint::runge_kutta4<ODEState> st;
                run([&](double t, double dt) { st.do_step(rhs, state, t, dt); });
                break;
            }
            case OdeMethod::AdamsBashforth5: {
                odeint::adams_bashforth<5, ODEState> st;   // self-starting (RK4 initialiser)
                run([&](double t, double dt) { st.do_step(rhs, state, t, dt); });
                break;
            }
            case OdeMethod::AdamsMoulton5: {
                odeint::adams_bashforth_moulton<5, ODEState> st;
                run([&](double t, double dt) { st.do_step(rhs, state, t, dt); });
                break;
            }
            default:
                throw std::logic_error("integrate_on_grid: unhandled fixed-step method");
        }
        return;
    }

    // Error-controlled methods. A drive caps the step (DriveSchedule::max_step)
    // unless the caller set one: from a stationary state the error estimate
    // is zero until the pulse arrives, and an uncapped step would jump over it.
    const double max_dt = (s.max_dt > 0.0) ? s.max_dt : drive.max_step();
    const vector<double> times = grid.times();
    size_t k = 0;
    auto obs = [&](const ODEState& x, double t) {
        observer(x.data(), k, t);
        ++k;
    };
    switch (method) {
        case OdeMethod::Dopri5:
            // Dense output: steps are chosen by the error controller alone and
            // the samples interpolated (4th order) at the exact grid times.
            odeint::integrate_times(
                odeint::make_dense_output(s.abs_tol, s.rel_tol, max_dt, odeint::runge_kutta_dopri5<ODEState>()),
                rhs, state, times.begin(), times.end(), s.dt, obs);
            break;
        case OdeMethod::BulirschStoer:
            odeint::integrate_times(
                odeint::bulirsch_stoer_dense_out<ODEState>(s.abs_tol, s.rel_tol, 1.0, 1.0, max_dt),
                rhs, state, times.begin(), times.end(), s.dt, obs);
            break;
        case OdeMethod::CashKarp54:
            // No dense output in odeint: step exactly onto every sample time.
            odeint::integrate_times(
                odeint::make_controlled(s.abs_tol, s.rel_tol, max_dt, odeint::runge_kutta_cash_karp54<ODEState>()),
                rhs, state, times.begin(), times.end(), s.dt, obs);
            break;
        case OdeMethod::Fehlberg78:
            odeint::integrate_times(
                odeint::make_controlled(s.abs_tol, s.rel_tol, max_dt, odeint::runge_kutta_fehlberg78<ODEState>()),
                rhs, state, times.begin(), times.end(), s.dt, obs);
            break;
        default:
            throw std::logic_error("integrate_on_grid: unhandled adaptive method");
    }
    if (k != grid.n)
        throw std::logic_error("integrate_on_grid: observer called " + std::to_string(k) + " times for " +
                               std::to_string(grid.n) + " samples");
}

vector<double> Lattice::record_magnetizations(ODEState x0, const TimeGrid& grid, const DriveSchedule& drive,
                                              const DynamicsSettings& settings,
                                              const std::function<void()>& on_sample) const {
    const size_t D3 = 3 * spin_dim;
    vector<double> out(grid.n * D3);
    vector<double> scratch(N_atoms * spin_dim);
    integrate_on_grid(x0, grid, drive, settings, [&](const double* x, size_t k, double t) {
        double* m = out.data() + k * D3;
        measure_magnetizations(x, m, scratch.data());
        // Every spin enters the sums, so a NaN/Inf anywhere shows up here.
        for (size_t c = 0; c < D3; ++c) {
            if (!std::isfinite(m[c]))
                throw std::runtime_error("spin dynamics diverged (non-finite state at t = " + std::to_string(t) +
                                         "); reduce the time step or tighten the tolerances");
        }
        if (on_sample) on_sample();
    });
    return out;
}

Lattice::PumpProbeTrajectory Lattice::to_trajectory(const vector<double>& flat, const TimeGrid& grid) const {
    const Eigen::Index D = Eigen::Index(spin_dim);
    const size_t D3 = 3 * spin_dim;
    if (flat.size() != grid.n * D3)
        throw std::logic_error("to_trajectory: series length does not match the grid");
    PumpProbeTrajectory traj;
    traj.reserve(grid.n);
    for (size_t k = 0; k < grid.n; ++k) {
        const double* m = flat.data() + k * D3;
        traj.push_back({grid[k], {SpinVector(Eigen::Map<const Eigen::VectorXd>(m, D)),
                                  SpinVector(Eigen::Map<const Eigen::VectorXd>(m + D, D)),
                                  SpinVector(Eigen::Map<const Eigen::VectorXd>(m + 2 * D, D))}});
    }
    return traj;
}

Lattice::PumpProbeTrajectory Lattice::trajectory_from_states(
    const std::vector<std::pair<double, std::vector<double>>>& raw) const {
    PumpProbeTrajectory traj;
    traj.reserve(raw.size());
    for (const auto& [t, x] : raw) traj.push_back({t, measure_magnetizations(x.data())});
    return traj;
}

Lattice::PumpProbeTrajectory Lattice::drive_trajectory(const DriveSchedule& drive, const TimeGrid& grid,
                                                       const DynamicsSettings& settings) const {
    return to_trajectory(record_magnetizations(spins_to_state(spins), grid, drive, settings), grid);
}

// ============================================================
// Molecular dynamics
// ============================================================

namespace {

#ifdef HDF5_ENABLED
// H5::Exception does not derive from std::exception: convert it, so callers
// (trial loops, MPI error agreement) see every I/O failure.
template<class F>
void hdf5_io(const char* what, F&& f) {
    try {
        f();
    } catch (const H5::Exception& e) {
        throw std::runtime_error(std::string(what) + ": HDF5: " + e.getDetailMsg());
    }
}

// Diagnostics and grid metadata appended to the trajectory file written by
// HDF5MDWriter (which owns the /trajectory datasets it creates).
void append_md_diagnostics(const std::string& file, const std::vector<double>& energy,
                           const std::vector<double>& norm_err, double dt_save,
                           double alpha, double temperature, const std::string& method,
                           const std::string& damping_form) {
    H5::H5File f(file, H5F_ACC_RDWR);
    H5::Group traj = f.openGroup("/trajectory");
    hsize_t dims[1] = {energy.size()};
    H5::DataSpace space(1, dims);
    traj.createDataSet("energy_density", H5::PredType::NATIVE_DOUBLE, space)
        .write(energy.data(), H5::PredType::NATIVE_DOUBLE);
    traj.createDataSet("max_norm_error", H5::PredType::NATIVE_DOUBLE, space)
        .write(norm_err.data(), H5::PredType::NATIVE_DOUBLE);
    H5::Group meta = f.openGroup("/metadata");
    H5::DataSpace scalar(H5S_SCALAR);
    auto attr = [&](const char* name, double v) {
        meta.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar).write(H5::PredType::NATIVE_DOUBLE, &v);
    };
    attr("dt_save", dt_save);
    attr("alpha_gilbert", alpha);
    attr("langevin_temperature", temperature);
    auto str_attr = [&](const char* name, const std::string& v) {
        H5::StrType str_type(H5::PredType::C_S1, v.size() + 1);
        meta.createAttribute(name, str_type, scalar).write(str_type, v.c_str());
    };
    str_attr("integrator", method);
    str_attr("damping_form", damping_form);
}
#endif

}  // namespace

void Lattice::molecular_dynamics(double T_start, double T_end, double dt_initial,
                                 string out_dir, size_t save_interval,
                                 string method, bool use_gpu,
                                 double abs_tol, double rel_tol) {
    namespace dyn = classical_spin::dynamics;
    if (save_interval == 0) throw std::invalid_argument("molecular_dynamics: save_interval must be >= 1");
    if (!(dt_initial > 0.0)) throw std::invalid_argument("molecular_dynamics: time step must be positive");
    const OdeMethod m = dyn::parse_ode_method(method);
    if (use_gpu) {
#ifdef CUDA_ENABLED
        if (gpu_usable(*this)) {
            molecular_dynamics_gpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
            return;
        }
#else
        warn_no_gpu();
#endif
    }
#ifndef HDF5_ENABLED
    if (!out_dir.empty())
        throw std::runtime_error("molecular_dynamics: HDF5 support is required for trajectory output");
#endif
    const TimeGrid grid = TimeGrid::covering(T_start, T_end, double(save_interval) * dt_initial,
                                             "molecular_dynamics output grid");
    const double default_tol = (m == OdeMethod::BulirschStoer) ? 1e-8 : 1e-6;
    const DynamicsSettings settings{method, dt_initial,
                                    abs_tol > 0.0 ? abs_tol : default_tol,
                                    rel_tol > 0.0 ? rel_tol : default_tol, 0.0};

    if (!out_dir.empty()) std::filesystem::create_directories(out_dir);
    cout << "Molecular dynamics: t = " << T_start << " -> " << grid.t_end() << ", integrator "
         << dyn::ode_method_name(m) << ", dt = " << dt_initial << ", " << grid.n
         << " samples every " << grid.dt;
    if (alpha_gilbert > 0.0) cout << ", alpha = " << alpha_gilbert;
    if (langevin_temperature > 0.0) cout << ", Langevin T = " << langevin_temperature;
    cout << endl;

#ifdef HDF5_ENABLED
    const string h5_file = out_dir.empty() ? string() : out_dir + "/trajectory.h5";
    std::unique_ptr<HDF5MDWriter> writer;
    if (!out_dir.empty()) {
        hdf5_io("molecular_dynamics: creating the trajectory file", [&] {
            writer = std::make_unique<HDF5MDWriter>(h5_file, lattice_size, spin_dim, N_atoms, dim1, dim2, dim3,
                                                    method, dt_initial, T_start, T_end, save_interval,
                                                    spin_length, &site_positions, grid.n);
        });
    }
#endif
    ODEState x = spins_to_state(spins);
    const size_t D = spin_dim;
    vector<double> energy(grid.n), norm_err(grid.n), mags(3 * D), scratch(N_atoms * D);
    const size_t report_every = std::max<size_t>(1, grid.n / 10);
    const double s_len = double(spin_length);

    integrate_on_grid(x, grid, active_drive, settings, [&](const double* xs, size_t k, double t) {
        measure_magnetizations(xs, mags.data(), scratch.data());
        energy[k] = total_energy_flat(xs) / double(lattice_size);
        if (!std::isfinite(energy[k]))
            throw std::runtime_error("molecular_dynamics diverged (non-finite energy at t = " + std::to_string(t) +
                                     "); reduce the time step or tighten the tolerances");
        double err = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            double n2 = 0.0;
            for (size_t d = 0; d < D; ++d) n2 += xs[i * D + d] * xs[i * D + d];
            err = std::max(err, std::abs(std::sqrt(n2) - s_len));
        }
        norm_err[k] = err;
#ifdef HDF5_ENABLED
        if (writer) {
            const Eigen::Index Di = Eigen::Index(D);
            hdf5_io("molecular_dynamics: writing a sample", [&] {
                writer->write_flat_step(t, Eigen::Map<const Eigen::VectorXd>(mags.data(), Di),
                                        Eigen::Map<const Eigen::VectorXd>(mags.data() + D, Di),
                                        Eigen::Map<const Eigen::VectorXd>(mags.data() + 2 * D, Di), xs);
            });
        }
#endif
        if (k % report_every == 0 || k + 1 == grid.n) {
            cout << "  t = " << t << "  E/N = " << std::setprecision(12) << energy[k]
                 << "  max||S|-s| = " << std::setprecision(3) << err << std::setprecision(6) << endl;
        }
    });

#ifdef HDF5_ENABLED
    if (writer) {
        hdf5_io("molecular_dynamics: finishing the trajectory file", [&] {
            writer->close();
            writer.reset();
            append_md_diagnostics(h5_file, energy, norm_err, grid.dt, alpha_gilbert, langevin_temperature,
                                  dyn::ode_method_name(m),
                                  damping_form == dyn::DampingForm::Gilbert ? "gilbert" : "landau_lifshitz");
        });
        cout << "Trajectory written to " << h5_file << " (" << grid.n << " samples)" << endl;
    }
#endif
    if (!out_dir.empty()) {
        // Final configuration, in the format of save_spin_config, so runs can be chained.
        std::ofstream fs(out_dir + "/final_spins.txt");
        fs << std::scientific << std::setprecision(16);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < D; ++d) fs << x[i * D + d] << (d + 1 < D ? " " : "\n");
        }
    }
}

// ---- Lattice::collect_energy_samples ----
    vector<double> Lattice::collect_energy_samples(size_t n_samples, size_t interval,
                                         double T, bool gaussian_move, double& sigma,
                                         size_t overrelaxation_rate) {
        vector<double> energies;
        energies.reserve(n_samples / interval + 1);

        for (size_t i = 0; i < n_samples; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation(T);
                if (i % overrelaxation_rate == 0) {
                    metropolis(T, gaussian_move, sigma);
                }
            } else {
                metropolis(T, gaussian_move, sigma);
            }

            if (i % interval == 0) {
                energies.push_back(total_energy(spins));
            }
        }

        return energies;
    }

// ============================================================
// Pulse drives
// ============================================================

Lattice::PumpProbeTrajectory Lattice::single_pulse_drive(
    const vector<SpinVector>& field_in, double t_B,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size,
    string method, bool use_gpu, bool /*pulse_window_chunking*/,
    double abs_tol, double rel_tol) {
    if (use_gpu) {
#ifdef CUDA_ENABLED
        if (gpu_usable(*this))
            return single_pulse_drive_gpu(field_in, t_B, pulse_amp, pulse_width, pulse_freq,
                                          T_start, T_end, step_size, method);
#else
        warn_no_gpu();
#endif
    }
    DriveSchedule drive = make_drive();
    add_pulse(drive, field_in, Pulse{t_B, pulse_amp, pulse_width, pulse_freq});
    return drive_trajectory(drive, TimeGrid::covering(T_start, T_end, step_size, "single_pulse_drive"),
                            DynamicsSettings{method, step_size, abs_tol, rel_tol, 0.0});
}

Lattice::PumpProbeTrajectory Lattice::double_pulse_drive(
    const vector<SpinVector>& field_in_1, double t_B_1,
    const vector<SpinVector>& field_in_2, double t_B_2,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size,
    string method, bool use_gpu, bool /*pulse_window_chunking*/,
    double abs_tol, double rel_tol) {
    if (use_gpu) {
#ifdef CUDA_ENABLED
        if (gpu_usable(*this))
            return double_pulse_drive_gpu(field_in_1, t_B_1, field_in_2, t_B_2,
                                          pulse_amp, pulse_width, pulse_freq,
                                          T_start, T_end, step_size, method);
#else
        warn_no_gpu();
#endif
    }
    DriveSchedule drive = make_drive();
    add_pulse(drive, field_in_1, Pulse{t_B_1, pulse_amp, pulse_width, pulse_freq});
    add_pulse(drive, field_in_2, Pulse{t_B_2, pulse_amp, pulse_width, pulse_freq});
    return drive_trajectory(drive, TimeGrid::covering(T_start, T_end, step_size, "double_pulse_drive"),
                            DynamicsSettings{method, step_size, abs_tol, rel_tol, 0.0});
}

// ============================================================
// Pump-probe / 2DCS delay scans
// ============================================================

void Lattice::synthesize_probe_response(const vector<double>& ref, long ref_j0, const double* baseline,
                                        long shift, size_t n, double* out) const {
    const size_t D3 = 3 * spin_dim;
    const long n_ref = long(ref.size() / D3);
    for (size_t k = 0; k < n; ++k) {
        const long j = long(k) - shift - ref_j0;   // index into ref of the time t_k - τ
        const double* src;
        if (j < 0) {
            src = baseline;
        } else if (j < n_ref) {
            src = ref.data() + size_t(j) * D3;
        } else {
            throw std::logic_error("synthesize_probe_response: reference trajectory too short");
        }
        std::copy_n(src, D3, out + k * D3);
    }
}

#ifdef HDF5_ENABLED
std::unique_ptr<HDF5PumpProbeWriter> Lattice::open_scan_writer(const PumpProbeScanSpec& spec, double E_ground,
                                                              const SpinVector& M_ground) const {
    std::filesystem::create_directories(spec.dir_name);
    return std::make_unique<HDF5PumpProbeWriter>(
        spec.dir_name + "/pump_probe_spectroscopy.h5",
        lattice_size, spin_dim, N_atoms, dim1, dim2, dim3, spin_length,
        spec.amp, spec.width, spec.freq,
        spec.grid.t0, spec.T_end_requested, spec.grid.dt, spec.settings.method,
        spec.tau_start, spec.tau_end, spec.tau_step,
        E_ground, M_ground, spec.Temp_start, spec.Temp_end, spec.n_anneal,
        spec.T_zero_quench, spec.quench_sweeps,
        &spec.field, &site_positions);
}
#endif

Lattice::PumpProbeScanSpec Lattice::make_scan_spec(
    const vector<SpinVector>& field_in, double pulse_amp, double pulse_width, double pulse_freq,
    double tau_start, double tau_end, double tau_step,
    double T_start, double T_end, double T_step,
    double Temp_start, double Temp_end, size_t n_anneal, bool T_zero_quench, size_t quench_sweeps,
    const string& dir_name, const string& method, bool reuse_m0_for_m1, double stationarity_tol,
    double abs_tol, double rel_tol) const {
    namespace dyn = classical_spin::dynamics;
    require_dynamics_dim(spin_dim, "pump_probe_spectroscopy");
    dyn::parse_ode_method(method);
    (void) local_polarisation(field_in);   // validates the pulse directions
    if (!(pulse_width > 0.0) || !std::isfinite(pulse_width))
        throw std::invalid_argument("pump_probe_spectroscopy: pulse width must be positive");
    if (!(stationarity_tol >= 0.0))
        throw std::invalid_argument("pump_probe_spectroscopy: stationarity_tol must be >= 0");
    if (langevin_temperature > 0.0)
        throw std::invalid_argument("pump_probe_spectroscopy: delay scans need deterministic dynamics "
                                    "(M_NL = M01 - M0 - M1 compares separate trajectories); set "
                                    "langevin_temperature = 0");
    PumpProbeScanSpec spec;
    spec.field = field_in;
    spec.amp = pulse_amp;
    spec.width = pulse_width;
    spec.freq = pulse_freq;
    spec.taus = dyn::delay_grid(tau_start, tau_end, tau_step, "pump-probe delay scan");
    spec.tau_start = tau_start;
    spec.tau_end = tau_end;
    spec.tau_step = tau_step;
    spec.grid = TimeGrid::covering(T_start, T_end, T_step, "pump-probe time grid");
    spec.settings = DynamicsSettings{method, T_step, abs_tol, rel_tol, 0.0};
    spec.reuse_m0_for_m1 = reuse_m0_for_m1;
    spec.stationarity_tol = stationarity_tol;
    spec.T_end_requested = T_end;
    spec.Temp_start = Temp_start;
    spec.Temp_end = Temp_end;
    spec.n_anneal = n_anneal;
    spec.T_zero_quench = T_zero_quench;
    spec.quench_sweeps = quench_sweeps;
    spec.dir_name = dir_name;
    return spec;
}

/**
 * State shared by every delay of one scan: the ground state, its
 * magnetisation baseline, M0 and the W1 reference.
 *
 * W1 (M1 by time translation). A probe at τ started from the stationary
 * ground state g at T_start is, shifted by τ, the response R to a pulse at 0
 * started from g at T_start - τ. R(t) does not depend on the start time as
 * long as the run starts from g before the pulse acts, i.e. before
 * -9 w (DriveSchedule support). Hence, for every τ with
 *     τ / T_step integer   and   T_start <= τ - 9 w,
 *     M1(t_k; τ) = R(t_k - τ)   (= baseline for t_k - τ before R starts),
 * where R is integrated ONCE from the last grid time <= -9 w (or from T_start
 * when T_start <= -9 w, in which case its first n samples ARE M0) up to
 * t_end - τ_min. The old code shifted M0 itself, which is wrong whenever the
 * pump at 0 is not entirely inside the window (T_start = 0 cut it in half),
 * clamped negative τ to the last M0 sample, rounded τ to the grid and used a
 * different "antiferro" baseline; delays that do not qualify are now simply
 * integrated.
 */
struct Lattice::PumpProbeScan {
    const Lattice& lat;
    const PumpProbeScanSpec& spec;
    size_t D3 = 0;
    ODEState ground;
    vector<double> baseline;
    vector<char> synth;          // per delay: M1 synthesised from the reference
    vector<long> shift;          // τ / T_step for synthesised delays
    vector<double> M0;
    vector<double> ref;          // W1 reference R, sample j at t0 + (ref_j0 + j) dt
    long ref_j0 = 0;
    double residual = 0.0;
    bool w1 = false;

    PumpProbeScan(const Lattice& l, const PumpProbeScanSpec& s, bool verbose) : lat(l), spec(s) {
        const TimeGrid& grid = spec.grid;
        const size_t n_tau = spec.taus.size();
        D3 = 3 * lat.spin_dim;
        ground = lat.spins_to_state(lat.spins);
        baseline.resize(D3);
        vector<double> scratch(lat.N_atoms * lat.spin_dim);
        lat.measure_magnetizations(ground.data(), baseline.data(), scratch.data());

        // ---- W1 eligibility ----
        const double half = classical_spin::dynamics::kPulseSupportWidths * spec.width;
        synth.assign(n_tau, 0);
        shift.assign(n_tau, 0);
        w1 = spec.reuse_m0_for_m1;
        if (w1) {
            residual = lat.stationarity_residual();
            if (residual > spec.stationarity_tol) {
                w1 = false;
                if (verbose)
                    cout << "  [W1] off: ground state not stationary (torque residual " << residual
                         << " > stationarity_tol " << spec.stationarity_tol << ")" << endl;
            }
        }
        if (w1) {
            for (size_t i = 0; i < n_tau && w1; ++i) {
                const double r = spec.taus[i] / grid.dt;
                shift[i] = std::lround(r);
                if (std::abs(r - double(shift[i])) > 1e-6) {
                    w1 = false;
                    if (verbose)
                        cout << "  [W1] off: delay " << spec.taus[i] << " is not a multiple of T_step "
                             << grid.dt << endl;
                }
            }
        }
        size_t n_synth = 0;
        long m_min = std::numeric_limits<long>::max();
        if (w1) {
            for (size_t i = 0; i < n_tau; ++i) {
                synth[i] = (grid.t0 <= spec.taus[i] - half) ? 1 : 0;
                if (synth[i]) { ++n_synth; m_min = std::min(m_min, shift[i]); }
            }
        }
        if (verbose && spec.reuse_m0_for_m1 && w1)
            cout << "  [W1] M1 synthesised for " << n_synth << " of " << n_tau
                 << " delays (the others start inside the probe window and are integrated)" << endl;

        // ---- M0 and the reference ----
        DriveSchedule pump = lat.make_drive();
        lat.add_pulse(pump, spec.field, Pulse{0.0, spec.amp, spec.width, spec.freq});
        const long n = long(grid.n);
        if (n_synth > 0 && grid.t0 <= -half) {
            const long j_hi = std::max(n - 1, n - 1 - m_min);
            const TimeGrid rgrid{grid.t0, grid.dt, size_t(j_hi + 1)};
            ref = lat.record_magnetizations(ground, rgrid, pump, spec.settings);
            ref_j0 = 0;
            M0.assign(ref.begin(), ref.begin() + grid.n * D3);
        } else {
            M0 = lat.record_magnetizations(ground, grid, pump, spec.settings);
            if (n_synth > 0) {
                ref_j0 = long(std::floor((-half - grid.t0) / grid.dt));
                const long j_hi = n - 1 - m_min;
                if (j_hi >= ref_j0) {
                    const TimeGrid rgrid{grid.t0 + double(ref_j0) * grid.dt, grid.dt, size_t(j_hi - ref_j0 + 1)};
                    ref = lat.record_magnetizations(ground, rgrid, pump, spec.settings);
                }
            }
        }
    }

    DriveSchedule probe(double tau) const {
        DriveSchedule d = lat.make_drive();
        lat.add_pulse(d, spec.field, Pulse{tau, spec.amp, spec.width, spec.freq});
        return d;
    }
    DriveSchedule pump_and_probe(double tau) const {
        DriveSchedule d = lat.make_drive();
        lat.add_pulse(d, spec.field, Pulse{0.0, spec.amp, spec.width, spec.freq});
        lat.add_pulse(d, spec.field, Pulse{tau, spec.amp, spec.width, spec.freq});
        return d;
    }

    /// M1 and M01 of delay i into M1, M01 (grid.n * D3 values each).
    void compute(size_t i, double* M1, double* M01, const std::function<void()>& on_sample) const {
        const double tau = spec.taus[i];
        const size_t len = spec.grid.n * D3;
        if (synth[i]) {
            lat.synthesize_probe_response(ref, ref_j0, baseline.data(), shift[i], spec.grid.n, M1);
        } else {
            const vector<double> m1 = lat.record_magnetizations(ground, spec.grid, probe(tau), spec.settings,
                                                                on_sample);
            std::copy_n(m1.data(), len, M1);
        }
        const vector<double> m01 = lat.record_magnetizations(ground, spec.grid, pump_and_probe(tau),
                                                             spec.settings, on_sample);
        std::copy_n(m01.data(), len, M01);
    }
};

namespace {

// Every rank throws together when any rank failed (no rank left in a collective).
void agree_or_throw(MPI_Comm comm, const std::string& local_error, const char* what) {
    int failed = local_error.empty() ? 0 : 1, any = 0;
    MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, comm);
    if (any) {
        throw std::runtime_error(local_error.empty()
                                     ? std::string(what) + " failed on another rank"
                                     : std::string(what) + ": " + local_error);
    }
}

}  // namespace

void Lattice::run_pump_probe_scan(const PumpProbeScanSpec& spec, MPI_Comm comm_in, int outer_omp_threads) const {
    MPI_Comm comm;
    MPI_Comm_dup(comm_in, &comm);   // private tag space for the scheduler
    struct CommGuard { MPI_Comm& c; ~CommGuard() { MPI_Comm_free(&c); } } guard{comm};
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const bool root = (rank == 0);
    const size_t n_tau = spec.taus.size();
    const size_t n = spec.grid.n;

    if (root) {
        cout << "\n=== Pump-probe delay scan ===\n"
             << "  pulse: A = " << spec.amp << ", w = " << spec.width << ", omega = " << spec.freq << "\n"
             << "  delays: " << n_tau << " (" << spec.taus.front() << " .. " << spec.taus.back() << ")\n"
             << "  grid: " << n << " samples, t = " << spec.grid.t0 << " .. " << spec.grid.t_end()
             << " step " << spec.grid.dt << "\n"
             << "  integrator: " << spec.settings.method << ", ranks: " << size << endl;
    }

    // ---- Setup (identical on every rank) and the writer on rank 0 ----
    std::unique_ptr<PumpProbeScan> scan;
#ifdef HDF5_ENABLED
    std::unique_ptr<HDF5PumpProbeWriter> writer;
#endif
    std::string error;
    try {
        scan = std::make_unique<PumpProbeScan>(*this, spec, root);
        if (root) {
            std::filesystem::create_directories(spec.dir_name);
            save_positions(spec.dir_name + "/positions.txt");
            save_spin_config(spec.dir_name + "/initial_spins.txt");
#ifdef HDF5_ENABLED
            writer = open_scan_writer(spec, energy_density(), magnetization_local());
            writer->write_reference_trajectory(to_trajectory(scan->M0, spec.grid));
#else
            std::cerr << "Warning: built without HDF5; pump-probe results are not written." << std::endl;
#endif
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
#ifdef HDF5_ENABLED
    catch (const H5::Exception& e) {
        error = "HDF5: " + e.getDetailMsg();
    }
#endif
    agree_or_throw(comm, error, "pump-probe scan setup");

    const size_t D3 = scan->D3;
    std::string failure;   // first failure seen by rank 0 (or by the OpenMP loop)
    size_t written = 0;
    // Called by one thread at a time on rank 0. HDF5 errors (not std::exceptions)
    // are converted so that every caller's handler sees them.
    auto write = [&](size_t i, const double* M1, const double* M01) {
#ifdef HDF5_ENABLED
        if (writer) {
            const vector<double> m1(M1, M1 + n * D3), m01(M01, M01 + n * D3);
            try {
                writer->write_tau_trajectory(int(i), spec.taus[i], to_trajectory(m1, spec.grid),
                                             to_trajectory(m01, spec.grid));
            } catch (const H5::Exception& e) {
                throw std::runtime_error("writing delay " + std::to_string(spec.taus[i]) + ": HDF5: " +
                                         e.getDetailMsg());
            }
        }
#else
        (void) i; (void) M1; (void) M01;
#endif
        if (++written % std::max<size_t>(1, n_tau / 10) == 0 || written == n_tau)
            cout << "  delays done: " << written << "/" << n_tau << endl;
    };

    if (size == 1) {
        // ---- Shared-memory path: OpenMP over delays, the lattice is read-only ----
        int n_outer = 1;
#ifdef _OPENMP
        n_outer = (outer_omp_threads <= 0) ? omp_get_max_threads() : outer_omp_threads;
        n_outer = std::max(1, std::min<int>(n_outer, int(n_tau)));
        // Inner (per-site) parallel loops run serially inside the τ threads.
        const int saved_levels = omp_get_max_active_levels();
        if (n_outer > 1) omp_set_max_active_levels(1);
#else
        (void) outer_omp_threads;
#endif
        auto one = [&](size_t i, vector<double>& M1, vector<double>& M01) {
            try {
                scan->compute(i, M1.data(), M01.data(), nullptr);
#ifdef _OPENMP
                #pragma omp critical(lattice_pump_probe_write)
#endif
                write(i, M1.data(), M01.data());
            } catch (const std::exception& e) {
#ifdef _OPENMP
                #pragma omp critical(lattice_pump_probe_error)
#endif
                if (failure.empty()) failure = "delay " + std::to_string(spec.taus[i]) + ": " + e.what();
            }
        };
        if (n_outer > 1) {
#ifdef _OPENMP
            #pragma omp parallel num_threads(n_outer)
            {
                vector<double> M1(n * D3), M01(n * D3);
                #pragma omp for schedule(dynamic, 1)
                for (long i = 0; i < long(n_tau); ++i) one(size_t(i), M1, M01);
            }
#endif
        } else {
            vector<double> M1(n * D3), M01(n * D3);
            for (size_t i = 0; i < n_tau && failure.empty(); ++i) one(i, M1, M01);
        }
#ifdef _OPENMP
        omp_set_max_active_levels(saved_levels);
#endif
    } else {
        // ---- MPI path: dynamic self-scheduling, rank 0 computes too ----
        // Workers ask for a delay index, return [index, M1, M01] and receive
        // the next index (-1 = stop). Rank 0 serves requests between its own
        // delays and from the observer of its own integrations (MPI_Iprobe
        // per sample), so a worker never waits for more than one sample of
        // rank-0 work. Every received result is written at once.
        constexpr int TAG_REQUEST = 7101, TAG_ASSIGN = 7102, TAG_RESULT = 7103, TAG_FAILURE = 7104;
        const size_t payload = 1 + 2 * n * D3;
        if (payload > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("pump-probe scan: trajectory too long for one MPI message");
        if (root) {
            size_t next = 0;
            int stopped = 0;
            vector<double> rbuf(payload);
            auto assign = [&](int dest) {
                int idx = (failure.empty() && next < n_tau) ? int(next++) : -1;
                if (idx < 0) ++stopped;
                MPI_Send(&idx, 1, MPI_INT, dest, TAG_ASSIGN, comm);
            };
            auto handle = [&](const MPI_Status& st) {
                const int src = st.MPI_SOURCE;
                if (st.MPI_TAG == TAG_RESULT) {
                    MPI_Recv(rbuf.data(), int(payload), MPI_DOUBLE, src, TAG_RESULT, comm, MPI_STATUS_IGNORE);
                    try {
                        write(size_t(rbuf[0]), rbuf.data() + 1, rbuf.data() + 1 + n * D3);
                    } catch (const std::exception& e) {
                        if (failure.empty()) failure = std::string("writing results: ") + e.what();
                    }
                } else if (st.MPI_TAG == TAG_FAILURE) {
                    int len = 0;
                    MPI_Get_count(&st, MPI_CHAR, &len);
                    std::string msg(size_t(len), '\0');
                    MPI_Recv(msg.data(), len, MPI_CHAR, src, TAG_FAILURE, comm, MPI_STATUS_IGNORE);
                    if (failure.empty()) failure = msg;
                } else {
                    int dummy = 0;
                    MPI_Recv(&dummy, 1, MPI_INT, src, st.MPI_TAG, comm, MPI_STATUS_IGNORE);
                }
                assign(src);   // always answer, even after a failure (then with -1)
            };
            auto serve = [&](bool block) {
                for (;;) {
                    MPI_Status st;
                    int flag = 0;
                    if (block) {
                        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &st);
                        flag = 1;
                        block = false;
                    } else {
                        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &st);
                    }
                    if (!flag) return;
                    handle(st);
                }
            };
            vector<double> M1(n * D3), M01(n * D3);
            size_t own = 0;
            for (;;) {
                serve(false);
                if (failure.empty() && next < n_tau) {
                    const size_t i = next++;
                    try {
                        scan->compute(i, M1.data(), M01.data(), [&] { serve(false); });
                        write(i, M1.data(), M01.data());
                        ++own;
                    } catch (const std::exception& e) {
                        if (failure.empty())
                            failure = "rank 0, delay " + std::to_string(spec.taus[i]) + ": " + e.what();
                    }
                } else if (stopped < size - 1) {
                    serve(true);
                } else {
                    break;
                }
            }
            cout << "  rank 0 computed " << own << " of " << n_tau << " delays" << endl;
        } else {
            vector<double> buf(payload);
            int idx = 0;
            MPI_Send(&idx, 1, MPI_INT, 0, TAG_REQUEST, comm);
            for (;;) {
                MPI_Recv(&idx, 1, MPI_INT, 0, TAG_ASSIGN, comm, MPI_STATUS_IGNORE);
                if (idx < 0) break;
                try {
                    buf[0] = double(idx);
                    scan->compute(size_t(idx), buf.data() + 1, buf.data() + 1 + n * D3, nullptr);
                    MPI_Send(buf.data(), int(payload), MPI_DOUBLE, 0, TAG_RESULT, comm);
                } catch (const std::exception& e) {
                    const std::string msg = "rank " + std::to_string(rank) + ", delay " +
                                            std::to_string(spec.taus[size_t(idx)]) + ": " + e.what();
                    MPI_Send(msg.data(), int(msg.size()), MPI_CHAR, 0, TAG_FAILURE, comm);
                }
            }
        }
    }

    // ---- Finish on rank 0, then one collective verdict ----
    if (root) {
        try {
#ifdef HDF5_ENABLED
            if (writer) {
                writer->close();
                writer.reset();
                // Provenance: which M1 were synthesised (W1) rather than integrated.
                H5::H5File f(spec.dir_name + "/pump_probe_spectroscopy.h5", H5F_ACC_RDWR);
                hsize_t dims[1] = {n_tau};
                vector<int> flags(scan->synth.begin(), scan->synth.end());
                f.openGroup("/tau_scan").createDataSet("m1_synthesized", H5::PredType::NATIVE_INT,
                                                       H5::DataSpace(1, dims))
                    .write(flags.data(), H5::PredType::NATIVE_INT);
                cout << "  written: " << spec.dir_name << "/pump_probe_spectroscopy.h5" << endl;
            }
#endif
        } catch (const std::exception& e) {
            if (failure.empty()) failure = std::string("closing the output file: ") + e.what();
        }
#ifdef HDF5_ENABLED
        catch (const H5::Exception& e) {
            if (failure.empty()) failure = "closing the output file: HDF5: " + e.getDetailMsg();
        }
#endif
    }
    int failed = failure.empty() ? 0 : 1;
    MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
    if (failed) {
        int len = int(failure.size());
        MPI_Bcast(&len, 1, MPI_INT, 0, comm);
        std::string msg(size_t(len), '\0');
        if (root) msg = failure;
        MPI_Bcast(msg.data(), len, MPI_CHAR, 0, comm);
        throw std::runtime_error("pump-probe scan failed: " + msg);
    }
}

void Lattice::broadcast_dynamical_state(MPI_Comm comm, int root) {
    const size_t D = spin_dim, D2 = D * D;
    vector<double> buf(lattice_size * D + 3 * D2 + 3 + 2);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        size_t o = 0;
        for (size_t i = 0; i < lattice_size; ++i)
            for (size_t d = 0; d < D; ++d) buf[o++] = spins[i](d);
        for (size_t a = 0; a < 3; ++a)
            for (size_t r = 0; r < D; ++r)
                for (size_t c = 0; c < D; ++c) buf[o++] = twist_matrices[a](r, c);
        for (size_t a = 0; a < 3; ++a) buf[o++] = twist_angles[a];
        buf[o++] = alpha_gilbert;
        buf[o++] = langevin_temperature;
    }
    MPI_Bcast(buf.data(), int(buf.size()), MPI_DOUBLE, root, comm);
    if (rank != root) {
        size_t o = 0;
        for (size_t i = 0; i < lattice_size; ++i)
            for (size_t d = 0; d < D; ++d) spins[i](d) = buf[o++];
        for (size_t a = 0; a < 3; ++a)
            for (size_t r = 0; r < D; ++r)
                for (size_t c = 0; c < D; ++c) twist_matrices[a](r, c) = buf[o++];
        for (size_t a = 0; a < 3; ++a) twist_angles[a] = buf[o++];
        alpha_gilbert = buf[o++];
        langevin_temperature = buf[o++];
        sync_twist_state();
    }
}

void Lattice::pump_probe_spectroscopy(const vector<SpinVector>& field_in,
                                      double pulse_amp, double pulse_width, double pulse_freq,
                                      double tau_start, double tau_end, double tau_step,
                                      double T_start, double T_end, double T_step,
                                      double Temp_start, double Temp_end, size_t n_anneal,
                                      bool T_zero_quench, size_t quench_sweeps,
                                      string dir_name, string method, bool use_gpu,
                                      bool reuse_m0_for_m1, double stationarity_tol,
                                      int outer_omp_threads, bool /*pulse_window_chunking*/,
                                      double abs_tol, double rel_tol) {
    const PumpProbeScanSpec spec = make_scan_spec(
        field_in, pulse_amp, pulse_width, pulse_freq, tau_start, tau_end, tau_step,
        T_start, T_end, T_step, Temp_start, Temp_end, n_anneal, T_zero_quench, quench_sweeps,
        dir_name, method, reuse_m0_for_m1, stationarity_tol, abs_tol, rel_tol);
    if (use_gpu) {
#ifdef CUDA_ENABLED
        if (gpu_usable(*this)) {
            pump_probe_spectroscopy_gpu_batched(spec);
            return;
        }
#else
        warn_no_gpu();
#endif
    }
    run_pump_probe_scan(spec, MPI_COMM_SELF, outer_omp_threads);
}

void Lattice::pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in,
                                          double pulse_amp, double pulse_width, double pulse_freq,
                                          double tau_start, double tau_end, double tau_step,
                                          double T_start, double T_end, double T_step,
                                          double Temp_start, double Temp_end, size_t n_anneal,
                                          bool T_zero_quench, size_t quench_sweeps,
                                          string dir_name, string method, bool use_gpu,
                                          bool reuse_m0_for_m1, double stationarity_tol,
                                          bool /*pulse_window_chunking*/,
                                          double abs_tol, double rel_tol, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    // Rank 0's configuration (and twist state, damping) defines the scan, and
    // rank 0 decides the backend (its GPU runs the batched path alone).
    broadcast_dynamical_state(comm, 0);
    int gpu_flag = use_gpu ? 1 : 0;
    MPI_Bcast(&gpu_flag, 1, MPI_INT, 0, comm);
    use_gpu = (gpu_flag != 0);

    PumpProbeScanSpec spec;
    std::string error;
    try {
        spec = make_scan_spec(field_in, pulse_amp, pulse_width, pulse_freq, tau_start, tau_end, tau_step,
                              T_start, T_end, T_step, Temp_start, Temp_end, n_anneal, T_zero_quench,
                              quench_sweeps, dir_name, method, reuse_m0_for_m1, stationarity_tol,
                              abs_tol, rel_tol);
    } catch (const std::exception& e) {
        error = e.what();
    }
    agree_or_throw(comm, error, "pump_probe_spectroscopy_mpi");

    if (use_gpu) {
#ifdef CUDA_ENABLED
        // One batched GPU launch on rank 0 replaces the τ distribution. The
        // capability test depends only on broadcast state: every rank agrees.
        std::string reason;
        if (!gpu_supports_model(reason)) {
            if (rank == 0)
                std::cerr << "Warning: the GPU right-hand side does not support " << reason
                          << "; running the delay scan on the CPU." << std::endl;
        } else {
            std::string gpu_error;
            if (rank == 0) {
                try { pump_probe_spectroscopy_gpu_batched(spec); }
                catch (const std::exception& e) { gpu_error = e.what(); }
#ifdef HDF5_ENABLED
                catch (const H5::Exception& e) { gpu_error = "HDF5: " + e.getDetailMsg(); }
#endif
                catch (...) { gpu_error = "unknown error in the GPU batched scan"; }
            }
            int failed = gpu_error.empty() ? 0 : 1;
            MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
            if (failed)
                throw std::runtime_error(rank == 0 ? gpu_error : "GPU batched pump-probe scan failed on rank 0");
            return;
        }
#else
        if (rank == 0) warn_no_gpu();
#endif
    }
    run_pump_probe_scan(spec, comm, 1);
}

// ============================================================
// Dynamical structure factor
// ============================================================

array<Eigen::Vector3d, 3> Lattice::reciprocal_vectors() const {
    Eigen::Matrix3d A;
    for (int i = 0; i < 3; ++i) A.col(i) = unit_cell.lattice_vectors[size_t(i)];
    if (std::abs(A.determinant()) < 1e-12)
        throw std::invalid_argument("reciprocal_vectors: the lattice vectors are linearly dependent");
    const Eigen::Matrix3d B = 2.0 * M_PI * A.inverse().transpose();   // a_i . b_j = 2π δ_ij
    return {Eigen::Vector3d(B.col(0)), Eigen::Vector3d(B.col(1)), Eigen::Vector3d(B.col(2))};
}

Lattice::DSSFResult Lattice::dynamical_structure_factor(const DSSFSettings& s) {
    namespace dyn = classical_spin::dynamics;
    using cplx = std::complex<double>;
    if (spin_dim != 3) throw std::invalid_argument("dynamical_structure_factor: spin_dim must be 3");
    if (s.q_points.empty()) throw std::invalid_argument("dynamical_structure_factor: no q points");
    if (s.n_samples == 0 || s.save_every == 0 || !(s.dt > 0.0) || !(s.t_max > 0.0) ||
        !(s.temperature >= 0.0) || !(s.alpha_sampling > 0.0) || s.t_equilibrate < 0.0 || s.t_decorrelate < 0.0)
        throw std::invalid_argument("dynamical_structure_factor: need n_samples, save_every >= 1, dt, t_max, "
                                    "alpha_sampling > 0 and non-negative temperature and times");
    dyn::parse_ode_method(s.method);

    const size_t N = lattice_size, n_q = s.q_points.size();
    const TimeGrid grid = TimeGrid::covering(0.0, s.t_max, double(s.save_every) * s.dt, "DSSF time grid");
    dyn::DSSFAccumulator acc(n_q, grid.n, grid.dt, s.hann_window);
    // Phase table e^{-i q.r_i} / sqrt(N) and the global frames per site.
    vector<cplx> phase(n_q * N);
    const double inv_sqrt_n = 1.0 / std::sqrt(double(N));
    for (size_t q = 0; q < n_q; ++q)
        for (size_t i = 0; i < N; ++i) phase[q * N + i] = std::polar(inv_sqrt_n, -s.q_points[q].dot(site_positions[i]));

    // Damping/bath are switched per stage and restored on every exit path.
    struct Restore {
        Lattice& l; double a, T; dyn::DampingForm f;
        ~Restore() { l.alpha_gilbert = a; l.langevin_temperature = T; l.damping_form = f; }
    } restore{*this, alpha_gilbert, langevin_temperature, damping_form};
    const DriveSchedule no_drive = make_drive();
    auto thermalise = [&](double t) {
        if (s.temperature <= 0.0 || t <= 0.0) return;
        alpha_gilbert = s.alpha_sampling;
        langevin_temperature = s.temperature;
        damping_form = dyn::DampingForm::LandauLifshitz;
        ODEState x = spins_to_state(spins);
        integrate_on_grid(x, TimeGrid{0.0, t, 2}, no_drive, DynamicsSettings{"spherical_midpoint", s.dt},
                          [](const double*, size_t, double) {});
        spins = state_to_spins(x);
    };

    thermalise(s.t_equilibrate);
    vector<cplx> A(n_q * grid.n * 3);
    vector<double> g(3 * N);
    for (size_t sample = 0; sample < s.n_samples; ++sample) {
        if (sample > 0) thermalise(s.t_decorrelate);
        alpha_gilbert = 0.0;
        langevin_temperature = 0.0;
        ODEState x = spins_to_state(spins);
        integrate_on_grid(x, grid, no_drive, DynamicsSettings{s.method, s.dt, 1e-10, 1e-10, 0.0},
                          [&](const double* xs, size_t k, double) {
            for (size_t i = 0; i < N; ++i) {   // global-frame spins
                const SpinMatrix& F = sublattice_frames[i % N_atoms];
                for (int a = 0; a < 3; ++a)
                    g[3 * i + a] = F(a, 0) * xs[3 * i] + F(a, 1) * xs[3 * i + 1] + F(a, 2) * xs[3 * i + 2];
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(n_q * N >= 65536)
#endif
            for (size_t q = 0; q < n_q; ++q) {
                cplx Aq[3] = {0.0, 0.0, 0.0};
                const cplx* ph = phase.data() + q * N;
                for (size_t i = 0; i < N; ++i)
                    for (int a = 0; a < 3; ++a) Aq[a] += ph[i] * g[3 * i + a];
                for (int a = 0; a < 3; ++a) A[(q * grid.n + k) * 3 + a] = Aq[a];
            }
        });
        acc.add_sample(A);
    }

    DSSFResult r;
    r.q = s.q_points;
    r.n_samples = acc.samples();
    r.temperature = s.temperature;
    r.dt_sample = grid.dt;
    r.t_max = grid.t_end();
    const size_t M = acc.n_omega();
    r.omega.resize(M);
    for (size_t j = 0; j < M; ++j) r.omega[j] = acc.omega(j);
    r.S.resize(n_q * M * 9);
    r.S_err.resize(n_q * M * 9);
    r.S_static.resize(n_q * 9);
    for (size_t q = 0; q < n_q; ++q) {
        for (size_t j = 0; j < M; ++j)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    const size_t idx = ((q * M + j) * 3 + a) * 3 + b;
                    r.S[idx] = acc.S(q, j, a, b);
                    r.S_err[idx] = acc.S_err(q, j, a, b);
                }
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) r.S_static[(q * 3 + a) * 3 + b] = acc.S_static(q, a, b);
    }
    return r;
}

void Lattice::write_dssf(const DSSFResult& r, const string& file) {
#ifdef HDF5_ENABLED
    hdf5_io("write_dssf", [&] {
        H5::H5File f(file, H5F_ACC_TRUNC);
        H5::Group g = f.createGroup("/dssf");
        const hsize_t nq = r.q.size(), nw = r.omega.size();
        auto dataset = [&](const char* name, const vector<double>& v, std::initializer_list<hsize_t> dims) {
            vector<hsize_t> d(dims);
            g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(int(d.size()), d.data()))
                .write(v.data(), H5::PredType::NATIVE_DOUBLE);
        };
        vector<double> q(nq * 3);
        for (size_t i = 0; i < nq; ++i)
            for (int a = 0; a < 3; ++a) q[i * 3 + a] = r.q[i](a);
        dataset("q", q, {nq, 3});
        dataset("omega", r.omega, {nw});
        vector<double> re(r.S.size()), im(r.S.size());
        for (size_t i = 0; i < r.S.size(); ++i) { re[i] = r.S[i].real(); im[i] = r.S[i].imag(); }
        dataset("S_re", re, {nq, nw, 3, 3});
        dataset("S_im", im, {nq, nw, 3, 3});
        dataset("S_err", r.S_err, {nq, nw, 3, 3});
        vector<double> sre(r.S_static.size()), sim(r.S_static.size());
        for (size_t i = 0; i < r.S_static.size(); ++i) { sre[i] = r.S_static[i].real(); sim[i] = r.S_static[i].imag(); }
        dataset("S_static_re", sre, {nq, 3, 3});
        dataset("S_static_im", sim, {nq, 3, 3});
        if (r.temperature > 0.0) {
            // Classical-to-quantum intensity factor βω / (1 - e^{-βω}) (detailed balance).
            vector<double> c2q(nw);
            for (size_t j = 0; j < nw; ++j) {
                const double x = r.omega[j] / r.temperature;
                c2q[j] = (std::abs(x) < 1e-12) ? 1.0 : x / (1.0 - std::exp(-x));
            }
            dataset("classical_to_quantum", c2q, {nw});
        }
        H5::DataSpace scalar(H5S_SCALAR);
        auto attr = [&](const char* name, double v) {
            g.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar).write(H5::PredType::NATIVE_DOUBLE, &v);
        };
        attr("temperature", r.temperature);
        attr("n_samples", double(r.n_samples));
        attr("dt_sample", r.dt_sample);
        attr("t_max", r.t_max);
    });
#else
    (void) r;
    throw std::runtime_error("write_dssf: HDF5 support is required (" + file + ")");
#endif
}
