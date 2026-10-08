#ifndef CLASSICAL_SPIN_GPU_ODE_INTEGRATOR_CUH
#define CLASSICAL_SPIN_GPU_ODE_INTEGRATOR_CUH

// =============================================================================
// gpu::ode -- in-house, header-only GPU ODE integrator module.
//
// One physics-agnostic implementation of the explicit Runge-Kutta family,
// shared by every GPU time propagator (single lattice, mixed lattice, batched
// 2DCS).
//
// Design
// ------
//   * State          : thrust::device_vector<double>  (flat GPU array).
//   * System concept : any callable with
//                           void operator()(const State& in, State& out, double t)
//                       that evaluates dS/dt = f(t, S) on the device. A single
//                       trajectory (LLG over lattice_size*spin_dim) and a batched
//                       ensemble (LLG over B*N*spin_dim) plug into the SAME
//                       steppers -- only the RHS functor differs.
//   * Workspace      : owns the reusable stage buffers (k1..k13, tmp, and the
//                       candidate / error vectors of the adaptive driver).
//   * Methods        : parsed once into gpu::ode::Method (rk_tableaux.h);
//                       unsupported names throw. Fixed step: euler, rk2, rk4,
//                       ssprk53 (hand-fused kernels). Error-controlled: rk5
//                       (Cash-Karp 5(4)), dopri5, rk78 (Fehlberg 7(8)), driven
//                       by the shared tableau constants of rk_tableaux.h, which
//                       tests/test_gpu_tableaux.cpp verifies on the CPU.
//   * Output grid    : integrate() samples t_k = T_start + k * dt_out computed
//                       from the integer k (dt_out = save_interval * dt), with
//                       as many samples as fit into [T_start, T_end] -- the
//                       TimeGrid::covering grid of the CPU drivers. The old
//                       loop accumulated t += dt and took one step past T_end.
// =============================================================================

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <iostream>

#include "classical_spin/gpu/gpu_common_helpers.cuh"
#include "classical_spin/gpu/ode/rk_tableaux.h"
#include "classical_spin/dynamics/time_grid.h"

namespace gpu {
namespace ode {

using State = thrust::device_vector<double>;

// -----------------------------------------------------------------------------
// Workspace: reusable RK stage buffers. `ensure` grows (never shrinks) the
// buffers needed for the requested number of stages.
// -----------------------------------------------------------------------------
struct Workspace {
    State k1, k2, k3, k4, k5, k6, k7;
    State k8, k9, k10, k11, k12, k13;
    State tmp;
    State ynew, err;   // adaptive driver: candidate solution and error estimate

    void ensure(size_t n, int stages, bool adaptive = false) {
        auto grow = [&](State& a) { if (a.size() < n) a.resize(n, 0.0); };
        grow(k1); grow(k2); grow(tmp);
        if (stages >= 3)  { grow(k3); }
        if (stages >= 4)  { grow(k4); }
        if (stages >= 5)  { grow(k5); }
        if (stages >= 6)  { grow(k6); }
        if (stages >= 7)  { grow(k7); }
        if (stages >= 8)  { grow(k8); }
        if (stages >= 9)  { grow(k9); }
        if (stages >= 10) { grow(k10); }
        if (stages >= 11) { grow(k11); }
        if (stages >= 12) { grow(k12); }
        if (stages >= 13) { grow(k13); }
        // The candidate is swapped with the state, so it must match it exactly.
        if (ynew.size() != n) ynew.resize(n, 0.0);
        if (adaptive) grow(err);
    }

    State& stage(int i) {
        State* k[13] = {&k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9, &k10, &k11, &k12, &k13};
        return *k[i];
    }
};

// max_i |err_i| / (atol + rtol (|y_i| + h |k1_i|)): the error norm of
// Boost.Odeint's default_error_checker (max norm), so a tolerance means the
// same on the GPU as in the CPU drivers. Non-finite ratios map to 1e300 (a
// rejected step) so that thrust::maximum cannot drop a NaN.
struct ErrorRatio {
    const double* err;
    const double* y;
    const double* k1;
    double atol, rtol, h;
    __host__ __device__ double operator()(size_t i) const {
        const double r = fabs(err[i]) / (atol + rtol * (fabs(y[i]) + h * fabs(k1[i])));
        return isfinite(r) ? r : 1e300;
    }
};

// One step of an explicit tableau from y: candidate in ws.ynew; with
// `with_error`, returns the error norm of the embedded pair (else 0).
template <class System>
double tableau_step(System& system, State& y, double t, double h, const ExplicitTableau& tab,
                    Workspace& ws, bool with_error, double atol, double rtol) {
    const size_t n = y.size();
    ws.ensure(n, tab.stages, with_error);
    const int BLOCK_SIZE = 256;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535))));
    const double* d_y = thrust::raw_pointer_cast(y.data());
    double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

    auto row = [&](const double* coef, int m_terms) {
        RKCombination comb{};
        comb.m = 0;
        for (int j = 0; j < m_terms; ++j) {
            if (coef[j] == 0.0) continue;
            comb.k[comb.m] = thrust::raw_pointer_cast(ws.stage(j).data());
            comb.coef[comb.m] = coef[j];
            ++comb.m;
        }
        return comb;
    };

    system(y, ws.stage(0), t);
    for (int i = 1; i < tab.stages; ++i) {
        ::rk_lincomb_kernel<<<grid, block>>>(d_tmp, d_y, 1.0, row(tab.a[i], i), h, n);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());
        system(ws.tmp, ws.stage(i), t + tab.c[i] * h);
    }
    ::rk_lincomb_kernel<<<grid, block>>>(thrust::raw_pointer_cast(ws.ynew.data()), d_y, 1.0,
                                         row(tab.b, tab.stages), h, n);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    if (!with_error) return 0.0;

    double e[ExplicitTableau::kMaxStages];
    for (int j = 0; j < tab.stages; ++j) e[j] = tab.b[j] - tab.b_hat[j];
    double* d_err = thrust::raw_pointer_cast(ws.err.data());
    ::rk_lincomb_kernel<<<grid, block>>>(d_err, nullptr, 0.0, row(e, tab.stages), h, n);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    const ErrorRatio ratio{d_err, d_y, thrust::raw_pointer_cast(ws.k1.data()), atol, rtol, h};
    return thrust::transform_reduce(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(n),
                                    ratio, 0.0, thrust::maximum<double>());
}

// -----------------------------------------------------------------------------
// step: advance `state` by ONE step of size `dt` (no error control; an
// error-controlled method takes one step of its propagated solution).
// -----------------------------------------------------------------------------
template <class System>
void step(System& system, State& state, double t, double dt, Method method, Workspace& ws) {
    const size_t array_size = state.size();

    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(static_cast<unsigned>((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE));

    double* d_state = thrust::raw_pointer_cast(state.data());

    switch (method) {
    case Method::Euler: {
        // y_{n+1} = y_n + h * f(t_n, y_n)  (1st order, 1 eval, 1 sync)
        ws.ensure(array_size, 2);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());

        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, dt, array_size);
        cudaDeviceSynchronize();
        break;
    }
    case Method::RK2: {
        // Modified midpoint (2nd order, 2 evals, 2 syncs)
        ws.ensure(array_size, 2);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2 = thrust::raw_pointer_cast(ws.k2.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k2, t + 0.5 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k2, dt, array_size);
        cudaDeviceSynchronize();
        break;
    }
    case Method::RK4: {
        // Classic RK4 (4th order, 4 evals, 4 syncs via fused final update)
        ws.ensure(array_size, 4);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2 = thrust::raw_pointer_cast(ws.k2.data());
        double* d_k3 = thrust::raw_pointer_cast(ws.k3.data());
        double* d_k4 = thrust::raw_pointer_cast(ws.k4.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        system(state, ws.k1, t);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k2, t + 0.5 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k2, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k3, t + 0.5 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k3, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k4, t + dt);

        ::rk4_final_update_kernel<<<grid, block>>>(d_state, d_k1, d_k2, d_k3, d_k4, dt / 6.0, array_size);
        cudaDeviceSynchronize();
        break;
    }
    case Method::SSPRK53: {
        // Strong-Stability-Preserving RK, 5 stages, 3rd order
        constexpr double a30 = 0.355909775063327;
        constexpr double a32 = 0.644090224936674;
        constexpr double a40 = 0.367933791638137;
        constexpr double a43 = 0.632066208361863;
        constexpr double a52 = 0.237593836598569;
        constexpr double a54 = 0.762406163401431;
        constexpr double b10 = 0.377268915331368;
        constexpr double b21 = 0.377268915331368;
        constexpr double b32 = 0.242995220537396;
        constexpr double b43 = 0.238458932846290;
        constexpr double b54 = 0.287632146308408;
        constexpr double c1 = 0.377268915331368;
        constexpr double c2 = 0.754537830662736;
        constexpr double c3 = 0.728985661612188;
        constexpr double c4 = 0.699226135931670;

        ws.ensure(array_size, 2);
        double* d_k   = thrust::raw_pointer_cast(ws.k1.data());        // k
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());       // tmp
        double* d_u   = thrust::raw_pointer_cast(ws.k2.data());        // u

        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k, b10 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k1, t + c1 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_u, d_tmp, 1.0, d_k, b21 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.k2, ws.k1, t + c2 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a30, d_u, a32, d_k, b32 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k1, t + c3 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a40, d_tmp, a43, d_k, b43 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k1, t + c4 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_state, d_u, a52, d_tmp, a54, d_k, b54 * dt, array_size);
        cudaDeviceSynchronize();
        break;
    }
    case Method::CashKarp54:
    case Method::Dopri5:
    case Method::Fehlberg78:
        tableau_step(system, state, t, dt, *adaptive_tableau(method), ws, false, 0.0, 0.0);
        state.swap(ws.ynew);
        break;
    }
    GPU_CHECK(cudaGetLastError());
}

template <class System>
void step(System& system, State& state, double t, double dt, const std::string& method, Workspace& ws) {
    step(system, state, t, dt, parse_method(method), ws);
}

// -----------------------------------------------------------------------------
// advance_adaptive: integrate from t0 to exactly t1 with an embedded pair,
// adapting the step h (in/out: the step to try next) to the tolerances.
// Accepted when the error norm (ErrorRatio) is <= 1; the step factor is
// 0.9 err^(-1/(q+1)) clamped to [0.2, 5], q the embedded order (Hairer,
// Norsett & Wanner I, Sec. II.4). The last step is shortened to land on t1
// (a shortened step does not shrink h). Throws std::runtime_error when h
// underflows (non-finite right-hand side or an unreachable tolerance).
// -----------------------------------------------------------------------------
template <class System>
void advance_adaptive(System& system, State& y, double t0, double t1, double& h, const ExplicitTableau& tab,
                      double atol, double rtol, Workspace& ws) {
    const double q = double(std::min(tab.order, tab.embedded_order));
    const double t_scale = std::max({1.0, std::abs(t0), std::abs(t1)});
    double t = t0;
    while (t1 - t > 1e-13 * t_scale) {
        const bool last = (h >= t1 - t);
        const double hs = last ? (t1 - t) : h;
        const double err = tableau_step(system, y, t, hs, tab, ws, true, atol, rtol);
        const double fac = (err > 0.0) ? 0.9 * std::pow(err, -1.0 / (q + 1.0)) : 5.0;
        if (err <= 1.0) {
            y.swap(ws.ynew);
            t = last ? t1 : t + hs;
            const double h_new = hs * std::min(5.0, std::max(0.2, fac));
            h = last ? std::max(h, h_new) : h_new;
        } else {
            h = hs * std::max(0.2, std::min(1.0, fac));
            if (!(h > 1e-14 * t_scale))
                throw std::runtime_error("gpu::ode: step size underflow at t = " + std::to_string(t) +
                                         " (error norm " + std::to_string(err) +
                                         "); the right-hand side is not finite or the tolerance unreachable");
        }
    }
}

// -----------------------------------------------------------------------------
// integrate: sample the solution on t_k = T_start + k * dt_out,
// dt_out = save_interval * dt, k = 0 .. n-1 (n = the samples that fit into
// [T_start, T_end], as TimeGrid::covering), calling `observe(t_k, state)` at
// each. Fixed-step methods take exactly save_interval steps of dt between
// samples; error-controlled methods integrate adaptively between samples (dt
// is the initial step) with the tolerances abs_tol / rel_tol.
// -----------------------------------------------------------------------------
template <class System, class Observer>
void integrate(System& system, State& state, double T_start, double T_end,
               double dt, size_t save_interval, Observer&& observe,
               const std::string& method, Workspace& ws,
               double abs_tol = 1e-8, double rel_tol = 1e-8) {
    const Method m = parse_method(method);
    if (save_interval == 0) throw std::invalid_argument("gpu::ode::integrate: save_interval must be >= 1");
    if (is_adaptive(m) && !(abs_tol > 0.0 && rel_tol >= 0.0))
        throw std::invalid_argument("gpu::ode::integrate: abs_tol must be > 0 and rel_tol >= 0");
    const classical_spin::dynamics::TimeGrid grid = classical_spin::dynamics::TimeGrid::covering(
        T_start, T_end, double(save_interval) * dt, "gpu::ode::integrate");
    observe(grid[0], state);
    double h = dt;
    for (size_t k = 1; k < grid.n; ++k) {
        const double t0 = grid[k - 1];
        if (is_adaptive(m)) {
            advance_adaptive(system, state, t0, grid[k], h, *adaptive_tableau(m), abs_tol, rel_tol, ws);
        } else {
            for (size_t s = 0; s < save_interval; ++s) step(system, state, t0 + double(s) * dt, dt, m, ws);
        }
        observe(grid[k], state);
    }
    GPU_CHECK_KERNEL();
}

} // namespace ode
} // namespace gpu

#endif // CLASSICAL_SPIN_GPU_ODE_INTEGRATOR_CUH
