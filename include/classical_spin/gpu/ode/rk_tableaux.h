#pragma once
/**
 * rk_tableaux.h — Butcher tableaux and method names of the GPU integrators.
 *
 * Plain C++ (no CUDA keywords): the device steppers in integrator.cuh read
 * these constants on the host to build their stage kernels, and
 * tests/test_gpu_tableaux.cpp runs the SAME constants through a CPU stepper
 * and measures the convergence order, so the GPU coefficients are verified
 * without a GPU.
 *
 * Embedded pairs (b propagates, b_hat estimates the error):
 *   - Fehlberg 7(8): E. Fehlberg, NASA TR R-287 (1968), Table X; Hairer,
 *     Nørsett & Wanner, Solving ODEs I, Table 5.4. 13 stages, the 8th-order
 *     solution is propagated (as Boost.Odeint runge_kutta_fehlberg78), the
 *     7th-order one (11 stages) estimates the error. The previous GPU code
 *     swapped a53 and a54 (order ~5) and propagated the 7th-order weights.
 *   - Dormand-Prince 5(4): J. R. Dormand & P. J. Prince, J. Comput. Appl.
 *     Math. 6, 19 (1980); 7 stages (FSAL), 5th order propagated.
 *   - Cash-Karp 5(4): J. R. Cash & A. H. Karp, ACM TOMS 16, 201 (1990);
 *     6 stages, 5th order propagated.
 */

#include <stdexcept>
#include <string>
#include <string_view>

namespace gpu {
namespace ode {

/** Explicit Runge-Kutta tableau with an optional embedded solution. */
struct ExplicitTableau {
    static constexpr int kMaxStages = 13;
    int stages;
    int order;           ///< order of the propagated solution b
    int embedded_order;  ///< order of b_hat (0: no embedded solution)
    double c[kMaxStages];
    double a[kMaxStages][kMaxStages];
    double b[kMaxStages];
    double b_hat[kMaxStages];
};

inline constexpr ExplicitTableau kFehlberg78 = {
    13, 8, 7,
    {0.0, 2.0 / 27.0, 1.0 / 9.0, 1.0 / 6.0, 5.0 / 12.0, 1.0 / 2.0, 5.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0, 1.0,
     0.0, 1.0},
    {
        {},
        {2.0 / 27.0},
        {1.0 / 36.0, 1.0 / 12.0},
        {1.0 / 24.0, 0.0, 1.0 / 8.0},
        {5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0},
        {1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0},
        {-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0},
        {31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0},
        {2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0},
        {-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0,
         -1.0 / 12.0},
        {2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -301.0 / 82.0, 2133.0 / 4100.0, 45.0 / 82.0,
         45.0 / 164.0, 18.0 / 41.0},
        {3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0, 3.0 / 41.0, 6.0 / 41.0, 0.0},
        {-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -289.0 / 82.0, 2193.0 / 4100.0, 51.0 / 82.0,
         33.0 / 164.0, 12.0 / 41.0, 0.0, 1.0},
    },
    {0.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0, 0.0, 41.0 / 840.0,
     41.0 / 840.0},
    {41.0 / 840.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0, 41.0 / 840.0,
     0.0, 0.0},
};

inline constexpr ExplicitTableau kDormandPrince54 = {
    7, 5, 4,
    {0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0},
    {
        {},
        {1.0 / 5.0},
        {3.0 / 40.0, 9.0 / 40.0},
        {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0},
        {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0},
        {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0},
        {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0},
    },
    {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0},
    {5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0},
};

inline constexpr ExplicitTableau kCashKarp54 = {
    6, 5, 4,
    {0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0},
    {
        {},
        {1.0 / 5.0},
        {3.0 / 40.0, 9.0 / 40.0},
        {3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0},
        {-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0},
        {1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0},
    },
    {37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0},
    {2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0},
};

/** Classical 4th-order Runge-Kutta (no embedded solution). */
inline constexpr ExplicitTableau kRK4 = {
    4, 4, 0,
    {0.0, 0.5, 0.5, 1.0},
    {{}, {0.5}, {0.0, 0.5}, {0.0, 0.0, 1.0}},
    {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},
    {},
};

/**
 * Integrators available on the GPU. Fixed-step methods take steps of the
 * given dt; the error-controlled ones (embedded pairs) adapt the step to
 * abs_tol / rel_tol like the CPU drivers, with dt as the initial step.
 */
enum class Method { Euler, RK2, RK4, SSPRK53, CashKarp54, Dopri5, Fehlberg78 };

inline bool is_adaptive(Method m) {
    return m == Method::CashKarp54 || m == Method::Dopri5 || m == Method::Fehlberg78;
}

/** Tableau of an error-controlled method (nullptr for the hand-fused fixed-step ones). */
inline const ExplicitTableau* adaptive_tableau(Method m) {
    switch (m) {
        case Method::CashKarp54: return &kCashKarp54;
        case Method::Dopri5:     return &kDormandPrince54;
        case Method::Fehlberg78: return &kFehlberg78;
        default:                 return nullptr;
    }
}

inline const char* gpu_method_names() {
    return "euler, rk2 (midpoint), rk4, ssprk53 (fixed step); rk5 (rkck54, rk54, rkf54), dopri5, "
           "rk78 (rkf78) (error-controlled)";
}

/**
 * Parse a GPU integrator name. Names the CPU accepts but the GPU does not
 * implement (bulirsch_stoer, Adams methods, geometric integrators) and
 * unknown names throw std::invalid_argument — they used to fall back silently
 * to a different method.
 */
inline Method parse_method(std::string_view name) {
    if (name == "euler") return Method::Euler;
    if (name == "rk2" || name == "midpoint") return Method::RK2;
    if (name == "rk4") return Method::RK4;
    if (name == "ssprk53") return Method::SSPRK53;
    if (name == "rk5" || name == "rkck54" || name == "rk54" || name == "rkf54") return Method::CashKarp54;
    if (name == "dopri5") return Method::Dopri5;
    if (name == "rk78" || name == "rkf78") return Method::Fehlberg78;
    if (name == "bulirsch_stoer" || name == "bs")
        throw std::invalid_argument("GPU integrator '" + std::string(name) +
                                    "' is not available: the GPU has no Bulirsch-Stoer extrapolation (the old "
                                    "'bs' was a 2nd-order midpoint step). Use dopri5 or rk78 (error-controlled) "
                                    "on the GPU, or run bulirsch_stoer on the CPU (use_gpu = false)");
    throw std::invalid_argument("GPU integrator '" + std::string(name) + "' is not available on the GPU (valid: " +
                                gpu_method_names() + "); run it on the CPU (use_gpu = false)");
}

}  // namespace ode
}  // namespace gpu
